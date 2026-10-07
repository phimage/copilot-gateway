//! Minimal Agent Client Protocol (ACP) client over the stdio of a child
//! process (newline delimited JSON-RPC 2.0).
//!
//! See <https://agentclientprotocol.com>.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, info, warn};

pub const PROTOCOL_VERSION: u64 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum PermissionPolicy {
    /// Reject every tool permission request made by the agent.
    Deny,
    /// Approve every tool permission request made by the agent (dangerous).
    Allow,
}

#[derive(Debug, Clone)]
pub struct AgentCommand {
    pub program: PathBuf,
    pub args: Vec<String>,
    pub env: Vec<(String, String)>,
    pub cwd: PathBuf,
}

#[derive(Debug)]
pub struct RpcError {
    pub code: i64,
    pub message: String,
    pub data: Option<Value>,
}

impl std::fmt::Display for RpcError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} (code {})", self.message, self.code)?;
        if let Some(d) = &self.data {
            write!(f, ": {d}")?;
        }
        Ok(())
    }
}

impl std::error::Error for RpcError {}

type Pending = Arc<Mutex<HashMap<u64, oneshot::Sender<Result<Value, RpcError>>>>>;
type Sessions = Arc<Mutex<HashMap<String, mpsc::UnboundedSender<Value>>>>;

/// A running ACP agent process.
pub struct AcpProcess {
    writer: mpsc::UnboundedSender<String>,
    pending: Pending,
    sessions: Sessions,
    next_id: AtomicU64,
    alive: Arc<AtomicBool>,
    child: Mutex<Option<Child>>,
    close_stdin: Mutex<Option<oneshot::Sender<()>>>,
    closing: Arc<AtomicBool>,
    /// Result of the `initialize` request.
    pub init: Value,
}

impl AcpProcess {
    pub async fn spawn(cmd: &AgentCommand, policy: PermissionPolicy) -> Result<Arc<Self>> {
        info!(program = %cmd.program.display(), args = ?cmd.args, "starting ACP agent");
        let mut command = Command::new(&cmd.program);
        command
            .args(&cmd.args)
            .current_dir(&cmd.cwd)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);
        for (k, v) in &cmd.env {
            command.env(k, v);
        }
        detach_from_terminal(&mut command);
        let mut child = command
            .spawn()
            .with_context(|| format!("failed to start `{}`", cmd.program.display()))?;

        let stdin = child.stdin.take().context("no stdin")?;
        let stdout = child.stdout.take().context("no stdout")?;
        let stderr = child.stderr.take().context("no stderr")?;

        let (writer, mut write_rx) = mpsc::unbounded_channel::<String>();
        let pending: Pending = Arc::default();
        let sessions: Sessions = Arc::default();
        let alive = Arc::new(AtomicBool::new(true));
        let closing = Arc::new(AtomicBool::new(false));

        // Writer task. Closing stdin (EOF) asks the agent to exit.
        let (close_tx, mut close_rx) = oneshot::channel::<()>();
        tokio::spawn(async move {
            let mut stdin = stdin;
            loop {
                let line = tokio::select! {
                    line = write_rx.recv() => line,
                    _ = &mut close_rx => None,
                };
                let Some(line) = line else { break };
                if stdin.write_all(line.as_bytes()).await.is_err()
                    || stdin.write_all(b"\n").await.is_err()
                    || stdin.flush().await.is_err()
                {
                    break;
                }
            }
        });

        tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                debug!(target: "copilot", "{line}");
            }
        });

        {
            let pending = pending.clone();
            let sessions = sessions.clone();
            let alive = alive.clone();
            let closing = closing.clone();
            let writer = writer.clone();
            tokio::spawn(async move {
                let mut lines = BufReader::new(stdout).lines();
                loop {
                    match lines.next_line().await {
                        Ok(Some(line)) => {
                            handle_line(&line, &pending, &sessions, &writer, policy);
                        }
                        Ok(None) => break,
                        Err(e) => {
                            warn!("error reading agent stdout: {e}");
                            break;
                        }
                    }
                }
                if closing.load(Ordering::SeqCst) {
                    debug!("ACP agent exited");
                } else {
                    warn!("ACP agent exited unexpectedly (see debug logs for its stderr)");
                }
                alive.store(false, Ordering::SeqCst);
                for (_, tx) in pending.lock().unwrap().drain() {
                    let _ = tx.send(Err(RpcError {
                        code: -32000,
                        message: "agent process exited".into(),
                        data: None,
                    }));
                }
                sessions.lock().unwrap().clear();
            });
        }

        let mut process = AcpProcess {
            writer,
            pending,
            sessions,
            next_id: AtomicU64::new(1),
            alive,
            child: Mutex::new(Some(child)),
            close_stdin: Mutex::new(Some(close_tx)),
            closing,
            init: Value::Null,
        };

        let init = process
            .request_timeout(
                "initialize",
                json!({
                    "protocolVersion": PROTOCOL_VERSION,
                    "clientCapabilities": {
                        "fs": {"readTextFile": false, "writeTextFile": false},
                        "terminal": false
                    },
                    "clientInfo": {
                        "name": "copilot-gateway",
                        "title": "Copilot Gateway",
                        "version": env!("CARGO_PKG_VERSION")
                    }
                }),
                Duration::from_secs(120),
            )
            .await
            .context("ACP initialize failed")?;
        let agent = init.get("agentInfo").cloned().unwrap_or_default();
        info!(%agent, "ACP agent initialized");
        process.init = init;
        Ok(Arc::new(process))
    }

    pub fn is_alive(&self) -> bool {
        self.alive.load(Ordering::SeqCst)
    }

    pub fn supports_close(&self) -> bool {
        self.init
            .pointer("/agentCapabilities/sessionCapabilities/close")
            .is_some_and(|v| !v.is_null())
    }

    pub fn supports_images(&self) -> bool {
        self.init
            .pointer("/agentCapabilities/promptCapabilities/image")
            .and_then(Value::as_bool)
            .unwrap_or(false)
    }

    fn send(&self, msg: Value) -> Result<()> {
        self.writer
            .send(msg.to_string())
            .map_err(|_| anyhow!("agent process is not running"))
    }

    pub async fn request(&self, method: &str, params: Value) -> Result<Value> {
        if !self.is_alive() {
            bail!("agent process is not running");
        }
        let id = self.next_id.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = oneshot::channel();
        self.pending.lock().unwrap().insert(id, tx);
        if let Err(e) = self.send(json!({"jsonrpc": "2.0", "id": id, "method": method, "params": params})) {
            self.pending.lock().unwrap().remove(&id);
            return Err(e);
        }
        match rx.await {
            Ok(Ok(v)) => Ok(v),
            Ok(Err(e)) => Err(anyhow::Error::new(e).context(format!("ACP `{method}` failed"))),
            Err(_) => bail!("ACP `{method}`: agent process exited"),
        }
    }

    pub async fn request_timeout(&self, method: &str, params: Value, timeout: Duration) -> Result<Value> {
        tokio::time::timeout(timeout, self.request(method, params))
            .await
            .map_err(|_| anyhow!("ACP `{method}` timed out after {timeout:?}"))?
    }

    pub fn notify(&self, method: &str, params: Value) -> Result<()> {
        self.send(json!({"jsonrpc": "2.0", "method": method, "params": params}))
    }

    /// Subscribe to `session/update` notifications of a session.
    pub fn subscribe(&self, session_id: &str) -> mpsc::UnboundedReceiver<Value> {
        let (tx, rx) = mpsc::unbounded_channel();
        self.sessions.lock().unwrap().insert(session_id.to_string(), tx);
        rx
    }

    pub fn unsubscribe(&self, session_id: &str) {
        self.sessions.lock().unwrap().remove(session_id);
    }

    /// Ask the agent to exit by closing its stdin, then kill it if needed.
    pub async fn shutdown(&self) {
        let child = self.child.lock().unwrap().take();
        let Some(mut child) = child else { return };
        self.closing.store(true, Ordering::SeqCst);
        if let Some(close) = self.close_stdin.lock().unwrap().take() {
            let _ = close.send(());
        }
        match tokio::time::timeout(Duration::from_secs(2), child.wait()).await {
            Ok(_) => {}
            Err(_) => {
                let _ = child.kill().await;
            }
        }
    }
}

impl Drop for AcpProcess {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.lock().unwrap().take() {
            let _ = child.start_kill();
        }
    }
}

fn handle_line(
    line: &str,
    pending: &Pending,
    sessions: &Sessions,
    writer: &mpsc::UnboundedSender<String>,
    policy: PermissionPolicy,
) {
    let line = line.trim();
    if line.is_empty() {
        return;
    }
    let msg: Value = match serde_json::from_str(line) {
        Ok(v) => v,
        Err(_) => {
            debug!(target: "copilot", "non JSON-RPC output: {line}");
            return;
        }
    };
    let id = msg.get("id").cloned().filter(|v| !v.is_null());
    match (msg.get("method").and_then(Value::as_str), id) {
        (Some(method), Some(id)) => {
            let params = msg.get("params").cloned().unwrap_or(Value::Null);
            let reply = match handle_agent_request(method, &params, policy) {
                Ok(result) => json!({"jsonrpc": "2.0", "id": id, "result": result}),
                Err((code, message)) => {
                    json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}})
                }
            };
            let _ = writer.send(reply.to_string());
        }
        (Some(method), None) => {
            if method == "session/update" {
                let params = msg.get("params").cloned().unwrap_or(Value::Null);
                let sid = params.get("sessionId").and_then(Value::as_str).unwrap_or("");
                if let Some(update) = params.get("update")
                    && let Some(tx) = sessions.lock().unwrap().get(sid)
                {
                    let _ = tx.send(update.clone());
                }
            } else {
                debug!("ignoring agent notification {method}");
            }
        }
        (None, Some(id)) => {
            let Some(id) = id.as_u64() else { return };
            let Some(tx) = pending.lock().unwrap().remove(&id) else {
                return;
            };
            let result = if let Some(err) = msg.get("error") {
                Err(RpcError {
                    code: err.get("code").and_then(Value::as_i64).unwrap_or(-32000),
                    message: err
                        .get("message")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown error")
                        .to_string(),
                    data: err.get("data").cloned(),
                })
            } else {
                Ok(msg.get("result").cloned().unwrap_or(Value::Null))
            };
            let _ = tx.send(result);
        }
        (None, None) => {}
    }
}

/// Requests sent by the agent to us (the client).
fn handle_agent_request(method: &str, params: &Value, policy: PermissionPolicy) -> Result<Value, (i64, String)> {
    match method {
        "session/request_permission" => {
            let options = params
                .get("options")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            let wanted: &[&str] = match policy {
                PermissionPolicy::Deny => &["reject_once", "reject_always"],
                PermissionPolicy::Allow => &["allow_once", "allow_always"],
            };
            let title = params.pointer("/toolCall/title").and_then(Value::as_str).unwrap_or("?");
            let chosen = wanted.iter().find_map(|kind| {
                options
                    .iter()
                    .find(|o| o.get("kind").and_then(Value::as_str) == Some(kind))
                    .and_then(|o| o.get("optionId").cloned())
            });
            info!(tool = title, ?policy, "agent permission request");
            Ok(match chosen {
                Some(option_id) => {
                    json!({"outcome": {"outcome": "selected", "optionId": option_id}})
                }
                None => json!({"outcome": {"outcome": "cancelled"}}),
            })
        }
        _ => Err((-32601, format!("method not supported by copilot-gateway: {method}"))),
    }
}

/// Keep the agent out of the terminal's foreground process group so that
/// Ctrl+C in a launched tool (Claude Code, Codex...) does not kill it.
fn detach_from_terminal(command: &mut Command) {
    #[cfg(unix)]
    {
        command.process_group(0);
    }
    #[cfg(windows)]
    {
        const CREATE_NEW_PROCESS_GROUP: u32 = 0x0000_0200;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn permission_policy() {
        let params = json!({
            "sessionId": "s",
            "toolCall": {"toolCallId": "t", "title": "Run ls"},
            "options": [
                {"optionId": "a", "name": "Allow", "kind": "allow_once"},
                {"optionId": "r", "name": "Reject", "kind": "reject_once"}
            ]
        });
        let deny = handle_agent_request("session/request_permission", &params, PermissionPolicy::Deny).unwrap();
        assert_eq!(deny["outcome"]["optionId"], "r");
        let allow = handle_agent_request("session/request_permission", &params, PermissionPolicy::Allow).unwrap();
        assert_eq!(allow["outcome"]["optionId"], "a");
        let none = handle_agent_request(
            "session/request_permission",
            &json!({"options": []}),
            PermissionPolicy::Deny,
        )
        .unwrap();
        assert_eq!(none["outcome"]["outcome"], "cancelled");
        assert!(handle_agent_request("fs/read_text_file", &json!({}), PermissionPolicy::Deny).is_err());
    }
}
