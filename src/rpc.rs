//! JSON-RPC 2.0 over the stdio of a child process, shared by the ACP
//! backend (newline delimited messages) and the Copilot SDK backend
//! (`Content-Length` framed messages, as in LSP).

use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::{Context, Result, anyhow, bail};
use serde_json::{Value, json};
use tokio::io::{AsyncBufRead, AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader};
use tokio::process::{Child, Command};
use tokio::sync::{mpsc, oneshot};
use tracing::{debug, info, warn};

#[derive(Debug, Clone)]
pub struct AgentCommand {
    pub program: PathBuf,
    pub args: Vec<String>,
    pub env: Vec<(String, String)>,
    pub cwd: PathBuf,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Framing {
    /// One JSON message per line (ACP).
    Lines,
    /// `Content-Length: N\r\n\r\n` header before each message (LSP style).
    ContentLength,
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

/// Answers requests sent by the child process to us.
pub type RequestHandler = Arc<dyn Fn(&str, &Value) -> Result<Value, (i64, String)> + Send + Sync>;

/// Extracts `(session id, payload)` from a notification to route it to the
/// subscriber of that session.
pub type Router = fn(&str, &Value) -> Option<(String, Value)>;

type Pending = Arc<Mutex<HashMap<u64, oneshot::Sender<Result<Value, RpcError>>>>>;
type Sessions = Arc<Mutex<HashMap<String, mpsc::UnboundedSender<Value>>>>;

/// A child process spoken to with JSON-RPC over stdin/stdout.
pub struct RpcProcess {
    name: &'static str,
    framing: Framing,
    writer: mpsc::UnboundedSender<String>,
    pending: Pending,
    sessions: Sessions,
    next_id: AtomicU64,
    alive: Arc<AtomicBool>,
    child: Mutex<Option<Child>>,
    close_stdin: Mutex<Option<oneshot::Sender<()>>>,
    closing: Arc<AtomicBool>,
}

impl RpcProcess {
    pub fn spawn(
        name: &'static str,
        cmd: &AgentCommand,
        framing: Framing,
        handler: RequestHandler,
        router: Router,
    ) -> Result<Self> {
        info!(program = %cmd.program.display(), args = ?cmd.args, "starting {name}");
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

        // Writer task. Closing stdin (EOF) asks the process to exit.
        let (close_tx, mut close_rx) = oneshot::channel::<()>();
        tokio::spawn(async move {
            let mut stdin = stdin;
            loop {
                let msg = tokio::select! {
                    msg = write_rx.recv() => msg,
                    _ = &mut close_rx => None,
                };
                let Some(msg) = msg else { break };
                let frame = match framing {
                    Framing::Lines => format!("{msg}\n"),
                    Framing::ContentLength => format!("Content-Length: {}\r\n\r\n{msg}", msg.len()),
                };
                if stdin.write_all(frame.as_bytes()).await.is_err() || stdin.flush().await.is_err() {
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
                let mut reader = BufReader::new(stdout);
                loop {
                    match read_message(&mut reader, framing).await {
                        Ok(Some(msg)) => dispatch(&msg, &pending, &sessions, &writer, &handler, router),
                        Ok(None) => break,
                        Err(e) => {
                            warn!("error reading {name} output: {e}");
                            break;
                        }
                    }
                }
                if closing.load(Ordering::SeqCst) {
                    debug!("{name} exited");
                } else {
                    warn!("{name} exited unexpectedly (see debug logs for its stderr)");
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

        Ok(RpcProcess {
            name,
            framing,
            writer,
            pending,
            sessions,
            next_id: AtomicU64::new(1),
            alive,
            child: Mutex::new(Some(child)),
            close_stdin: Mutex::new(Some(close_tx)),
            closing,
        })
    }

    pub fn framing(&self) -> Framing {
        self.framing
    }

    pub fn is_alive(&self) -> bool {
        self.alive.load(Ordering::SeqCst)
    }

    fn send(&self, msg: Value) -> Result<()> {
        self.writer
            .send(msg.to_string())
            .map_err(|_| anyhow!("{} is not running", self.name))
    }

    pub async fn request(&self, method: &str, params: Value) -> Result<Value> {
        if !self.is_alive() {
            bail!("{} is not running", self.name);
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
            Ok(Err(e)) => Err(anyhow::Error::new(e).context(format!("`{method}` failed"))),
            Err(_) => bail!("`{method}`: {} exited", self.name),
        }
    }

    pub async fn request_timeout(&self, method: &str, params: Value, timeout: Duration) -> Result<Value> {
        tokio::time::timeout(timeout, self.request(method, params))
            .await
            .map_err(|_| anyhow!("`{method}` timed out after {timeout:?}"))?
    }

    pub fn notify(&self, method: &str, params: Value) -> Result<()> {
        self.send(json!({"jsonrpc": "2.0", "method": method, "params": params}))
    }

    /// Subscribe to the notifications routed to a session.
    pub fn subscribe(&self, session_id: &str) -> mpsc::UnboundedReceiver<Value> {
        let (tx, rx) = mpsc::unbounded_channel();
        self.sessions.lock().unwrap().insert(session_id.to_string(), tx);
        rx
    }

    pub fn unsubscribe(&self, session_id: &str) {
        self.sessions.lock().unwrap().remove(session_id);
    }

    /// Ask the process to exit by closing its stdin, then kill it if needed.
    pub async fn shutdown(&self) {
        let child = self.child.lock().unwrap().take();
        let Some(mut child) = child else { return };
        self.closing.store(true, Ordering::SeqCst);
        if let Some(close) = self.close_stdin.lock().unwrap().take() {
            let _ = close.send(());
        }
        if tokio::time::timeout(Duration::from_secs(2), child.wait())
            .await
            .is_err()
        {
            let _ = child.kill().await;
        }
    }
}

impl Drop for RpcProcess {
    fn drop(&mut self) {
        if let Some(mut child) = self.child.lock().unwrap().take() {
            let _ = child.start_kill();
        }
    }
}

/// Read one message. `Ok(None)` on end of stream.
pub async fn read_message<R: AsyncBufRead + Unpin>(reader: &mut R, framing: Framing) -> std::io::Result<Option<Value>> {
    loop {
        match framing {
            Framing::Lines => {
                let mut line = String::new();
                if reader.read_line(&mut line).await? == 0 {
                    return Ok(None);
                }
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                match serde_json::from_str(line) {
                    Ok(v) => return Ok(Some(v)),
                    Err(_) => debug!(target: "copilot", "non JSON-RPC output: {line}"),
                }
            }
            Framing::ContentLength => {
                let mut length: Option<usize> = None;
                loop {
                    let mut line = String::new();
                    if reader.read_line(&mut line).await? == 0 {
                        return Ok(None);
                    }
                    let line = line.trim();
                    if line.is_empty() {
                        if length.is_some() {
                            break;
                        }
                        continue;
                    }
                    if let Some((k, v)) = line.split_once(':')
                        && k.trim().eq_ignore_ascii_case("content-length")
                    {
                        length = v.trim().parse().ok();
                    }
                }
                let mut body = vec![0u8; length.unwrap_or(0)];
                reader.read_exact(&mut body).await?;
                match serde_json::from_slice(&body) {
                    Ok(v) => return Ok(Some(v)),
                    Err(e) => debug!(target: "copilot", "invalid JSON-RPC message: {e}"),
                }
            }
        }
    }
}

/// Encode one message with the given framing.
pub fn encode_message(msg: &Value, framing: Framing) -> String {
    let body = msg.to_string();
    match framing {
        Framing::Lines => format!("{body}\n"),
        Framing::ContentLength => format!("Content-Length: {}\r\n\r\n{body}", body.len()),
    }
}

fn dispatch(
    msg: &Value,
    pending: &Pending,
    sessions: &Sessions,
    writer: &mpsc::UnboundedSender<String>,
    handler: &RequestHandler,
    router: Router,
) {
    let id = msg.get("id").cloned().filter(|v| !v.is_null());
    let params = msg.get("params").cloned().unwrap_or(Value::Null);
    match (msg.get("method").and_then(Value::as_str), id) {
        (Some(method), Some(id)) => {
            let reply = match handler(method, &params) {
                Ok(result) => json!({"jsonrpc": "2.0", "id": id, "result": result}),
                Err((code, message)) => {
                    json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}})
                }
            };
            let _ = writer.send(reply.to_string());
        }
        (Some(method), None) => match router(method, &params) {
            Some((sid, payload)) => {
                if let Some(tx) = sessions.lock().unwrap().get(&sid) {
                    let _ = tx.send(payload);
                }
            }
            None => debug!("ignoring notification {method}"),
        },
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

    #[tokio::test]
    async fn framing_roundtrip() {
        for framing in [Framing::Lines, Framing::ContentLength] {
            let a = json!({"jsonrpc": "2.0", "id": 1, "result": {"text": "é\nx"}});
            let b = json!({"jsonrpc": "2.0", "method": "m", "params": {}});
            let data = format!("{}{}", encode_message(&a, framing), encode_message(&b, framing));
            let mut reader = BufReader::new(data.as_bytes());
            assert_eq!(read_message(&mut reader, framing).await.unwrap(), Some(a));
            assert_eq!(read_message(&mut reader, framing).await.unwrap(), Some(b));
            assert_eq!(read_message(&mut reader, framing).await.unwrap(), None);
        }
    }
}
