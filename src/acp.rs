//! Agent Client Protocol (ACP) client over the stdio of a child process
//! (newline delimited JSON-RPC 2.0).
//!
//! See <https://agentclientprotocol.com>.

use std::ops::Deref;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use serde_json::{Value, json};
use tracing::info;

pub use crate::rpc::AgentCommand;
use crate::rpc::{Framing, RpcProcess};

pub const PROTOCOL_VERSION: u64 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum PermissionPolicy {
    /// Reject every tool permission request made by the agent.
    Deny,
    /// Approve every tool permission request made by the agent (dangerous).
    Allow,
}

/// A running ACP agent process.
pub struct AcpProcess {
    rpc: RpcProcess,
    /// Result of the `initialize` request.
    pub init: Value,
}

impl Deref for AcpProcess {
    type Target = RpcProcess;

    fn deref(&self) -> &RpcProcess {
        &self.rpc
    }
}

impl AcpProcess {
    pub async fn spawn(cmd: &AgentCommand, policy: PermissionPolicy) -> Result<Arc<Self>> {
        let rpc = RpcProcess::spawn(
            "ACP agent",
            cmd,
            Framing::Lines,
            Arc::new(move |method, params| handle_agent_request(method, params, policy)),
            route_update,
        )?;
        let init = rpc
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
        Ok(Arc::new(AcpProcess { rpc, init }))
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
}

/// `session/update` notifications are routed to the session's subscriber.
fn route_update(method: &str, params: &Value) -> Option<(String, Value)> {
    if method != "session/update" {
        return None;
    }
    let sid = params.get("sessionId")?.as_str()?.to_string();
    Some((sid, params.get("update")?.clone()))
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
