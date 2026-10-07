//! A tiny fake ACP agent used by the integration tests (and handy to try
//! the gateway without a Copilot subscription). Enabled by running the
//! gateway binary with `COPILOT_GATEWAY_MOCK_AGENT=1`.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::{mpsc, oneshot};

pub const ENV_VAR: &str = "COPILOT_GATEWAY_MOCK_AGENT";

#[derive(Default)]
struct State {
    models: HashMap<String, String>,
    cancelled: HashSet<String>,
    pending: HashMap<String, oneshot::Sender<Value>>,
    next_session: u64,
}

fn config_options(current: &str) -> Value {
    json!([
        {"id": "model", "name": "Model", "category": "model", "type": "select", "currentValue": current,
         "options": [
            {"value": "gpt-5", "name": "GPT-5"},
            {"value": "claude-sonnet-4.5", "name": "Claude Sonnet 4.5"},
            {"value": "claude-haiku-4.5", "name": "Claude Haiku 4.5"}
         ]},
        {"id": "reasoning", "name": "Reasoning", "category": "thought_level", "type": "select",
         "currentValue": "medium",
         "options": [{"value": "low", "name": "Low"}, {"value": "medium", "name": "Medium"}, {"value": "high", "name": "High"}]}
    ])
}

pub async fn run() -> anyhow::Result<()> {
    let (out_tx, mut out_rx) = mpsc::unbounded_channel::<Value>();
    tokio::spawn(async move {
        let mut stdout = tokio::io::stdout();
        while let Some(msg) = out_rx.recv().await {
            let mut line = msg.to_string();
            line.push('\n');
            if stdout.write_all(line.as_bytes()).await.is_err() {
                break;
            }
            let _ = stdout.flush().await;
        }
    });
    let state = Arc::new(Mutex::new(State::default()));
    let require_auth = std::env::var_os("MOCK_REQUIRE_AUTH").is_some();

    let mut lines = BufReader::new(tokio::io::stdin()).lines();
    while let Some(line) = lines.next_line().await? {
        let Ok(msg) = serde_json::from_str::<Value>(&line) else {
            continue;
        };
        let id = msg.get("id").cloned();
        let params = msg.get("params").cloned().unwrap_or(Value::Null);
        let method = msg.get("method").and_then(Value::as_str).map(str::to_string);
        let reply = |result: Value| json!({"jsonrpc": "2.0", "id": id, "result": result});

        match method.as_deref() {
            None => {
                // Response to one of our requests.
                let key = id.map(|v| v.to_string()).unwrap_or_default();
                if let Some(tx) = state.lock().unwrap().pending.remove(&key) {
                    let _ = tx.send(msg.get("result").cloned().unwrap_or(Value::Null));
                }
            }
            Some("initialize") => {
                let _ = out_tx.send(reply(json!({
                    "protocolVersion": 1,
                    "agentCapabilities": {
                        "loadSession": false,
                        "promptCapabilities": {"image": true, "audio": false, "embeddedContext": true},
                        "sessionCapabilities": {"close": {}}
                    },
                    "agentInfo": {"name": "mock", "version": "0"},
                    "authMethods": []
                })));
            }
            Some("session/new") => {
                if require_auth {
                    let _ = out_tx.send(json!({"jsonrpc": "2.0", "id": id,
                        "error": {"code": -32000, "message": "Authentication required"}}));
                    continue;
                }
                let sid = {
                    let mut s = state.lock().unwrap();
                    s.next_session += 1;
                    let sid = format!("mock-{}", s.next_session);
                    s.models.insert(sid.clone(), "gpt-5".into());
                    sid
                };
                let _ = out_tx.send(reply(
                    json!({"sessionId": sid, "configOptions": config_options("gpt-5")}),
                ));
            }
            Some("session/set_config_option") => {
                let sid = params["sessionId"].as_str().unwrap_or_default().to_string();
                let value = params["value"].as_str().unwrap_or_default().to_string();
                if params["configId"] == "model" {
                    state.lock().unwrap().models.insert(sid, value.clone());
                }
                let _ = out_tx.send(reply(json!({"configOptions": config_options(&value)})));
            }
            Some("session/close") => {
                let _ = out_tx.send(reply(json!({})));
            }
            Some("session/cancel") => {
                let sid = params["sessionId"].as_str().unwrap_or_default().to_string();
                state.lock().unwrap().cancelled.insert(sid);
            }
            Some("session/prompt") => {
                let out_tx = out_tx.clone();
                let state = state.clone();
                tokio::spawn(async move {
                    let result = prompt(&params, &out_tx, &state).await;
                    let _ = out_tx.send(json!({"jsonrpc": "2.0", "id": id, "result": result}));
                });
            }
            Some(other) => {
                if id.is_some() {
                    let _ = out_tx.send(json!({"jsonrpc": "2.0", "id": id,
                        "error": {"code": -32601, "message": format!("unknown method {other}")}}));
                }
            }
        }
    }
    Ok(())
}

async fn prompt(params: &Value, out: &mpsc::UnboundedSender<Value>, state: &Arc<Mutex<State>>) -> Value {
    let sid = params["sessionId"].as_str().unwrap_or_default().to_string();
    let blocks = params["prompt"].as_array().cloned().unwrap_or_default();
    let text: String = blocks
        .iter()
        .filter_map(|b| b.get("text").and_then(Value::as_str))
        .collect();
    let images = blocks.iter().filter(|b| b["type"] == "image").count();
    let model = state.lock().unwrap().models.get(&sid).cloned().unwrap_or_default();

    let update = |update: Value| {
        let _ = out.send(json!({"jsonrpc": "2.0", "method": "session/update",
            "params": {"sessionId": sid, "update": update}}));
    };

    let mut permission = String::new();
    if text.contains("PERMISSION") {
        let (tx, rx) = oneshot::channel();
        let req_id = json!(format!("perm-{sid}"));
        state.lock().unwrap().pending.insert(req_id.to_string(), tx);
        let _ = out.send(
            json!({"jsonrpc": "2.0", "id": req_id, "method": "session/request_permission",
            "params": {"sessionId": sid,
                "toolCall": {"toolCallId": "t1", "title": "rm -rf /", "kind": "execute"},
                "options": [
                    {"optionId": "yes", "name": "Allow", "kind": "allow_once"},
                    {"optionId": "no", "name": "Reject", "kind": "reject_once"}
                ]}}),
        );
        let res = rx.await.unwrap_or(Value::Null);
        permission = format!(
            " permission={}",
            res["outcome"]["optionId"].as_str().unwrap_or("cancelled")
        );
    }

    update(json!({"sessionUpdate": "agent_thought_chunk", "content": {"type": "text", "text": "Thinking..."}}));

    let reply = if text.contains("RUN_BASH") && text.contains("<tool_result id=") {
        let result = text
            .rsplit_once("<tool_result id=")
            .and_then(|(_, r)| r.split_once('>'))
            .and_then(|(_, r)| r.split_once("</tool_result>"))
            .map(|(r, _)| r.trim().to_string())
            .unwrap_or_default();
        format!("The command printed: {result}")
    } else if text.contains("RUN_BASH") && text.contains("\"name\":\"exec_command\"") {
        "<tool_call>{\"name\": \"exec_command\", \"arguments\": {\"cmd\": \"echo gateway-ok\"}}</tool_call>".to_string()
    } else if text.contains("RUN_BASH") && text.contains("<tools>") {
        "<tool_call>{\"name\": \"Bash\", \"arguments\": {\"command\": \"echo gateway-ok\", \"description\": \"Print a marker\"}}</tool_call>"
            .to_string()
    } else if text.contains("<tools>") && text.contains("weather") {
        "Let me check.\n<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Paris\"}}\n</tool_call>\n"
            .to_string()
    } else if text.contains("<tools>") && text.contains("apply_patch") && text.contains("patch it") {
        "<tool_call>{\"name\": \"apply_patch\", \"arguments\": \"*** Begin Patch\\n*** End Patch\"}</tool_call>"
            .to_string()
    } else {
        format!("Hello from {model}! images={images}{permission}")
    };

    let slow = text.contains("SLOW");
    let chars: Vec<char> = reply.chars().collect();
    for chunk in chars.chunks(5) {
        if state.lock().unwrap().cancelled.contains(&sid) {
            return json!({"stopReason": "cancelled"});
        }
        update(json!({"sessionUpdate": "agent_message_chunk", "messageId": "m1",
            "content": {"type": "text", "text": chunk.iter().collect::<String>()}}));
        if slow {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
    }
    json!({"stopReason": "end_turn", "usage": {"inputTokens": 11, "outputTokens": 7, "totalTokens": 18}})
}
