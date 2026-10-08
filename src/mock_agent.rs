//! Tiny fake Copilot agents used by the integration tests (and handy to try
//! the gateway without a Copilot subscription). Enabled by running the
//! gateway binary with `COPILOT_GATEWAY_MOCK_AGENT=acp` (ACP agent, the
//! default for any other value) or `COPILOT_GATEWAY_MOCK_AGENT=sdk` (Copilot
//! SDK protocol server).

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::sync::{mpsc, oneshot};

pub const ENV_VAR: &str = "COPILOT_GATEWAY_MOCK_AGENT";

/// Run the mock selected by the value of [`ENV_VAR`].
pub async fn run_from_env() -> anyhow::Result<()> {
    match std::env::var(ENV_VAR).as_deref() {
        Ok("sdk") => sdk::run().await,
        _ => run().await,
    }
}

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

/// Mock of `copilot --headless --stdio` (Copilot SDK protocol).
mod sdk {
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use serde_json::{Value, json};
    use tokio::io::{AsyncWriteExt, BufReader};
    use tokio::sync::{mpsc, oneshot};

    use crate::rpc::{Framing, encode_message, read_message};

    struct MockSession {
        model: String,
        tools: Vec<String>,
        system_replaced: bool,
        sends: u32,
        aborted: bool,
    }

    #[derive(Default)]
    struct State {
        sessions: HashMap<String, MockSession>,
        created: u32,
        waiting: HashMap<String, oneshot::Sender<Value>>,
        next: u64,
    }

    type Shared = Arc<Mutex<State>>;
    type Out = mpsc::UnboundedSender<Value>;

    fn models() -> Value {
        json!({"models": [
            {"id": "gpt-5", "name": "GPT-5", "billing": {"multiplier": 1.0},
             "supportedReasoningEfforts": ["low", "medium", "high"], "capabilities": {}},
            {"id": "claude-sonnet-4.5", "name": "Claude Sonnet 4.5", "billing": {"multiplier": 1.0}, "capabilities": {}},
            {"id": "claude-haiku-4.5", "name": "Claude Haiku 4.5", "billing": {"multiplier": 0.33}, "capabilities": {}},
            {"id": "disabled-model", "name": "Disabled", "policy": {"state": "disabled"}, "capabilities": {}}
        ]})
    }

    pub async fn run() -> anyhow::Result<()> {
        let (out, mut out_rx) = mpsc::unbounded_channel::<Value>();
        tokio::spawn(async move {
            let mut stdout = tokio::io::stdout();
            while let Some(msg) = out_rx.recv().await {
                if stdout
                    .write_all(encode_message(&msg, Framing::ContentLength).as_bytes())
                    .await
                    .is_err()
                {
                    break;
                }
                let _ = stdout.flush().await;
            }
        });
        let state: Shared = Arc::default();
        let require_auth = std::env::var_os("MOCK_REQUIRE_AUTH").is_some();
        let mut reader = BufReader::new(tokio::io::stdin());
        while let Some(msg) = read_message(&mut reader, Framing::ContentLength).await? {
            let id = msg.get("id").cloned();
            let params = msg.get("params").cloned().unwrap_or(Value::Null);
            let Some(method) = msg.get("method").and_then(Value::as_str).map(str::to_string) else {
                continue;
            };
            let reply = |result: Value| json!({"jsonrpc": "2.0", "id": id, "result": result});
            let error =
                |message: &str| json!({"jsonrpc": "2.0", "id": id, "error": {"code": -32603, "message": message}});
            let sid = params["sessionId"].as_str().unwrap_or_default().to_string();
            let response = match method.as_str() {
                "connect" => reply(json!({"ok": true, "protocolVersion": 3, "version": "mock"})),
                "models.list" if require_auth => {
                    error("Request models.list failed with message: Not authenticated. Please authenticate first.")
                }
                "models.list" => reply(models()),
                "session.create" => {
                    let mut st = state.lock().unwrap();
                    st.created += 1;
                    let tools = params["tools"]
                        .as_array()
                        .map(|a| {
                            a.iter()
                                .filter_map(|t| t["name"].as_str().map(str::to_string))
                                .collect()
                        })
                        .unwrap_or_default();
                    st.sessions.insert(
                        sid.clone(),
                        MockSession {
                            model: params["model"].as_str().unwrap_or("gpt-5").to_string(),
                            tools,
                            system_replaced: params["systemMessage"]["mode"] == "replace"
                                && params["availableTools"] == json!(["custom:*"]),
                            sends: 0,
                            aborted: false,
                        },
                    );
                    reply(json!({"sessionId": sid}))
                }
                "session.options.update" => reply(json!({})),
                "session.send" => {
                    let known = {
                        let mut st = state.lock().unwrap();
                        st.next += 1;
                        match st.sessions.get_mut(&sid) {
                            Some(s) => {
                                s.sends += 1;
                                s.aborted = false;
                                true
                            }
                            None => false,
                        }
                    };
                    if known {
                        let state = state.clone();
                        let out = out.clone();
                        let params = params.clone();
                        tokio::spawn(async move { turn(state, out, sid, params).await });
                        reply(json!({"messageId": "msg-1"}))
                    } else {
                        error("unknown session")
                    }
                }
                "session.tools.handlePendingToolCall" | "session.permissions.handlePendingPermissionRequest" => {
                    let request_id = params["requestId"].as_str().unwrap_or_default().to_string();
                    let value = params
                        .get("result")
                        .cloned()
                        .unwrap_or_else(|| json!({"error": params["error"]}));
                    if let Some(tx) = state.lock().unwrap().waiting.remove(&request_id) {
                        let _ = tx.send(value);
                    }
                    reply(json!({"success": true}))
                }
                "session.abort" => {
                    if let Some(s) = state.lock().unwrap().sessions.get_mut(&sid) {
                        s.aborted = true;
                    }
                    // Release any tool or permission wait of the session.
                    let mut st = state.lock().unwrap();
                    let keys: Vec<String> = st.waiting.keys().filter(|k| k.starts_with(&sid)).cloned().collect();
                    for k in keys {
                        if let Some(tx) = st.waiting.remove(&k) {
                            let _ = tx.send(json!({"aborted": true}));
                        }
                    }
                    reply(json!({}))
                }
                "session.delete" => {
                    state.lock().unwrap().sessions.remove(&sid);
                    reply(json!({"success": true}))
                }
                other => error(&format!("unknown method {other}")),
            };
            let _ = out.send(response);
        }
        Ok(())
    }

    fn emit(out: &Out, sid: &str, kind: &str, data: Value) {
        let _ = out.send(json!({"jsonrpc": "2.0", "method": "session.event", "params": {
            "sessionId": sid,
            "event": {"id": uuid::Uuid::new_v4().to_string(), "parentId": null,
                      "timestamp": "2026-01-01T00:00:00Z", "type": kind, "data": data}
        }}));
    }

    fn aborted(state: &Shared, sid: &str) -> bool {
        state.lock().unwrap().sessions.get(sid).is_none_or(|s| s.aborted)
    }

    async fn wait_for(state: &Shared, key: String) -> Value {
        let (tx, rx) = oneshot::channel();
        state.lock().unwrap().waiting.insert(key, tx);
        rx.await.unwrap_or(Value::Null)
    }

    /// Stream `text` as deltas then the final message.
    async fn say(state: &Shared, out: &Out, sid: &str, text: &str, slow: bool, tool_requests: Value) -> bool {
        let message_id = uuid::Uuid::new_v4().to_string();
        let chars: Vec<char> = text.chars().collect();
        for chunk in chars.chunks(5) {
            if aborted(state, sid) {
                return false;
            }
            emit(
                out,
                sid,
                "assistant.message_delta",
                json!({"messageId": message_id, "deltaContent": chunk.iter().collect::<String>()}),
            );
            if slow {
                tokio::time::sleep(Duration::from_millis(100)).await;
            }
        }
        emit(
            out,
            sid,
            "assistant.message",
            json!({"messageId": message_id, "content": text, "toolRequests": tool_requests}),
        );
        emit(
            out,
            sid,
            "assistant.usage",
            json!({"model": "mock", "inputTokens": 11, "outputTokens": 7}),
        );
        true
    }

    async fn turn(state: Shared, out: Out, sid: String, params: Value) {
        let prompt = params["prompt"].as_str().unwrap_or_default().to_string();
        let images = params["attachments"].as_array().map_or(0, Vec::len);
        let (model, tools, system_replaced, sends, created) = {
            let st = state.lock().unwrap();
            let s = &st.sessions[&sid];
            (s.model.clone(), s.tools.clone(), s.system_replaced, s.sends, st.created)
        };
        let has = |t: &str| tools.iter().any(|x| x == t);
        emit(&out, &sid, "assistant.turn_start", json!({"turnId": "t1"}));
        emit(
            &out,
            &sid,
            "assistant.reasoning_delta",
            json!({"reasoningId": "r1", "deltaContent": "Thinking..."}),
        );

        let mut permission = String::new();
        if prompt.contains("PERMISSION") {
            let request_id = format!("{sid}-perm");
            emit(
                &out,
                &sid,
                "permission.requested",
                json!({"requestId": request_id, "permissionRequest": {"kind": "shell"}}),
            );
            let res = wait_for(&state, request_id).await;
            permission = format!(" permission={}", res["kind"].as_str().unwrap_or("none"));
        }

        let calls: Vec<(&str, Value)> = if prompt.contains("RUN_BASH") && has("Bash") {
            vec![(
                "Bash",
                json!({"command": "echo gateway-ok", "description": "Print a marker"}),
            )]
        } else if prompt.contains("RUN_BASH") && has("exec_command") {
            vec![("exec_command", json!({"cmd": "echo gateway-ok"}))]
        } else if prompt.contains("TWO_TOOLS") && has("get_weather") {
            vec![
                ("get_weather", json!({"city": "Paris"})),
                ("get_weather", json!({"city": "Lyon"})),
            ]
        } else if prompt.contains("weather") && has("get_weather") {
            vec![("get_weather", json!({"city": "Paris"}))]
        } else if prompt.contains("patch it") && has("apply_patch") {
            vec![("apply_patch", json!({"input": "*** Begin Patch\n*** End Patch"}))]
        } else {
            vec![]
        };

        if !calls.is_empty() {
            let requests: Vec<Value> = calls
                .iter()
                .enumerate()
                .map(|(i, (name, args))| json!({"toolCallId": format!("call_{sid}_{i}"), "name": name, "arguments": args}))
                .collect();
            if !say(&state, &out, &sid, "Let me check.\n", false, json!(requests)).await {
                return finish_aborted(&out, &sid);
            }
            let mut waits = Vec::new();
            for (i, req) in requests.iter().enumerate() {
                let request_id = format!("{sid}-tool-{i}");
                let (tx, rx) = oneshot::channel();
                state.lock().unwrap().waiting.insert(request_id.clone(), tx);
                emit(
                    &out,
                    &sid,
                    "external_tool.requested",
                    json!({
                        "requestId": request_id, "sessionId": sid, "toolCallId": req["toolCallId"],
                        "toolName": req["name"], "arguments": req["arguments"]
                    }),
                );
                waits.push(rx);
            }
            let mut results = Vec::new();
            let mut result_images = 0;
            for rx in waits {
                let r = rx.await.unwrap_or(Value::Null);
                if r.get("aborted").is_some() {
                    return finish_aborted(&out, &sid);
                }
                result_images += r["binaryResultsForLlm"].as_array().map_or(0, Vec::len);
                let text = match &r {
                    Value::String(s) => s.clone(),
                    other => other["textResultForLlm"].as_str().unwrap_or("?").to_string(),
                };
                let kind = r["resultType"].as_str().unwrap_or("success");
                results.push(if kind == "success" {
                    text
                } else {
                    format!("{kind}: {text}")
                });
            }
            let text = if prompt.contains("RUN_BASH") {
                format!("The command printed: {}", results.join(", "))
            } else {
                format!("Results: {} (images={result_images})", results.join(" | "))
            };
            if !say(&state, &out, &sid, &text, false, json!([])).await {
                return finish_aborted(&out, &sid);
            }
        } else {
            let text = format!(
                "Hello from {model}! images={images} sends={sends} sessions={created} system={}{permission}",
                if system_replaced { "replaced" } else { "default" }
            );
            if !say(&state, &out, &sid, &text, prompt.contains("SLOW"), json!([])).await {
                return finish_aborted(&out, &sid);
            }
        }
        emit(&out, &sid, "assistant.turn_end", json!({"turnId": "t1"}));
        emit(&out, &sid, "session.idle", json!({}));
    }

    fn finish_aborted(out: &Out, sid: &str) {
        emit(out, sid, "abort", json!({"reason": "user"}));
        emit(out, sid, "session.idle", json!({"aborted": true}));
    }
}
