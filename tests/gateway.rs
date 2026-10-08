//! End-to-end tests: real HTTP server + real agent process (the built-in mock
//! agents, started from the gateway binary itself), for both backends.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use copilot_gateway::acp::{AgentCommand, PermissionPolicy};
use copilot_gateway::backend::{Backend, BackendConfig};
use copilot_gateway::engine::{BackendKind, Engine};
use copilot_gateway::sdk::SdkBackend;
use copilot_gateway::server::{self, AppState};
use futures::StreamExt;
use serde_json::{Value, json};

const KEY: &str = "test-key";

async fn start(extra_env: &[(&str, &str)], max_concurrent: usize) -> String {
    start_backend(BackendKind::Acp, extra_env, max_concurrent).await
}

async fn start_sdk(extra_env: &[(&str, &str)], max_concurrent: usize) -> String {
    start_backend(BackendKind::Sdk, extra_env, max_concurrent).await
}

async fn start_backend(kind: BackendKind, extra_env: &[(&str, &str)], max_concurrent: usize) -> String {
    let mock = match kind {
        BackendKind::Acp => "acp",
        BackendKind::Sdk => "sdk",
    };
    let mut env = vec![(copilot_gateway::mock_agent::ENV_VAR.to_string(), mock.to_string())];
    env.extend(extra_env.iter().map(|(k, v)| (k.to_string(), v.to_string())));
    let config = BackendConfig {
        agent: AgentCommand {
            program: PathBuf::from(env!("CARGO_BIN_EXE_copilot-gateway")),
            args: vec![],
            env,
            cwd: std::env::temp_dir(),
        },
        permission: PermissionPolicy::Deny,
        default_model: None,
        model_map: vec![],
        max_concurrent,
        session_ttl: Duration::from_secs(600),
        max_sessions: 8,
    };
    let backend = match kind {
        BackendKind::Acp => Engine::Acp(Backend::new(config)),
        BackendKind::Sdk => Engine::Sdk(SdkBackend::new(config)),
    };
    let state = AppState {
        backend: Arc::new(backend),
        api_key: Some(KEY.into()),
    };
    let (tx, rx) = tokio::sync::oneshot::channel::<SocketAddr>();
    tokio::spawn(async move {
        server::serve(
            state,
            "127.0.0.1:0".parse().unwrap(),
            move |a| {
                let _ = tx.send(a);
            },
            std::future::pending(),
        )
        .await
        .unwrap();
    });
    format!("http://{}", rx.await.unwrap())
}

fn client() -> reqwest::Client {
    reqwest::Client::new()
}

async fn post(base: &str, path: &str, body: Value) -> reqwest::Response {
    client()
        .post(format!("{base}{path}"))
        .bearer_auth(KEY)
        .json(&body)
        .send()
        .await
        .unwrap()
}

/// Parse an SSE body into (event name, data) pairs.
async fn sse(resp: reqwest::Response) -> Vec<(String, String)> {
    let text = resp.text().await.unwrap();
    let mut out = Vec::new();
    for block in text.split("\n\n") {
        let mut name = String::new();
        let mut data = String::new();
        for line in block.lines() {
            if let Some(v) = line.strip_prefix("event: ") {
                name = v.to_string();
            } else if let Some(v) = line.strip_prefix("data: ") {
                data.push_str(v);
            }
        }
        if !data.is_empty() {
            out.push((name, data));
        }
    }
    out
}

#[tokio::test]
async fn anthropic_non_streaming() {
    let base = start(&[], 4).await;
    let resp = post(
        &base,
        "/v1/messages",
        json!({
            "model": "claude-sonnet-4-5-20250929",
            "max_tokens": 100,
            "messages": [{"role": "user", "content": "hi"}]
        }),
    )
    .await;
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["type"], "message");
    assert_eq!(body["model"], "claude-sonnet-4-5-20250929");
    assert_eq!(body["content"][0]["text"], "Hello from claude-sonnet-4.5! images=0");
    assert_eq!(body["stop_reason"], "end_turn");
    assert_eq!(body["usage"]["input_tokens"], 11);
}

#[tokio::test]
async fn anthropic_streaming_tool_use() {
    let base = start(&[], 4).await;
    let resp = post(
        &base,
        "/v1/messages",
        json!({
            "model": "claude-haiku-4-5",
            "max_tokens": 100,
            "stream": true,
            "system": "You are helpful.",
            "tools": [{"name": "get_weather", "description": "weather", "input_schema": {"type": "object"}}],
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "weather in Paris?"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}}
            ]}]
        }),
    )
    .await;
    assert_eq!(resp.status(), 200);
    let events = sse(resp).await;
    let names: Vec<&str> = events.iter().map(|(n, _)| n.as_str()).collect();
    assert_eq!(names.first(), Some(&"message_start"));
    assert_eq!(names.last(), Some(&"message_stop"));

    let mut text = String::new();
    let mut tool_json = String::new();
    let mut tool_name = String::new();
    let mut stop = String::new();
    for (name, data) in &events {
        let v: Value = serde_json::from_str(data).unwrap();
        match name.as_str() {
            "content_block_start" if v["content_block"]["type"] == "tool_use" => {
                tool_name = v["content_block"]["name"].as_str().unwrap().to_string();
                assert!(!v["content_block"]["id"].as_str().unwrap().is_empty());
            }
            "content_block_delta" => match v["delta"]["type"].as_str().unwrap() {
                "text_delta" => text.push_str(v["delta"]["text"].as_str().unwrap()),
                "input_json_delta" => tool_json.push_str(v["delta"]["partial_json"].as_str().unwrap()),
                other => panic!("unexpected delta {other}"),
            },
            "message_delta" => stop = v["delta"]["stop_reason"].as_str().unwrap().to_string(),
            _ => {}
        }
    }
    assert_eq!(text, "Let me check.\n");
    assert_eq!(tool_name, "get_weather");
    assert_eq!(
        serde_json::from_str::<Value>(&tool_json).unwrap(),
        json!({"city": "Paris"})
    );
    assert_eq!(stop, "tool_use");
}

#[tokio::test]
async fn anthropic_thinking_and_count_tokens() {
    let base = start(&[], 4).await;
    let resp = post(
        &base,
        "/v1/messages",
        json!({
            "model": "gpt-5",
            "max_tokens": 100,
            "thinking": {"type": "enabled", "budget_tokens": 2000},
            "messages": [{"role": "user", "content": "hi"}]
        }),
    )
    .await;
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["content"][0]["type"], "thinking");
    assert_eq!(body["content"][0]["thinking"], "Thinking...");
    assert_eq!(body["content"][1]["text"], "Hello from gpt-5! images=0");

    let resp = post(
        &base,
        "/v1/messages/count_tokens",
        json!({"model": "x", "messages": [{"role": "user", "content": "hello world"}]}),
    )
    .await;
    let body: Value = resp.json().await.unwrap();
    assert!(body["input_tokens"].as_u64().unwrap() > 0);
}

#[tokio::test]
async fn openai_chat_completions() {
    let base = start(&[], 4).await;
    // Non-streaming with tools.
    let resp = post(
        &base,
        "/v1/chat/completions",
        json!({
            "model": "gpt-5",
            "messages": [{"role": "user", "content": "what is the weather?"}],
            "tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]
        }),
    )
    .await;
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    let choice = &body["choices"][0];
    assert_eq!(choice["finish_reason"], "tool_calls");
    assert_eq!(choice["message"]["tool_calls"][0]["function"]["name"], "get_weather");
    assert_eq!(
        choice["message"]["tool_calls"][0]["function"]["arguments"],
        "{\"city\":\"Paris\"}"
    );
    assert_eq!(body["usage"]["completion_tokens"], 7);

    // Streaming plain text.
    let resp = post(
        &base,
        "/v1/chat/completions",
        json!({
            "model": "claude-sonnet-4.5",
            "stream": true,
            "stream_options": {"include_usage": true},
            "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        }),
    )
    .await;
    let events = sse(resp).await;
    assert_eq!(events.last().unwrap().1, "[DONE]");
    let mut text = String::new();
    let mut finish = String::new();
    let mut usage = Value::Null;
    for (_, data) in &events[..events.len() - 1] {
        let v: Value = serde_json::from_str(data).unwrap();
        if let Some(c) = v.pointer("/choices/0/delta/content").and_then(Value::as_str) {
            text.push_str(c);
        }
        if let Some(f) = v.pointer("/choices/0/finish_reason").and_then(Value::as_str) {
            finish = f.to_string();
        }
        if !v["usage"].is_null() {
            usage = v["usage"].clone();
        }
    }
    assert_eq!(text, "Hello from claude-sonnet-4.5! images=0");
    assert_eq!(finish, "stop");
    assert_eq!(usage["total_tokens"], 18);
}

#[tokio::test]
async fn openai_responses_streaming_custom_tool() {
    let base = start(&[], 4).await;
    let resp = post(
        &base,
        "/v1/responses",
        json!({
            "model": "gpt-5-codex",
            "instructions": "You are Codex.",
            "stream": true,
            "reasoning": {"effort": "high", "summary": "auto"},
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "patch it"}]}],
            "tools": [{"type": "custom", "name": "apply_patch", "description": "Apply a patch",
                       "format": {"type": "grammar", "syntax": "lark", "definition": "start: begin_patch"}}]
        }),
    )
    .await;
    assert_eq!(resp.status(), 200);
    let events = sse(resp).await;
    assert_eq!(events[0].0, "response.created");
    let (last_name, last_data) = events.last().unwrap();
    assert_eq!(last_name, "response.completed");
    let completed: Value = serde_json::from_str(last_data).unwrap();
    let output = completed["response"]["output"].as_array().unwrap();
    assert_eq!(output[0]["type"], "reasoning");
    assert_eq!(output[1]["type"], "custom_tool_call");
    assert_eq!(output[1]["name"], "apply_patch");
    assert_eq!(output[1]["input"], "*** Begin Patch\n*** End Patch");
    assert_eq!(completed["response"]["usage"]["total_tokens"], 18);
    for (i, (_, data)) in events.iter().enumerate() {
        let v: Value = serde_json::from_str(data).unwrap();
        assert_eq!(v["sequence_number"], i);
    }
}

#[tokio::test]
async fn openai_responses_non_streaming() {
    let base = start(&[], 4).await;
    let resp = post(&base, "/v1/responses", json!({"model": "gpt-5", "input": "hi"})).await;
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["status"], "completed");
    assert_eq!(body["output"][0]["content"][0]["text"], "Hello from gpt-5! images=0");
}

#[tokio::test]
async fn models_auth_and_permissions() {
    let base = start(&[], 4).await;
    let resp = client().get(format!("{base}/v1/models")).send().await.unwrap();
    assert_eq!(resp.status(), 401);

    let resp = client()
        .get(format!("{base}/v1/models"))
        .header("x-api-key", KEY)
        .send()
        .await
        .unwrap();
    let body: Value = resp.json().await.unwrap();
    let ids: Vec<&str> = body["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, vec!["gpt-5", "claude-sonnet-4.5", "claude-haiku-4.5"]);

    // The agent asks for permission to run one of its own tools: denied.
    let resp = post(
        &base,
        "/v1/chat/completions",
        json!({"model": "gpt-5", "messages": [{"role": "user", "content": "PERMISSION"}]}),
    )
    .await;
    let body: Value = resp.json().await.unwrap();
    assert_eq!(
        body["choices"][0]["message"]["content"],
        "Hello from gpt-5! images=0 permission=no"
    );
}

#[tokio::test]
async fn authentication_error_is_reported() {
    let base = start(&[("MOCK_REQUIRE_AUTH", "1")], 4).await;
    let resp = post(
        &base,
        "/v1/messages",
        json!({"model": "x", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]}),
    )
    .await;
    assert_eq!(resp.status(), 502);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["type"], "error");
    assert!(body["error"]["message"].as_str().unwrap().contains("copilot login"));
}

#[tokio::test]
async fn client_disconnect_cancels_turn() {
    // With a single slot, a second request only succeeds if the first,
    // abandoned turn was cancelled and released its slot.
    let base = start(&[], 1).await;
    let resp = post(
        &base,
        "/v1/chat/completions",
        json!({"model": "gpt-5", "stream": true, "messages": [{"role": "user", "content": "SLOW"}]}),
    )
    .await;
    let mut stream = resp.bytes_stream();
    let _ = stream.next().await;
    drop(stream);

    let resp = tokio::time::timeout(
        Duration::from_secs(10),
        post(
            &base,
            "/v1/chat/completions",
            json!({"model": "gpt-5", "messages": [{"role": "user", "content": "hi"}]}),
        ),
    )
    .await
    .expect("second request blocked: first turn was not cancelled");
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["choices"][0]["message"]["content"], "Hello from gpt-5! images=0");
}

// ---------------------------------------------------------------------------
// Copilot SDK backend
// ---------------------------------------------------------------------------

/// Collect an Anthropic SSE stream into (text, tool uses, stop reason).
async fn anthropic_stream(resp: reqwest::Response) -> (String, Vec<(String, String, Value)>, String) {
    let mut text = String::new();
    let mut tools: Vec<(String, String, String)> = Vec::new();
    let mut stop = String::new();
    for (name, data) in sse(resp).await {
        let v: Value = serde_json::from_str(&data).unwrap();
        match name.as_str() {
            "content_block_start" if v["content_block"]["type"] == "tool_use" => tools.push((
                v["content_block"]["id"].as_str().unwrap().to_string(),
                v["content_block"]["name"].as_str().unwrap().to_string(),
                String::new(),
            )),
            "content_block_delta" => match v["delta"]["type"].as_str().unwrap() {
                "text_delta" => text.push_str(v["delta"]["text"].as_str().unwrap()),
                "input_json_delta" => tools
                    .last_mut()
                    .unwrap()
                    .2
                    .push_str(v["delta"]["partial_json"].as_str().unwrap()),
                _ => {}
            },
            "message_delta" => stop = v["delta"]["stop_reason"].as_str().unwrap().to_string(),
            "error" => panic!("stream error: {data}"),
            _ => {}
        }
    }
    let tools = tools
        .into_iter()
        .map(|(id, name, json)| (id, name, serde_json::from_str(&json).unwrap()))
        .collect();
    (text, tools, stop)
}

const WEATHER_TOOL: &str = r#"{"name": "get_weather", "description": "weather", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}}"#;

#[tokio::test]
async fn sdk_simple_message_and_models() {
    let base = start_sdk(&[], 4).await;
    let resp = post(
        &base,
        "/v1/messages",
        json!({
            "model": "claude-sonnet-4-5-20250929",
            "max_tokens": 100,
            "system": "You are terse.",
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "hi"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}}
            ]}]
        }),
    )
    .await;
    assert_eq!(resp.status(), 200);
    let body: Value = resp.json().await.unwrap();
    assert_eq!(
        body["content"][0]["text"],
        "Hello from claude-sonnet-4.5! images=1 sends=1 sessions=1 system=replaced"
    );
    assert_eq!(body["usage"]["input_tokens"], 11);
    assert_eq!(body["usage"]["output_tokens"], 7);

    let resp = client()
        .get(format!("{base}/v1/models"))
        .bearer_auth(KEY)
        .send()
        .await
        .unwrap();
    let body: Value = resp.json().await.unwrap();
    let ids: Vec<&str> = body["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, vec!["gpt-5", "claude-sonnet-4.5", "claude-haiku-4.5"]);
}

#[tokio::test]
async fn sdk_native_tool_loop_stays_in_one_turn() {
    let base = start_sdk(&[], 4).await;
    let tool: Value = serde_json::from_str(WEATHER_TOOL).unwrap();
    let user = json!({"role": "user", "content": "TWO_TOOLS weather please"});
    let resp = post(
        &base,
        "/v1/messages",
        json!({"model": "gpt-5", "max_tokens": 100, "stream": true, "tools": [tool], "messages": [user]}),
    )
    .await;
    let (text, tools, stop) = anthropic_stream(resp).await;
    assert_eq!(text, "Let me check.\n");
    assert_eq!(stop, "tool_use");
    assert_eq!(tools.len(), 2);
    assert_eq!(tools[0].1, "get_weather");
    assert_eq!(tools[0].2, json!({"city": "Paris"}));
    assert_eq!(tools[1].2, json!({"city": "Lyon"}));

    // The client runs the tools and sends the results back: the same Copilot
    // turn continues (no new session, no new prompt).
    let assistant = json!({"role": "assistant", "content": [
        {"type": "text", "text": text},
        {"type": "tool_use", "id": tools[0].0, "name": "get_weather", "input": tools[0].2},
        {"type": "tool_use", "id": tools[1].0, "name": "get_weather", "input": tools[1].2}
    ]});
    let results = json!({"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": tools[0].0, "content": "sunny"},
        {"type": "tool_result", "tool_use_id": tools[1].0, "content": [{"type": "text", "text": "rainy"}], "is_error": true},
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}}
    ]});
    let resp = post(
        &base,
        "/v1/messages",
        json!({"model": "gpt-5", "max_tokens": 100, "stream": true, "tools": [tool],
               "messages": [user, assistant, results]}),
    )
    .await;
    let (text, tools2, stop) = anthropic_stream(resp).await;
    assert!(tools2.is_empty());
    assert_eq!(stop, "end_turn");
    assert_eq!(text, "Results: sunny | failure: rainy (images=1)");

    // Follow-up message: the idle session is reused (sends=2, sessions=1).
    let final_assistant = json!({"role": "assistant", "content": [{"type": "text", "text": text}]});
    let resp = post(
        &base,
        "/v1/messages",
        json!({"model": "gpt-5", "max_tokens": 100, "tools": [tool],
               "messages": [user, assistant, results, final_assistant, {"role": "user", "content": "thanks"}]}),
    )
    .await;
    let body: Value = resp.json().await.unwrap();
    assert_eq!(
        body["content"][0]["text"],
        "Hello from gpt-5! images=0 sends=2 sessions=1 system=replaced"
    );

    // A conversation the gateway doesn't know gets a new session.
    let resp = post(
        &base,
        "/v1/messages",
        json!({"model": "gpt-5", "max_tokens": 100, "tools": [tool], "messages": [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "something else"},
            {"role": "user", "content": "second"}
        ]}),
    )
    .await;
    let body: Value = resp.json().await.unwrap();
    assert_eq!(
        body["content"][0]["text"],
        "Hello from gpt-5! images=0 sends=1 sessions=2 system=replaced"
    );
}

#[tokio::test]
async fn sdk_chat_completions_and_responses() {
    let base = start_sdk(&[], 4).await;
    // Chat Completions: tool call then result.
    let tools = json!([{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]);
    let user = json!({"role": "user", "content": "what is the weather?"});
    let resp = post(
        &base,
        "/v1/chat/completions",
        json!({"model": "gpt-5", "messages": [user], "tools": tools}),
    )
    .await;
    let body: Value = resp.json().await.unwrap();
    let msg = &body["choices"][0]["message"];
    assert_eq!(body["choices"][0]["finish_reason"], "tool_calls");
    let call = msg["tool_calls"][0].clone();
    assert_eq!(call["function"]["arguments"], "{\"city\":\"Paris\"}");
    let resp = post(
        &base,
        "/v1/chat/completions",
        json!({"model": "gpt-5", "tools": tools, "messages": [
            user, msg, {"role": "tool", "tool_call_id": call["id"], "content": "sunny"}
        ]}),
    )
    .await;
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["choices"][0]["message"]["content"], "Results: sunny (images=0)");

    // Responses API with a freeform (custom) tool and a namespaced tool.
    let tools = json!([
        {"type": "custom", "name": "apply_patch", "description": "Apply a patch",
         "format": {"type": "grammar", "syntax": "lark", "definition": "start: x"}},
        {"type": "namespace", "name": "multi_agent", "tools": [{"type": "function", "name": "spawn", "parameters": {"type": "object"}}]}
    ]);
    let input = json!([{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "patch it"}]}]);
    let resp = post(
        &base,
        "/v1/responses",
        json!({"model": "gpt-5", "instructions": "You are Codex.", "stream": true, "input": input, "tools": tools}),
    )
    .await;
    let events = sse(resp).await;
    let (last, data) = events.last().unwrap();
    assert_eq!(last, "response.completed");
    let completed: Value = serde_json::from_str(data).unwrap();
    let call = completed["response"]["output"]
        .as_array()
        .unwrap()
        .iter()
        .find(|i| i["type"] == "custom_tool_call")
        .unwrap()
        .clone();
    assert_eq!(call["name"], "apply_patch");
    assert_eq!(call["input"], "*** Begin Patch\n*** End Patch");
    let mut input2 = input.as_array().unwrap().clone();
    input2.push(call.clone());
    input2.push(json!({"type": "custom_tool_call_output", "call_id": call["call_id"], "output": "Done!"}));
    let resp = post(
        &base,
        "/v1/responses",
        json!({"model": "gpt-5", "instructions": "You are Codex.", "input": input2, "tools": tools}),
    )
    .await;
    let body: Value = resp.json().await.unwrap();
    assert_eq!(body["output"][0]["content"][0]["text"], "Results: Done! (images=0)");
}

#[tokio::test]
async fn sdk_permissions_auth_and_cancel() {
    let base = start_sdk(&[], 1).await;
    // Copilot asking for a permission is refused.
    let resp = post(
        &base,
        "/v1/chat/completions",
        json!({"model": "gpt-5", "messages": [{"role": "user", "content": "PERMISSION"}]}),
    )
    .await;
    let body: Value = resp.json().await.unwrap();
    assert!(
        body["choices"][0]["message"]["content"]
            .as_str()
            .unwrap()
            .ends_with("permission=reject")
    );

    // Client disconnect aborts the turn and frees the only slot.
    let resp = post(
        &base,
        "/v1/chat/completions",
        json!({"model": "gpt-5", "stream": true, "messages": [{"role": "user", "content": "SLOW"}]}),
    )
    .await;
    let mut stream = resp.bytes_stream();
    let _ = stream.next().await;
    drop(stream);
    let resp = tokio::time::timeout(
        Duration::from_secs(10),
        post(
            &base,
            "/v1/chat/completions",
            json!({"model": "gpt-5", "messages": [{"role": "user", "content": "hi"}]}),
        ),
    )
    .await
    .expect("second request blocked: first turn was not cancelled");
    assert_eq!(resp.status(), 200);

    // Not logged in: clear error.
    let base = start_sdk(&[("MOCK_REQUIRE_AUTH", "1")], 1).await;
    let resp = post(
        &base,
        "/v1/messages",
        json!({"model": "x", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}]}),
    )
    .await;
    assert_eq!(resp.status(), 502);
    let body: Value = resp.json().await.unwrap();
    assert!(body["error"]["message"].as_str().unwrap().contains("copilot login"));
}
