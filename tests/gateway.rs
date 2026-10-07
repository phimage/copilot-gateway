//! End-to-end tests: real HTTP server + real ACP process (the built-in mock
//! agent, started from the gateway binary itself).

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use copilot_gateway::acp::{AgentCommand, PermissionPolicy};
use copilot_gateway::backend::{Backend, BackendConfig};
use copilot_gateway::server::{self, AppState};
use futures::StreamExt;
use serde_json::{Value, json};

const KEY: &str = "test-key";

async fn start(extra_env: &[(&str, &str)], max_concurrent: usize) -> String {
    let mut env = vec![(copilot_gateway::mock_agent::ENV_VAR.to_string(), "1".to_string())];
    env.extend(extra_env.iter().map(|(k, v)| (k.to_string(), v.to_string())));
    let backend = Backend::new(BackendConfig {
        agent: AgentCommand {
            program: PathBuf::from(env!("CARGO_BIN_EXE_copilot-gateway")),
            args: vec!["--acp".into()],
            env,
            cwd: std::env::temp_dir(),
        },
        permission: PermissionPolicy::Deny,
        default_model: None,
        model_map: vec![],
        max_concurrent,
    });
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
                assert!(v["content_block"]["id"].as_str().unwrap().starts_with("toolu_"));
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
