//! OpenAI Chat Completions API (`POST /v1/chat/completions`).

use std::collections::HashMap;
use std::convert::Infallible;

use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::response::sse::Event;
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};
use tracing::debug;

use crate::chat::{ChatRequest, Part, Role, ToolChoice, ToolDef, content_to_text, image_from_url};
use crate::output::{FinalUsage, Finish, Item, OutEvent, OutputStream, new_id, now_secs};
use crate::server::{ApiError, AppState, Flavor, parse_json, sse};

const FLAVOR: Flavor = Flavor::OpenAi;

pub async fn chat_completions(State(state): State<AppState>, body: Bytes) -> Result<Response, ApiError> {
    let body = parse_json(FLAVOR, &body)?;
    tracing::trace!(%body, "chat completion request body");
    let req = to_chat_request(&body).map_err(|e| ApiError::bad_request(FLAVOR, e))?;
    let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let include_usage = body
        .pointer("/stream_options/include_usage")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let model = req.model.clone();
    debug!(
        model,
        stream,
        messages = req.messages.len(),
        tools = req.tools.len(),
        "chat completion request"
    );

    let turn = state
        .backend
        .start_turn(&req)
        .await
        .map_err(|e| ApiError::upstream(FLAVOR, e))?;
    let out = OutputStream::new(turn, &req);

    if stream {
        return Ok(sse(stream_chunks(out, model, include_usage)));
    }

    let c = out.collect().await.map_err(|e| ApiError::upstream(FLAVOR, e))?;
    let text = c.text();
    let tool_calls: Vec<Value> = c
        .items
        .iter()
        .filter_map(|i| match i {
            Item::ToolCall { id, name, arguments } => Some(tool_call_json(id, name, arguments)),
            _ => None,
        })
        .collect();
    let mut message = json!({
        "role": "assistant",
        "content": if text.is_empty() && !tool_calls.is_empty() { Value::Null } else { json!(text) },
    });
    if !tool_calls.is_empty() {
        message["tool_calls"] = json!(tool_calls);
    }
    if !c.thinking.is_empty() {
        message["reasoning_content"] = json!(c.thinking);
    }
    Ok(Json(json!({
        "id": new_id("chatcmpl-"),
        "object": "chat.completion",
        "created": now_secs(),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason(c.finish)}],
        "usage": usage_json(&c.usage),
    }))
    .into_response())
}

fn tool_call_json(id: &str, name: &str, arguments: &Value) -> Value {
    json!({
        "id": id,
        "type": "function",
        "function": {"name": name, "arguments": arguments_string(arguments)}
    })
}

fn arguments_string(arguments: &Value) -> String {
    match arguments {
        Value::String(s) => s.clone(),
        other => other.to_string(),
    }
}

fn finish_reason(finish: Finish) -> &'static str {
    match finish {
        Finish::ToolCalls => "tool_calls",
        Finish::Length => "length",
        Finish::Refusal => "content_filter",
        Finish::Stop | Finish::Cancelled => "stop",
    }
}

fn usage_json(u: &FinalUsage) -> Value {
    json!({
        "prompt_tokens": u.input_tokens,
        "completion_tokens": u.output_tokens,
        "total_tokens": u.input_tokens + u.output_tokens,
        "prompt_tokens_details": {"cached_tokens": u.cached_tokens}
    })
}

fn stream_chunks(
    mut out: OutputStream,
    model: String,
    include_usage: bool,
) -> impl futures::Stream<Item = Result<Event, Infallible>> {
    let id = new_id("chatcmpl-");
    let created = now_secs();
    async_stream::stream! {
        let chunk = |delta: Value, finish: Option<&str>| {
            Event::default().data(json!({
                "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]
            }).to_string())
        };
        yield Ok(chunk(json!({"role": "assistant", "content": ""}), None));
        let mut tool_index = 0;
        while let Some(ev) = out.next().await {
            match ev {
                OutEvent::Text(t) => yield Ok(chunk(json!({"content": t}), None)),
                OutEvent::Thought(t) => yield Ok(chunk(json!({"reasoning_content": t}), None)),
                OutEvent::ToolCall { id, name, arguments } => {
                    let mut call = tool_call_json(&id, &name, &arguments);
                    call["index"] = json!(tool_index);
                    tool_index += 1;
                    yield Ok(chunk(json!({"tool_calls": [call]}), None));
                }
                OutEvent::Done { finish, usage } => {
                    yield Ok(chunk(json!({}), Some(finish_reason(finish))));
                    if include_usage {
                        yield Ok(Event::default().data(json!({
                            "id": id, "object": "chat.completion.chunk", "created": created, "model": model,
                            "choices": [], "usage": usage_json(&usage)
                        }).to_string()));
                    }
                }
                OutEvent::Error(e) => {
                    tracing::error!("agent error during stream: {e}");
                    yield Ok(Event::default().data(json!({
                        "error": {"message": e, "type": "server_error", "param": null, "code": null}
                    }).to_string()));
                }
            }
        }
        yield Ok(Event::default().data("[DONE]"));
    }
}

/// Convert a Chat Completions request into a [`ChatRequest`].
pub fn to_chat_request(body: &Value) -> Result<ChatRequest, String> {
    let mut req = ChatRequest {
        model: body
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        reasoning_effort: body.get("reasoning_effort").and_then(Value::as_str).map(str::to_string),
        ..Default::default()
    };
    let messages = body
        .get("messages")
        .and_then(Value::as_array)
        .ok_or("`messages` must be an array")?;
    let mut tool_names: HashMap<String, String> = HashMap::new();
    for msg in messages {
        let content = msg.get("content").unwrap_or(&Value::Null);
        match msg.get("role").and_then(Value::as_str).unwrap_or("user") {
            "system" | "developer" => req.system.push(content_to_text(content)),
            "assistant" => {
                let text = content_to_text(content);
                if !text.is_empty() {
                    req.push_part(Role::Assistant, Part::Text(text));
                }
                if let Some(calls) = msg.get("tool_calls").and_then(Value::as_array) {
                    for call in calls {
                        let id = call.get("id").and_then(Value::as_str).unwrap_or_default().to_string();
                        let name = call
                            .pointer("/function/name")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_string();
                        let raw = call.pointer("/function/arguments").cloned().unwrap_or(Value::Null);
                        let arguments = match &raw {
                            Value::String(s) => serde_json::from_str(s).unwrap_or(raw.clone()),
                            _ => raw,
                        };
                        tool_names.insert(id.clone(), name.clone());
                        req.push_part(Role::Assistant, Part::ToolCall { id, name, arguments });
                    }
                }
            }
            "tool" | "function" => {
                let id = msg
                    .get("tool_call_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                let name = tool_names
                    .get(&id)
                    .cloned()
                    .or_else(|| msg.get("name").and_then(Value::as_str).map(str::to_string));
                req.push_part(
                    Role::User,
                    Part::ToolResult {
                        id,
                        name,
                        content: content_to_text(content),
                        is_error: false,
                    },
                );
            }
            _ => match content {
                Value::Array(parts) => {
                    for p in parts {
                        match p.get("type").and_then(Value::as_str) {
                            Some("image_url") => {
                                let url = p
                                    .pointer("/image_url/url")
                                    .or_else(|| p.get("image_url"))
                                    .and_then(Value::as_str)
                                    .unwrap_or_default();
                                req.push_part(Role::User, image_from_url(url));
                            }
                            _ => {
                                let text = content_to_text(&Value::Array(vec![p.clone()]));
                                req.push_part(Role::User, Part::Text(text));
                            }
                        }
                    }
                }
                other => req.push_part(Role::User, Part::Text(content_to_text(other))),
            },
        }
    }

    if let Some(tools) = body.get("tools").and_then(Value::as_array) {
        for t in tools {
            match t.get("type").and_then(Value::as_str) {
                Some("function") | None => {
                    let f = t.get("function").unwrap_or(t);
                    let Some(name) = f.get("name").and_then(Value::as_str) else {
                        continue;
                    };
                    req.tools.push(ToolDef {
                        name: name.to_string(),
                        description: f.get("description").and_then(Value::as_str).map(str::to_string),
                        schema: f.get("parameters").cloned(),
                        freeform: false,
                    });
                }
                Some("custom") => {
                    let f = t.get("custom").unwrap_or(t);
                    let Some(name) = f.get("name").and_then(Value::as_str) else {
                        continue;
                    };
                    req.tools.push(ToolDef {
                        name: name.to_string(),
                        description: f.get("description").and_then(Value::as_str).map(str::to_string),
                        schema: None,
                        freeform: true,
                    });
                }
                Some(other) => debug!("skipping unsupported tool type {other}"),
            }
        }
    }

    req.tool_choice = parse_tool_choice(body.get("tool_choice"));
    Ok(req)
}

pub fn parse_tool_choice(v: Option<&Value>) -> ToolChoice {
    match v {
        Some(Value::String(s)) => match s.as_str() {
            "none" => ToolChoice::None,
            "required" => ToolChoice::Required,
            _ => ToolChoice::Auto,
        },
        Some(obj @ Value::Object(_)) => obj
            .pointer("/function/name")
            .or_else(|| obj.get("name"))
            .and_then(Value::as_str)
            .map(|n| ToolChoice::Specific(n.to_string()))
            .unwrap_or(ToolChoice::Auto),
        _ => ToolChoice::Auto,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_request() {
        let body = json!({
            "model": "gpt-5",
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": [{"type": "text", "text": "hi"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}]},
                {"role": "assistant", "content": null, "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{\"a\":1}"}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "ok"}
            ],
            "tools": [{"type": "function", "function": {"name": "f", "parameters": {"type": "object"}}}],
            "tool_choice": {"type": "function", "function": {"name": "f"}}
        });
        let req = to_chat_request(&body).unwrap();
        assert_eq!(req.system, vec!["sys"]);
        assert_eq!(req.messages.len(), 3);
        assert_eq!(req.messages[0].parts.len(), 2);
        assert_eq!(
            req.messages[1].parts[0],
            Part::ToolCall {
                id: "c1".into(),
                name: "f".into(),
                arguments: json!({"a": 1})
            }
        );
        assert!(matches!(&req.messages[2].parts[0], Part::ToolResult { name: Some(n), .. } if n == "f"));
        assert_eq!(req.tool_choice, ToolChoice::Specific("f".into()));
    }
}
