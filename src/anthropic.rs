//! Anthropic Messages API (`POST /v1/messages`), as used by Claude Code.

use std::collections::HashMap;
use std::convert::Infallible;

use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::response::sse::Event;
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};
use tracing::debug;

use crate::chat::{ChatRequest, Part, Role, ToolChoice, ToolDef, content_to_text, estimate_tokens};
use crate::output::{Finish, Item, OutEvent, OutputStream, new_id};
use crate::server::{ApiError, AppState, Flavor, parse_json, sse};

const FLAVOR: Flavor = Flavor::Anthropic;

pub async fn messages(State(state): State<AppState>, body: Bytes) -> Result<Response, ApiError> {
    let body = parse_json(FLAVOR, &body)?;
    tracing::trace!(%body, "anthropic request body");
    let req = to_chat_request(&body).map_err(|e| ApiError::bad_request(FLAVOR, e))?;
    let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let thinking = matches!(
        body.pointer("/thinking/type").and_then(Value::as_str),
        Some("enabled") | Some("adaptive")
    );
    let model = req.model.clone();
    debug!(
        model,
        stream,
        messages = req.messages.len(),
        tools = req.tools.len(),
        "anthropic request"
    );

    let turn = state
        .backend
        .start_turn(&req)
        .await
        .map_err(|e| ApiError::upstream(FLAVOR, e))?;
    let out = OutputStream::new(turn, &req);

    if stream {
        let input_estimate = estimate_tokens(body.to_string().len());
        Ok(sse(stream_events(out, model, input_estimate, thinking)))
    } else {
        let c = out.collect().await.map_err(|e| ApiError::upstream(FLAVOR, e))?;
        let mut content = Vec::new();
        if thinking && !c.thinking.is_empty() {
            content.push(json!({"type": "thinking", "thinking": c.thinking, "signature": ""}));
        }
        for item in &c.items {
            match item {
                Item::Text(t) => content.push(json!({"type": "text", "text": t})),
                Item::ToolCall { id, name, arguments } => content.push(json!({
                    "type": "tool_use", "id": id, "name": name, "input": arguments
                })),
            }
        }
        Ok(Json(json!({
            "id": new_id("msg_"),
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": content,
            "stop_reason": stop_reason(c.finish),
            "stop_sequence": null,
            "usage": {
                "input_tokens": c.usage.input_tokens,
                "output_tokens": c.usage.output_tokens,
                "cache_read_input_tokens": c.usage.cached_tokens,
                "cache_creation_input_tokens": 0
            }
        }))
        .into_response())
    }
}

pub async fn count_tokens(Json(body): Json<Value>) -> Json<Value> {
    // No tokenizer for the remote models: estimate from the payload size.
    let mut chars = 0;
    for key in ["system", "messages", "tools"] {
        if let Some(v) = body.get(key) {
            chars += v.to_string().len();
        }
    }
    Json(json!({"input_tokens": estimate_tokens(chars)}))
}

fn stop_reason(finish: Finish) -> &'static str {
    match finish {
        Finish::ToolCalls => "tool_use",
        Finish::Length => "max_tokens",
        Finish::Refusal => "refusal",
        Finish::Stop | Finish::Cancelled => "end_turn",
    }
}

fn event(name: &str, data: Value) -> Event {
    Event::default().event(name).data(data.to_string())
}

#[derive(PartialEq, Clone, Copy)]
enum Block {
    Text,
    Thinking,
}

/// Encodes [`OutEvent`]s as Anthropic streaming events.
struct Encoder {
    index: usize,
    open: Option<Block>,
    thinking: bool,
}

impl Encoder {
    fn close(&mut self, out: &mut Vec<Event>) {
        if self.open.take().is_some() {
            out.push(event(
                "content_block_stop",
                json!({"type": "content_block_stop", "index": self.index}),
            ));
            self.index += 1;
        }
    }

    fn open(&mut self, kind: Block, out: &mut Vec<Event>) {
        if self.open == Some(kind) {
            return;
        }
        self.close(out);
        let block = match kind {
            Block::Text => json!({"type": "text", "text": ""}),
            Block::Thinking => json!({"type": "thinking", "thinking": "", "signature": ""}),
        };
        out.push(event(
            "content_block_start",
            json!({"type": "content_block_start", "index": self.index, "content_block": block}),
        ));
        self.open = Some(kind);
    }

    fn delta(&self, delta: Value) -> Event {
        event(
            "content_block_delta",
            json!({"type": "content_block_delta", "index": self.index, "delta": delta}),
        )
    }

    fn encode(&mut self, ev: OutEvent) -> Vec<Event> {
        let mut out = Vec::new();
        match ev {
            OutEvent::Thought(t) => {
                if self.thinking {
                    self.open(Block::Thinking, &mut out);
                    out.push(self.delta(json!({"type": "thinking_delta", "thinking": t})));
                }
            }
            OutEvent::Text(t) => {
                self.open(Block::Text, &mut out);
                out.push(self.delta(json!({"type": "text_delta", "text": t})));
            }
            OutEvent::ToolCall { id, name, arguments } => {
                self.close(&mut out);
                out.push(event(
                    "content_block_start",
                    json!({
                        "type": "content_block_start", "index": self.index,
                        "content_block": {"type": "tool_use", "id": id, "name": name, "input": {}}
                    }),
                ));
                out.push(self.delta(json!({"type": "input_json_delta", "partial_json": arguments.to_string()})));
                out.push(event(
                    "content_block_stop",
                    json!({"type": "content_block_stop", "index": self.index}),
                ));
                self.index += 1;
            }
            OutEvent::Done { finish, usage } => {
                self.close(&mut out);
                out.push(event(
                    "message_delta",
                    json!({
                        "type": "message_delta",
                        "delta": {"stop_reason": stop_reason(finish), "stop_sequence": null},
                        "usage": {
                            "input_tokens": usage.input_tokens,
                            "output_tokens": usage.output_tokens,
                            "cache_read_input_tokens": usage.cached_tokens
                        }
                    }),
                ));
                out.push(event("message_stop", json!({"type": "message_stop"})));
            }
            OutEvent::Error(e) => {
                self.close(&mut out);
                tracing::error!("agent error during stream: {e}");
                out.push(event(
                    "error",
                    json!({"type": "error", "error": {"type": "api_error", "message": e}}),
                ));
            }
        }
        out
    }
}

fn stream_events(
    mut out: OutputStream,
    model: String,
    input_estimate: u64,
    thinking: bool,
) -> impl futures::Stream<Item = Result<Event, Infallible>> {
    async_stream::stream! {
        yield Ok(event("message_start", json!({
            "type": "message_start",
            "message": {
                "id": new_id("msg_"),
                "type": "message",
                "role": "assistant",
                "model": model,
                "content": [],
                "stop_reason": null,
                "stop_sequence": null,
                "usage": {"input_tokens": input_estimate, "output_tokens": 1}
            }
        })));
        yield Ok(event("ping", json!({"type": "ping"})));
        let mut encoder = Encoder { index: 0, open: None, thinking };
        while let Some(ev) = out.next().await {
            for e in encoder.encode(ev) {
                yield Ok(e);
            }
        }
    }
}

/// Convert an Anthropic Messages request into a [`ChatRequest`].
pub fn to_chat_request(body: &Value) -> Result<ChatRequest, String> {
    let mut req = ChatRequest {
        model: body
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        ..Default::default()
    };

    match body.get("system") {
        Some(Value::String(s)) => req.system.push(s.clone()),
        Some(v @ Value::Array(_)) => req.system.push(content_to_text(v)),
        _ => {}
    }

    let messages = body
        .get("messages")
        .and_then(Value::as_array)
        .ok_or("`messages` must be an array")?;
    let mut tool_names: HashMap<String, String> = HashMap::new();
    for msg in messages {
        let role = match msg.get("role").and_then(Value::as_str) {
            Some("assistant") => Role::Assistant,
            _ => Role::User,
        };
        match msg.get("content") {
            Some(Value::String(s)) => req.push_part(role, Part::Text(s.clone())),
            Some(Value::Array(blocks)) => {
                for block in blocks {
                    for part in convert_block(block, &mut tool_names) {
                        req.push_part(role, part);
                    }
                }
            }
            _ => {}
        }
    }

    if let Some(tools) = body.get("tools").and_then(Value::as_array) {
        for t in tools {
            let Some(name) = t.get("name").and_then(Value::as_str) else {
                continue;
            };
            let kind = t.get("type").and_then(Value::as_str).unwrap_or("custom");
            if kind != "custom" && t.get("input_schema").is_none() {
                debug!("skipping Anthropic server tool {name} ({kind})");
                continue;
            }
            req.tools.push(ToolDef {
                name: name.to_string(),
                description: t.get("description").and_then(Value::as_str).map(str::to_string),
                schema: t.get("input_schema").cloned(),
                freeform: false,
            });
        }
    }

    req.tool_choice = match body.pointer("/tool_choice/type").and_then(Value::as_str) {
        Some("any") => ToolChoice::Required,
        Some("none") => ToolChoice::None,
        Some("tool") => body
            .pointer("/tool_choice/name")
            .and_then(Value::as_str)
            .map(|n| ToolChoice::Specific(n.to_string()))
            .unwrap_or(ToolChoice::Required),
        _ => ToolChoice::Auto,
    };

    if let Some(budget) = body.pointer("/thinking/budget_tokens").and_then(Value::as_u64) {
        req.reasoning_effort = Some(
            match budget {
                0..4096 => "low",
                4096..16384 => "medium",
                _ => "high",
            }
            .to_string(),
        );
    }
    if let Some(effort) = body.pointer("/output_config/effort").and_then(Value::as_str) {
        req.reasoning_effort = Some(effort.to_string());
    }
    Ok(req)
}

fn convert_block(block: &Value, tool_names: &mut HashMap<String, String>) -> Vec<Part> {
    let kind = block.get("type").and_then(Value::as_str).unwrap_or("text");
    match kind {
        "text" => block
            .get("text")
            .and_then(Value::as_str)
            .map(|t| vec![Part::Text(t.to_string())])
            .unwrap_or_default(),
        "image" => image_part(block.get("source")).into_iter().collect(),
        "document" => {
            let source = block.get("source");
            match source.and_then(|s| s.get("type")).and_then(Value::as_str) {
                Some("text") => vec![Part::Text(
                    source
                        .and_then(|s| s.get("data"))
                        .and_then(Value::as_str)
                        .unwrap_or("")
                        .to_string(),
                )],
                Some("content") => vec![Part::Text(content_to_text(
                    source.and_then(|s| s.get("content")).unwrap_or(&Value::Null),
                ))],
                _ => {
                    let title = block.get("title").and_then(Value::as_str).unwrap_or("document");
                    vec![Part::Text(format!(
                        "[{title}: binary document not supported by the gateway]"
                    ))]
                }
            }
        }
        "tool_use" | "server_tool_use" => {
            let id = block.get("id").and_then(Value::as_str).unwrap_or_default().to_string();
            let name = block
                .get("name")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            tool_names.insert(id.clone(), name.clone());
            vec![Part::ToolCall {
                id,
                name,
                arguments: block.get("input").cloned().unwrap_or_else(|| json!({})),
            }]
        }
        "tool_result" => {
            let id = block
                .get("tool_use_id")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            let content = block.get("content").cloned().unwrap_or(Value::Null);
            let mut parts = vec![Part::ToolResult {
                name: tool_names.get(&id).cloned(),
                id,
                content: content_to_text(&content),
                is_error: block.get("is_error").and_then(Value::as_bool).unwrap_or(false),
            }];
            if let Value::Array(items) = &content {
                parts.extend(
                    items
                        .iter()
                        .filter(|i| i.get("type").and_then(Value::as_str) == Some("image"))
                        .filter_map(|i| image_part(i.get("source"))),
                );
            }
            parts
        }
        // Thinking blocks produced by previous turns are not replayed.
        "thinking" | "redacted_thinking" => vec![],
        _ => {
            let text = content_to_text(&Value::Array(vec![block.clone()]));
            if text.is_empty() {
                vec![]
            } else {
                vec![Part::Text(text)]
            }
        }
    }
}

fn image_part(source: Option<&Value>) -> Option<Part> {
    let source = source?;
    match source.get("type").and_then(Value::as_str)? {
        "base64" => Some(Part::Image {
            mime: source
                .get("media_type")
                .and_then(Value::as_str)
                .unwrap_or("image/png")
                .to_string(),
            data: source.get("data").and_then(Value::as_str)?.to_string(),
            url: None,
        }),
        "url" => Some(crate::chat::image_from_url(source.get("url").and_then(Value::as_str)?)),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_claude_code_style_request() {
        let body = json!({
            "model": "claude-sonnet-4-5-20250929",
            "system": [{"type": "text", "text": "You are Claude Code."}],
            "messages": [
                {"role": "user", "content": "list files"},
                {"role": "assistant", "content": [
                    {"type": "thinking", "thinking": "hmm", "signature": "x"},
                    {"type": "text", "text": "Sure."},
                    {"type": "tool_use", "id": "toolu_1", "name": "Bash", "input": {"command": "ls"}}
                ]},
                {"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": [{"type": "text", "text": "a.txt"}]},
                    {"type": "text", "text": "thanks"}
                ]}
            ],
            "tools": [
                {"name": "Bash", "description": "Run a command", "input_schema": {"type": "object"}},
                {"type": "web_search_20250305", "name": "web_search"}
            ],
            "tool_choice": {"type": "any"},
            "thinking": {"type": "enabled", "budget_tokens": 31999}
        });
        let req = to_chat_request(&body).unwrap();
        assert_eq!(req.system, vec!["You are Claude Code."]);
        assert_eq!(req.messages.len(), 3);
        assert_eq!(req.messages[1].parts.len(), 2);
        assert_eq!(
            req.messages[2].parts[0],
            Part::ToolResult {
                id: "toolu_1".into(),
                name: Some("Bash".into()),
                content: "a.txt".into(),
                is_error: false
            }
        );
        assert_eq!(req.tools.len(), 1);
        assert_eq!(req.tool_choice, ToolChoice::Required);
        assert_eq!(req.reasoning_effort.as_deref(), Some("high"));
    }
}
