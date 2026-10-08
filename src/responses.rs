//! OpenAI Responses API (`POST /v1/responses`), as used by Codex CLI.

use std::collections::HashSet;
use std::convert::Infallible;

use axum::Json;
use axum::body::Bytes;
use axum::extract::State;
use axum::response::sse::Event;
use axum::response::{IntoResponse, Response};
use serde_json::{Value, json};
use tracing::debug;

use crate::chat::{ChatRequest, Part, Role, ToolDef, content_to_text, image_from_url};
use crate::openai::parse_tool_choice;
use crate::output::{FinalUsage, Item, OutEvent, OutputStream, new_id, now_secs};
use crate::server::{ApiError, AppState, Flavor, parse_json, sse};

const FLAVOR: Flavor = Flavor::OpenAi;
const LOCAL_SHELL: &str = "local_shell";

pub async fn create(State(state): State<AppState>, body: Bytes) -> Result<Response, ApiError> {
    let body = parse_json(FLAVOR, &body)?;
    tracing::trace!(%body, "responses request body");
    let req = to_chat_request(&body).map_err(|e| ApiError::bad_request(FLAVOR, e))?;
    let stream = body.get("stream").and_then(Value::as_bool).unwrap_or(false);
    let reasoning = body.get("reasoning").is_some_and(|r| !r.is_null());
    debug!(
        model = req.model,
        stream,
        messages = req.messages.len(),
        tools = req.tools.len(),
        "responses request"
    );

    let turn = state
        .backend
        .start_turn(&req)
        .await
        .map_err(|e| ApiError::upstream(FLAVOR, e))?;
    let out = OutputStream::new(turn, &req);
    let mut encoder = Encoder::new(req.model.clone(), reasoning, req.namespaces.clone());

    if stream {
        return Ok(sse(stream_events(out, encoder)));
    }
    let c = out.collect().await.map_err(|e| ApiError::upstream(FLAVOR, e))?;
    if reasoning && !c.thinking.is_empty() {
        encoder.output.push(reasoning_item(new_id("rs_"), &c.thinking));
    }
    for item in c.items {
        let ev = match item {
            Item::Text(t) => OutEvent::Text(t),
            Item::ToolCall { id, name, arguments } => OutEvent::ToolCall { id, name, arguments },
        };
        encoder.encode(ev);
    }
    encoder.close_open();
    Ok(Json(encoder.response("completed", Some(&c.usage))).into_response())
}

fn stream_events(
    mut out: OutputStream,
    mut encoder: Encoder,
) -> impl futures::Stream<Item = Result<Event, Infallible>> {
    async_stream::stream! {
        let created = encoder.response("in_progress", None);
        yield Ok(encoder.event("response.created", json!({"response": created.clone()})));
        yield Ok(encoder.event("response.in_progress", json!({"response": created})));
        while let Some(ev) = out.next().await {
            for e in encoder.encode(ev) {
                yield Ok(e);
            }
        }
    }
}

enum Open {
    Message { id: String, text: String },
    Reasoning { id: String, text: String },
}

/// Encodes [`OutEvent`]s as Responses API items and streaming events.
struct Encoder {
    id: String,
    model: String,
    created_at: u64,
    reasoning: bool,
    seq: u64,
    output: Vec<Value>,
    open: Option<Open>,
    pending: Vec<Event>,
    namespaces: HashSet<String>,
}

fn message_item(id: &str, text: &str, status: &str) -> Value {
    json!({
        "type": "message", "id": id, "status": status, "role": "assistant",
        "content": [{"type": "output_text", "text": text, "annotations": []}]
    })
}

fn reasoning_item(id: String, text: &str) -> Value {
    json!({
        "type": "reasoning", "id": id,
        "summary": [{"type": "summary_text", "text": text}]
    })
}

impl Encoder {
    fn new(model: String, reasoning: bool, namespaces: HashSet<String>) -> Self {
        Self {
            namespaces,
            id: new_id("resp_"),
            model,
            created_at: now_secs(),
            reasoning,
            seq: 0,
            output: Vec::new(),
            open: None,
            pending: Vec::new(),
        }
    }

    fn event(&mut self, kind: &str, mut data: Value) -> Event {
        data["type"] = json!(kind);
        data["sequence_number"] = json!(self.seq);
        self.seq += 1;
        Event::default().event(kind).data(data.to_string())
    }

    fn emit(&mut self, kind: &str, data: Value) {
        let e = self.event(kind, data);
        self.pending.push(e);
    }

    fn response(&self, status: &str, usage: Option<&FinalUsage>) -> Value {
        json!({
            "id": self.id,
            "object": "response",
            "created_at": self.created_at,
            "status": status,
            "model": self.model,
            "output": if status == "in_progress" { json!([]) } else { json!(self.output) },
            "error": null,
            "incomplete_details": null,
            "parallel_tool_calls": true,
            "tool_choice": "auto",
            "tools": [],
            "usage": usage.map(|u| json!({
                "input_tokens": u.input_tokens,
                "input_tokens_details": {"cached_tokens": u.cached_tokens},
                "output_tokens": u.output_tokens,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": u.input_tokens + u.output_tokens
            })),
        })
    }

    fn close_open(&mut self) {
        let output_index = self.output.len();
        match self.open.take() {
            Some(Open::Message { id, text }) => {
                self.emit(
                    "response.output_text.done",
                    json!({
                        "item_id": id, "output_index": output_index, "content_index": 0, "text": text
                    }),
                );
                self.emit(
                    "response.content_part.done",
                    json!({
                        "item_id": id, "output_index": output_index, "content_index": 0,
                        "part": {"type": "output_text", "text": text, "annotations": []}
                    }),
                );
                let item = message_item(&id, &text, "completed");
                self.emit(
                    "response.output_item.done",
                    json!({"output_index": output_index, "item": item.clone()}),
                );
                self.output.push(item);
            }
            Some(Open::Reasoning { id, text }) => {
                self.emit(
                    "response.reasoning_summary_text.done",
                    json!({
                        "item_id": id, "output_index": output_index, "summary_index": 0, "text": text
                    }),
                );
                self.emit(
                    "response.reasoning_summary_part.done",
                    json!({
                        "item_id": id, "output_index": output_index, "summary_index": 0,
                        "part": {"type": "summary_text", "text": text}
                    }),
                );
                let item = reasoning_item(id, &text);
                self.emit(
                    "response.output_item.done",
                    json!({"output_index": output_index, "item": item.clone()}),
                );
                self.output.push(item);
            }
            None => {}
        }
    }

    fn encode(&mut self, ev: OutEvent) -> Vec<Event> {
        match ev {
            OutEvent::Text(t) => {
                if !matches!(self.open, Some(Open::Message { .. })) {
                    self.close_open();
                    let id = new_id("msg_");
                    let output_index = self.output.len();
                    self.emit("response.output_item.added", json!({
                        "output_index": output_index,
                        "item": {"type": "message", "id": id, "status": "in_progress", "role": "assistant", "content": []}
                    }));
                    self.emit(
                        "response.content_part.added",
                        json!({
                            "item_id": id, "output_index": output_index, "content_index": 0,
                            "part": {"type": "output_text", "text": "", "annotations": []}
                        }),
                    );
                    self.open = Some(Open::Message {
                        id,
                        text: String::new(),
                    });
                }
                if let Some(Open::Message { id, text }) = &mut self.open {
                    text.push_str(&t);
                    let id = id.clone();
                    self.emit(
                        "response.output_text.delta",
                        json!({
                            "item_id": id, "output_index": self.output.len(), "content_index": 0, "delta": t
                        }),
                    );
                }
            }
            OutEvent::Thought(t) => {
                if !self.reasoning {
                    return std::mem::take(&mut self.pending);
                }
                if !matches!(self.open, Some(Open::Reasoning { .. })) {
                    self.close_open();
                    let id = new_id("rs_");
                    let output_index = self.output.len();
                    self.emit(
                        "response.output_item.added",
                        json!({
                            "output_index": output_index,
                            "item": {"type": "reasoning", "id": id, "summary": []}
                        }),
                    );
                    self.emit(
                        "response.reasoning_summary_part.added",
                        json!({
                            "item_id": id, "output_index": output_index, "summary_index": 0,
                            "part": {"type": "summary_text", "text": ""}
                        }),
                    );
                    self.open = Some(Open::Reasoning {
                        id,
                        text: String::new(),
                    });
                }
                if let Some(Open::Reasoning { id, text }) = &mut self.open {
                    text.push_str(&t);
                    let id = id.clone();
                    self.emit(
                        "response.reasoning_summary_text.delta",
                        json!({
                            "item_id": id, "output_index": self.output.len(), "summary_index": 0, "delta": t
                        }),
                    );
                }
            }
            OutEvent::ToolCall { id, name, arguments } => {
                self.close_open();
                let output_index = self.output.len();
                let call_id = id;
                let item = if name == LOCAL_SHELL {
                    let action = json!({
                        "type": "exec",
                        "command": arguments.get("command").cloned().unwrap_or(json!([])),
                        "timeout_ms": arguments.get("timeout_ms").cloned().unwrap_or(Value::Null),
                        "working_directory": arguments.get("workdir")
                            .or_else(|| arguments.get("working_directory")).cloned().unwrap_or(Value::Null),
                        "env": arguments.get("env").cloned().unwrap_or(Value::Null),
                        "user": null
                    });
                    json!({"type": "local_shell_call", "id": new_id("lsh_"), "call_id": call_id,
                           "status": "completed", "action": action})
                } else {
                    let (namespace, tool) = split_namespace(&name, &self.namespaces);
                    let mut item = if let Value::String(input) = &arguments {
                        json!({"type": "custom_tool_call", "id": new_id("ctc_"), "call_id": call_id,
                               "name": tool, "input": input, "status": "completed"})
                    } else {
                        json!({"type": "function_call", "id": new_id("fc_"), "call_id": call_id,
                               "name": tool, "arguments": arguments.to_string(), "status": "completed"})
                    };
                    if let Some(ns) = namespace {
                        item["namespace"] = json!(ns);
                    }
                    item
                };
                let id = item["id"].clone();
                let mut added = item.clone();
                match item["type"].as_str() {
                    Some("function_call") => {
                        added["arguments"] = json!("");
                        added["status"] = json!("in_progress");
                        self.emit(
                            "response.output_item.added",
                            json!({"output_index": output_index, "item": added}),
                        );
                        self.emit(
                            "response.function_call_arguments.delta",
                            json!({
                                "item_id": id, "output_index": output_index, "delta": item["arguments"]
                            }),
                        );
                        self.emit(
                            "response.function_call_arguments.done",
                            json!({
                                "item_id": id, "output_index": output_index, "arguments": item["arguments"]
                            }),
                        );
                    }
                    Some("custom_tool_call") => {
                        added["input"] = json!("");
                        added["status"] = json!("in_progress");
                        self.emit(
                            "response.output_item.added",
                            json!({"output_index": output_index, "item": added}),
                        );
                        self.emit(
                            "response.custom_tool_call_input.delta",
                            json!({
                                "item_id": id, "output_index": output_index, "delta": item["input"]
                            }),
                        );
                        self.emit(
                            "response.custom_tool_call_input.done",
                            json!({
                                "item_id": id, "output_index": output_index, "input": item["input"]
                            }),
                        );
                    }
                    _ => {
                        self.emit(
                            "response.output_item.added",
                            json!({"output_index": output_index, "item": added}),
                        );
                    }
                }
                self.emit(
                    "response.output_item.done",
                    json!({"output_index": output_index, "item": item.clone()}),
                );
                self.output.push(item);
            }
            OutEvent::Done { usage, .. } => {
                self.close_open();
                let response = self.response("completed", Some(&usage));
                self.emit("response.completed", json!({"response": response}));
            }
            OutEvent::Error(e) => {
                self.close_open();
                tracing::error!("agent error during stream: {e}");
                let mut response = self.response("failed", None);
                response["error"] = json!({"code": "server_error", "message": e});
                self.emit("response.failed", json!({"response": response}));
            }
        }
        std::mem::take(&mut self.pending)
    }
}

/// Convert a Responses API request into a [`ChatRequest`].
pub fn to_chat_request(body: &Value) -> Result<ChatRequest, String> {
    let mut req = ChatRequest {
        model: body
            .get("model")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string(),
        reasoning_effort: body
            .pointer("/reasoning/effort")
            .and_then(Value::as_str)
            .map(str::to_string),
        ..Default::default()
    };
    if let Some(instructions) = body.get("instructions").and_then(Value::as_str) {
        req.system.push(instructions.to_string());
    }

    match body.get("input") {
        Some(Value::String(s)) => req.push_part(Role::User, Part::Text(s.clone())),
        Some(Value::Array(items)) => {
            let mut names = std::collections::HashMap::new();
            for item in items {
                convert_item(&mut req, item, &mut names);
            }
        }
        _ => return Err("`input` must be a string or an array".into()),
    }

    if let Some(tools) = body.get("tools").and_then(Value::as_array) {
        add_tools(&mut req, tools, None);
    }
    req.tool_choice = parse_tool_choice(body.get("tool_choice"));
    Ok(req)
}

/// Register tool definitions. Tools inside a `namespace` are exposed to the
/// model as `namespace.tool`.
fn add_tools(req: &mut ChatRequest, tools: &[Value], namespace: Option<&str>) {
    for t in tools {
        let kind = t.get("type").and_then(Value::as_str).unwrap_or("function");
        let name = t.get("name").and_then(Value::as_str).map(|n| match namespace {
            Some(ns) => format!("{ns}.{n}"),
            None => n.to_string(),
        });
        let description = t.get("description").and_then(Value::as_str).map(str::to_string);
        match (kind, name) {
            ("namespace", Some(_)) => {
                let ns = t.get("name").and_then(Value::as_str).unwrap_or_default();
                if let Some(nested) = t.get("tools").and_then(Value::as_array) {
                    req.namespaces.insert(ns.to_string());
                    add_tools(req, nested, Some(ns));
                }
            }
            ("function", Some(name)) => req.tools.push(ToolDef {
                name,
                description,
                schema: t.get("parameters").cloned(),
                freeform: false,
            }),
            ("custom", Some(name)) => {
                let mut description = description.unwrap_or_default();
                if let Some(def) = t.pointer("/format/definition").and_then(Value::as_str) {
                    let syntax = t.pointer("/format/syntax").and_then(Value::as_str).unwrap_or("grammar");
                    description.push_str(&format!("\n\nThe input must follow this {syntax} grammar:\n{def}"));
                }
                req.tools.push(ToolDef {
                    name,
                    description: Some(description),
                    schema: None,
                    freeform: true,
                });
            }
            ("local_shell", _) => req.tools.push(ToolDef {
                name: LOCAL_SHELL.into(),
                description: Some("Run a shell command on the user's machine.".into()),
                schema: Some(json!({
                    "type": "object",
                    "properties": {
                        "command": {"type": "array", "items": {"type": "string"},
                                    "description": "Command and arguments, e.g. [\"bash\", \"-lc\", \"ls\"]"},
                        "workdir": {"type": "string"},
                        "timeout_ms": {"type": "integer"}
                    },
                    "required": ["command"]
                })),
                freeform: false,
            }),
            (other, _) => debug!("skipping unsupported tool type {other}"),
        }
    }
}

/// Split `namespace.tool` back into its parts for known namespaces.
fn split_namespace<'a>(name: &'a str, namespaces: &HashSet<String>) -> (Option<&'a str>, &'a str) {
    match name.split_once('.') {
        Some((ns, tool)) if namespaces.contains(ns) => (Some(ns), tool),
        _ => (None, name),
    }
}

fn qualified_name(item: &Value) -> String {
    let name = item.get("name").and_then(Value::as_str).unwrap_or_default();
    match item.get("namespace").and_then(Value::as_str) {
        Some(ns) if !ns.is_empty() => format!("{ns}.{name}"),
        _ => name.to_string(),
    }
}

fn convert_item(req: &mut ChatRequest, item: &Value, names: &mut std::collections::HashMap<String, String>) {
    let kind = item.get("type").and_then(Value::as_str).unwrap_or("message");
    let call_id = || {
        item.get("call_id")
            .and_then(Value::as_str)
            .unwrap_or_default()
            .to_string()
    };
    match kind {
        "message" => {
            let content = item.get("content").unwrap_or(&Value::Null);
            match item.get("role").and_then(Value::as_str).unwrap_or("user") {
                "system" | "developer" => req.system.push(content_to_text(content)),
                role => {
                    let role = if role == "assistant" {
                        Role::Assistant
                    } else {
                        Role::User
                    };
                    match content {
                        Value::Array(parts) => {
                            for p in parts {
                                match p.get("type").and_then(Value::as_str) {
                                    Some("input_image") => {
                                        let url = p.get("image_url").and_then(Value::as_str).unwrap_or_default();
                                        req.push_part(role, image_from_url(url));
                                    }
                                    _ => {
                                        let text = content_to_text(&Value::Array(vec![p.clone()]));
                                        if !text.is_empty() {
                                            req.push_part(role, Part::Text(text));
                                        }
                                    }
                                }
                            }
                        }
                        other => req.push_part(role, Part::Text(content_to_text(other))),
                    }
                }
            }
        }
        "additional_tools" => {
            if let Some(tools) = item.get("tools").and_then(Value::as_array) {
                add_tools(req, tools, None);
            }
        }
        "function_call" => {
            let name = qualified_name(item);
            let raw = item.get("arguments").cloned().unwrap_or(Value::Null);
            let arguments = match &raw {
                Value::String(s) => serde_json::from_str(s).unwrap_or(raw.clone()),
                _ => raw,
            };
            names.insert(call_id(), name.clone());
            req.push_part(
                Role::Assistant,
                Part::ToolCall {
                    id: call_id(),
                    name,
                    arguments,
                },
            );
        }
        "custom_tool_call" => {
            let name = qualified_name(item);
            names.insert(call_id(), name.clone());
            req.push_part(
                Role::Assistant,
                Part::ToolCall {
                    id: call_id(),
                    name,
                    arguments: item.get("input").cloned().unwrap_or(json!("")),
                },
            );
        }
        "local_shell_call" => {
            let action = item.get("action").cloned().unwrap_or(Value::Null);
            names.insert(call_id(), LOCAL_SHELL.into());
            req.push_part(
                Role::Assistant,
                Part::ToolCall {
                    id: call_id(),
                    name: LOCAL_SHELL.into(),
                    arguments: json!({
                        "command": action.get("command").cloned().unwrap_or(json!([])),
                        "workdir": action.get("working_directory").cloned().unwrap_or(Value::Null),
                    }),
                },
            );
        }
        "function_call_output" | "custom_tool_call_output" | "local_shell_call_output" => {
            let id = call_id();
            let output = item.get("output").unwrap_or(&Value::Null);
            let content = match output {
                Value::Object(o) => o
                    .get("content")
                    .map(content_to_text)
                    .unwrap_or_else(|| output.to_string()),
                other => content_to_text(other),
            };
            req.push_part(
                Role::User,
                Part::ToolResult {
                    name: names.get(&id).cloned(),
                    id,
                    content,
                    is_error: false,
                },
            );
        }
        // Reasoning and hosted tool items are not replayed.
        _ => debug!("skipping input item of type {kind}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::output::Finish;

    #[test]
    fn converts_codex_style_request() {
        let body = json!({
            "model": "gpt-5-codex",
            "instructions": "You are Codex.",
            "input": [
                {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "env"}]},
                {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "fix it"}]},
                {"type": "reasoning", "id": "rs_1", "summary": []},
                {"type": "function_call", "name": "shell", "arguments": "{\"command\":[\"ls\"]}", "call_id": "c1"},
                {"type": "function_call_output", "call_id": "c1", "output": "a.rs"},
                {"type": "custom_tool_call", "name": "apply_patch", "input": "*** Begin Patch", "call_id": "c2"},
                {"type": "custom_tool_call_output", "call_id": "c2", "output": "Done"}
            ],
            "tools": [
                {"type": "function", "name": "shell", "parameters": {"type": "object"}},
                {"type": "custom", "name": "apply_patch", "description": "patch",
                 "format": {"type": "grammar", "syntax": "lark", "definition": "start: x"}},
                {"type": "web_search"},
                {"type": "namespace", "name": "multi_agent", "tools": [
                    {"type": "function", "name": "spawn", "parameters": {"type": "object"}}]}
            ],
            "reasoning": {"effort": "medium", "summary": "auto"},
            "stream": true
        });
        let req = to_chat_request(&body).unwrap();
        assert_eq!(req.system, vec!["You are Codex.", "env"]);
        assert_eq!(req.messages.len(), 5);
        assert_eq!(req.tools.len(), 3);
        assert_eq!(req.tools[2].name, "multi_agent.spawn");
        assert!(req.namespaces.contains("multi_agent"));
        assert!(req.tools[1].freeform);
        assert!(req.tools[1].description.as_ref().unwrap().contains("start: x"));
        assert_eq!(req.reasoning_effort.as_deref(), Some("medium"));
        assert!(matches!(&req.messages[4].parts[0], Part::ToolResult { name: Some(n), .. } if n == "apply_patch"));
    }

    #[test]
    fn encoder_produces_items() {
        let mut enc = Encoder::new("m".into(), true, HashSet::from(["multi_agent".to_string()]));
        enc.encode(OutEvent::Thought("think".into()));
        enc.encode(OutEvent::Text("Hello ".into()));
        enc.encode(OutEvent::Text("world".into()));
        enc.encode(OutEvent::ToolCall {
            id: "c0".into(),
            name: "shell".into(),
            arguments: json!({"command": ["ls"]}),
        });
        enc.encode(OutEvent::ToolCall {
            id: "c1".into(),
            name: "apply_patch".into(),
            arguments: json!("*** Begin Patch"),
        });
        enc.encode(OutEvent::ToolCall {
            id: "c2".into(),
            name: "multi_agent.spawn".into(),
            arguments: json!({}),
        });
        enc.encode(OutEvent::Done {
            finish: Finish::ToolCalls,
            usage: FinalUsage::default(),
        });
        let types: Vec<&str> = enc.output.iter().map(|i| i["type"].as_str().unwrap()).collect();
        assert_eq!(
            types,
            vec![
                "reasoning",
                "message",
                "function_call",
                "custom_tool_call",
                "function_call"
            ]
        );
        assert_eq!(enc.output[4]["name"], "spawn");
        assert_eq!(enc.output[4]["namespace"], "multi_agent");
        assert_eq!(enc.output[1]["content"][0]["text"], "Hello world");
        assert_eq!(enc.output[2]["arguments"], "{\"command\":[\"ls\"]}");
        assert_eq!(enc.seq, 26);
    }
}
