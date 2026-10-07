//! Provider-neutral representation of a chat request.
//!
//! The Anthropic, OpenAI Chat Completions and OpenAI Responses front-ends all
//! convert their payloads into a [`ChatRequest`], which is then rendered into
//! an ACP prompt by [`crate::prompt`].

use serde_json::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    User,
    Assistant,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Part {
    Text(String),
    /// Base64 encoded image, or a remote URL when `data` is empty.
    Image {
        mime: String,
        data: String,
        url: Option<String>,
    },
    ToolCall {
        id: String,
        name: String,
        /// JSON arguments (object for function tools, string for freeform tools).
        arguments: Value,
    },
    ToolResult {
        id: String,
        name: Option<String>,
        content: String,
        is_error: bool,
    },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Message {
    pub role: Role,
    pub parts: Vec<Part>,
}

impl Message {
    pub fn new(role: Role) -> Self {
        Self {
            role,
            parts: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct ToolDef {
    pub name: String,
    pub description: Option<String>,
    /// JSON schema of the arguments (function tools).
    pub schema: Option<Value>,
    /// Freeform tools take a raw string as input instead of a JSON object.
    pub freeform: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub enum ToolChoice {
    #[default]
    Auto,
    None,
    Required,
    Specific(String),
}

#[derive(Debug, Clone, Default)]
pub struct ChatRequest {
    pub model: String,
    pub system: Vec<String>,
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDef>,
    pub tool_choice: ToolChoice,
    /// Requested reasoning effort (low/medium/high...), if any.
    pub reasoning_effort: Option<String>,
    /// Tool namespaces (Responses API); their tools are named `namespace.tool`.
    pub namespaces: std::collections::HashSet<String>,
}

impl ChatRequest {
    /// Push a part to the last message if it has the given role, otherwise
    /// start a new message. Keeps consecutive same-role content together.
    pub fn push_part(&mut self, role: Role, part: Part) {
        match self.messages.last_mut() {
            Some(last) if last.role == role => last.parts.push(part),
            _ => self.messages.push(Message {
                role,
                parts: vec![part],
            }),
        }
    }

    pub fn tool_is_freeform(&self, name: &str) -> bool {
        self.tools.iter().any(|t| t.name == name && t.freeform)
    }
}

/// Collect text out of the most common "content" shapes: a plain string, or
/// an array of `{type: "...text", text}` blocks.
pub fn content_to_text(content: &Value) -> String {
    match content {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        Value::Array(items) => {
            let mut out = Vec::new();
            for item in items {
                match item {
                    Value::String(s) => out.push(s.clone()),
                    Value::Object(obj) => {
                        if let Some(t) = obj.get("text").and_then(Value::as_str) {
                            out.push(t.to_string());
                        } else if let Some(t) = obj.get("type").and_then(Value::as_str)
                            && t.contains("image")
                        {
                            out.push("[image]".to_string());
                        }
                    }
                    _ => {}
                }
            }
            out.join("\n")
        }
        other => other.to_string(),
    }
}

/// Parse a `data:` URL into (mime, base64 data).
pub fn parse_data_url(url: &str) -> Option<(String, String)> {
    let rest = url.strip_prefix("data:")?;
    let (meta, data) = rest.split_once(',')?;
    let mime = meta.strip_suffix(";base64")?;
    Some((mime.to_string(), data.to_string()))
}

pub fn image_from_url(url: &str) -> Part {
    match parse_data_url(url) {
        Some((mime, data)) => Part::Image { mime, data, url: None },
        None => Part::Image {
            mime: String::new(),
            data: String::new(),
            url: Some(url.to_string()),
        },
    }
}

/// Rough token estimate (no tokenizer available for the remote models).
pub fn estimate_tokens(chars: usize) -> u64 {
    (chars as u64).div_ceil(4)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn data_url() {
        assert_eq!(
            parse_data_url("data:image/png;base64,AAAA"),
            Some(("image/png".into(), "AAAA".into()))
        );
        assert_eq!(parse_data_url("https://x/y.png"), None);
    }

    #[test]
    fn content_text() {
        assert_eq!(content_to_text(&json!("hi")), "hi");
        assert_eq!(
            content_to_text(&json!([{"type":"text","text":"a"},{"type":"input_text","text":"b"}])),
            "a\nb"
        );
    }
}
