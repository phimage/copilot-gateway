//! Rendering of a [`ChatRequest`] into an ACP prompt, and parsing of the
//! agent's textual output back into text / tool calls.
//!
//! ACP agents such as `copilot --acp` own their tools and do not expose the
//! raw model API, so client-side tools (Claude Code's `Bash`, Codex's
//! `shell`, ...) are emulated: tool definitions are described in the prompt
//! and the model is asked to emit `<tool_call>{json}</tool_call>` blocks,
//! which [`ToolCallParser`] turns back into structured tool calls.

use serde_json::{Value, json};

use crate::chat::{ChatRequest, Part, Role, ToolChoice};

pub const TOOL_CALL_OPEN: &str = "<tool_call>";
pub const TOOL_CALL_CLOSE: &str = "</tool_call>";

const GATEWAY_PREAMBLE: &str = "You are acting as the language model behind an API gateway. \
A client application (for example a coding assistant) drives the conversation below and \
provides its own system prompt and tools. Follow the client's system prompt. Do NOT use any \
of your own built-in tools (no file access, no shell commands, no web access, no sub-agents): \
answer directly with text, and only use the client tools described below, through the exact \
text protocol described.";

/// Render the request into ACP `ContentBlock`s.
pub fn render(req: &ChatRequest) -> Vec<Value> {
    let mut blocks = PromptBuilder::default();

    let simple = req.system.is_empty()
        && req.tools.is_empty()
        && req.messages.len() == 1
        && req.messages[0].role == Role::User
        && req.messages[0]
            .parts
            .iter()
            .all(|p| matches!(p, Part::Text(_) | Part::Image { .. }));

    if simple {
        // Plain single-turn chat: forward the user message as is.
        for part in &req.messages[0].parts {
            blocks.part(part);
        }
        return blocks.finish();
    }

    blocks.text(GATEWAY_PREAMBLE);
    blocks.text("\n\n");

    let system = req.system.join("\n\n");
    if !system.trim().is_empty() {
        blocks.text("<system>\n");
        blocks.text(system.trim());
        blocks.text("\n</system>\n\n");
    }

    let tools_enabled = !req.tools.is_empty() && req.tool_choice != ToolChoice::None;
    if tools_enabled {
        blocks.text(&tool_instructions(req));
        blocks.text("\n\n");
    }

    blocks.text("<conversation>\n");
    for msg in &req.messages {
        let tag = match msg.role {
            Role::User => "user",
            Role::Assistant => "assistant",
        };
        blocks.text(&format!("<{tag}>\n"));
        for part in &msg.parts {
            blocks.part(part);
        }
        blocks.text(&format!("\n</{tag}>\n"));
    }
    blocks.text("</conversation>\n\n");

    let mut closing = String::from(
        "Write the assistant's next reply in this conversation now. Output only the content \
of the reply itself (no <assistant> tags, no commentary about this format).",
    );
    match &req.tool_choice {
        ToolChoice::Required if tools_enabled => closing.push_str(" You MUST call at least one tool in this reply."),
        ToolChoice::Specific(name) if tools_enabled => {
            closing.push_str(&format!(" You MUST call the tool `{name}` in this reply."))
        }
        _ => {}
    }
    blocks.text(&closing);
    blocks.finish()
}

fn tool_instructions(req: &ChatRequest) -> String {
    let mut s = String::from(
        "# Client tools\n\
You can call the following tools. They are executed by the client application, not by you. \
They are the ONLY tools you may use.\n\n<tools>\n",
    );
    for tool in &req.tools {
        let mut def = serde_json::Map::new();
        def.insert("name".into(), json!(tool.name));
        if let Some(d) = &tool.description {
            def.insert("description".into(), json!(d));
        }
        if tool.freeform {
            def.insert("freeform".into(), json!(true));
        } else {
            def.insert(
                "parameters".into(),
                tool.schema.clone().unwrap_or_else(|| json!({"type": "object"})),
            );
        }
        s.push_str(&Value::Object(def).to_string());
        s.push('\n');
    }
    s.push_str(
        "</tools>\n\n\
To call a tool, write a block exactly like this in your reply:\n\
<tool_call>\n\
{\"name\": \"<tool name>\", \"arguments\": {<arguments matching the tool parameters schema>}}\n\
</tool_call>\n\n\
Rules:\n\
- You may write a short text before the tool call blocks. You may call several tools in one reply, one block per call.\n\
- The content of a block must be a single valid JSON object with \"name\" and \"arguments\".\n\
- For tools marked \"freeform\": true, \"arguments\" must be a JSON string containing the raw input.\n\
- After your tool call blocks, stop and end your reply immediately. Never invent tool results: \
the client runs the tools and sends the results back in the next user turn inside <tool_result> blocks.\n\
- If no tool is needed, just answer with text.",
    );
    s
}

#[derive(Default)]
struct PromptBuilder {
    blocks: Vec<Value>,
    text: String,
}

impl PromptBuilder {
    fn text(&mut self, s: &str) {
        self.text.push_str(s);
    }

    fn flush(&mut self) {
        if !self.text.is_empty() {
            self.blocks
                .push(json!({"type": "text", "text": std::mem::take(&mut self.text)}));
        }
    }

    fn part(&mut self, part: &Part) {
        match part {
            Part::Text(t) => self.text(t),
            Part::Image { mime, data, url } => {
                if data.is_empty() {
                    // Remote images are passed as a link (the agent may not fetch it).
                    self.text(&format!("[image: {}]", url.as_deref().unwrap_or("")));
                } else {
                    self.flush();
                    self.blocks
                        .push(json!({"type": "image", "mimeType": mime, "data": data}));
                }
            }
            Part::ToolCall { id: _, name, arguments } => {
                let call = json!({"name": name, "arguments": arguments});
                self.text(&format!("\n{TOOL_CALL_OPEN}\n{call}\n{TOOL_CALL_CLOSE}\n"));
            }
            Part::ToolResult {
                id,
                name,
                content,
                is_error,
            } => {
                let mut attrs = format!("id=\"{id}\"");
                if let Some(n) = name {
                    attrs.push_str(&format!(" name=\"{n}\""));
                }
                if *is_error {
                    attrs.push_str(" error=\"true\"");
                }
                self.text(&format!("\n<tool_result {attrs}>\n{content}\n</tool_result>\n"));
            }
        }
    }

    fn finish(mut self) -> Vec<Value> {
        self.flush();
        if self.blocks.is_empty() {
            self.blocks.push(json!({"type": "text", "text": ""}));
        }
        self.blocks
    }
}

/// Output of the streaming parser.
#[derive(Debug, Clone, PartialEq)]
pub enum Parsed {
    Text(String),
    ToolCall { name: String, arguments: Value },
}

/// Incremental parser extracting `<tool_call>` blocks from streamed text.
pub struct ToolCallParser {
    enabled: bool,
    buf: String,
    in_call: bool,
    seen_call: bool,
}

impl ToolCallParser {
    pub fn new(enabled: bool) -> Self {
        Self {
            enabled,
            buf: String::new(),
            in_call: false,
            seen_call: false,
        }
    }

    pub fn push(&mut self, chunk: &str) -> Vec<Parsed> {
        if !self.enabled {
            return if chunk.is_empty() {
                vec![]
            } else {
                vec![Parsed::Text(chunk.to_string())]
            };
        }
        self.buf.push_str(chunk);
        let mut out = Vec::new();
        loop {
            if self.in_call {
                match self.buf.find(TOOL_CALL_CLOSE) {
                    Some(idx) => {
                        let body = self.buf[..idx].to_string();
                        self.buf.drain(..idx + TOOL_CALL_CLOSE.len());
                        self.in_call = false;
                        out.push(self.parse_call(&body));
                    }
                    None => break,
                }
            } else {
                match self.buf.find(TOOL_CALL_OPEN) {
                    Some(idx) => {
                        let text = self.buf[..idx].to_string();
                        self.emit_text(&mut out, text);
                        self.buf.drain(..idx + TOOL_CALL_OPEN.len());
                        self.in_call = true;
                    }
                    None => {
                        // Keep back a suffix that could be the beginning of the tag.
                        let keep = partial_suffix_len(&self.buf, TOOL_CALL_OPEN);
                        let cut = self.buf.len() - keep;
                        let text = self.buf[..cut].to_string();
                        self.buf.drain(..cut);
                        self.emit_text(&mut out, text);
                        break;
                    }
                }
            }
        }
        out
    }

    pub fn finish(&mut self) -> Vec<Parsed> {
        let mut out = Vec::new();
        let rest = std::mem::take(&mut self.buf);
        if self.in_call {
            self.in_call = false;
            // Unterminated block: accept it if it is valid JSON.
            match parse_call_json(&rest) {
                Some((name, arguments)) => {
                    self.seen_call = true;
                    out.push(Parsed::ToolCall { name, arguments });
                }
                None => out.push(Parsed::Text(format!("{TOOL_CALL_OPEN}{rest}"))),
            }
        } else {
            self.emit_text(&mut out, rest);
        }
        out
    }

    fn emit_text(&mut self, out: &mut Vec<Parsed>, text: String) {
        if text.is_empty() || (self.seen_call && text.trim().is_empty()) {
            return;
        }
        out.push(Parsed::Text(text));
    }

    fn parse_call(&mut self, body: &str) -> Parsed {
        match parse_call_json(body) {
            Some((name, arguments)) => {
                self.seen_call = true;
                Parsed::ToolCall { name, arguments }
            }
            None => Parsed::Text(format!("{TOOL_CALL_OPEN}{body}{TOOL_CALL_CLOSE}")),
        }
    }
}

fn partial_suffix_len(buf: &str, tag: &str) -> usize {
    let max = tag.len().saturating_sub(1).min(buf.len());
    for len in (1..=max).rev() {
        if buf.is_char_boundary(buf.len() - len) && tag.starts_with(&buf[buf.len() - len..]) {
            return len;
        }
    }
    0
}

fn parse_call_json(body: &str) -> Option<(String, Value)> {
    let mut s = body.trim();
    if let Some(rest) = s.strip_prefix("```") {
        // Strip a markdown fence (```json ... ```).
        let rest = rest.split_once('\n').map(|(_, r)| r).unwrap_or(rest);
        s = rest.trim_end().strip_suffix("```").unwrap_or(rest).trim();
    }
    let v: Value = serde_json::from_str(s).ok()?;
    let obj = v.as_object()?;
    let name = obj.get("name")?.as_str()?.to_string();
    let arguments = obj
        .get("arguments")
        .or_else(|| obj.get("input"))
        .or_else(|| obj.get("parameters"))
        .cloned()
        .unwrap_or_else(|| json!({}));
    Some((name, arguments))
}

/// Normalize tool arguments according to the tool kind: function tools get a
/// JSON object, freeform tools get a raw string.
pub fn normalize_arguments(arguments: Value, freeform: bool) -> Value {
    match (freeform, arguments) {
        (true, Value::String(s)) => Value::String(s),
        (true, other) => match other {
            Value::Object(ref o) if o.len() == 1 => match o.values().next() {
                Some(Value::String(s)) => Value::String(s.clone()),
                _ => Value::String(other.to_string()),
            },
            _ => Value::String(other.to_string()),
        },
        (false, Value::String(s)) => match serde_json::from_str::<Value>(&s) {
            Ok(v @ Value::Object(_)) => v,
            _ => json!({ "input": s }),
        },
        (false, Value::Null) => json!({}),
        (false, other) => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::chat::{Message, ToolDef};

    fn feed(parser: &mut ToolCallParser, text: &str, step: usize) -> Vec<Parsed> {
        let chars: Vec<char> = text.chars().collect();
        let mut out = Vec::new();
        for chunk in chars.chunks(step) {
            out.extend(parser.push(&chunk.iter().collect::<String>()));
        }
        out.extend(parser.finish());
        // Merge consecutive text events for easier assertions.
        let mut merged: Vec<Parsed> = Vec::new();
        for p in out {
            match (merged.last_mut(), p) {
                (Some(Parsed::Text(a)), Parsed::Text(b)) => a.push_str(&b),
                (_, p) => merged.push(p),
            }
        }
        merged
    }

    #[test]
    fn parses_tool_calls_across_chunks() {
        let text = "Let me check.\n<tool_call>\n{\"name\": \"Bash\", \"arguments\": {\"command\": \"ls\"}}\n</tool_call>\n<tool_call>{\"name\":\"Read\",\"arguments\":{\"path\":\"a<b\"}}</tool_call>\n";
        for step in [1, 2, 3, 7, 100] {
            let mut p = ToolCallParser::new(true);
            let out = feed(&mut p, text, step);
            assert_eq!(
                out,
                vec![
                    Parsed::Text("Let me check.\n".into()),
                    Parsed::ToolCall {
                        name: "Bash".into(),
                        arguments: json!({"command": "ls"})
                    },
                    Parsed::ToolCall {
                        name: "Read".into(),
                        arguments: json!({"path": "a<b"})
                    },
                ],
                "step {step}"
            );
        }
    }

    #[test]
    fn keeps_plain_text_and_lone_brackets() {
        let mut p = ToolCallParser::new(true);
        let out = feed(&mut p, "a < b and <tool is fine <", 1);
        assert_eq!(out, vec![Parsed::Text("a < b and <tool is fine <".into())]);
    }

    #[test]
    fn invalid_block_is_text() {
        let mut p = ToolCallParser::new(true);
        let out = feed(&mut p, "x<tool_call>not json</tool_call>", 4);
        assert_eq!(out, vec![Parsed::Text("x<tool_call>not json</tool_call>".into())]);
    }

    #[test]
    fn fenced_and_unterminated() {
        let mut p = ToolCallParser::new(true);
        let out = feed(
            &mut p,
            "<tool_call>```json\n{\"name\":\"a\",\"input\":{\"x\":1}}\n```</tool_call><tool_call>{\"name\":\"b\"}",
            5,
        );
        assert_eq!(
            out,
            vec![
                Parsed::ToolCall {
                    name: "a".into(),
                    arguments: json!({"x": 1})
                },
                Parsed::ToolCall {
                    name: "b".into(),
                    arguments: json!({})
                },
            ]
        );
    }

    #[test]
    fn disabled_parser_passes_through() {
        let mut p = ToolCallParser::new(false);
        let out = feed(&mut p, "<tool_call>{\"name\":\"a\"}</tool_call>", 3);
        assert_eq!(
            out,
            vec![Parsed::Text("<tool_call>{\"name\":\"a\"}</tool_call>".into())]
        );
    }

    #[test]
    fn normalize() {
        assert_eq!(normalize_arguments(json!("{\"a\":1}"), false), json!({"a": 1}));
        assert_eq!(normalize_arguments(json!({"input": "patch"}), true), json!("patch"));
        assert_eq!(normalize_arguments(json!("raw"), true), json!("raw"));
    }

    #[test]
    fn render_simple_and_full() {
        let mut req = ChatRequest::default();
        req.push_part(Role::User, Part::Text("hello".into()));
        assert_eq!(render(&req), vec![json!({"type":"text","text":"hello"})]);

        req.system.push("Be terse.".into());
        req.tools.push(ToolDef {
            name: "Bash".into(),
            description: Some("run".into()),
            schema: Some(json!({"type":"object"})),
            freeform: false,
        });
        req.messages.push(Message {
            role: Role::Assistant,
            parts: vec![Part::ToolCall {
                id: "1".into(),
                name: "Bash".into(),
                arguments: json!({"command":"ls"}),
            }],
        });
        req.push_part(
            Role::User,
            Part::ToolResult {
                id: "1".into(),
                name: Some("Bash".into()),
                content: "file.txt".into(),
                is_error: false,
            },
        );
        req.push_part(
            Role::User,
            Part::Image {
                mime: "image/png".into(),
                data: "AAAA".into(),
                url: None,
            },
        );
        let blocks = render(&req);
        assert_eq!(blocks.len(), 3);
        let text = blocks[0]["text"].as_str().unwrap();
        assert!(text.contains("<system>\nBe terse.\n</system>"));
        assert!(text.contains("\"name\":\"Bash\""));
        assert!(text.contains("<tool_result id=\"1\" name=\"Bash\">\nfile.txt\n</tool_result>"));
        assert_eq!(blocks[1]["type"], "image");
        assert!(blocks[2]["text"].as_str().unwrap().contains("next reply"));
    }
}
