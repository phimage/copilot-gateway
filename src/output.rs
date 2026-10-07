//! Turns the raw agent event stream into API-level events (text, thinking,
//! tool calls, completion), shared by every HTTP front-end.

use std::collections::{HashSet, VecDeque};

use serde_json::Value;

use crate::backend::{Turn, TurnEvent};
use crate::chat::{ChatRequest, estimate_tokens};
use crate::prompt::{Parsed, ToolCallParser, normalize_arguments};

#[derive(Debug, Clone, Copy, Default)]
pub struct FinalUsage {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cached_tokens: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Finish {
    Stop,
    ToolCalls,
    Length,
    Refusal,
    Cancelled,
}

#[derive(Debug, Clone)]
pub enum OutEvent {
    Text(String),
    Thought(String),
    ToolCall { name: String, arguments: Value },
    Done { finish: Finish, usage: FinalUsage },
    Error(String),
}

pub struct OutputStream {
    turn: Turn,
    parser: ToolCallParser,
    freeform: HashSet<String>,
    queue: VecDeque<OutEvent>,
    finished: bool,
    prompt_chars: usize,
    output_chars: usize,
    tool_calls: usize,
}

impl OutputStream {
    pub fn new(turn: Turn, req: &ChatRequest, prompt: &[Value]) -> Self {
        let tools_enabled = !req.tools.is_empty() && req.tool_choice != crate::chat::ToolChoice::None;
        let prompt_chars = prompt
            .iter()
            .map(|b| b.get("text").and_then(Value::as_str).map_or(1000, str::len))
            .sum();
        Self {
            turn,
            parser: ToolCallParser::new(tools_enabled),
            freeform: req
                .tools
                .iter()
                .filter(|t| t.freeform)
                .map(|t| t.name.clone())
                .collect(),
            queue: VecDeque::new(),
            finished: false,
            prompt_chars,
            output_chars: 0,
            tool_calls: 0,
        }
    }

    pub fn model(&self) -> Option<&str> {
        self.turn.model.as_deref()
    }

    fn push_parsed(&mut self, parsed: Vec<Parsed>) {
        for p in parsed {
            match p {
                Parsed::Text(t) => self.queue.push_back(OutEvent::Text(t)),
                Parsed::ToolCall { name, arguments } => {
                    self.tool_calls += 1;
                    let freeform = self.freeform.contains(&name);
                    self.queue.push_back(OutEvent::ToolCall {
                        arguments: normalize_arguments(arguments, freeform),
                        name,
                    });
                }
            }
        }
    }

    pub async fn next(&mut self) -> Option<OutEvent> {
        loop {
            if let Some(ev) = self.queue.pop_front() {
                return Some(ev);
            }
            if self.finished {
                return None;
            }
            match self.turn.events.recv().await {
                Some(TurnEvent::Text(t)) => {
                    self.output_chars += t.len();
                    let parsed = self.parser.push(&t);
                    self.push_parsed(parsed);
                }
                Some(TurnEvent::Thought(t)) => {
                    self.output_chars += t.len();
                    self.queue.push_back(OutEvent::Thought(t));
                }
                Some(TurnEvent::Done { stop_reason, usage }) => {
                    let parsed = self.parser.finish();
                    self.push_parsed(parsed);
                    let finish = if self.tool_calls > 0 {
                        Finish::ToolCalls
                    } else {
                        match stop_reason.as_str() {
                            "max_tokens" | "max_turn_requests" => Finish::Length,
                            "refusal" => Finish::Refusal,
                            "cancelled" => Finish::Cancelled,
                            _ => Finish::Stop,
                        }
                    };
                    let usage = FinalUsage {
                        input_tokens: usage.input_tokens.unwrap_or_else(|| estimate_tokens(self.prompt_chars)),
                        output_tokens: usage
                            .output_tokens
                            .unwrap_or_else(|| estimate_tokens(self.output_chars)),
                        cached_tokens: usage.cached_read_tokens.unwrap_or(0),
                    };
                    self.queue.push_back(OutEvent::Done { finish, usage });
                    self.finished = true;
                }
                Some(TurnEvent::Error(e)) => {
                    let parsed = self.parser.finish();
                    self.push_parsed(parsed);
                    self.queue.push_back(OutEvent::Error(e));
                    self.finished = true;
                }
                None => {
                    self.queue
                        .push_back(OutEvent::Error("agent turn ended unexpectedly".into()));
                    self.finished = true;
                }
            }
        }
    }

    /// Collect the whole turn (non-streaming responses).
    pub async fn collect(mut self) -> Result<Collected, String> {
        let mut c = Collected::default();
        while let Some(ev) = self.next().await {
            match ev {
                OutEvent::Text(t) => match c.items.last_mut() {
                    Some(Item::Text(prev)) => prev.push_str(&t),
                    _ => c.items.push(Item::Text(t)),
                },
                OutEvent::Thought(t) => c.thinking.push_str(&t),
                OutEvent::ToolCall { name, arguments } => c.items.push(Item::ToolCall { name, arguments }),
                OutEvent::Done { finish, usage } => {
                    c.finish = finish;
                    c.usage = usage;
                }
                OutEvent::Error(e) => return Err(e),
            }
        }
        c.model = self.turn.model.clone();
        Ok(c)
    }
}

#[derive(Debug, Clone)]
pub enum Item {
    Text(String),
    ToolCall { name: String, arguments: Value },
}

#[derive(Debug, Clone)]
pub struct Collected {
    pub items: Vec<Item>,
    pub thinking: String,
    pub finish: Finish,
    pub usage: FinalUsage,
    pub model: Option<String>,
}

impl Default for Collected {
    fn default() -> Self {
        Self {
            items: Vec::new(),
            thinking: String::new(),
            finish: Finish::Stop,
            usage: FinalUsage::default(),
            model: None,
        }
    }
}

impl Collected {
    pub fn text(&self) -> String {
        self.items
            .iter()
            .filter_map(|i| match i {
                Item::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect()
    }
}

pub fn new_id(prefix: &str) -> String {
    format!("{prefix}{}", uuid::Uuid::new_v4().simple())
}

pub fn now_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
