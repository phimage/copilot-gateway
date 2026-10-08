//! Turns the raw agent event stream into API-level events (text, thinking,
//! tool calls, completion), shared by every HTTP front-end.

use std::collections::VecDeque;

use serde_json::Value;

use crate::backend::{Turn, TurnEvent};
use crate::chat::{ChatRequest, Part, estimate_tokens};

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
    ToolCall { id: String, name: String, arguments: Value },
    Done { finish: Finish, usage: FinalUsage },
    Error(String),
}

pub struct OutputStream {
    turn: Turn,
    queue: VecDeque<OutEvent>,
    finished: bool,
    prompt_chars: usize,
    output_chars: usize,
    tool_calls: usize,
}

/// Size of the request content, used to estimate tokens when the backend
/// does not report usage.
fn request_chars(req: &ChatRequest) -> usize {
    let mut chars: usize = req.system.iter().map(String::len).sum();
    for m in &req.messages {
        for p in &m.parts {
            chars += match p {
                Part::Text(t) => t.len(),
                Part::Image { .. } => 1000,
                Part::ToolCall { name, arguments, .. } => name.len() + arguments.to_string().len(),
                Part::ToolResult { content, .. } => content.len(),
            };
        }
    }
    for t in &req.tools {
        chars += t.name.len()
            + t.description.as_ref().map_or(0, String::len)
            + t.schema.as_ref().map_or(0, |s| s.to_string().len());
    }
    chars
}

impl OutputStream {
    pub fn new(turn: Turn, req: &ChatRequest) -> Self {
        Self {
            turn,
            queue: VecDeque::new(),
            finished: false,
            prompt_chars: request_chars(req),
            output_chars: 0,
            tool_calls: 0,
        }
    }

    pub fn model(&self) -> Option<&str> {
        self.turn.model.as_deref()
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
                    if !t.is_empty() {
                        self.queue.push_back(OutEvent::Text(t));
                    }
                }
                Some(TurnEvent::Thought(t)) => {
                    self.output_chars += t.len();
                    self.queue.push_back(OutEvent::Thought(t));
                }
                Some(TurnEvent::ToolCall { id, name, arguments }) => {
                    self.tool_calls += 1;
                    self.output_chars += arguments.to_string().len();
                    self.queue.push_back(OutEvent::ToolCall { id, name, arguments });
                }
                Some(TurnEvent::Done { stop_reason, usage }) => {
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
                OutEvent::ToolCall { id, name, arguments } => c.items.push(Item::ToolCall { id, name, arguments }),
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
    ToolCall { id: String, name: String, arguments: Value },
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
