//! copilot-gateway: expose an Agent Client Protocol agent (by default
//! `copilot --acp`, the GitHub Copilot CLI) as Anthropic- and
//! OpenAI-compatible HTTP APIs.

pub mod acp;
pub mod anthropic;
pub mod backend;
pub mod chat;
pub mod launch;
pub mod mock_agent;
pub mod openai;
pub mod output;
pub mod prompt;
pub mod responses;
pub mod server;
