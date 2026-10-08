//! copilot-gateway: expose the GitHub Copilot CLI (through the Copilot SDK
//! protocol, `copilot --headless`, or the Agent Client Protocol,
//! `copilot --acp`) as Anthropic- and OpenAI-compatible HTTP APIs.

pub mod acp;
pub mod anthropic;
pub mod backend;
pub mod chat;
pub mod engine;
pub mod launch;
pub mod mock_agent;
pub mod openai;
pub mod output;
pub mod prompt;
pub mod responses;
pub mod rpc;
pub mod sdk;
pub mod server;
