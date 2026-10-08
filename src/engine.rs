//! The backend driving Copilot: the Copilot SDK protocol (default) or ACP.

use anyhow::Result;

use crate::backend::{Backend, ModelInfo, Turn};
use crate::chat::ChatRequest;
use crate::sdk::SdkBackend;

#[derive(Debug, Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
pub enum BackendKind {
    /// `copilot --headless --stdio`: Copilot SDK protocol, native client tools.
    Sdk,
    /// `copilot --acp`: Agent Client Protocol, client tools emulated in the prompt.
    Acp,
}

pub enum Engine {
    Sdk(SdkBackend),
    Acp(Backend),
}

impl Engine {
    pub fn kind(&self) -> BackendKind {
        match self {
            Engine::Sdk(_) => BackendKind::Sdk,
            Engine::Acp(_) => BackendKind::Acp,
        }
    }

    /// Start a chat turn. Errors before the turn is running (agent not
    /// installed, authentication...) are returned here so that HTTP handlers
    /// can answer with a proper error status.
    pub async fn start_turn(&self, req: &ChatRequest) -> Result<Turn> {
        match self {
            Engine::Sdk(b) => b.start_turn(req).await,
            Engine::Acp(b) => b.start_chat(req).await,
        }
    }

    pub async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        match self {
            Engine::Sdk(b) => b.list_models().await,
            Engine::Acp(b) => b.list_models().await,
        }
    }

    /// Model to hand to launched tools that need an explicit model name.
    pub async fn agent_default_model(&self) -> Option<String> {
        match self {
            Engine::Sdk(b) => b.default_model().await,
            Engine::Acp(b) => b.agent_default_model().await,
        }
    }

    /// Start the agent eagerly (to surface configuration errors at startup).
    pub async fn warm_up(&self) -> Result<()> {
        match self {
            Engine::Sdk(b) => b.warm_up().await,
            Engine::Acp(b) => b.warm_up().await,
        }
    }

    pub async fn shutdown(&self) {
        match self {
            Engine::Sdk(b) => b.shutdown().await,
            Engine::Acp(b) => b.shutdown().await,
        }
    }

    pub fn image_support(&self) -> Option<bool> {
        match self {
            Engine::Sdk(_) => Some(true),
            Engine::Acp(b) => b.image_support(),
        }
    }
}
