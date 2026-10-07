//! HTTP server: routing, authentication, errors and shared endpoints.

use std::net::SocketAddr;
use std::sync::Arc;

use axum::extract::{Request, State};
use axum::http::{HeaderMap, StatusCode};
use axum::middleware::{self, Next};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tracing::{error, info};

use crate::backend::Backend;
use crate::{anthropic, openai, responses};

#[derive(Clone)]
pub struct AppState {
    pub backend: Arc<Backend>,
    pub api_key: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Flavor {
    Anthropic,
    OpenAi,
}

#[derive(Debug)]
pub struct ApiError {
    pub flavor: Flavor,
    pub status: StatusCode,
    pub message: String,
}

impl ApiError {
    pub fn bad_request(flavor: Flavor, message: impl Into<String>) -> Self {
        Self {
            flavor,
            status: StatusCode::BAD_REQUEST,
            message: message.into(),
        }
    }

    pub fn upstream(flavor: Flavor, err: impl std::fmt::Display) -> Self {
        let message = format!("{err:#}");
        error!("agent error: {message}");
        Self {
            flavor,
            status: StatusCode::BAD_GATEWAY,
            message,
        }
    }

    pub fn body(&self) -> Value {
        error_body(self.flavor, self.status, &self.message)
    }
}

pub fn error_body(flavor: Flavor, status: StatusCode, message: &str) -> Value {
    match flavor {
        Flavor::Anthropic => {
            let kind = match status.as_u16() {
                400 => "invalid_request_error",
                401 => "authentication_error",
                404 => "not_found_error",
                429 => "rate_limit_error",
                503 | 529 => "overloaded_error",
                _ => "api_error",
            };
            json!({"type": "error", "error": {"type": kind, "message": message}})
        }
        Flavor::OpenAi => {
            let kind = match status.as_u16() {
                400 => "invalid_request_error",
                401 => "authentication_error",
                _ => "server_error",
            };
            json!({"error": {"message": message, "type": kind, "param": null, "code": null}})
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        (self.status, Json(self.body())).into_response()
    }
}

pub fn router(state: AppState) -> Router {
    let api = Router::new()
        // Anthropic Messages API
        .route("/v1/messages", post(anthropic::messages))
        .route("/v1/messages/count_tokens", post(anthropic::count_tokens))
        // OpenAI Chat Completions & Responses APIs
        .route("/v1/chat/completions", post(openai::chat_completions))
        .route("/chat/completions", post(openai::chat_completions))
        .route("/v1/responses", post(responses::create))
        .route("/responses", post(responses::create))
        .route("/v1/models", get(models))
        .route("/models", get(models))
        .route_layer(middleware::from_fn_with_state(state.clone(), auth));

    Router::new()
        .route("/", get(index))
        .route("/health", get(health))
        .merge(api)
        .with_state(state)
}

async fn auth(State(state): State<AppState>, req: Request, next: Next) -> Response {
    if let Some(expected) = &state.api_key
        && !is_authorized(req.headers(), expected)
    {
        let flavor = if req.headers().contains_key("anthropic-version") || req.uri().path().starts_with("/v1/messages")
        {
            Flavor::Anthropic
        } else {
            Flavor::OpenAi
        };
        return (
            StatusCode::UNAUTHORIZED,
            Json(error_body(flavor, StatusCode::UNAUTHORIZED, "invalid API key")),
        )
            .into_response();
    }
    next.run(req).await
}

fn is_authorized(headers: &HeaderMap, expected: &str) -> bool {
    let bearer = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer ").or_else(|| v.strip_prefix("bearer ")))
        .map(str::trim);
    let api_key = headers
        .get("x-api-key")
        .or_else(|| headers.get("api-key"))
        .and_then(|v| v.to_str().ok())
        .map(str::trim);
    bearer == Some(expected) || api_key == Some(expected)
}

async fn index() -> Json<Value> {
    Json(json!({
        "name": "copilot-gateway",
        "version": env!("CARGO_PKG_VERSION"),
        "endpoints": [
            "POST /v1/messages",
            "POST /v1/messages/count_tokens",
            "POST /v1/chat/completions",
            "POST /v1/responses",
            "GET /v1/models",
            "GET /health"
        ]
    }))
}

async fn health(State(state): State<AppState>) -> Json<Value> {
    Json(json!({"status": "ok", "image_input": state.backend.image_support()}))
}

/// Model list compatible with both the OpenAI and the Anthropic formats.
async fn models(State(state): State<AppState>, headers: HeaderMap) -> Result<Json<Value>, ApiError> {
    let flavor = if headers.contains_key("anthropic-version") {
        Flavor::Anthropic
    } else {
        Flavor::OpenAi
    };
    let models = state
        .backend
        .list_models()
        .await
        .map_err(|e| ApiError::upstream(flavor, e))?;
    let data: Vec<Value> = models
        .iter()
        .map(|m| {
            json!({
                "id": m.id,
                "object": "model",
                "type": "model",
                "created": 0,
                "created_at": "1970-01-01T00:00:00Z",
                "owned_by": "github-copilot",
                "display_name": m.name,
                "description": m.description,
            })
        })
        .collect();
    Ok(Json(json!({
        "object": "list",
        "first_id": models.first().map(|m| m.id.clone()),
        "last_id": models.last().map(|m| m.id.clone()),
        "has_more": false,
        "data": data,
    })))
}

/// Bind and serve until `shutdown` resolves. Returns the bound address via
/// `on_bound` before serving.
pub async fn serve(
    state: AppState,
    addr: SocketAddr,
    on_bound: impl FnOnce(SocketAddr),
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> anyhow::Result<()> {
    let listener = TcpListener::bind(addr).await?;
    let local = listener.local_addr()?;
    info!("copilot-gateway listening on http://{local}");
    on_bound(local);
    axum::serve(listener, router(state))
        .with_graceful_shutdown(shutdown)
        .await?;
    Ok(())
}

/// Wrap an SSE stream with keep-alive comments (agents can think for a while).
pub fn sse<S>(stream: S) -> Response
where
    S: futures::Stream<Item = Result<axum::response::sse::Event, std::convert::Infallible>> + Send + 'static,
{
    axum::response::Sse::new(stream)
        .keep_alive(axum::response::sse::KeepAlive::default())
        .into_response()
}

pub fn parse_json(flavor: Flavor, body: &[u8]) -> Result<Value, ApiError> {
    serde_json::from_slice(body).map_err(|e| ApiError::bad_request(flavor, format!("invalid JSON body: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn authorization_headers() {
        let mut h = HeaderMap::new();
        assert!(!is_authorized(&h, "k"));
        h.insert("authorization", "Bearer k".parse().unwrap());
        assert!(is_authorized(&h, "k"));
        let mut h = HeaderMap::new();
        h.insert("x-api-key", "k".parse().unwrap());
        assert!(is_authorized(&h, "k"));
        assert!(!is_authorized(&h, "other"));
    }
}
