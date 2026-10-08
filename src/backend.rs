//! Runs prompt turns on the ACP agent: one ACP session per HTTP request.

use std::sync::{Arc, Mutex as StdMutex};
use std::time::Duration;

use anyhow::{Context, Result, anyhow};
use serde::Serialize;
use serde_json::{Value, json};
use tokio::sync::{Mutex, OwnedSemaphorePermit, Semaphore, mpsc};
use tracing::{debug, info, warn};

use crate::acp::{AcpProcess, AgentCommand, PermissionPolicy};

#[derive(Debug, Clone)]
pub struct BackendConfig {
    pub agent: AgentCommand,
    pub permission: PermissionPolicy,
    /// Model used when the requested one cannot be matched.
    pub default_model: Option<String>,
    /// `(pattern, target)` rewrites applied to requested model names.
    /// A pattern ending with `*` matches by prefix, `*` alone matches all.
    pub model_map: Vec<(String, String)>,
    pub max_concurrent: usize,
}

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
}

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct Usage {
    pub input_tokens: Option<u64>,
    pub output_tokens: Option<u64>,
    pub cached_read_tokens: Option<u64>,
}

#[derive(Debug, Clone)]
pub enum TurnEvent {
    Text(String),
    Thought(String),
    Done { stop_reason: String, usage: Usage },
    Error(String),
}

/// A running prompt turn. Dropping it cancels the turn.
pub struct Turn {
    pub events: mpsc::Receiver<TurnEvent>,
    /// The agent model actually used, when known.
    pub model: Option<String>,
}

#[derive(Debug, Clone, Default)]
pub struct TurnOptions {
    pub model: String,
    pub reasoning_effort: Option<String>,
}

pub struct Backend {
    cfg: BackendConfig,
    process: Mutex<Option<Arc<AcpProcess>>>,
    models: StdMutex<Option<Vec<ModelInfo>>>,
    agent_model: StdMutex<Option<String>>,
    limiter: Arc<Semaphore>,
}

/// How the agent lets us pick a model in a session.
#[derive(Debug, Clone, PartialEq)]
enum ModelSelector {
    /// `session/set_config_option` with this config id.
    ConfigOption(String),
    /// Legacy `session/set_model`.
    SetModel,
}

#[derive(Debug, Default)]
struct SessionInfo {
    models: Vec<ModelInfo>,
    current_model: Option<String>,
    selector: Option<ModelSelector>,
    /// (config id, allowed values) of the reasoning effort selector.
    thought_level: Option<(String, Vec<String>)>,
}

impl Backend {
    pub fn new(cfg: BackendConfig) -> Self {
        let limiter = Arc::new(Semaphore::new(cfg.max_concurrent.max(1)));
        Self {
            cfg,
            process: Mutex::new(None),
            models: StdMutex::new(None),
            agent_model: StdMutex::new(None),
            limiter,
        }
    }

    pub fn config(&self) -> &BackendConfig {
        &self.cfg
    }

    /// Get the agent process, (re)starting it if needed.
    async fn process(&self) -> Result<Arc<AcpProcess>> {
        let mut guard = self.process.lock().await;
        if let Some(p) = guard.as_ref() {
            if p.is_alive() {
                return Ok(p.clone());
            }
            warn!("ACP agent died, restarting it");
        }
        let p = AcpProcess::spawn(&self.cfg.agent, self.cfg.permission).await?;
        *guard = Some(p.clone());
        Ok(p)
    }

    /// Start the agent eagerly (to surface configuration errors at startup).
    pub async fn warm_up(&self) -> Result<()> {
        self.process().await.map(|_| ())
    }

    pub async fn shutdown(&self) {
        if let Some(p) = self.process.lock().await.take() {
            p.shutdown().await;
        }
    }

    async fn new_session(&self, proc: &AcpProcess) -> Result<(String, SessionInfo)> {
        let res = proc
            .request_timeout(
                "session/new",
                json!({"cwd": self.cfg.agent.cwd, "mcpServers": []}),
                Duration::from_secs(180),
            )
            .await
            .map_err(explain_auth_error)?;
        let sid = res
            .get("sessionId")
            .and_then(Value::as_str)
            .context("session/new returned no sessionId")?
            .to_string();
        let info = parse_session_info(&res);
        if info.current_model.is_some() {
            *self.agent_model.lock().unwrap() = info.current_model.clone();
        }
        if !info.models.is_empty() {
            *self.models.lock().unwrap() = Some(info.models.clone());
        }
        Ok((sid, info))
    }

    fn close_session(proc: &Arc<AcpProcess>, sid: &str) {
        proc.unsubscribe(sid);
        if proc.supports_close() {
            let proc = proc.clone();
            let sid = sid.to_string();
            tokio::spawn(async move {
                let _ = proc
                    .request_timeout("session/close", json!({"sessionId": sid}), Duration::from_secs(30))
                    .await;
            });
        }
    }

    /// Models advertised by the agent (cached after the first session).
    pub async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        if let Some(m) = self.models.lock().unwrap().clone() {
            return Ok(m);
        }
        let proc = self.process().await?;
        let (sid, info) = self.new_session(&proc).await?;
        Self::close_session(&proc, &sid);
        let mut models = info.models;
        if models.is_empty()
            && let Some(m) = info.current_model.or(self.cfg.default_model.clone())
        {
            models.push(ModelInfo {
                id: m.clone(),
                name: m,
                description: None,
            });
        }
        *self.models.lock().unwrap() = Some(models.clone());
        Ok(models)
    }

    /// The model a new agent session uses by default (known after the first session).
    pub async fn agent_default_model(&self) -> Option<String> {
        if self.agent_model.lock().unwrap().is_none() {
            let _ = self.list_models().await;
            if self.agent_model.lock().unwrap().is_none() {
                // Model list was cached without a session: probe one.
                if let Ok(proc) = self.process().await
                    && let Ok((sid, _)) = self.new_session(&proc).await
                {
                    Self::close_session(&proc, &sid);
                }
            }
        }
        self.agent_model.lock().unwrap().clone()
    }

    pub fn image_support(&self) -> Option<bool> {
        self.process
            .try_lock()
            .ok()
            .and_then(|p| p.as_ref().map(|p| p.supports_images()))
    }

    /// Start a prompt turn. Errors before the turn is running (agent not
    /// installed, authentication...) are returned here so that HTTP handlers
    /// can answer with a proper error status.
    pub async fn start_turn(&self, prompt: Vec<Value>, opts: TurnOptions) -> Result<Turn> {
        let permit = self
            .limiter
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| anyhow!("backend is shutting down"))?;
        let proc = self.process().await?;
        let (sid, info) = self.new_session(&proc).await?;

        let model = match self.select_model(&proc, &sid, &info, &opts.model).await {
            Ok(m) => m,
            Err(e) => {
                Self::close_session(&proc, &sid);
                return Err(e);
            }
        };
        if let (Some(effort), Some((config_id, values))) = (&opts.reasoning_effort, &info.thought_level)
            && let Some(value) = values.iter().find(|v| v.eq_ignore_ascii_case(effort))
        {
            let r = proc
                .request_timeout(
                    "session/set_config_option",
                    json!({"sessionId": sid, "configId": config_id, "value": value}),
                    Duration::from_secs(30),
                )
                .await;
            if let Err(e) = r {
                warn!("could not set reasoning effort: {e:#}");
            }
        }

        let updates = proc.subscribe(&sid);
        let (tx, rx) = mpsc::channel(256);
        tokio::spawn(run_turn(proc, sid, prompt, updates, tx, permit));
        Ok(Turn { events: rx, model })
    }

    async fn select_model(
        &self,
        proc: &AcpProcess,
        sid: &str,
        info: &SessionInfo,
        requested: &str,
    ) -> Result<Option<String>> {
        let target = resolve_model(
            requested,
            &info.models,
            &self.cfg.model_map,
            self.cfg.default_model.as_deref(),
        );
        let Some(target) = target else {
            return Ok(info.current_model.clone());
        };
        if info.current_model.as_deref() == Some(target.as_str()) {
            return Ok(Some(target));
        }
        let res = match &info.selector {
            Some(ModelSelector::ConfigOption(id)) => {
                proc.request_timeout(
                    "session/set_config_option",
                    json!({"sessionId": sid, "configId": id, "value": target}),
                    Duration::from_secs(60),
                )
                .await
            }
            Some(ModelSelector::SetModel) | None => {
                proc.request_timeout(
                    "session/set_model",
                    json!({"sessionId": sid, "modelId": target}),
                    Duration::from_secs(60),
                )
                .await
            }
        };
        match res {
            Ok(_) => {
                info!(requested, model = %target, "model selected");
                Ok(Some(target))
            }
            Err(e) => {
                warn!(requested, model = %target, "could not select model, using the agent default: {e:#}");
                Ok(info.current_model.clone())
            }
        }
    }
}

async fn run_turn(
    proc: Arc<AcpProcess>,
    sid: String,
    prompt: Vec<Value>,
    mut updates: mpsc::UnboundedReceiver<Value>,
    tx: mpsc::Sender<TurnEvent>,
    _permit: OwnedSemaphorePermit,
) {
    let request = {
        let proc = proc.clone();
        let params = json!({"sessionId": sid, "prompt": prompt});
        async move { proc.request("session/prompt", params).await }
    };
    tokio::pin!(request);

    let mut last_message_id: Option<String> = None;
    let mut emitted_text = false;
    let mut cancelled = false;

    let result = loop {
        tokio::select! {
            biased;
            Some(update) = updates.recv() => {
                if let Some(ev) = convert_update(&update, &mut last_message_id, &mut emitted_text)
                    && tx.send(ev).await.is_err() && !cancelled {
                        cancelled = true;
                        let _ = proc.notify("session/cancel", json!({"sessionId": sid}));
                    }
            }
            _ = tx.closed(), if !cancelled => {
                debug!(session = %sid, "client went away, cancelling turn");
                cancelled = true;
                let _ = proc.notify("session/cancel", json!({"sessionId": sid}));
            }
            res = &mut request => break res,
        }
    };

    // Updates sent before the response are already queued.
    while let Ok(update) = updates.try_recv() {
        if let Some(ev) = convert_update(&update, &mut last_message_id, &mut emitted_text) {
            let _ = tx.send(ev).await;
        }
    }

    let final_event = match result {
        Ok(res) => {
            let stop_reason = res
                .get("stopReason")
                .and_then(Value::as_str)
                .unwrap_or("end_turn")
                .to_string();
            let u = res.get("usage");
            let get = |k: &str| u.and_then(|u| u.get(k)).and_then(Value::as_u64);
            TurnEvent::Done {
                stop_reason,
                usage: Usage {
                    input_tokens: get("inputTokens"),
                    output_tokens: get("outputTokens"),
                    cached_read_tokens: get("cachedReadTokens"),
                },
            }
        }
        Err(e) => TurnEvent::Error(format!("{e:#}")),
    };
    let _ = tx.send(final_event).await;
    Backend::close_session(&proc, &sid);
}

fn convert_update(update: &Value, last_message_id: &mut Option<String>, emitted_text: &mut bool) -> Option<TurnEvent> {
    let kind = update.get("sessionUpdate").and_then(Value::as_str)?;
    match kind {
        "agent_message_chunk" => {
            let text = content_text(update.get("content")?)?;
            let mut out = String::new();
            // A new agent message (e.g. after the agent used one of its own
            // tools) is separated from the previous one.
            if let Some(mid) = update.get("messageId").and_then(Value::as_str) {
                if last_message_id.as_deref().is_some_and(|l| l != mid) && *emitted_text {
                    out.push_str("\n\n");
                }
                *last_message_id = Some(mid.to_string());
            }
            out.push_str(&text);
            if out.is_empty() {
                return None;
            }
            *emitted_text = true;
            Some(TurnEvent::Text(out))
        }
        "agent_thought_chunk" => content_text(update.get("content")?).map(TurnEvent::Thought),
        "tool_call" | "tool_call_update" => {
            let title = update.get("title").and_then(Value::as_str).unwrap_or("");
            let status = update.get("status").and_then(Value::as_str).unwrap_or("");
            debug!(title, status, "agent tool activity");
            None
        }
        _ => None,
    }
}

fn content_text(content: &Value) -> Option<String> {
    match content.get("type").and_then(Value::as_str) {
        Some("text") => content.get("text").and_then(Value::as_str).map(str::to_string),
        Some("resource_link") => content.get("uri").and_then(Value::as_str).map(|u| format!("[{u}]")),
        _ => None,
    }
}

fn parse_session_info(res: &Value) -> SessionInfo {
    let mut info = SessionInfo::default();
    if let Some(options) = res.get("configOptions").and_then(Value::as_array) {
        for opt in options {
            let id = opt.get("id").and_then(Value::as_str).unwrap_or_default();
            let category = opt.get("category").and_then(Value::as_str).unwrap_or_default();
            let values = select_values(opt.get("options"));
            let current = opt.get("currentValue").and_then(Value::as_str);
            if category == "model" || (category.is_empty() && id == "model") {
                info.selector = Some(ModelSelector::ConfigOption(id.to_string()));
                info.current_model = current.map(str::to_string);
                info.models = values;
            } else if category == "thought_level" {
                info.thought_level = Some((id.to_string(), values.into_iter().map(|m| m.id).collect()));
            }
        }
    }
    if info.selector.is_none()
        && let Some(models) = res.get("models")
    {
        info.selector = Some(ModelSelector::SetModel);
        info.current_model = models.get("currentModelId").and_then(Value::as_str).map(str::to_string);
        info.models = models
            .get("availableModels")
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(|m| {
                        let id = m.get("modelId")?.as_str()?.to_string();
                        Some(ModelInfo {
                            name: m.get("name").and_then(Value::as_str).unwrap_or(&id).to_string(),
                            description: m.get("description").and_then(Value::as_str).map(str::to_string),
                            id,
                        })
                    })
                    .collect()
            })
            .unwrap_or_default();
    }
    info
}

fn select_values(options: Option<&Value>) -> Vec<ModelInfo> {
    let mut out = Vec::new();
    let Some(options) = options.and_then(Value::as_array) else {
        return out;
    };
    for o in options {
        if let Some(group) = o.get("options").and_then(Value::as_array) {
            out.extend(group.iter().filter_map(select_value));
        } else if let Some(v) = select_value(o) {
            out.push(v);
        }
    }
    out
}

fn select_value(o: &Value) -> Option<ModelInfo> {
    let id = o.get("value")?.as_str()?.to_string();
    Some(ModelInfo {
        name: o.get("name").and_then(Value::as_str).unwrap_or(&id).to_string(),
        description: o.get("description").and_then(Value::as_str).map(str::to_string),
        id,
    })
}

fn explain_auth_error(e: anyhow::Error) -> anyhow::Error {
    let msg = format!("{e:#}");
    if msg.to_lowercase().contains("auth") {
        e.context("the Copilot CLI is not authenticated: run `copilot login` (or set COPILOT_GITHUB_TOKEN) and retry")
    } else {
        e
    }
}

/// Normalize a model name for fuzzy comparison:
/// `claude-sonnet-4-5-20250929` and `claude-sonnet-4.5` both give
/// `claude-sonnet-4-5`.
pub fn normalize_model(name: &str) -> String {
    let mut s = name.trim().to_lowercase();
    if let Some((_, rest)) = s.rsplit_once('/') {
        s = rest.to_string();
    }
    if let Some(idx) = s.find('[') {
        s.truncate(idx);
    }
    s = s.replace(['.', '_', ' '], "-");
    if let Some(stripped) = s.strip_suffix("-latest") {
        s = stripped.to_string();
    }
    // Trailing date: -20250929 or -2025-09-29.
    let parts: Vec<&str> = s.split('-').collect();
    let n = parts.len();
    if n > 1 && parts[n - 1].len() == 8 && parts[n - 1].chars().all(|c| c.is_ascii_digit()) {
        s = parts[..n - 1].join("-");
    } else if n > 3 && parts[n - 3].len() == 4 && parts[n - 3..].iter().all(|p| p.chars().all(|c| c.is_ascii_digit())) {
        s = parts[..n - 3].join("-");
    }
    s
}

fn apply_model_map(requested: &str, map: &[(String, String)]) -> Option<String> {
    for (pattern, target) in map {
        let hit = if pattern == "*" {
            true
        } else if let Some(prefix) = pattern.strip_suffix('*') {
            requested.starts_with(prefix)
        } else {
            requested == pattern
        };
        if hit {
            return Some(target.clone());
        }
    }
    None
}

/// Pick the agent model for a requested model name. `None` means "keep the
/// agent's current model".
pub fn resolve_model(
    requested: &str,
    available: &[ModelInfo],
    map: &[(String, String)],
    default: Option<&str>,
) -> Option<String> {
    let requested = apply_model_map(requested, map).unwrap_or_else(|| requested.to_string());
    let fallback = || default.map(str::to_string);
    if requested.is_empty() || requested == "default" {
        return fallback();
    }
    if available.is_empty() {
        return Some(requested);
    }
    if let Some(m) = available.iter().find(|m| m.id.eq_ignore_ascii_case(&requested)) {
        return Some(m.id.clone());
    }
    let norm = normalize_model(&requested);
    if let Some(m) = available.iter().find(|m| normalize_model(&m.id) == norm) {
        return Some(m.id.clone());
    }
    // Same family (e.g. a newer/older version of the requested model): pick
    // the highest version of the longest matching prefix.
    let mut best: Option<(&ModelInfo, (usize, usize, String))> = None;
    for m in available {
        let mn = normalize_model(&m.id);
        let common = common_prefix_tokens(&mn, &norm);
        if common == 0 {
            continue;
        }
        let shared = mn.split('-').filter(|t| norm.split('-').any(|x| x == *t)).count();
        let score = (common, shared, mn);
        if best.as_ref().is_none_or(|(_, b)| score > *b) {
            best = Some((m, score));
        }
    }
    if let Some((m, (common, _, _))) = best {
        // Require more than the vendor prefix ("claude", "gpt") to match.
        let family_words = ["opus", "sonnet", "haiku", "mini", "codex", "flash", "pro"];
        let req_tokens: Vec<&str> = norm.split('-').collect();
        let m_norm = normalize_model(&m.id);
        let fam_ok = req_tokens
            .iter()
            .any(|t| family_words.contains(t) && m_norm.split('-').any(|x| x == *t));
        if common >= 2 || fam_ok {
            return Some(m.id.clone());
        }
    }
    // Family keyword anywhere (e.g. "claude-3-5-haiku" vs "claude-haiku-4.5").
    for word in ["opus", "sonnet", "haiku"] {
        if norm.split('-').any(|t| t == word) {
            let mut candidates: Vec<&ModelInfo> = available
                .iter()
                .filter(|m| normalize_model(&m.id).split('-').any(|t| t == word))
                .collect();
            candidates.sort_by_key(|m| std::cmp::Reverse(normalize_model(&m.id)));
            if let Some(m) = candidates.first() {
                return Some(m.id.clone());
            }
        }
    }
    fallback()
}

fn common_prefix_tokens(a: &str, b: &str) -> usize {
    a.split('-').zip(b.split('-')).take_while(|(x, y)| x == y).count()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn models(ids: &[&str]) -> Vec<ModelInfo> {
        ids.iter()
            .map(|id| ModelInfo {
                id: id.to_string(),
                name: id.to_string(),
                description: None,
            })
            .collect()
    }

    #[test]
    fn normalize() {
        assert_eq!(normalize_model("claude-sonnet-4-5-20250929"), "claude-sonnet-4-5");
        assert_eq!(normalize_model("claude-sonnet-4.5"), "claude-sonnet-4-5");
        assert_eq!(normalize_model("anthropic/Claude-Opus-4.1[1m]"), "claude-opus-4-1");
        assert_eq!(normalize_model("gpt-4o-2024-08-06"), "gpt-4o");
        assert_eq!(normalize_model("claude-3-5-haiku-latest"), "claude-3-5-haiku");
    }

    #[test]
    fn resolve() {
        let av = models(&[
            "claude-sonnet-4.5",
            "claude-sonnet-4",
            "claude-opus-4.1",
            "claude-haiku-4.5",
            "gpt-5",
            "gpt-5-mini",
            "gpt-5-codex",
        ]);
        let r = |q: &str| resolve_model(q, &av, &[], Some("gpt-5"));
        assert_eq!(r("claude-sonnet-4-5-20250929").as_deref(), Some("claude-sonnet-4.5"));
        assert_eq!(r("claude-sonnet-4-20250514").as_deref(), Some("claude-sonnet-4"));
        assert_eq!(r("claude-opus-4-5").as_deref(), Some("claude-opus-4.1"));
        assert_eq!(r("claude-3-5-haiku-20241022").as_deref(), Some("claude-haiku-4.5"));
        assert_eq!(r("GPT-5").as_deref(), Some("gpt-5"));
        assert_eq!(r("gpt-5-codex").as_deref(), Some("gpt-5-codex"));
        assert_eq!(r("gpt-5.1-codex").as_deref(), Some("gpt-5-codex"));
        assert_eq!(r("llama-3").as_deref(), Some("gpt-5"));
        assert_eq!(r("").as_deref(), Some("gpt-5"));
        assert_eq!(resolve_model("", &av, &[], None), None);

        let map = vec![("claude-3-5-haiku*".to_string(), "gpt-5-mini".to_string())];
        assert_eq!(
            resolve_model("claude-3-5-haiku-20241022", &av, &map, None).as_deref(),
            Some("gpt-5-mini")
        );
        assert_eq!(resolve_model("anything", &[], &[], None).as_deref(), Some("anything"));
    }

    #[test]
    fn session_info_from_config_options() {
        let res = json!({
            "sessionId": "s1",
            "configOptions": [
                {"id": "mode", "name": "Mode", "category": "mode", "type": "select", "currentValue": "agent",
                 "options": [{"value": "agent", "name": "Agent"}]},
                {"id": "model", "name": "Model", "category": "model", "type": "select", "currentValue": "gpt-5",
                 "options": [{"group": "g", "name": "G", "options": [{"value": "gpt-5", "name": "GPT-5"}]},
                             {"group": "h", "name": "H", "options": [{"value": "claude-sonnet-4.5", "name": "Sonnet"}]}]},
                {"id": "effort", "name": "Effort", "category": "thought_level", "type": "select", "currentValue": "medium",
                 "options": [{"value": "low", "name": "Low"}, {"value": "high", "name": "High"}]}
            ]
        });
        let info = parse_session_info(&res);
        assert_eq!(info.selector, Some(ModelSelector::ConfigOption("model".into())));
        assert_eq!(info.current_model.as_deref(), Some("gpt-5"));
        assert_eq!(info.models.len(), 2);
        assert_eq!(info.thought_level.unwrap().1, vec!["low", "high"]);

        let legacy = parse_session_info(&json!({
            "sessionId": "s",
            "models": {"currentModelId": "a", "availableModels": [{"modelId": "a", "name": "A"}]}
        }));
        assert_eq!(legacy.selector, Some(ModelSelector::SetModel));
        assert_eq!(legacy.models[0].id, "a");
    }
}
