//! Copilot SDK backend: drives `copilot --headless --stdio`, the JSON-RPC
//! protocol used by the official GitHub Copilot SDK (protocol version 3,
//! `Content-Length` framed messages).
//!
//! Compared to ACP this protocol lets the gateway:
//! - declare the client's tools (Claude Code's `Bash`, Codex's `exec_command`...)
//!   as native tools without handler: the model calls them, the runtime emits
//!   `external_tool.requested`, and the result is given back with
//!   `session.tools.handlePendingToolCall` once the client sends it;
//! - replace Copilot's system prompt with the client's one
//!   (`systemMessage: {mode: "replace"}`) and disable Copilot's own tools;
//! - keep one Copilot session per conversation: a client tool loop stays in
//!   one Copilot turn, and follow-up messages reuse the idle session;
//! - report real token usage (`assistant.usage`).
//!
//! Methods used: `connect`, `models.list`, `session.create`,
//! `session.options.update`, `session.send`, `session.tools.handlePendingToolCall`,
//! `session.permissions.handlePendingPermissionRequest`, `session.abort`,
//! `session.delete`; notifications `session.event` (`{sessionId, event}`).

use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex as StdMutex};
use std::time::{Duration, Instant};

use anyhow::{Result, anyhow, bail};
use serde_json::{Value, json};
use tokio::sync::{Mutex, OwnedSemaphorePermit, Semaphore, mpsc};
use tracing::{debug, info, warn};

use crate::backend::{BackendConfig, ModelInfo, Turn, TurnEvent, Usage, explain_auth_error, resolve_model};
use crate::chat::{ChatRequest, Message, Part, Role, ToolChoice, ToolDef};
use crate::prompt::{TOOL_CALL_CLOSE, TOOL_CALL_OPEN, normalize_arguments};
use crate::rpc::{Framing, RpcProcess};

pub const PROTOCOL_VERSION: u64 = 3;

/// How long to wait for further tool requests of the same model response
/// when their number is not known yet.
const TOOL_BATCH_WINDOW: Duration = Duration::from_millis(300);

#[derive(Clone)]
struct SdkModel {
    info: ModelInfo,
    efforts: Vec<String>,
}

/// Mapping between the client's tool names and the names declared to
/// Copilot (restricted to `[A-Za-z0-9_-]{1,64}`).
#[derive(Default)]
struct ToolNames {
    to_client: HashMap<String, String>,
    to_wire: HashMap<String, String>,
    freeform: HashSet<String>,
}

impl ToolNames {
    fn new(tools: &[ToolDef]) -> Self {
        let mut names = ToolNames::default();
        for t in tools {
            let mut base: String = t
                .name
                .chars()
                .map(|c| {
                    if c.is_ascii_alphanumeric() || c == '_' || c == '-' {
                        c
                    } else {
                        '_'
                    }
                })
                .take(60)
                .collect();
            if base.is_empty() {
                base = "tool".into();
            }
            let mut wire = base.clone();
            let mut n = 2;
            while names.to_client.contains_key(&wire) {
                wire = format!("{base}_{n}");
                n += 1;
            }
            names.to_client.insert(wire.clone(), t.name.clone());
            names.to_wire.insert(t.name.clone(), wire);
            if t.freeform {
                names.freeform.insert(t.name.clone());
            }
        }
        names
    }

    fn client(&self, wire: &str) -> String {
        self.to_client.get(wire).cloned().unwrap_or_else(|| wire.to_string())
    }

    fn wire(&self, client: &str) -> String {
        self.to_wire.get(client).cloned().unwrap_or_else(|| client.to_string())
    }
}

struct PendingTool {
    tool_call_id: String,
    request_id: String,
}

/// A Copilot session kept alive between two HTTP requests.
struct Parked {
    sid: String,
    proc: Arc<RpcProcess>,
    config_key: u64,
    names: Arc<ToolNames>,
    events: mpsc::UnboundedReceiver<Value>,
    /// Tool calls waiting for the client's results (turn paused).
    pending: Vec<PendingTool>,
    /// Fingerprint of the conversation after the last completed turn.
    fingerprint: Option<u64>,
    model: Option<String>,
    last_used: Instant,
}

type Store = Arc<StdMutex<Vec<Parked>>>;

pub struct SdkBackend {
    cfg: BackendConfig,
    process: Mutex<Option<Arc<RpcProcess>>>,
    models: StdMutex<Option<Vec<SdkModel>>>,
    parked: Store,
    limiter: Arc<Semaphore>,
}

impl SdkBackend {
    pub fn new(cfg: BackendConfig) -> Self {
        let limiter = Arc::new(Semaphore::new(cfg.max_concurrent.max(1)));
        Self {
            cfg,
            process: Mutex::new(None),
            models: StdMutex::new(None),
            parked: Arc::default(),
            limiter,
        }
    }

    /// Get the Copilot process, (re)starting it and doing the `connect`
    /// handshake if needed.
    async fn process(&self) -> Result<Arc<RpcProcess>> {
        let mut guard = self.process.lock().await;
        if let Some(p) = guard.as_ref() {
            if p.is_alive() {
                return Ok(p.clone());
            }
            warn!("Copilot process died, restarting it");
            self.parked.lock().unwrap().clear();
        }
        let rpc = RpcProcess::spawn(
            "Copilot",
            &self.cfg.agent,
            Framing::ContentLength,
            Arc::new(|method, _params| Err((-32601, format!("method not supported by copilot-gateway: {method}")))),
            route_event,
        )?;
        let res = rpc
            .request_timeout(
                "connect",
                json!({"token": null, "supportedTaskKinds": ["agent", "client", "shell"]}),
                Duration::from_secs(120),
            )
            .await
            .map_err(|e| {
                e.context(
                    "the Copilot CLI did not answer the SDK handshake: update it (`npm install -g @github/copilot`) \
                     or use `--backend acp`",
                )
            })?;
        let version = res.get("protocolVersion").and_then(Value::as_u64).unwrap_or(0);
        if version != PROTOCOL_VERSION {
            warn!(
                version,
                expected = PROTOCOL_VERSION,
                "unexpected Copilot SDK protocol version"
            );
        }
        let cli = res.get("version").and_then(Value::as_str).unwrap_or("?");
        info!(cli, protocol = version, "connected to Copilot (SDK protocol)");
        let rpc = Arc::new(rpc);
        *guard = Some(rpc.clone());
        Ok(rpc)
    }

    pub async fn warm_up(&self) -> Result<()> {
        self.process().await.map(|_| ())
    }

    pub async fn shutdown(&self) {
        let parked: Vec<Parked> = std::mem::take(&mut *self.parked.lock().unwrap());
        for p in parked {
            discard(p.proc, p.sid, !p.pending.is_empty()).await;
        }
        if let Some(p) = self.process.lock().await.take() {
            p.shutdown().await;
        }
    }

    async fn models(&self, proc: &RpcProcess) -> Result<Vec<SdkModel>> {
        if let Some(m) = self.models.lock().unwrap().clone() {
            return Ok(m);
        }
        let res = proc
            .request_timeout("models.list", json!({}), Duration::from_secs(60))
            .await
            .map_err(explain_auth_error)?;
        let models: Vec<SdkModel> = res
            .get("models")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(parse_model).collect())
            .unwrap_or_default();
        if models.is_empty() {
            bail!("Copilot returned no models (is your Copilot plan active?)");
        }
        *self.models.lock().unwrap() = Some(models.clone());
        Ok(models)
    }

    pub async fn list_models(&self) -> Result<Vec<ModelInfo>> {
        let proc = self.process().await?;
        Ok(self.models(&proc).await?.into_iter().map(|m| m.info).collect())
    }

    pub async fn default_model(&self) -> Option<String> {
        let models = self.list_models().await.ok()?;
        resolve_model("", &models, &[], self.cfg.default_model.as_deref())
            .or_else(|| models.first().map(|m| m.id.clone()))
    }

    /// Discard parked sessions that expired or exceed the session cap.
    fn evict(&self) {
        let mut evicted = Vec::new();
        {
            let mut parked = self.parked.lock().unwrap();
            let ttl = self.cfg.session_ttl;
            let (keep, old): (Vec<Parked>, Vec<Parked>) = parked.drain(..).partition(|p| p.last_used.elapsed() < ttl);
            *parked = keep;
            evicted.extend(old);
            parked.sort_by_key(|p| p.last_used);
            while parked.len() > self.cfg.max_sessions.max(1) {
                evicted.push(parked.remove(0));
            }
        }
        for p in evicted {
            debug!(session = %p.sid, "evicting Copilot session");
            tokio::spawn(discard(p.proc, p.sid, !p.pending.is_empty()));
        }
    }

    fn take_parked(&self, pred: impl Fn(&Parked) -> bool) -> Option<Parked> {
        let mut parked = self.parked.lock().unwrap();
        let idx = parked.iter().position(|p| p.proc.is_alive() && pred(p))?;
        Some(parked.remove(idx))
    }

    pub async fn start_turn(&self, req: &ChatRequest) -> Result<Turn> {
        let permit = self
            .limiter
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| anyhow!("backend is shutting down"))?;
        self.evict();
        let proc = self.process().await?;
        let models = self.models(&proc).await?;
        let infos: Vec<ModelInfo> = models.iter().map(|m| m.info.clone()).collect();
        let model = resolve_model(
            &req.model,
            &infos,
            &self.cfg.model_map,
            self.cfg.default_model.as_deref(),
        );
        let names = Arc::new(ToolNames::new(&req.tools));
        let key = config_key(model.as_deref(), &req.system, &req.tools);

        let (history, trailing) = match req.messages.split_last() {
            Some((last, rest)) if last.role == Role::User => (rest, Some(last)),
            _ => (&req.messages[..], None),
        };
        let required_tool = match &req.tool_choice {
            ToolChoice::Specific(name) => Some(names.wire(name)),
            _ => None,
        };

        // 1. Tool results for a turn paused on client tool calls: resume it.
        if let Some(trailing) = trailing {
            let results = tool_results(trailing);
            if !results.is_empty() {
                // Matched by tool call ids only: the client's tool list may
                // change between requests (e.g. an MCP server connecting).
                let ids: HashSet<&str> = results.iter().map(|r| r.id.as_str()).collect();
                if let Some(parked) =
                    self.take_parked(|p| p.pending.iter().any(|t| ids.contains(t.tool_call_id.as_str())))
                {
                    debug!(session = %parked.sid, "resuming paused Copilot turn with tool results");
                    let extra = extra_content(trailing);
                    if let Err(e) = resolve_pending(&parked, &results, extra).await {
                        tokio::spawn(discard(parked.proc, parked.sid, true));
                        return Err(e);
                    }
                    return Ok(self.drive(parked, req.messages.clone(), permit));
                }
            }
        }

        // 2. A follow-up message of a conversation whose last turn completed.
        if let Some(trailing) = trailing {
            let fp = fingerprint(key, history);
            if let Some(parked) = self.take_parked(|p| p.pending.is_empty() && p.fingerprint == Some(fp)) {
                debug!(session = %parked.sid, "continuing Copilot session");
                let (prompt, attachments) = render_user_message(trailing);
                if let Err(e) = send(&parked.proc, &parked.sid, prompt, attachments, required_tool).await {
                    tokio::spawn(discard(parked.proc, parked.sid, false));
                    return Err(e);
                }
                return Ok(self.drive(parked, req.messages.clone(), permit));
            }
        }

        // 3. New session; earlier history (if any) is given as a transcript.
        let sid = format!("cgw-{}", uuid::Uuid::new_v4().simple());
        let events = proc.subscribe(&sid);
        let effort = req.reasoning_effort.as_ref().and_then(|e| {
            let m = models.iter().find(|m| Some(&m.info.id) == model.as_ref())?;
            m.efforts.iter().find(|x| x.eq_ignore_ascii_case(e)).cloned()
        });
        let create = session_create_params(&sid, model.as_deref(), effort.as_deref(), req, &names, &self.cfg);
        if let Err(e) = proc
            .request_timeout("session.create", create, Duration::from_secs(180))
            .await
        {
            proc.unsubscribe(&sid);
            return Err(explain_auth_error(e));
        }
        let options = json!({
            "sessionId": sid,
            "skipCustomInstructions": true,
            "customAgentsLocalOnly": true,
            "coauthorEnabled": false,
            "manageScheduleEnabled": false,
            "installedPlugins": [],
            "includedBuiltinSkills": []
        });
        if let Err(e) = proc
            .request_timeout("session.options.update", options, Duration::from_secs(30))
            .await
        {
            debug!("session.options.update failed (ignored): {e:#}");
        }
        info!(session = %sid, model = model.as_deref().unwrap_or("default"), requested = req.model, tools = req.tools.len(), "new Copilot session");

        let (mut prompt, attachments) = match trailing {
            Some(t) => render_user_message(t),
            None => (String::new(), Vec::new()),
        };
        if !history.is_empty() {
            prompt = format!(
                "Earlier messages of this conversation, for context:\n<conversation>\n{}</conversation>\n\n{}",
                render_transcript(history),
                if prompt.is_empty() { "Continue." } else { &prompt }
            );
        }
        let parked = Parked {
            sid: sid.clone(),
            proc: proc.clone(),
            config_key: key,
            names,
            events,
            pending: Vec::new(),
            fingerprint: None,
            model: model.clone(),
            last_used: Instant::now(),
        };
        if let Err(e) = send(&proc, &sid, prompt, attachments, required_tool).await {
            tokio::spawn(discard(proc, sid, false));
            return Err(e);
        }
        Ok(self.drive(parked, req.messages.clone(), permit))
    }

    /// Stream the session's events of the current turn as [`TurnEvent`]s,
    /// until the turn completes or pauses on client tool calls.
    fn drive(&self, session: Parked, messages: Vec<Message>, permit: OwnedSemaphorePermit) -> Turn {
        let (tx, rx) = mpsc::channel(256);
        let model = session.model.clone();
        let store = self.parked.clone();
        let max_sessions = self.cfg.max_sessions;
        tokio::spawn(drive_turn(session, messages, tx, store, max_sessions, permit));
        Turn { events: rx, model }
    }
}

/// `session.event` notifications are routed to the session's subscriber.
fn route_event(method: &str, params: &Value) -> Option<(String, Value)> {
    if method != "session.event" {
        return None;
    }
    let sid = params.get("sessionId")?.as_str()?.to_string();
    Some((sid, params.get("event")?.clone()))
}

fn parse_model(m: &Value) -> Option<SdkModel> {
    let id = m.get("id")?.as_str()?.to_string();
    if m.pointer("/policy/state").and_then(Value::as_str) == Some("disabled") {
        return None;
    }
    let name = m.get("name").and_then(Value::as_str).unwrap_or(&id).to_string();
    let description = m
        .pointer("/billing/multiplier")
        .and_then(Value::as_f64)
        .map(|x| format!("premium request multiplier: {x}"));
    let efforts = m
        .get("supportedReasoningEfforts")
        .and_then(Value::as_array)
        .map(|a| a.iter().filter_map(|e| e.as_str().map(str::to_string)).collect())
        .unwrap_or_default();
    Some(SdkModel {
        info: ModelInfo { id, name, description },
        efforts,
    })
}

fn session_create_params(
    sid: &str,
    model: Option<&str>,
    effort: Option<&str>,
    req: &ChatRequest,
    names: &ToolNames,
    cfg: &BackendConfig,
) -> Value {
    let tools: Vec<Value> = req
        .tools
        .iter()
        .map(|t| {
            let mut description = t.description.clone().unwrap_or_default();
            let parameters = if t.freeform {
                description.push_str("\n\nPass the raw input as the `input` string argument.");
                json!({"type": "object", "properties": {"input": {"type": "string"}}, "required": ["input"]})
            } else {
                t.schema
                    .clone()
                    .unwrap_or_else(|| json!({"type": "object", "properties": {}}))
            };
            json!({
                "name": names.wire(&t.name),
                "description": description,
                "parameters": parameters,
                "overridesBuiltInTool": true,
                "skipPermission": true,
                "defer": "never"
            })
        })
        .collect();
    let system = req.system.join("\n\n");
    let system = if system.trim().is_empty() {
        "You are a helpful assistant.".to_string()
    } else {
        system
    };
    let mut params = json!({
        "sessionId": sid,
        "clientName": "copilot-gateway",
        "tools": tools,
        "toolSearch": {"enabled": false},
        "systemMessage": {"mode": "replace", "content": system},
        // Only the client's tools: Copilot's own tools are not available.
        "availableTools": ["custom:*"],
        "excludedTools": [],
        "toolFilterPrecedence": "excluded",
        "workingDirectory": cfg.agent.cwd,
        "streaming": true,
        "includeSubAgentStreamingEvents": false,
        "requestPermission": false,
        "requestUserInput": false,
        "requestElicitation": false,
        "requestExitPlanMode": false,
        "requestAutoModeSwitch": false,
        "hooks": false,
        "isExperimentalMode": false,
        "enableSessionTelemetry": false,
        "skipEmbeddingRetrieval": true,
        "enableOnDemandInstructionDiscovery": false,
        "enableFileHooks": false,
        "enableHostGitOperations": false,
        "enableSessionStore": false,
        "enableSkills": false,
        "memory": {"enabled": false},
        "customAgentsLocalOnly": true,
        "envValueMode": "direct"
    });
    if let Some(m) = model {
        params["model"] = json!(m);
    }
    if let Some(e) = effort {
        params["reasoningEffort"] = json!(e);
    }
    params
}

async fn send(
    proc: &RpcProcess,
    sid: &str,
    prompt: String,
    attachments: Vec<Value>,
    required_tool: Option<String>,
) -> Result<()> {
    let prompt = if prompt.trim().is_empty() && !attachments.is_empty() {
        "(see the attached image)".to_string()
    } else {
        prompt
    };
    let mut params = json!({"sessionId": sid, "prompt": prompt});
    if !attachments.is_empty() {
        params["attachments"] = json!(attachments);
    }
    if let Some(t) = required_tool {
        params["requiredTool"] = json!(t);
    }
    proc.request_timeout("session.send", params, Duration::from_secs(120))
        .await
        .map(|_| ())
        .map_err(explain_auth_error)
}

/// Abort (if a turn is running) and delete a session.
async fn discard(proc: Arc<RpcProcess>, sid: String, abort: bool) {
    proc.unsubscribe(&sid);
    if !proc.is_alive() {
        return;
    }
    if abort {
        let _ = proc
            .request_timeout("session.abort", json!({"sessionId": sid}), Duration::from_secs(10))
            .await;
    }
    let _ = proc
        .request_timeout("session.delete", json!({"sessionId": sid}), Duration::from_secs(10))
        .await;
}

struct ToolResultIn {
    id: String,
    content: String,
    is_error: bool,
    images: Vec<(String, String)>,
}

/// Tool results of a user message, each with the images following it.
fn tool_results(msg: &Message) -> Vec<ToolResultIn> {
    let mut out: Vec<ToolResultIn> = Vec::new();
    for part in &msg.parts {
        match part {
            Part::ToolResult {
                id, content, is_error, ..
            } => out.push(ToolResultIn {
                id: id.clone(),
                content: content.clone(),
                is_error: *is_error,
                images: Vec::new(),
            }),
            Part::Image { mime, data, .. } if !data.is_empty() => {
                if let Some(last) = out.last_mut() {
                    last.images.push((mime.clone(), data.clone()));
                }
            }
            _ => {}
        }
    }
    out
}

/// Text and images of a user message that come with tool results but are
/// not part of them (e.g. reminders added by the client, or a new instruction).
fn extra_content(msg: &Message) -> (String, Vec<(String, String)>) {
    let mut text = Vec::new();
    let mut images = Vec::new();
    let mut seen_result = false;
    for part in &msg.parts {
        match part {
            Part::ToolResult { .. } => seen_result = true,
            Part::Text(t) if !t.trim().is_empty() => text.push(t.clone()),
            Part::Image { mime, data, .. } if !data.is_empty() && !seen_result => {
                images.push((mime.clone(), data.clone()))
            }
            _ => {}
        }
    }
    (text.join("\n\n"), images)
}

/// Give the client's tool results to the paused turn.
async fn resolve_pending(
    parked: &Parked,
    results: &[ToolResultIn],
    extra: (String, Vec<(String, String)>),
) -> Result<()> {
    let (extra_text, extra_images) = extra;
    let n = parked.pending.len();
    for (i, pending) in parked.pending.iter().enumerate() {
        let last = i + 1 == n;
        let result = match results.iter().find(|r| r.id == pending.tool_call_id) {
            Some(r) => {
                let mut text = r.content.clone();
                let mut images = r.images.clone();
                if last {
                    if !extra_text.is_empty() {
                        text.push_str("\n\n");
                        text.push_str(&extra_text);
                    }
                    images.extend(extra_images.iter().cloned());
                }
                let mut result = json!({
                    "textResultForLlm": text,
                    "resultType": if r.is_error { "failure" } else { "success" },
                });
                if !images.is_empty() {
                    result["binaryResultsForLlm"] = json!(
                        images
                            .iter()
                            .map(|(mime, data)| json!({"type": "image", "data": data, "mimeType": mime}))
                            .collect::<Vec<_>>()
                    );
                }
                result
            }
            None => json!({
                "textResultForLlm": "No result was provided for this tool call.",
                "resultType": "failure"
            }),
        };
        parked
            .proc
            .request_timeout(
                "session.tools.handlePendingToolCall",
                json!({"sessionId": parked.sid, "requestId": pending.request_id, "result": result}),
                Duration::from_secs(30),
            )
            .await?;
    }
    Ok(())
}

/// The prompt text and attachments of a user message.
fn render_user_message(msg: &Message) -> (String, Vec<Value>) {
    let mut text = String::new();
    let mut attachments = Vec::new();
    for part in &msg.parts {
        match part {
            Part::Image { mime, data, url } => {
                if data.is_empty() {
                    text.push_str(&format!("[image: {}]", url.as_deref().unwrap_or("")));
                } else {
                    attachments.push(json!({
                        "type": "blob",
                        "mimeType": mime,
                        "data": data,
                        "displayName": format!("image-{}", attachments.len() + 1)
                    }));
                }
            }
            other => render_part(other, &mut text),
        }
    }
    (text, attachments)
}

fn render_part(part: &Part, out: &mut String) {
    match part {
        Part::Text(t) => out.push_str(t),
        Part::Image { url, .. } => out.push_str(&format!(
            "[image{}]",
            url.as_deref().map(|u| format!(": {u}")).unwrap_or_default()
        )),
        Part::ToolCall { name, arguments, .. } => {
            let call = json!({"name": name, "arguments": arguments});
            out.push_str(&format!("\n{TOOL_CALL_OPEN}\n{call}\n{TOOL_CALL_CLOSE}\n"));
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
            out.push_str(&format!("\n<tool_result {attrs}>\n{content}\n</tool_result>\n"));
        }
    }
}

fn render_transcript(messages: &[Message]) -> String {
    let mut out = String::new();
    for m in messages {
        let tag = match m.role {
            Role::User => "user",
            Role::Assistant => "assistant",
        };
        out.push_str(&format!("<{tag}>\n"));
        for p in &m.parts {
            render_part(p, &mut out);
        }
        out.push_str(&format!("\n</{tag}>\n"));
    }
    out
}

/// Identifies the session configuration: a session is only reused for a
/// request with the same model, system prompt and tools.
fn config_key(model: Option<&str>, system: &[String], tools: &[ToolDef]) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    model.hash(&mut h);
    system.hash(&mut h);
    for t in tools {
        t.name.hash(&mut h);
        t.description.hash(&mut h);
        t.schema.as_ref().map(Value::to_string).hash(&mut h);
        t.freeform.hash(&mut h);
    }
    h.finish()
}

/// Fingerprint of a conversation, robust to formatting differences in how
/// clients echo it back (whitespace, thinking blocks, tool arguments).
fn fingerprint(key: u64, messages: &[Message]) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    key.hash(&mut h);
    let mut normalized: Vec<(bool, Vec<String>)> = Vec::new();
    for m in messages {
        let mut items = Vec::new();
        for p in &m.parts {
            match p {
                Part::Text(t) => {
                    let t = t.split_whitespace().collect::<Vec<_>>().join(" ");
                    if !t.is_empty() {
                        items.push(t);
                    }
                }
                Part::Image { .. } => items.push("[image]".into()),
                Part::ToolCall { id, name, .. } => items.push(format!("[call {id} {name}]")),
                Part::ToolResult { id, .. } => items.push(format!("[result {id}]")),
            }
        }
        if items.is_empty() {
            continue;
        }
        let user = m.role == Role::User;
        match normalized.last_mut() {
            Some((role, prev)) if *role == user => prev.extend(items),
            _ => normalized.push((user, items)),
        }
    }
    for (user, items) in normalized {
        user.hash(&mut h);
        // Join texts of a message: clients may split or merge text blocks.
        let mut text = String::new();
        for item in items {
            if item.starts_with("[call ") || item.starts_with("[result ") || item == "[image]" {
                text.hash(&mut h);
                text.clear();
                item.hash(&mut h);
            } else {
                if !text.is_empty() {
                    text.push(' ');
                }
                text.push_str(&item);
            }
        }
        text.hash(&mut h);
    }
    h.finish()
}

fn is_nested(event: &Value) -> bool {
    event.get("agentId").is_some_and(|v| !v.is_null())
        || event.pointer("/data/parentToolCallId").is_some_and(|v| !v.is_null())
}

fn add(acc: &mut Option<u64>, v: Option<u64>) {
    if let Some(v) = v {
        *acc = Some(acc.unwrap_or(0) + v);
    }
}

enum Outcome {
    /// The turn completed: the session is idle.
    Completed,
    /// The model called client tools: the turn waits for their results.
    ToolCalls(Vec<PendingTool>),
    /// The session cannot be reused.
    Failed,
}

async fn drive_turn(
    mut session: Parked,
    mut messages: Vec<Message>,
    tx: mpsc::Sender<TurnEvent>,
    store: Store,
    max_sessions: usize,
    _permit: OwnedSemaphorePermit,
) {
    let mut usage = Usage::default();
    let mut text_out = String::new();
    let mut streamed_messages: HashSet<String> = HashSet::new();
    let mut streamed_reasoning = false;
    let mut expected_tools = 0usize;
    let mut pending: Vec<PendingTool> = Vec::new();
    let mut batch_deadline: Option<tokio::time::Instant> = None;
    let sid = session.sid.clone();
    let proc = session.proc.clone();

    let outcome = loop {
        let deadline = batch_deadline;
        let event = tokio::select! {
            ev = session.events.recv() => ev,
            _ = async { tokio::time::sleep_until(deadline.unwrap()).await }, if deadline.is_some() => {
                break Outcome::ToolCalls(std::mem::take(&mut pending));
            }
            _ = tx.closed() => {
                debug!(session = %sid, "client went away, aborting Copilot turn");
                tokio::spawn(discard(proc.clone(), sid.clone(), true));
                return;
            }
        };
        let Some(event) = event else {
            let _ = tx.send(TurnEvent::Error("the Copilot process exited".into())).await;
            break Outcome::Failed;
        };
        let kind = event.get("type").and_then(Value::as_str).unwrap_or("");
        let data = event.get("data").cloned().unwrap_or(Value::Null);
        if is_nested(&event) {
            continue;
        }
        let str_of = |k: &str| data.get(k).and_then(Value::as_str).unwrap_or("").to_string();
        let ev = match kind {
            "assistant.message_delta" => {
                streamed_messages.insert(str_of("messageId"));
                let t = str_of("deltaContent");
                text_out.push_str(&t);
                Some(TurnEvent::Text(t))
            }
            "assistant.reasoning_delta" => {
                streamed_reasoning = true;
                Some(TurnEvent::Thought(str_of("deltaContent")))
            }
            "assistant.message" => {
                expected_tools = data.get("toolRequests").and_then(Value::as_array).map_or(0, Vec::len);
                let mut evs = Vec::new();
                if !streamed_reasoning && let Some(r) = data.get("reasoningText").and_then(Value::as_str) {
                    evs.push(TurnEvent::Thought(r.to_string()));
                }
                let content = str_of("content");
                if !streamed_messages.contains(&str_of("messageId")) && !content.is_empty() {
                    text_out.push_str(&content);
                    evs.push(TurnEvent::Text(content));
                }
                for ev in evs {
                    let _ = tx.send(ev).await;
                }
                if expected_tools > 0 && pending.len() >= expected_tools {
                    break Outcome::ToolCalls(std::mem::take(&mut pending));
                }
                None
            }
            "assistant.usage" => {
                let get = |k: &str| data.get(k).and_then(Value::as_u64);
                add(&mut usage.input_tokens, get("inputTokens"));
                add(&mut usage.output_tokens, get("outputTokens"));
                add(&mut usage.cached_read_tokens, get("cacheReadTokens"));
                None
            }
            "external_tool.requested" => {
                let name = session.names.client(&str_of("toolName"));
                let arguments = normalize_arguments(
                    data.get("arguments").cloned().unwrap_or(Value::Null),
                    session.names.freeform.contains(&name),
                );
                let id = str_of("toolCallId");
                pending.push(PendingTool {
                    tool_call_id: id.clone(),
                    request_id: str_of("requestId"),
                });
                let _ = tx.send(TurnEvent::ToolCall { id, name, arguments }).await;
                if expected_tools > 0 && pending.len() >= expected_tools {
                    break Outcome::ToolCalls(std::mem::take(&mut pending));
                }
                batch_deadline = Some(tokio::time::Instant::now() + TOOL_BATCH_WINDOW);
                None
            }
            "permission.requested" => {
                // Only client tools are available and they skip permissions;
                // anything else Copilot would want to do is refused.
                let request_id = str_of("requestId");
                let proc = proc.clone();
                let sid = sid.clone();
                tokio::spawn(async move {
                    let _ = proc
                        .request_timeout(
                            "session.permissions.handlePendingPermissionRequest",
                            json!({"sessionId": sid, "requestId": request_id,
                                   "result": {"kind": "reject", "feedback": "Not allowed through copilot-gateway."}}),
                            Duration::from_secs(10),
                        )
                        .await;
                });
                None
            }
            "session.error" => {
                let message = str_of("message");
                let kind = str_of("errorType");
                let _ = tx
                    .send(TurnEvent::Error(if kind.is_empty() {
                        message
                    } else {
                        format!("{kind}: {message}")
                    }))
                    .await;
                break Outcome::Failed;
            }
            "session.idle" => {
                if data.get("aborted").and_then(Value::as_bool) == Some(true) {
                    let _ = tx
                        .send(TurnEvent::Done {
                            stop_reason: "cancelled".into(),
                            usage,
                        })
                        .await;
                    break Outcome::Failed;
                }
                break Outcome::Completed;
            }
            _ => None,
        };
        if let Some(ev) = ev
            && tx.send(ev).await.is_err()
        {
            tokio::spawn(discard(proc.clone(), sid.clone(), true));
            return;
        }
    };

    match outcome {
        Outcome::Failed => {
            tokio::spawn(discard(proc, sid, true));
        }
        Outcome::Completed => {
            let _ = tx
                .send(TurnEvent::Done {
                    stop_reason: "end_turn".into(),
                    usage,
                })
                .await;
            if !text_out.is_empty() {
                messages.push(Message {
                    role: Role::Assistant,
                    parts: vec![Part::Text(text_out)],
                });
            }
            session.fingerprint = Some(fingerprint(session.config_key, &messages));
            session.pending.clear();
            park(session, &store, max_sessions);
        }
        Outcome::ToolCalls(pending) => {
            let _ = tx
                .send(TurnEvent::Done {
                    stop_reason: "tool_use".into(),
                    usage,
                })
                .await;
            session.fingerprint = None;
            session.pending = pending;
            park(session, &store, max_sessions);
        }
    }
}

fn park(mut session: Parked, store: &Store, max_sessions: usize) {
    session.last_used = Instant::now();
    let evicted = {
        let mut parked = store.lock().unwrap();
        parked.push(session);
        let mut evicted = Vec::new();
        while parked.len() > max_sessions.max(1) {
            let oldest = (0..parked.len()).min_by_key(|&i| parked[i].last_used).unwrap();
            evicted.push(parked.remove(oldest));
        }
        evicted
    };
    for p in evicted {
        tokio::spawn(discard(p.proc, p.sid, !p.pending.is_empty()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn msg(role: Role, parts: Vec<Part>) -> Message {
        Message { role, parts }
    }

    #[test]
    fn tool_names_are_sanitized_and_unique() {
        let tools = ["multi_agent.spawn", "multi_agent_spawn", "Bash", "a b"]
            .iter()
            .map(|n| ToolDef {
                name: n.to_string(),
                description: None,
                schema: None,
                freeform: false,
            })
            .collect::<Vec<_>>();
        let names = ToolNames::new(&tools);
        assert_eq!(names.wire("multi_agent.spawn"), "multi_agent_spawn");
        assert_eq!(names.wire("multi_agent_spawn"), "multi_agent_spawn_2");
        assert_eq!(names.wire("Bash"), "Bash");
        assert_eq!(names.client("a_b"), "a b");
        assert_eq!(names.client("multi_agent_spawn"), "multi_agent.spawn");
    }

    #[test]
    fn fingerprint_tolerates_formatting() {
        let a = vec![
            msg(Role::User, vec![Part::Text("hello  world".into())]),
            msg(
                Role::Assistant,
                vec![
                    Part::Text("Sure.\n".into()),
                    Part::ToolCall {
                        id: "c1".into(),
                        name: "Bash".into(),
                        arguments: json!({"command": "ls"}),
                    },
                ],
            ),
        ];
        let b = vec![
            msg(Role::User, vec![Part::Text("hello world".into())]),
            msg(Role::Assistant, vec![Part::Text("Sure.".into())]),
            msg(
                Role::Assistant,
                vec![Part::ToolCall {
                    id: "c1".into(),
                    name: "Bash".into(),
                    arguments: json!({}),
                }],
            ),
        ];
        assert_eq!(fingerprint(1, &a), fingerprint(1, &b));
        assert_ne!(fingerprint(1, &a), fingerprint(2, &a));
        let c = vec![msg(Role::User, vec![Part::Text("hello there".into())])];
        assert_ne!(fingerprint(1, &a), fingerprint(1, &c));
    }

    #[test]
    fn tool_results_and_extra_content() {
        let m = msg(
            Role::User,
            vec![
                Part::ToolResult {
                    id: "c1".into(),
                    name: None,
                    content: "ok".into(),
                    is_error: false,
                },
                Part::Image {
                    mime: "image/png".into(),
                    data: "AAA".into(),
                    url: None,
                },
                Part::Text("<system-reminder>x</system-reminder>".into()),
            ],
        );
        let r = tool_results(&m);
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].images.len(), 1);
        let (text, images) = extra_content(&m);
        assert_eq!(text, "<system-reminder>x</system-reminder>");
        assert!(images.is_empty());
    }

    #[test]
    fn create_params() {
        let req = ChatRequest {
            system: vec!["Be brief.".into()],
            tools: vec![ToolDef {
                name: "apply_patch".into(),
                description: Some("patch".into()),
                schema: None,
                freeform: true,
            }],
            ..Default::default()
        };
        let names = ToolNames::new(&req.tools);
        let cfg = BackendConfig {
            agent: crate::rpc::AgentCommand {
                program: "copilot".into(),
                args: vec![],
                env: vec![],
                cwd: "/tmp".into(),
            },
            permission: crate::acp::PermissionPolicy::Deny,
            default_model: None,
            model_map: vec![],
            max_concurrent: 1,
            session_ttl: Duration::from_secs(60),
            max_sessions: 4,
        };
        let p = session_create_params("s", Some("gpt-5"), Some("high"), &req, &names, &cfg);
        assert_eq!(p["systemMessage"], json!({"mode": "replace", "content": "Be brief."}));
        assert_eq!(p["availableTools"], json!(["custom:*"]));
        assert_eq!(p["tools"][0]["parameters"]["required"], json!(["input"]));
        assert_eq!(p["tools"][0]["skipPermission"], true);
        assert_eq!(p["model"], "gpt-5");
        assert_eq!(p["reasoningEffort"], "high");
    }
}
