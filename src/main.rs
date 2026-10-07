use std::net::SocketAddr;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;

use anyhow::{Context, Result};
use clap::{Args, Parser, Subcommand};
use copilot_gateway::acp::{AgentCommand, PermissionPolicy};
use copilot_gateway::backend::{Backend, BackendConfig};
use copilot_gateway::launch::{self, LaunchOptions, Tool};
use copilot_gateway::mock_agent;
use copilot_gateway::server::{self, AppState};
use tracing::{info, warn};

const DEFAULT_PORT: u16 = 4141;

#[derive(Parser)]
#[command(
    name = "copilot-gateway",
    version,
    about = "Expose GitHub Copilot CLI (`copilot --acp`) as Anthropic and OpenAI compatible HTTP APIs",
    after_help = "Examples:\n  \
copilot-gateway serve                     # HTTP server on http://127.0.0.1:4141\n  \
copilot-gateway claude                    # run Claude Code on your Copilot subscription\n  \
copilot-gateway codex                     # run Codex CLI on your Copilot subscription\n  \
copilot-gateway --model gpt-5 claude -- --resume\n  \
copilot-gateway exec -- my-tool --flag    # any tool reading OPENAI_*/ANTHROPIC_* variables\n  \
copilot-gateway models                    # list the models of your Copilot plan"
)]
struct Cli {
    #[command(flatten)]
    opts: Options,
    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Args, Clone)]
struct Options {
    /// Address to listen on.
    #[arg(long, env = "COPILOT_GATEWAY_HOST", default_value = "127.0.0.1", global = true)]
    host: String,

    /// Port to listen on [default: 4141 for `serve`, random for launched tools].
    #[arg(long, env = "COPILOT_GATEWAY_PORT", global = true)]
    port: Option<u16>,

    /// API key clients must send (`Authorization: Bearer` or `x-api-key`).
    /// Without it any key is accepted. Launched tools get a random key.
    #[arg(long, env = "COPILOT_GATEWAY_API_KEY", global = true, hide_env_values = true)]
    api_key: Option<String>,

    /// Copilot CLI executable (name in PATH or full path).
    #[arg(long, env = "COPILOT_GATEWAY_COPILOT_BIN", default_value = "copilot", global = true)]
    copilot_bin: String,

    /// Extra argument passed to the agent (repeatable), e.g. `--copilot-arg=--model=gpt-5`.
    #[arg(long = "copilot-arg", allow_hyphen_values = true, global = true)]
    copilot_args: Vec<String>,

    /// Replace the default agent arguments entirely (use with --copilot-arg).
    #[arg(long, global = true)]
    no_default_copilot_args: bool,

    /// Extra environment variable for the agent, KEY=VALUE (repeatable).
    #[arg(long = "copilot-env", value_parser = parse_key_value, global = true)]
    copilot_env: Vec<(String, String)>,

    /// Keep Copilot's own built-in tools enabled (file edits, shell...).
    /// By default they are disabled so Copilot behaves as a plain model.
    #[arg(long, global = true)]
    agent_tools: bool,

    /// Answer to the agent's own tool permission requests.
    #[arg(long, value_enum, default_value = "deny", global = true)]
    permission: PermissionPolicy,

    /// Default model, used when a requested model is not available.
    #[arg(long, short = 'm', env = "COPILOT_GATEWAY_MODEL", global = true)]
    model: Option<String>,

    /// Model for small/background tasks of launched tools (Claude Code "haiku" slot).
    #[arg(long, env = "COPILOT_GATEWAY_SMALL_MODEL", global = true)]
    small_model: Option<String>,

    /// Rewrite a requested model: PATTERN=MODEL, PATTERN may end with `*` (repeatable).
    #[arg(long = "model-map", value_parser = parse_key_value, global = true)]
    model_map: Vec<(String, String)>,

    /// Working directory of the agent sessions [default: a private temp directory].
    #[arg(long, env = "COPILOT_GATEWAY_WORKDIR", global = true)]
    workdir: Option<PathBuf>,

    /// Maximum number of concurrent requests sent to the agent.
    #[arg(long, default_value_t = 4, global = true)]
    max_concurrent: usize,

    /// Log filter (error, warn, info, debug, trace).
    #[arg(long, env = "COPILOT_GATEWAY_LOG", default_value = "info", global = true)]
    log_level: String,

    /// Log file [default: stderr for `serve`, a temp file for launched tools].
    #[arg(long, global = true)]
    log_file: Option<PathBuf>,
}

#[derive(Subcommand, Clone)]
enum Commands {
    /// Run the HTTP gateway (default command).
    Serve,
    /// Start the gateway and run Claude Code against it.
    Claude {
        /// Arguments passed to `claude`.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Start the gateway and run Codex CLI against it.
    Codex {
        /// Arguments passed to `codex`.
        #[arg(trailing_var_arg = true, allow_hyphen_values = true)]
        args: Vec<String>,
    },
    /// Start the gateway and run any command with OPENAI_* / ANTHROPIC_* variables set.
    Exec {
        /// Command and its arguments.
        #[arg(required = true, trailing_var_arg = true, allow_hyphen_values = true)]
        command: Vec<String>,
    },
    /// List the models available through Copilot.
    Models,
}

fn parse_key_value(s: &str) -> Result<(String, String), String> {
    s.split_once('=')
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .ok_or_else(|| format!("expected KEY=VALUE, got `{s}`"))
}

fn main() -> ExitCode {
    let runtime = tokio::runtime::Runtime::new().expect("tokio runtime");
    if std::env::var_os(mock_agent::ENV_VAR).is_some() {
        return match runtime.block_on(mock_agent::run()) {
            Ok(()) => ExitCode::SUCCESS,
            Err(_) => ExitCode::FAILURE,
        };
    }
    let cli = Cli::parse();
    match runtime.block_on(run(cli)) {
        Ok(code) => ExitCode::from(code.clamp(0, 255) as u8),
        Err(e) => {
            eprintln!("error: {e:#}");
            ExitCode::FAILURE
        }
    }
}

fn init_logging(opts: &Options, launched: bool) -> Result<Option<PathBuf>> {
    use tracing_subscriber::EnvFilter;
    let filter = EnvFilter::try_new(&opts.log_level).unwrap_or_else(|_| EnvFilter::new("info"));
    // A launched tool owns the terminal: log to a file instead of stderr.
    let file = opts
        .log_file
        .clone()
        .or_else(|| launched.then(|| std::env::temp_dir().join("copilot-gateway.log")));
    match &file {
        Some(path) => {
            let f = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)
                .with_context(|| format!("cannot open log file {}", path.display()))?;
            tracing_subscriber::fmt()
                .with_env_filter(filter)
                .with_ansi(false)
                .with_writer(std::sync::Mutex::new(f))
                .init();
        }
        None => tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_writer(std::io::stderr)
            .init(),
    }
    Ok(file)
}

fn backend_config(opts: &Options) -> Result<BackendConfig> {
    let program = launch::resolve_program(&opts.copilot_bin).with_context(|| {
        "GitHub Copilot CLI not found: install it with `npm install -g @github/copilot` \
         (or see https://github.com/github/copilot-cli), run `copilot login`, \
         or pass --copilot-bin <path>"
    })?;

    let mut args = Vec::new();
    if !opts.no_default_copilot_args {
        args.extend(
            [
                "--acp",
                "--disable-builtin-mcps",
                "--no-custom-instructions",
                "--no-ask-user",
            ]
            .map(String::from),
        );
        if !opts.agent_tools {
            args.push("--available-tools=".into());
        }
    }
    args.extend(opts.copilot_args.iter().cloned());

    let cwd = match &opts.workdir {
        Some(dir) => dir.clone(),
        None => std::env::temp_dir().join("copilot-gateway").join("workspace"),
    };
    std::fs::create_dir_all(&cwd).with_context(|| format!("cannot create {}", cwd.display()))?;
    let cwd = std::fs::canonicalize(&cwd).unwrap_or(cwd);
    let cwd = strip_verbatim(cwd);

    Ok(BackendConfig {
        agent: AgentCommand {
            program,
            args,
            env: opts.copilot_env.clone(),
            cwd,
        },
        permission: opts.permission,
        default_model: opts.model.clone(),
        model_map: opts.model_map.clone(),
        max_concurrent: opts.max_concurrent,
    })
}

/// `canonicalize` returns `\\?\C:\...` paths on Windows; agents expect plain ones.
fn strip_verbatim(p: PathBuf) -> PathBuf {
    let s = p.to_string_lossy();
    match s.strip_prefix(r"\\?\") {
        Some(rest) if !rest.starts_with("UNC") => PathBuf::from(rest),
        _ => p,
    }
}

async fn run(cli: Cli) -> Result<i32> {
    let command = cli.command.unwrap_or(Commands::Serve);
    let opts = cli.opts;
    let launched = matches!(
        command,
        Commands::Claude { .. } | Commands::Codex { .. } | Commands::Exec { .. }
    );
    let log_file = init_logging(&opts, launched)?;

    let backend = Arc::new(Backend::new(backend_config(&opts)?));

    match command {
        Commands::Models => {
            let models = backend.list_models().await?;
            for m in models {
                println!("{:<32} {}", m.id, m.name);
            }
            backend.shutdown().await;
            Ok(0)
        }
        Commands::Serve => {
            let addr = socket_addr(&opts, DEFAULT_PORT)?;
            let state = AppState {
                backend: backend.clone(),
                api_key: opts.api_key.clone(),
            };
            {
                let backend = backend.clone();
                tokio::spawn(async move {
                    if let Err(e) = backend.warm_up().await {
                        warn!("could not start the agent yet (will retry on first request): {e:#}");
                    }
                });
            }
            let key = opts.api_key.clone().unwrap_or_else(|| "anything".into());
            server::serve(
                state,
                addr,
                |local| {
                    info!("Anthropic API: ANTHROPIC_BASE_URL=http://{local} ANTHROPIC_AUTH_TOKEN={key}");
                    info!("OpenAI API:    OPENAI_BASE_URL=http://{local}/v1 OPENAI_API_KEY={key}");
                },
                async {
                    let _ = tokio::signal::ctrl_c().await;
                    info!("shutting down");
                },
            )
            .await?;
            backend.shutdown().await;
            Ok(0)
        }
        Commands::Claude { args } => launch_tool(Tool::Claude, args, &opts, backend, log_file).await,
        Commands::Codex { args } => launch_tool(Tool::Codex, args, &opts, backend, log_file).await,
        Commands::Exec { mut command } => {
            let program = command.remove(0);
            launch_tool(Tool::Exec(program), command, &opts, backend, log_file).await
        }
    }
}

fn socket_addr(opts: &Options, default_port: u16) -> Result<SocketAddr> {
    let port = opts.port.unwrap_or(default_port);
    let host = if opts.host == "localhost" {
        "127.0.0.1"
    } else {
        &opts.host
    };
    format!("{host}:{port}")
        .parse::<SocketAddr>()
        .or_else(|_| format!("[{host}]:{port}").parse())
        .with_context(|| format!("invalid listen address {host}:{port}"))
}

async fn launch_tool(
    tool: Tool,
    args: Vec<String>,
    opts: &Options,
    backend: Arc<Backend>,
    log_file: Option<PathBuf>,
) -> Result<i32> {
    let api_key = opts
        .api_key
        .clone()
        .unwrap_or_else(|| format!("cgw-{}", uuid::Uuid::new_v4().simple()));
    let addr = socket_addr(opts, 0)?;
    let listener_addr = Arc::new(std::sync::Mutex::new(None));
    let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();
    let (bound_tx, bound_rx) = tokio::sync::oneshot::channel::<SocketAddr>();

    let state = AppState {
        backend: backend.clone(),
        api_key: Some(api_key.clone()),
    };
    let server = tokio::spawn({
        let listener_addr = listener_addr.clone();
        async move {
            server::serve(
                state,
                addr,
                move |local| {
                    *listener_addr.lock().unwrap() = Some(local);
                    let _ = bound_tx.send(local);
                },
                async {
                    let _ = shutdown_rx.await;
                },
            )
            .await
        }
    });
    let local = match bound_rx.await {
        Ok(a) => a,
        Err(_) => {
            return Err(server
                .await?
                .err()
                .unwrap_or_else(|| anyhow::anyhow!("server failed to start")));
        }
    };

    // Start Copilot and open a session now, so that installation and
    // authentication problems show up before the tool starts.
    if let Err(e) = backend.list_models().await {
        let _ = shutdown_tx.send(());
        return Err(e.context("cannot start the Copilot CLI in ACP mode"));
    }
    if let Some(f) = &log_file {
        eprintln!("copilot-gateway: listening on http://{local} (logs: {})", f.display());
    }

    // Codex adapts its tool set to the model name: give it a real Copilot
    // model rather than its own default.
    let model = match (&tool, &opts.model) {
        (Tool::Codex, None) => backend.agent_default_model().await,
        _ => opts.model.clone(),
    };
    let launch_opts = LaunchOptions {
        base_url: format!("http://{local}"),
        api_key,
        model,
        small_model: opts.small_model.clone(),
    };
    let result = launch::run(&tool, &args, &launch_opts).await;
    let _ = shutdown_tx.send(());
    let _ = server.await;
    backend.shutdown().await;
    result
}
