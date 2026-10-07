//! Launch a client tool (Claude Code, Codex CLI, or any command) wired to the
//! gateway through environment variables / config overrides.

use std::ffi::OsString;
use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use tokio::process::Command;
use tracing::info;

#[derive(Debug, Clone)]
pub enum Tool {
    Claude,
    Codex,
    /// Any command; receives both the Anthropic and OpenAI variables.
    Exec(String),
}

#[derive(Debug, Clone)]
pub struct LaunchOptions {
    /// Gateway root URL, e.g. `http://127.0.0.1:41234`.
    pub base_url: String,
    pub api_key: String,
    /// Main model to ask the tool to use.
    pub model: Option<String>,
    /// Model for background / small tasks (Claude Code's "haiku" slot).
    pub small_model: Option<String>,
}

/// Resolve a program name through `PATH` (and `PATHEXT` on Windows, so that
/// npm `.cmd` shims such as `claude.cmd` are found).
pub fn resolve_program(name: &str) -> Result<PathBuf> {
    which::which(name).with_context(|| format!("`{name}` was not found in PATH"))
}

pub struct Prepared {
    pub program: PathBuf,
    pub args: Vec<OsString>,
    pub env: Vec<(String, String)>,
    pub env_remove: Vec<String>,
}

pub fn prepare(tool: &Tool, args: &[String], opts: &LaunchOptions) -> Result<Prepared> {
    let mut env = Vec::new();
    let mut env_remove = Vec::new();
    let mut final_args: Vec<OsString> = Vec::new();
    let v1 = format!("{}/v1", opts.base_url);

    let name = match tool {
        Tool::Claude => {
            env.push(("ANTHROPIC_BASE_URL".into(), opts.base_url.clone()));
            env.push(("ANTHROPIC_AUTH_TOKEN".into(), opts.api_key.clone()));
            // An API key would take precedence over the auth token (and make
            // Claude Code ask whether to use it).
            env_remove.push("ANTHROPIC_API_KEY".into());
            if std::env::var_os("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC").is_none() {
                env.push(("CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC".into(), "1".into()));
            }
            if let Some(m) = &opts.model {
                env.push(("ANTHROPIC_MODEL".into(), m.clone()));
            }
            if let Some(m) = &opts.small_model {
                env.push(("ANTHROPIC_DEFAULT_HAIKU_MODEL".into(), m.clone()));
                env.push(("ANTHROPIC_SMALL_FAST_MODEL".into(), m.clone()));
            }
            "claude".to_string()
        }
        Tool::Codex => {
            env.push(("COPILOT_GATEWAY_API_KEY".into(), opts.api_key.clone()));
            let provider = format!(
                "model_providers.copilot_gateway={{ name = \"Copilot Gateway\", base_url = \"{v1}\", \
                 env_key = \"COPILOT_GATEWAY_API_KEY\", wire_api = \"responses\" }}"
            );
            for c in [provider, "model_provider=\"copilot_gateway\"".to_string()] {
                final_args.push("-c".into());
                final_args.push(c.into());
            }
            if let Some(m) = &opts.model {
                final_args.push("-c".into());
                final_args.push(format!("model=\"{m}\"").into());
            }
            "codex".to_string()
        }
        Tool::Exec(cmd) => {
            for (k, v) in [
                ("COPILOT_GATEWAY_URL", opts.base_url.clone()),
                ("COPILOT_GATEWAY_API_KEY", opts.api_key.clone()),
                ("ANTHROPIC_BASE_URL", opts.base_url.clone()),
                ("ANTHROPIC_API_KEY", opts.api_key.clone()),
                ("ANTHROPIC_AUTH_TOKEN", opts.api_key.clone()),
                ("OPENAI_BASE_URL", v1.clone()),
                ("OPENAI_API_BASE", v1.clone()),
                ("OPENAI_API_KEY", opts.api_key.clone()),
            ] {
                env.push((k.into(), v));
            }
            if let Some(m) = &opts.model {
                env.push(("ANTHROPIC_MODEL".into(), m.clone()));
                env.push(("OPENAI_MODEL".into(), m.clone()));
            }
            cmd.clone()
        }
    };
    final_args.extend(args.iter().map(OsString::from));
    let program = resolve_program(&name).with_context(|| match tool {
        Tool::Claude => "install Claude Code with `npm install -g @anthropic-ai/claude-code`",
        Tool::Codex => "install Codex CLI with `npm install -g @openai/codex`",
        Tool::Exec(_) => "check the command name",
    })?;
    Ok(Prepared {
        program,
        args: final_args,
        env,
        env_remove,
    })
}

/// Run the tool attached to the current terminal and return its exit code.
pub async fn run(tool: &Tool, args: &[String], opts: &LaunchOptions) -> Result<i32> {
    let p = prepare(tool, args, opts)?;
    info!(program = %p.program.display(), "launching");
    let mut cmd = Command::new(&p.program);
    cmd.args(&p.args);
    for (k, v) in &p.env {
        cmd.env(k, v);
    }
    for k in &p.env_remove {
        cmd.env_remove(k);
    }
    let mut child = cmd
        .spawn()
        .with_context(|| format!("failed to start {}", p.program.display()))?;

    // Ctrl+C belongs to the interactive tool: the gateway must survive it.
    let ignore_ctrl_c = tokio::spawn(async {
        loop {
            if tokio::signal::ctrl_c().await.is_err() {
                break;
            }
        }
    });
    let status = child.wait().await?;
    ignore_ctrl_c.abort();
    match status.code() {
        Some(code) => Ok(code),
        None => {
            if status.success() {
                Ok(0)
            } else {
                bail!("{} was terminated by a signal", p.program.display())
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn opts() -> LaunchOptions {
        LaunchOptions {
            base_url: "http://127.0.0.1:1234".into(),
            api_key: "k".into(),
            model: Some("gpt-5".into()),
            small_model: Some("gpt-5-mini".into()),
        }
    }

    #[test]
    fn exec_env() {
        // `cargo` is always available while running the tests.
        let p = prepare(&Tool::Exec("cargo".into()), &["--version".into()], &opts()).unwrap();
        assert!(
            p.env
                .contains(&("OPENAI_BASE_URL".into(), "http://127.0.0.1:1234/v1".into()))
        );
        assert!(
            p.env
                .contains(&("ANTHROPIC_BASE_URL".into(), "http://127.0.0.1:1234".into()))
        );
        assert_eq!(p.args, vec![OsString::from("--version")]);
    }
}
