# copilot-gateway

Use your **GitHub Copilot subscription** from tools that speak the **Anthropic** or **OpenAI** API, such as **Claude Code** and **Codex CLI**.

> [!IMPORTANT]
> This is an unofficial, early project. Tool calls are emulated (not native), every request may count against your Copilot premium requests, and several API parameters are ignored. Read the **[Limitations](#limitations)** before using it.

`copilot-gateway` is a single small binary (Rust, no runtime needed) for **macOS, Linux and Windows**. It starts the GitHub Copilot CLI in [Agent Client Protocol](https://agentclientprotocol.com) mode (`copilot --acp`) and exposes it as an HTTP server:

| API | Endpoint | Used by |
| --- | --- | --- |
| Anthropic Messages | `POST /v1/messages` (+ `/v1/messages/count_tokens`) | Claude Code, Anthropic SDKs |
| OpenAI Chat Completions | `POST /v1/chat/completions` | most OpenAI-compatible tools and SDKs |
| OpenAI Responses | `POST /v1/responses` | Codex CLI, OpenAI SDKs |
| Models | `GET /v1/models` | both |

It can also **launch Claude Code or Codex for you**, already configured to use the gateway.

```
 Claude Code ─┐                           ┌───────────────────────┐
 Codex CLI  ──┼─ HTTP ─▶ copilot-gateway ─┼─ stdio (ACP) ─▶ copilot --acp ─▶ GitHub Copilot models
 your app   ──┘ (Anthropic / OpenAI API)  └───────────────────────┘
```

---

## Quick start

### 1. Install and log in to the GitHub Copilot CLI (once)

You need an active GitHub Copilot plan.

```sh
npm install -g @github/copilot     # requires Node.js 22+
copilot login                      # opens a browser to authenticate
```

Other installation methods (Homebrew, WinGet, install script) are described in the [Copilot CLI documentation](https://docs.github.com/copilot/how-tos/copilot-cli). Instead of `copilot login`, you can also set `COPILOT_GITHUB_TOKEN` (or `GH_TOKEN`).

### 2. Install copilot-gateway

**Download a binary** from the [Releases page](https://github.com/phimage/copilot-gateway/releases):

| OS | File |
| --- | --- |
| macOS (Apple Silicon) | `copilot-gateway-<version>-aarch64-apple-darwin.tar.gz` |
| macOS (Intel) | `copilot-gateway-<version>-x86_64-apple-darwin.tar.gz` |
| Linux x64 (static) | `copilot-gateway-<version>-x86_64-unknown-linux-musl.tar.gz` |
| Linux ARM64 (static) | `copilot-gateway-<version>-aarch64-unknown-linux-musl.tar.gz` |
| Windows x64 | `copilot-gateway-<version>-x86_64-pc-windows-msvc.zip` |
| Windows ARM64 | `copilot-gateway-<version>-aarch64-pc-windows-msvc.zip` |

Extract it and put `copilot-gateway` (`copilot-gateway.exe` on Windows) somewhere in your `PATH`.

macOS marks downloaded binaries as quarantined. If macOS refuses to run it:

```sh
xattr -d com.apple.quarantine ./copilot-gateway
```

**Or build from source** with [Rust](https://rustup.rs) 1.88+:

```sh
cargo install --git https://github.com/phimage/copilot-gateway
```

### 3. Use it

```sh
copilot-gateway models     # check that everything works and list your models

copilot-gateway claude     # Claude Code, running on Copilot
copilot-gateway codex      # Codex CLI, running on Copilot
copilot-gateway serve      # HTTP server on http://127.0.0.1:4141
```

Claude Code (`npm install -g @anthropic-ai/claude-code`) and Codex CLI (`npm install -g @openai/codex`) must be installed for the `claude` and `codex` commands.

---

## Launching Claude Code / Codex

```sh
copilot-gateway claude                              # interactive Claude Code
copilot-gateway claude -- --resume                  # arguments after `--` go to claude
copilot-gateway --model claude-sonnet-4.5 claude    # force a Copilot model
copilot-gateway --small-model gpt-5-mini claude     # model for Claude Code's background tasks

copilot-gateway codex                               # interactive Codex
copilot-gateway --model gpt-5 codex -- exec "explain this repository"

copilot-gateway exec -- my-tool --flag              # any other tool
```

The launcher:

1. starts the gateway on `127.0.0.1` with a random port and a random API key,
2. starts `copilot --acp` and checks that you are logged in,
3. runs the tool in your terminal with the right configuration:
   - **claude**: `ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN`, plus `ANTHROPIC_MODEL` / `ANTHROPIC_DEFAULT_HAIKU_MODEL` when `--model` / `--small-model` are given,
   - **codex**: a `copilot_gateway` model provider (Responses API) passed with `-c` options. The model defaults to Copilot's default model.
   - **exec**: `OPENAI_BASE_URL`, `OPENAI_API_KEY`, `ANTHROPIC_BASE_URL`, `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, `COPILOT_GATEWAY_URL`,
4. stops everything when the tool exits and returns the tool's exit code.

While a tool runs, gateway logs go to `copilot-gateway.log` in the system temp directory, so they don't disturb the terminal UI. Use `--log-file` to change it.

## Running the server

```sh
copilot-gateway serve                          # http://127.0.0.1:4141, no API key required
copilot-gateway serve --port 8080 --api-key my-secret
copilot-gateway serve --host 0.0.0.0 --api-key my-secret   # expose on the network (use a key!)
```

### Claude Code against a running server

macOS / Linux:

```sh
export ANTHROPIC_BASE_URL=http://127.0.0.1:4141
export ANTHROPIC_AUTH_TOKEN=my-secret        # any value if the server has no --api-key
claude
```

Windows (PowerShell):

```powershell
$env:ANTHROPIC_BASE_URL = "http://127.0.0.1:4141"
$env:ANTHROPIC_AUTH_TOKEN = "my-secret"
claude
```

You can also put these variables in the `env` section of Claude Code's `~/.claude/settings.json`.

### Codex against a running server

`~/.codex/config.toml`:

```toml
model = "gpt-5"                 # a model from `copilot-gateway models`
model_provider = "copilot"

[model_providers.copilot]
name = "Copilot Gateway"
base_url = "http://127.0.0.1:4141/v1"
wire_api = "responses"
# env_key = "COPILOT_GATEWAY_API_KEY"   # only if the server uses --api-key
```

Set `model` explicitly. Codex's own default model makes it use an experimental tool format that works poorly with emulated tool calls.

### Any OpenAI / Anthropic client

```sh
curl http://127.0.0.1:4141/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "gpt-5", "messages": [{"role": "user", "content": "Hello!"}]}'

curl http://127.0.0.1:4141/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model": "claude-sonnet-4.5", "max_tokens": 1024, "messages": [{"role": "user", "content": "Hello!"}]}'
```

```python
from openai import OpenAI
client = OpenAI(base_url="http://127.0.0.1:4141/v1", api_key="anything")
print(client.chat.completions.create(model="gpt-5", messages=[{"role": "user", "content": "Hi"}]).choices[0].message.content)
```

Streaming (`"stream": true`), tool/function calling and base64 images are supported on all endpoints.

## Models

`copilot-gateway models` (or `GET /v1/models`) lists the models of your Copilot plan. The `model` sent by a client is matched to a Copilot model:

1. `--model-map` rules (`--model-map 'claude-3-5-haiku*=gpt-5-mini'`, `*` = any),
2. exact match,
3. normalized match: `claude-sonnet-4-5-20250929` → `claude-sonnet-4.5`,
4. same family: `claude-opus-4-5` → newest available `claude-opus-*`, `gpt-5.1-codex` → `gpt-5-codex`,
5. otherwise `--model` if given, else Copilot's default model.

The selected model is logged (`model selected requested=... model=...`).

## Options

All options can go before or after the command. With `claude` / `codex` / `exec`, put the tool's own arguments after `--`.

| Option | Env variable | Default | Description |
| --- | --- | --- | --- |
| `--host` | `COPILOT_GATEWAY_HOST` | `127.0.0.1` | Listen address |
| `--port` | `COPILOT_GATEWAY_PORT` | `4141` (`serve`), random (launchers) | Listen port |
| `--api-key` | `COPILOT_GATEWAY_API_KEY` | none | Key required from clients (`Authorization: Bearer` or `x-api-key`) |
| `-m, --model` | `COPILOT_GATEWAY_MODEL` | Copilot default | Default / forced model |
| `--small-model` | `COPILOT_GATEWAY_SMALL_MODEL` | none | Claude Code background-task model |
| `--model-map PATTERN=MODEL` | | | Rewrite requested model names (repeatable) |
| `--copilot-bin` | `COPILOT_GATEWAY_COPILOT_BIN` | `copilot` | Copilot CLI executable |
| `--copilot-arg ARG` | | | Extra argument for `copilot` (repeatable), e.g. `--copilot-arg=--reasoning-effort=high` |
| `--copilot-env KEY=VALUE` | | | Extra environment variable for `copilot` (repeatable) |
| `--no-default-copilot-args` | | | Don't pass the default `copilot` arguments (below) |
| `--agent-tools` | | off | Keep Copilot's built-in tools (shell, file edits...) enabled |
| `--permission deny\|allow` | | `deny` | Answer to Copilot's own tool permission requests |
| `--workdir` | `COPILOT_GATEWAY_WORKDIR` | `<temp>/copilot-gateway/workspace` | Working directory of Copilot sessions |
| `--max-concurrent` | | `4` | Maximum parallel requests sent to Copilot |
| `--log-level` | `COPILOT_GATEWAY_LOG` | `info` | `error`, `warn`, `info`, `debug`, `trace` (`trace` logs request bodies) |
| `--log-file` | | stderr / temp file | Log destination |

Default `copilot` arguments: `--acp --disable-builtin-mcps --no-custom-instructions --no-ask-user --available-tools=` (the last one is omitted with `--agent-tools`).

## How it works

- At startup the gateway spawns `copilot --acp` and talks JSON-RPC to it over stdin/stdout. If the process dies, it is restarted on the next request.
- Each HTTP request creates a new ACP session in an empty working directory, selects the model (and reasoning effort when Copilot offers it), sends the whole conversation as one prompt, streams the reply back in the requested API format, and closes the session. If the client disconnects, the turn is cancelled.
- **Client tools** (Claude Code's `Bash`, `Edit`..., Codex's `exec_command`, `apply_patch`...) are **emulated**. Their definitions are written into the prompt, and the model is asked to answer with `<tool_call>{"name": ..., "arguments": ...}</tool_call>` blocks. The gateway parses these while streaming and turns them into native `tool_use` / `tool_calls` / `function_call` items. The client runs the tool and sends the result back in the next request, as with a real API.
- Copilot's **own** tools are disabled by default, and any permission request Copilot makes is denied, so Copilot behaves as a plain model and never touches your files. The client tool does that, with its own permission system.

## Development

```sh
cargo test                 # unit tests + end-to-end HTTP tests against a built-in mock ACP agent
cargo clippy --all-targets
```

The end-to-end tests run the gateway against a fake ACP agent built into the binary (enabled with the `COPILOT_GATEWAY_MOCK_AGENT=1` environment variable on the agent process). You can use it to try the gateway without a Copilot subscription:

```sh
copilot-gateway serve --copilot-bin "$(which copilot-gateway)" --no-default-copilot-args \
  --copilot-env COPILOT_GATEWAY_MOCK_AGENT=1
```

**Releases**: push a tag such as `v0.1.0`. The [release workflow](.github/workflows/release.yml) builds the binaries for macOS (arm64, x64), Linux (static musl x64, arm64) and Windows (x64, arm64), and attaches them with SHA-256 checksums to a GitHub release. CI runs tests on all three operating systems for every push and pull request.

## Limitations

- **Unofficial.** This project is not affiliated with GitHub, Anthropic or OpenAI. Make sure your usage complies with the GitHub Copilot terms of service.
- **Tool calling is emulated, not native.** Copilot's ACP mode does not expose the raw model API, so tool calls go through the text protocol described above. Strong models follow it well, but a model may sometimes write a malformed call, which is then returned as plain text. Very large tool sets (many MCP servers in Claude Code) make the prompt longer and calls less reliable.
- **Copilot's own system prompt is still present.** The model is told to follow the client's system prompt and to act as a plain model, but it is still running inside the Copilot agent, so behavior can differ slightly from the vendor's API.
- **Stateless, no prompt caching.** Each request opens a new Copilot session and resends the whole conversation. Long conversations mean big prompts and slower first tokens.
- **Premium requests.** Each HTTP request is a Copilot prompt and may count against your premium request quota, with the model's multiplier. Claude Code and Codex send many requests: one per tool step, plus background tasks such as titles and summaries. Use `--small-model` with a model that is included in your plan for Claude Code's background tasks, and check your usage on GitHub.
- **Approximate token usage.** Token counts come from Copilot when it reports them, otherwise they are estimated as characters ÷ 4. `count_tokens` is always an estimate.
- **Ignored parameters:** `max_tokens`, `temperature`, `top_p`, stop sequences, `n`, `logprobs`, structured output (`response_format` / JSON schema), `parallel_tool_calls`. The model decides when to stop.
- **Thinking / reasoning** is forwarded only when the client asks for it (Anthropic `thinking`, OpenAI `reasoning`). Anthropic thinking blocks carry no signature, and reasoning sent back by clients is not replayed to the model.
- **Content types:** text and base64 images are supported. Remote image URLs are not fetched (only the URL is passed on). PDFs and other binary documents are not supported. Server-side/hosted tools (Anthropic `web_search`, OpenAI `web_search`, code interpreter...) are ignored.
- **Codex's default model** uses an experimental tool format. `copilot-gateway codex` therefore selects Copilot's default model. With `serve`, set `model` in `config.toml`.
- **Model names** are matched approximately (see [Models](#models)). An unknown model silently falls back to the default. Check the logs if you need to know which model answered.
- **Concurrency** is limited to `--max-concurrent` parallel requests (default 4) on a single `copilot` process.
- **Early version.** The protocol handling is covered by end-to-end tests with a scripted ACP agent, and has been exercised with real Claude Code and Codex CLI clients. Copilot's ACP mode is itself young and may change between Copilot CLI releases.
