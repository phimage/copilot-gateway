# copilot-gateway

Use your **GitHub Copilot subscription** from tools that speak the **Anthropic** or **OpenAI** API, such as **Claude Code** and **Codex CLI**.

> [!IMPORTANT]
> This is an unofficial, early project. It relies on an undocumented Copilot CLI protocol, requests may count against your Copilot premium requests, and some API parameters are ignored. Read the **[Limitations](#limitations)** (per backend: `sdk` or `acp`) before using it.

`copilot-gateway` is a single small binary (Rust, no runtime needed) for **macOS, Linux and Windows**. It starts the GitHub Copilot CLI in headless mode and exposes it as an HTTP server:

| API | Endpoint | Used by |
| --- | --- | --- |
| Anthropic Messages | `POST /v1/messages` (+ `/v1/messages/count_tokens`) | Claude Code, Anthropic SDKs |
| OpenAI Chat Completions | `POST /v1/chat/completions` | most OpenAI-compatible tools and SDKs |
| OpenAI Responses | `POST /v1/responses` | Codex CLI, OpenAI SDKs |
| Models | `GET /v1/models` | both |

It can also **launch Claude Code or Codex for you**, already configured to use the gateway.

```
 Claude Code ─┐                                   stdio (JSON-RPC)
 Codex CLI  ──┼─ HTTP ─▶ copilot-gateway ─────────────────────────▶ copilot --headless ─▶ GitHub Copilot models
 your app   ──┘ (Anthropic / OpenAI API)    (or copilot --acp with --backend acp)
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
2. starts the Copilot CLI and checks that you are logged in,
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

Set `model` explicitly: Codex adapts its tool set to the model name, and its own default model uses an experimental "code mode" tool format.

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

The selected model is logged (`new Copilot session ... model=... requested=...`).

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
| `--backend sdk\|acp` | `COPILOT_GATEWAY_BACKEND` | `sdk` | How to drive Copilot (see [Backends](#backends)) |
| `--copilot-bin` | `COPILOT_GATEWAY_COPILOT_BIN` | `copilot` | Copilot CLI executable |
| `--copilot-arg ARG` | | | Extra argument for `copilot` (repeatable), e.g. `--copilot-arg=--reasoning-effort=high` |
| `--copilot-env KEY=VALUE` | | | Extra environment variable for `copilot` (repeatable) |
| `--no-default-copilot-args` | | | Don't pass the default `copilot` arguments (below) |
| `--agent-tools` | | off | ACP only: keep Copilot's built-in tools (shell, file edits...) enabled |
| `--permission deny\|allow` | | `deny` | ACP only: answer to Copilot's own tool permission requests |
| `--workdir` | `COPILOT_GATEWAY_WORKDIR` | `<temp>/copilot-gateway/workspace` | Working directory of Copilot sessions |
| `--max-concurrent` | | `4` | Maximum parallel requests sent to Copilot |
| `--session-ttl SECONDS` | | `1800` | SDK only: how long an idle conversation keeps its Copilot session |
| `--max-sessions` | | `16` | SDK only: Copilot sessions kept alive between requests |
| `--log-level` | `COPILOT_GATEWAY_LOG` | `info` | `error`, `warn`, `info`, `debug`, `trace` (`trace` logs request bodies) |
| `--log-file` | | stderr / temp file | Log destination |

Default `copilot` arguments: `--headless --stdio --no-auto-update` (SDK backend), or `--acp --disable-builtin-mcps --no-custom-instructions --no-ask-user --available-tools=` (ACP backend; the last one is omitted with `--agent-tools`).

## Backends

### `sdk` (default): Copilot SDK protocol

The gateway runs `copilot --headless --stdio` and speaks the JSON-RPC protocol used by the official [GitHub Copilot SDK](https://github.com/github/copilot-sdk):

- **Native client tools.** The client's tools (Claude Code's `Bash`, `Edit`..., Codex's `exec_command`, `apply_patch`...) are declared to Copilot as real tools. When the model calls one, the gateway returns a `tool_use` / `tool_calls` / `function_call` to the client, the client runs it, and its next request carries the result back to Copilot.
- **The client's system prompt replaces Copilot's.** Copilot's own tools, custom instructions, skills and memory are disabled. Copilot behaves as a plain model and never touches your files; the client tool does that, with its own permission system.
- **One Copilot session per conversation.** A tool loop (model → tool → result → model...) stays inside a single Copilot turn. A follow-up message continues the same session instead of resending the whole conversation. Idle sessions are kept for `--session-ttl` seconds.
- **Real token usage** is reported, as given by Copilot.
- Images are passed as attachments, including images in tool results (screenshots).

### `acp`: Agent Client Protocol

`--backend acp` runs `copilot --acp` ([Agent Client Protocol](https://agentclientprotocol.com)) instead. ACP has no way to declare client tools, so this backend **emulates** them: their definitions are written into the prompt, the model answers with `<tool_call>{"name": ..., "arguments": ...}</tool_call>` blocks, and the gateway turns them into native tool calls. Each request opens a new session and resends the whole conversation, and Copilot's own system prompt stays in place. Use it as a fallback if the SDK protocol changes in a Copilot CLI update.

### Common to both

- The Copilot process is started once and restarted if it dies. Sessions run in an empty working directory.
- If the client disconnects, the running turn is cancelled.

## Development

```sh
cargo test                 # unit tests + end-to-end HTTP tests against built-in mock Copilot agents
cargo clippy --all-targets
```

The end-to-end tests run the gateway against fake Copilot agents built into the binary, enabled with the `COPILOT_GATEWAY_MOCK_AGENT=sdk` (or `=acp`) environment variable on the agent process. You can use them to try the gateway without a Copilot subscription:

```sh
copilot-gateway serve --copilot-bin "$(which copilot-gateway)" --no-default-copilot-args \
  --copilot-env COPILOT_GATEWAY_MOCK_AGENT=sdk
```

**Releases**: push a tag such as `v0.1.0`. The [release workflow](.github/workflows/release.yml) builds the binaries for macOS (arm64, x64), Linux (static musl x64, arm64) and Windows (x64, arm64), and attaches them with SHA-256 checksums to a GitHub release. CI runs tests on all three operating systems for every push and pull request.

## Limitations

### At a glance

| | `sdk` (default) | `acp` (`--backend acp`) |
| --- | --- | --- |
| Client tools (Bash, Edit, exec_command...) | Native Copilot tools | Emulated through the prompt |
| System prompt | The client's, replacing Copilot's | The client's, inside Copilot's agent prompt |
| Copilot's own tools | Disabled | Disabled by `--available-tools=`, permission requests denied |
| Conversation | One Copilot session per conversation | New session per request, whole conversation resent |
| Tool loop (model → tool → result → model) | One Copilot prompt | One Copilot prompt per step |
| Token usage | Reported by Copilot | Reported by Copilot when available, else estimated |
| Protocol | Undocumented Copilot SDK protocol (experimental) | Agent Client Protocol (public spec) |

### Both backends

- **Unofficial.** This project is not affiliated with GitHub, Anthropic or OpenAI. Make sure your usage complies with the GitHub Copilot terms of service.
- **Premium requests.** Every new prompt sent to Copilot may count against your premium request quota, with the model's multiplier (see each backend below for what counts as a new prompt). Claude Code also sends background requests (titles, summaries); use `--small-model` with a model included in your plan for those, and check your usage on GitHub.
- **Ignored parameters:** `max_tokens`, `temperature`, `top_p`, stop sequences, `n`, `logprobs`, structured output (`response_format` / JSON schema), `parallel_tool_calls`. Copilot does not expose them; the model decides when to stop.
- **Thinking / reasoning** is forwarded only when the client asks for it (Anthropic `thinking`, OpenAI `reasoning`). Anthropic thinking blocks carry no signature, and reasoning sent back by clients is not replayed. The requested reasoning effort is applied only when the Copilot model supports that value.
- **Content types:** text and base64 images are supported, including images in tool results. Remote image URLs are not fetched (only the URL is passed on). PDFs and other binary documents are not supported. Server-side/hosted tools (Anthropic `web_search`, OpenAI `web_search`, code interpreter...) are ignored.
- **`count_tokens`** is always an estimate (characters ÷ 4): there is no tokenizer for Copilot's models.
- **Model names** are matched approximately (see [Models](#models)). An unknown model silently falls back to the default; check the logs to know which model answered.
- **Early version.** Protocol handling is covered by end-to-end tests with scripted mock agents and has been exercised with real Claude Code and Codex CLI clients, but not yet at scale with real Copilot models.

### `sdk` backend (default)

- **Undocumented protocol.** It speaks the JSON-RPC protocol of the GitHub Copilot SDK (version 3), which is not publicly documented and is marked experimental. A Copilot CLI update could break it; `--backend acp` is the fallback.
- **New prompts** (premium requests): each new conversation, each follow-up user message, and each request the gateway cannot match to a live session. Tool results continue the running turn and do not start a new prompt.
- **Conversation history.** Copilot only accepts history through its own sessions. When a request doesn't continue a session the gateway knows (gateway restart, session older than `--session-ttl`, more than `--max-sessions` conversations, history edited or compacted by the client, different model, system prompt or tools), a new session starts and the earlier messages are given to the model once, as a text transcript.
- **Sessions live in memory**: restarting the gateway loses them (see above). A paused tool call whose result never comes back is discarded after `--session-ttl`.
- **Tool choice:** "must call tool X" is forwarded; "must call any tool" and "no tools" are not enforced.

### `acp` backend

- **Tool calls are emulated.** ACP cannot declare client tools, so their definitions are written into the prompt and the model answers with `<tool_call>` text blocks that the gateway converts. A model may occasionally write a malformed call, which is returned as plain text; very large tool sets make the prompt longer and calls less reliable.
- **Copilot's agent prompt stays in place.** The client's system prompt is given inside the conversation, so behavior can differ from the vendor's API more than with `sdk`.
- **Stateless.** Every request opens a new Copilot session and resends the whole conversation as one prompt: slower on long conversations, no prompt caching, and **every request is a new prompt** (premium requests), including each step of a tool loop.
- **Copilot's own tools** are disabled with `--available-tools=` and any permission request is denied (`--permission`); the effect of the flag depends on the Copilot CLI version.
- **Tool choice:** "no tools" leaves the tool definitions out of the prompt; "must call a tool" and "must call tool X" are only requested in the prompt, not enforced.
- **Token usage** comes from Copilot when it reports it, otherwise it is estimated (characters ÷ 4).
