# Local AI Orchestrator: dual-GPU external memory

Run two local LLMs on two GPUs as one system:

- **GPU A, the primary model**, does the actual work: chat, coding, agent tool calls.
- **GPU B, the memory model**, works in the background. It maintains a persistent project
  memory, rolling session summaries and digests of large tool output, so the primary gets the
  context it needs in far fewer tokens.

```
your client ──► 127.0.0.1:8000  orchestrator ──► Ollama A (GPU A)  primary model
(OpenClaw, ai chat,     ├ injects relevant memory + session summary
 any Ollama client)     ├ trims old history / compresses old tool output (cache-friendly)
                        ├ strips the primary's <memory_flag> hints
                        └ queues finished turns ──► worker ──► Ollama B (GPU B)  memory model
                                                      └► validator ─► Markdown memory + SQLite
```

To your client, the orchestrator looks like a normal Ollama server. Everything is local; nothing
leaves the machine.

**Contents:** 1 Requirements · 2 Plan your setup · 3 Install · 4 Start two pinned Ollama instances ·
5 Configure · 6 Start and verify · 7 Connect a client · 8 Tune for your VRAM · 9 How it works ·
10 CLI and API · 11 Measure · 12 Troubleshooting · 13 Limitations

---

## 1. Requirements

- **Two GPUs**, any combination that Ollama supports. The included launcher script is for
  **Windows + NVIDIA**. Linux and AMD work too, with the manual commands in §4.
- **Ollama**, recent enough for both of your cards. New GPU generations need recent builds.
- **Python 3.11+**.
- Enough **PSU** headroom for both cards at full load, with separate PCIe power cables per card.
  Check both cards' rated board power plus your CPU.
- The second card can sit in an x4-electrical slot. Inference barely uses PCIe once a model is loaded.

One GPU also works (see §4.4), just without the isolation benefits.

## 2. Plan your setup

### 2.1 Which GPU does what

| Role | Give it | Why |
|---|---|---|
| Primary | The GPU with **more VRAM** (if equal, the faster one) | Answer quality and speed come from here |
| Memory | The other GPU | Its work happens in the background, so it can be older or slower |

The memory GPU needs to be good at following instructions and producing JSON, not at deep reasoning.

### 2.2 Sizing models to VRAM

Every model must fit **entirely** on its GPU. With partial CPU offload, performance collapses.
Rough rules for Q4_K_M quantization:

- **Weights** ≈ 0.6 GB per billion parameters (8B ≈ 5 GB, 14B ≈ 9 GB, 32B ≈ 20 GB).
- **KV cache** (the context) with `q8_0` cache type ≈ 0.04–0.1 GB per 1,000 tokens for 7–14B models.
  It is roughly double that with the default f16 cache.
- **Overhead** ≈ 0.5–1 GB. If the GPU also drives your monitors, subtract what the desktop uses
  (check `nvidia-smi` at idle).

Starting points (verify with §6; the measurement wins over this table):

| GPU VRAM | Primary model (num_ctx) | Memory model (num_ctx) |
|---|---|---|
| 4–6 GB | not recommended | `qwen3:4b` (4k–8k) or `qwen3:1.7b` |
| 8 GB | `qwen3:8b` (8k) | `qwen3:8b` (8k) |
| 12 GB | `qwen3:14b` (16k) | `qwen3:8b` (16k) |
| 16 GB | `qwen3:14b` (32k) | `qwen3:14b` (8k) |
| 24 GB | `qwen3:32b` (8k–16k) or `qwen3:14b` (64k) | overkill; consider giving it more work (§13) |

Any Ollama model works; the Qwen3 family is just a consistent example. For the **memory** model,
prefer models that handle structured output well. Smaller memory models produce more rejected
changes; you can see that in `ai memory changes` (§11).

## 3. Install

Pick an install folder (examples use `D:\AI`; any path works). Copy this project so the folder
contains `app\`, `config\`, `prompts\`, `scripts\`, `memory\`.

**Windows:**
```powershell
cd D:\AI
powershell -ExecutionPolicy Bypass -File .\scripts\start-orchestrator.ps1 -Test
```
This creates `.venv`, installs dependencies and runs the test suite.

**Linux:**
```bash
cd ~/ai
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-dev.txt && python -m pytest -q
```

Then edit `config\config.yaml` and set `paths.root` to your install folder. Alternatively, set the
`AI_ROOT` environment variable; `AI_CONFIG` points at an alternative config file.

**Seed memory:** `memory\*.md` ships with example entries describing one specific setup. Edit them
to describe yours, or delete the `.md` files and they'll be recreated empty on first start.

## 4. Start two pinned Ollama instances

The idea: two `ollama serve` processes, each limited to one GPU and listening on its own port, and
both reading the same models folder (weights are stored once).

### 4.1 Windows + NVIDIA (script)

First stop the Ollama tray app and disable its autostart (Task Manager → Startup apps). It runs its
own unpinned server on port 11434, which would grab the port and both GPUs.

```powershell
.\scripts\start-ollama-instances.ps1 -ListGpus           # names, UUIDs, VRAM
.\scripts\start-ollama-instances.ps1 -ModelsDir D:\AI\models
```

How roles are chosen: with exactly two GPUs of different VRAM, the **larger is primary** automatically.
Otherwise pick them by name or UUID:

```powershell
.\scripts\start-ollama-instances.ps1 -PrimaryMatch "4070" -MemoryMatch "3060"
.\scripts\start-ollama-instances.ps1 -PrimaryGpu GPU-1a2b... -MemoryGpu GPU-9f8e...
```

Other options: `-PrimaryPort/-MemoryPort` (11434/11435), `-PrimaryContext/-MemoryContext` (default
context per instance), `-KvCacheType` (`q8_0`; use `f16` if a model misbehaves), `-LogDir`,
`-Force` (stop running Ollama processes first), `-Status`, `-Stop`.

Each instance gets: `CUDA_VISIBLE_DEVICES=<GPU UUID>` (UUIDs, because CUDA indices don't reliably
follow slot order), flash attention on, `q8_0` KV cache, `OLLAMA_NUM_PARALLEL=1` (each parallel slot
costs a full KV cache) and `OLLAMA_MAX_LOADED_MODELS=1`.

### 4.2 Linux + NVIDIA (manual)

```bash
nvidia-smi -L                                   # get the UUIDs
common="OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 OLLAMA_NUM_PARALLEL=1 OLLAMA_MAX_LOADED_MODELS=1 OLLAMA_MODELS=/srv/ai/models"
env $common CUDA_VISIBLE_DEVICES=GPU-aaaa OLLAMA_HOST=127.0.0.1:11434 OLLAMA_CONTEXT_LENGTH=16384 ollama serve &
env $common CUDA_VISIBLE_DEVICES=GPU-bbbb OLLAMA_HOST=127.0.0.1:11435 OLLAMA_CONTEXT_LENGTH=8192  ollama serve &
```

To run them permanently, create two systemd services (or override the stock `ollama.service` for
instance A and copy it as `ollama-memory.service` for B), each with those values as `Environment=`
lines. Stop the stock service first if it isn't one of the two, since it binds 11434 unpinned.

### 4.3 AMD

Same idea, but Ollama selects AMD GPUs with `ROCR_VISIBLE_DEVICES` / `HIP_VISIBLE_DEVICES` instead of
`CUDA_VISIBLE_DEVICES`; see Ollama's GPU documentation for your platform. The Windows script is
NVIDIA-only, so start the instances manually as in §4.2. Mixing an NVIDIA and an AMD card also works
in principle, since each instance only needs its own GPU, but it is untested here.

### 4.4 One GPU only

Point both `ollama.primary.base_url` and `ollama.memory.base_url` at the same instance and set
`OLLAMA_MAX_LOADED_MODELS=2` on it. Both models must fit in VRAM together, or Ollama will swap them
in and out on every background task, which is very slow. A small memory model (`qwen3:1.7b`/`4b`)
helps. `/health` will warn that there is no isolation.

### 4.5 Pull models and verify pinning

Pull once; both instances share the models folder:
```powershell
$env:OLLAMA_HOST="127.0.0.1:11434"; ollama pull qwen3:14b; ollama pull qwen3:8b; Remove-Item Env:OLLAMA_HOST
```
(Linux: `OLLAMA_HOST=127.0.0.1:11434 ollama pull ...`)

Load each model on its own instance, then check:
```powershell
$env:OLLAMA_HOST="127.0.0.1:11434"; ollama run qwen3:14b "hi"
$env:OLLAMA_HOST="127.0.0.1:11435"; ollama run qwen3:8b "hi"; Remove-Item Env:OLLAMA_HOST
.\scripts\start-ollama-instances.ps1 -Status        # Linux: OLLAMA_HOST=127.0.0.1:1143x ollama ps ; nvidia-smi
```
**Both models must show 100% GPU, and each on the GPU you intended.** If not, lower that instance's
context or pick a smaller model or quant (§2.2).

## 5. Configure

`config\config.yaml` controls everything. The settings you must check:

| Setting | Set it to |
|---|---|
| `paths.root` | your install folder |
| `ollama.primary.model` / `num_ctx` | your primary model and the context that fits (§2.2) |
| `ollama.memory.model` / `num_ctx` | your memory model and its context |
| `ollama.*.base_url` | your two instances' ports |
| `ollama.memory.think` | `false` for models with a thinking mode (faster, cleaner JSON); remove for others |
| `memory.max_context_tokens` | ~15% of the primary's `num_ctx` (2,500 at 16k) |

Everything else has working defaults; §8 explains when to change them.

## 6. Start and verify

```powershell
.\scripts\start-orchestrator.ps1        # Linux: python -m app.main
.\ai status                             # Linux: python -m app.cli status
.\ai chat -v
```

`ai status` shows both instances, whether each loaded model is 100% on GPU, the background queue and
memory-file health. `ai chat -v` shows which memory entries were injected for each message.

The orchestrator listens on `127.0.0.1:8000` and has no authentication. Don't expose it to your LAN.

## 7. Connect a client

### 7.1 Any Ollama-native client

Point the client's Ollama URL at `http://127.0.0.1:8000` instead of `:11434`. It will list the primary
instance's models and chat as usual, with memory added automatically.

Memory only works on Ollama's **native** `/api/chat` endpoint. Clients that use the OpenAI-compatible
`/v1` endpoints are passed through **without** memory. If a client offers both, choose "Ollama".

Conversation ids: native requests don't carry one, so the orchestrator derives one from the model and
the first user message. Clients that can send an `X-Conversation-Id` header get exact session tracking.

### 7.2 OpenClaw

Merge `examples\openclaw-provider.json5` into `~/.openclaw/openclaw.json`, set your model id and
`contextWindow` (equal to `ollama.primary.num_ctx`), then run `openclaw gateway restart`. Keep
`api: "ollama"`. Check the snippet against your OpenClaw version's provider docs.

### 7.3 Clients inside WSL

To reach `127.0.0.1:8000` on Windows from WSL, copy `examples\wslconfig.example` to
`%UserProfile%\.wslconfig` (mirrored networking, plus a RAM cap so WSL doesn't take half your memory)
and run `wsl --shutdown`.

### 7.4 Your own code

```
POST http://127.0.0.1:8000/chat   {"conversation_id": "optional", "message": "..."}
```
This returns the answer, the conversation id and which memory entries were used. It keeps the last
`conversation.recent_turns` exchanges verbatim and lets memory and the session summary carry the rest.

## 8. Tune for your VRAM

All context-saving features are on by default and scale with your settings.

| Feature | Settings | Rule of thumb |
|---|---|---|
| Memory injection | `memory.max_context_tokens` | ~15% of primary `num_ctx`. Clients with large system prompts or many tools (agents) need more free room. |
| Stepped history trimming | `proxy.trim_trigger_user_turns` / `trim_keep_user_turns` | Small context (8–16k): 10 / 4. Large context (32k+): 20 / 8, or `0` to disable. |
| Tool-output compression | `compression.min_result_tokens`, `keep_recent_user_turns`, `digest_max_tokens` | Defaults suit most setups. `keep_recent` must be smaller than `trim_keep`. |
| Session summaries | `session.summary_max_tokens` | 400; raise it if long sessions lose details. |
| Memory flags | `flags.enabled` | On; costs the primary ~20–50 tokens only on turns that flag something. |

If `ai status` ever shows less than 100% GPU, **lower `num_ctx` first**. Keep the client's own
context-window setting (e.g. OpenClaw `contextWindow`) equal to `ollama.primary.num_ctx`.

## 9. How it works

### 9.1 Per request (`/api/chat`)

1. The turn plan is fixed once per user turn and reused for every tool-call round trip: the trim
   point, which old tool results become digests, and the memory block.
2. Old history is trimmed in steps, and old large tool results are swapped for digests (§9.4).
3. Relevant memory and the session summary are wrapped as `<PROJECT_MEMORY>…</PROJECT_MEMORY>` in
   the **latest** user message, so earlier history stays byte-identical and Ollama's prompt cache
   keeps hitting.
4. `prompts\primary_system.txt` (and the flag instruction) is appended to the client's system prompt,
   and `num_ctx` is set if the client didn't.
5. The reply streams back with tool calls intact and `<memory_flag>` tags removed.
6. When the reply has no pending tool calls, the turn is over. It is written to raw JSONL, and the
   background queue gets, in priority order: the session-summary update, digests for large tool
   results, and memory extraction (if the turn was flagged or matched the trigger heuristics).

### 9.2 Persistent memory

`memory\*.md` holds one file per category: STATE, OBJECTIVES, CONSTRAINTS, DECISIONS, LESSONS,
DISCOVERIES, ENVIRONMENT. Each entry has a stable id:

```markdown
### L-003 — HTTP 404 debugging
<!-- meta: status=active created=... updated=... source=... -->
A successful HTTP connection does not prove the requested endpoint exists.
```

- You can edit entry text by hand; keep the `###` heading and the meta line. The store refuses to
  rewrite a file with sections it doesn't recognise rather than lose your text. `ai memory validate`
  tells you what is wrong.
- Every write is: backup to `memory\history\` → re-parse and verify → atomic replace.
- The memory model only **proposes** changes, as JSON constrained by Ollama's structured output. A
  validator treats that output as untrusted. It rejects bad schemas, path-like titles, prompt-injection
  phrasing, dangerous commands and anything that looks like a secret, and it dedupes against existing
  memory. Every proposal, including rejections and the reason, is logged in SQLite.
- Raw conversation history (`conversations\*.jsonl`) is never deleted, so memory can always be rebuilt
  from it (`ai memory rebuild --replay`).

### 9.3 Memory flags

The primary is told to end a reply with up to three lines like
`<memory_flag category="lesson">MCP route is /mcp, not /sse</memory_flag>` when something durable
happened. They are stripped from the stream (even when split across chunks), always trigger
extraction, and reach the memory model as hints to verify. Keyword heuristics (English + Portuguese)
remain as a fallback for turns the primary forgets to flag.

### 9.4 Session summaries, trimming and compression

- **Session summary:** after every turn the memory model updates a rolling summary of that
  conversation. Summaries only move forward (a late retry can't overwrite a newer one).
- **Stepped trimming:** past `trim_trigger_user_turns`, the oldest turns are cut back to about
  `trim_keep_user_turns`, and the summary is injected as `SESSION SO FAR`. The cut point moves in steps
  (with 10/4: turns 11, 16, 22, …), so the prompt prefix stays identical in between and the cache hits.
  Turns the summary doesn't cover yet are never cut.
- **Tool-output compression:** large tool results from older turns are replaced by digests labelled
  `[compressed tool output: original ~N tokens; re-run the tool if exact output is needed]`. The
  current turn and the last `keep_recent_user_turns` are never touched. A digest is only used if it
  exists, respected its length limit and saves at least half. The boundary moves on the same turns as
  the trim point, and the set of compressed results is frozen in between.
- Cuts only happen at user-message boundaries, so tool-call chains are never split. The client keeps its
  full history on its side; only what is sent to the model shrinks.

## 10. CLI and API

`ai` is `ai.cmd` on Windows and `python -m app.cli` on Linux.

```
ai serve | chat [-v] | status | metrics
ai memory show [category] | search "text" | context "prompt" | changes | tasks | sessions
ai memory validate | backup | rebuild [--replay [--reset]] | consolidate
```

`ai memory context "prompt"` shows exactly what would be injected for a prompt. `rebuild` alone
rebuilds SQLite from the Markdown; `--replay` re-queues all raw history through the memory model;
`--reset` starts from empty memory first. A snapshot to `backups\` is always taken beforehand. Stop the
server (or use the API) before CLI rebuilds.

| Method | Path | |
|---|---|---|
| POST | `/chat` | spec API (§7.4) |
| GET | `/health` | both instances, GPU %, worker, queue, memory-file health, warnings |
| GET | `/metrics` | per-request tokens, prefill/generation time, TTFT, tok/s, trimming/compression savings |
| GET | `/memory/state`, `/memory/search?q=`, `/memory/context?q=` | inspect memory |
| GET | `/memory/changes`, `/memory/tasks`, `/memory/sessions` | audit trail, queue, summaries |
| POST | `/memory/rebuild`, `/memory/backup` | maintenance |
| POST | `/memory/consolidate` | not implemented yet (501) |
| * | `/api/*`, `/v1/*`, `/` | passthrough to the primary instance (memory only on `/api/chat`) |

Logs: `logs\orchestrator.log`, `primary.log`, `memory.log` (JSON lines), plus each Ollama
instance's log when started by the script.

## 11. Measure

Every feature can be switched off in `config.yaml`. Compare runs of the same kind of session with a
feature on and off in `/metrics`: `prompt_tokens` (tokens Ollama actually had to process; cache hits
lower it), `prefill_time`, `time_to_first_token`, `memory_tokens`, `user_turns_dropped`,
`tool_tokens_saved`, `memory_flags`.

Quality signals worth checking weekly:
- `ai memory changes`: a high rejection rate means the memory model is too small or confused.
- `ai memory sessions`: whether summaries keep the specifics you care about.
- Whether the primary re-runs tools it already ran. If so, digests are dropping something; raise
  `digest_max_tokens` or add the tool to `never_compress_tools`.

## 12. Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| Port 11434 already in use | The Ollama tray app or the stock service is running; stop it, or use `-Force` |
| A model is under 100% GPU | Context too large for the VRAM: lower `num_ctx`, use `q8_0` KV cache or a smaller model |
| Both models on the same GPU | Instances not pinned. Check `-Status` / `nvidia-smi`, and pin by UUID |
| Client works but no memory appears | Client uses `/v1` (OpenAI mode); switch it to the Ollama provider |
| Client in WSL can't reach :8000 | Mirrored networking not enabled (§7.3) |
| Memory changes are mostly rejected | Memory model too small or thinking mode on; set `think: false` or try a larger model |
| `memory file errors` in status | A hand edit broke the format; run `ai memory validate` and fix the reported line |
| History never gets trimmed | The summary isn't keeping up (check `ai memory tasks`), or `trim_trigger_user_turns: 0` |
| Client compacts or refuses at small context | Its own context precheck; raise `num_ctx` and the client's context window together if VRAM allows |

## 13. Limitations and ideas

- **Not built yet:** embedding-based retrieval (keyword search is used; `MemoryRetriever` is the
  interface to implement), memory consolidation, multi-project namespaces (the schema already has
  `project_id`), and an evaluation harness that replays sessions with features on and off.
- A large second GPU could take on more: embeddings, a bigger memory model, or splitting one large
  primary model across both cards instead. That trades the memory system for raw model size.
- Tested with fake Ollama instances (86 tests: validator, atomic writes, streaming, flag stripping,
  trimming and compression cache stability, retries…). Real-GPU behaviour (pinning, VRAM fit, a given
  model's JSON quality) can only be verified on your machine (§4.5, §11).
