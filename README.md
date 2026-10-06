# Local AI Orchestrator — dual-GPU external memory

Implementation of `local_ai_dual_gpu_memory_build_spec.md`, **Phases 1 and 2**, adapted to:

| | Spec assumed | This build |
|---|---|---|
| Primary GPU | RTX 5070 Ti 16 GB | **RTX 5070 12 GB** |
| Memory GPU | RTX 2070 SUPER 8 GB | RTX 2070 SUPER 8 GB |
| System RAM | — | 16 GB DDR4 |
| Front end | orchestrator's own API | **OpenClaw (WSL)** via an Ollama-compatible proxy, plus the spec's `/chat` API |

```
OpenClaw (WSL) ──► 127.0.0.1:8000  orchestrator  ──► 11434  Ollama A  RTX 5070       qwen3:14b
   ai chat     ──►   ├ injects <PROJECT_MEMORY>
                     ├ logs raw JSONL
                     └ queues finished turns ──► worker ──► 11435  Ollama B  RTX 2070 SUPER  qwen3:8b
                                                    └► validator ─► Markdown (canonical) + SQLite (index)
```

Copy this folder so that it **is** `G:\AI` (so `G:\AI\app`, `G:\AI\config`, `G:\AI\memory`, ...).
Everything else (paths, models, ports, budgets) is in `config\config.yaml`.

---

## 1. Hardware checks before installing the 2070 SUPER

- **PSU**: RTX 5070 ≈ 250 W + RTX 2070 SUPER ≈ 215 W, plus CPU. Aim for a quality 850 W unit, with separate PCIe power cables per card (no daisy-chained pigtails).
- **Slot**: the second card can sit in an x4-electrical slot; inference barely touches PCIe once a model is loaded.
- **Driver**: one current NVIDIA driver covers both Blackwell and Turing. After installing, `nvidia-smi -L` must list both cards.
- **Ollama**: the 5070 needs a recent Ollama build (Blackwell/CUDA 12.8 support).

## 2. Install

Requires Python 3.11+ on Windows.

```powershell
cd G:\AI
powershell -ExecutionPolicy Bypass -File .\scripts\start-orchestrator.ps1 -Test   # creates .venv, installs, runs tests
```

## 3. Start the two pinned Ollama instances

The Ollama tray app runs its own **unpinned** server on 11434. Quit it and disable its autostart
(Task Manager → Startup apps), otherwise it grabs the port and both GPUs.

```powershell
.\scripts\start-ollama-instances.ps1 -ListGpus          # shows names + UUIDs
.\scripts\start-ollama-instances.ps1 -ModelsDir G:\AI\models   # or wherever your models already are
```

The script auto-detects the GPUs by name ("5070" / "2070"), pins each instance with
`CUDA_VISIBLE_DEVICES=<GPU UUID>` (indices are unreliable), and sets per instance:
flash attention on, `OLLAMA_KV_CACHE_TYPE=q8_0`, `OLLAMA_NUM_PARALLEL=1`
(each parallel slot costs a full KV cache), and a default context length. Both instances share one
models folder, so weights are stored once.

Pull models once (either instance works, the folder is shared):

```powershell
$env:OLLAMA_HOST="127.0.0.1:11434"; ollama pull qwen3:14b; ollama pull qwen3:8b; Remove-Item Env:OLLAMA_HOST
```

Load one model on each and check pinning. **Both must show 100% GPU:**

```powershell
$env:OLLAMA_HOST="127.0.0.1:11434"; ollama run qwen3:14b "hi" --verbose
$env:OLLAMA_HOST="127.0.0.1:11435"; ollama run qwen3:8b "hi" --verbose; Remove-Item Env:OLLAMA_HOST
.\scripts\start-ollama-instances.ps1 -Status
```

Until the 2070 SUPER is installed, point `ollama.memory.base_url` at `http://127.0.0.1:11434` and
consider `qwen3:4b` as the memory model. `/health` will then warn that there is no GPU isolation.

## 4. Start the orchestrator

```powershell
.\scripts\start-orchestrator.ps1
.\ai status          # both instances, GPU residency, queue, memory file health
.\ai chat -v         # talk through /chat; -v shows which memory entries were injected
```

## 5. Connect OpenClaw (WSL)

1. Copy `examples\wslconfig.example` to `%UserProfile%\.wslconfig` and run `wsl --shutdown`.
   Mirrored networking lets WSL reach `127.0.0.1:8000` on Windows, so the orchestrator never has to
   listen on a LAN-facing address. This also caps WSL's RAM, which matters with 16 GB total.
2. Merge `examples\openclaw-provider.json5` into `~/.openclaw/openclaw.json` in the distro and run
   `openclaw gateway restart`. Check it against your OpenClaw version's Ollama provider docs.
   **Keep `api: "ollama"`**: memory is only injected on the native `/api/chat` route. `/v1/*` is
   passed through untouched.
3. From WSL: `curl http://127.0.0.1:8000/api/tags` should list the primary instance's models.

What the proxy does to each `/api/chat` request:

- Trims old history **in steps** (§6b) and puts the session summary in its place.
- Wraps the **latest** user message as `<PROJECT_MEMORY>…</PROJECT_MEMORY>` + `<USER_REQUEST>…</USER_REQUEST>`.
  Earlier history is left byte-identical, so Ollama's prompt cache keeps hitting on the history prefix.
  The same memory block is reused for every tool-call round trip of one turn, for the same reason.
- Appends `prompts\primary_system.txt` to OpenClaw's system prompt (`proxy.append_system_prompt`).
- Sets `options.num_ctx` from config if OpenClaw didn't.
- Streams the response through with tool calls intact, removing the primary's `<memory_flag>` tags (§6a).
  When flags are disabled, chunks are relayed byte-for-byte.
- When the model's reply has **no** pending tool calls, the turn is over. It is written to JSONL
  (user message, tool calls, tool results, answer, flags) and two tasks are queued: a session-summary
  update (high priority) and memory extraction (if flagged or triggered). Intermediate tool steps are
  not queued separately; flags raised during them are carried to the end of the turn.

Conversation ids: OpenClaw doesn't send one, so the proxy derives `oc-<hash>` from the model and the
first user message. Send an `X-Conversation-Id` header if you have a better id.

OpenClaw's own background calls (for example compaction summaries) also pass through `/api/chat`.
They get memory injected and are logged like any other turn; the memory model usually returns no changes for them.

## 6a. Memory flags (primary → helper)

The primary is told to end a reply with up to three lines like

```
<memory_flag category="lesson">MCP route is /mcp, not /sse</memory_flag>
```

when the turn produced something durable (`prompts\primary_flags.txt`). The proxy strips them from
the stream before OpenClaw sees them. Tags split across chunks at any character are handled, and only
the few characters that could still be the start of a tag are held back. Then:

- A flagged turn **always** goes to the memory model, even if the regex heuristics would skip it.
- The flags go to the memory model as *hints to verify*, not facts. The extractor still writes the
  entry itself and the validator still applies every rule, because flag text is model output.
- The heuristics stay on as a fallback for turns the primary forgets to flag.

This costs ~20–50 generated tokens on turns that flag something and nothing on the rest. Disable it
with `flags.enabled: false`. To judge whether flags help, look at `memory_flags` per request in
`/metrics` and at what lands in `ai memory changes`.

## 6b. Session summaries and stepped trimming

After every finished turn the memory model updates a rolling summary of that conversation: goal,
what was tried, what worked or failed, specifics, next steps. It's capped at `session.summary_max_tokens`.
Summaries are stored in SQLite, only move forward (a late retry can never overwrite a newer one),
and jump the queue ahead of extraction so they are ready for your next prompt.
Read them with `ai memory sessions`.

Once OpenClaw's history exceeds `proxy.trim_trigger_user_turns` (10), the proxy drops the oldest
user turns in steps back to about `trim_keep_user_turns` (4). The summary goes into the memory block as
`SESSION SO FAR`, taking at most half the memory budget so durable memory always keeps room.

- **Why steps:** with trigger 10 / keep 4 the cut point is fixed for 5–6 turns at a time
  (turns 11–15 drop 6, 16–21 drop 12, …). In between cuts the prompt prefix is identical and Ollama's
  cache hits; only the turn where the cut moves pays a full prefill.
- **Safety:** with `trim_requires_summary: true`, turns the summary hasn't covered yet are never
  cut. If the helper lags or is down, history simply isn't trimmed until it catches up.
- **Stable within a turn:** the trim point, memory block and summary are fixed at the first request
  of a turn and reused for every tool-call round trip, even if a summary update lands mid-turn.
- Cuts only happen at user-message boundaries, so tool-call chains are never split.
- OpenClaw keeps its full history on its side; only what is sent to the model is trimmed.

Set `trim_trigger_user_turns: 0` to turn trimming off. Compare `prompt_tokens`, `prefill_time` and
`user_turns_dropped` in `/metrics` with it on and off.

## 6c. Tool-result compression

In agent sessions the context fills with tool output (file dumps, logs, command output), and all of it
is re-sent on every request. After each turn, the memory model writes a short digest of every tool result
over `compression.min_result_tokens` (400). Digests are stored by content hash, so the same output
is never digested twice. When the primary is called, results from **older** turns are sent as:

```
[compressed tool output: original ~1840 tokens; re-run the tool if exact output is needed]
- contents of app/config.py, 180 lines; pydantic models PathsConfig, OllamaEndpoint, ...
```

Accuracy safeguards:
- Results from the current turn and the last `keep_recent_user_turns` (2) are never touched.
- A result is only replaced once its digest exists. If the helper is behind, it stays verbatim.
- A digest is **unusable** if the helper overshot the length limit (nothing gets cut off arbitrarily)
  or if it doesn't save at least half. Those results stay verbatim.
- The digest prompt keeps error messages, paths, names, versions, ports and IDs verbatim, and the
  marker tells the model it can re-run the tool if it needs exact output.
- `never_compress_tools` keeps specific tools' output verbatim, e.g. if you notice the model needing
  old file contents exactly.

Cache safeguards: the compression boundary moves on the **same turns as the trim point** (before trimming
starts, every `step_turns`), so the cache is invalidated once per step, not twice. The set of results
replaced is frozen when the boundary moves, so a digest finishing mid-window doesn't change the prefix.
With the defaults, the boundary moves at turns 8, 11, 16, 22, …

Measure it: `tool_results_compressed` and `tool_tokens_saved` per request in `/metrics`, and the
`tool_digests` totals (usable vs not). If the model starts re-running tools it already ran, digests
are dropping something it needs. Raise `digest_max_tokens` or exclude that tool.

## 6. Fitting a 12 GB card

qwen3:14b Q4_K_M is roughly 9 GB of weights. With a q8_0 KV cache, 16k context adds roughly
1.3 GB, which leaves little headroom. Hence:

- `ollama.primary.num_ctx: 16384`, and OpenClaw's `contextWindow` set to the same value.
- `memory.max_context_tokens: 2500` (the spec's 6000 assumed 16 GB). OpenClaw's own system prompt
  and tool schemas already take a few thousand tokens.
- Stepped trimming (§6b) keeps OpenClaw's history from filling the window in long sessions.
- If `ai status` reports anything under 100% GPU, lower `num_ctx` before anything else.
- There is an OpenClaw issue report (#65465) of its context precheck using a 16384-token reserve
  regardless of config. If OpenClaw compacts constantly or refuses requests at a 16k window, that is
  the first thing to check.

To compare with and without memory (spec §35), run the same session with `proxy.inject_memory`
on and off and compare `prompt_tokens`, `prefill_time` and `time_to_first_token` in `GET /metrics`.

## 7. Memory

`memory\*.md` is canonical and human-editable. Entries look like:

```markdown
### L-003 — HTTP 404 debugging
<!-- meta: status=active created=... updated=... source=oc-... -->
A successful HTTP connection does not prove the requested endpoint exists.
```

You can edit entry text freely; keep the `###` heading and the meta line. The store **refuses to
rewrite** a file containing sections it doesn't manage, rather than silently dropping your text.
`ai memory validate` tells you what is wrong. Every write is: backup to `memory\history\` → render →
re-parse and verify → write temp → fsync → atomic replace.

The seed files describe this setup (hardware, constraints, OpenClaw decision). Edit them to taste.

How updates flow:

1. A finished turn passes the trigger heuristics (English + Portuguese: corrections, errors,
   successes, config changes, decisions, tool use, …). Otherwise it is recorded as `skipped`.
   Set `memory.trigger_mode: always` to send everything.
2. The task is persisted in SQLite (`memory_tasks`), so queued work survives restarts. It is
   processed one at a time and retried with exponential backoff up to `max_attempts`.
3. qwen3:8b is called with Ollama **structured outputs** (the change schema is passed as `format`)
   and thinking off, so it is constrained to valid JSON.
4. The validator treats the output as hostile. It rejects: invalid JSON (no code-fence stripping or
   repair), unknown fields, categories and operations; empty, oversized or low-confidence entries;
   path-like titles and path traversal; prompt-injection phrasing and delimiter tags; dangerous
   commands (`rm -rf`, `iex`, encoded PowerShell, …; ordinary commands like `nvidia-smi -L` are
   allowed as facts); and anything that looks like a secret. It also dedupes against existing memory:
   same title → update, near-identical content → reject, update of a missing entry → add.
5. Accepted changes go to Markdown and are mirrored to SQLite. Every proposal, including rejected
   ones and the reason, is kept in `memory_changes`.

## 8. CLI and API

```
ai serve | chat | status | metrics
ai memory show [category] | search "MCP 404" | context "prompt" | changes | tasks | sessions
ai memory validate | backup | rebuild [--replay [--reset]] | consolidate
```

`rebuild` alone rebuilds SQLite from Markdown. `--replay` re-queues every turn from the raw JSONL,
and `--reset` starts from empty memory files first. All of them take a snapshot to
`backups\YYYY-MM-DD_HH-MM-SS\` beforehand. Stop the server first, or use the API endpoint, so the
CLI and the server don't write at the same moment.

| Method | Path | |
|---|---|---|
| POST | `/chat` | `{"conversation_id"?, "message"}` → `{"conversation_id","response","memory_update_queued",…}` |
| GET | `/health` | both instances, GPU %, worker, queue, memory-file health, isolation warnings |
| GET | `/memory/state` | STATE entries + Markdown |
| GET | `/memory/search?q=` | keyword search |
| GET | `/memory/context?q=` | exact block that would be injected, with token estimate |
| GET | `/memory/changes`, `/memory/tasks` | audit trail and queue |
| GET | `/memory/sessions` | rolling session summaries |
| POST | `/memory/rebuild` | `{"replay": false, "reset": false}` |
| POST | `/memory/backup` | snapshot |
| POST | `/memory/consolidate` | 501, Phase 4 |
| GET | `/metrics` | tokens, prefill/generation time, TTFT, tok/s, memory update time |
| * | `/api/*`, `/v1/*`, `/` | Ollama passthrough to the primary instance (only `/api/chat` gets memory) |

Logs are JSON lines in `logs\orchestrator.log`, `primary.log` and `memory.log`. Ollama's own logs
are in `logs\ollama-primary.log` and `logs\ollama-memory.log`.

## 9. Differences from the spec

- **Defaults tuned for 12 GB**: see §6 above.
- **Added**: primary-model memory flags, session summaries, stepped trimming and tool-result compression (§6a–§6c); the Ollama-compatible proxy for OpenClaw; `/memory/context`, `/memory/changes`,
  `/memory/tasks`, `/memory/backup`; a persistent task table; secret rejection; Portuguese trigger
  keywords; `conversation.recent_turns` (the last 2 exchanges go verbatim with `/chat`, so
  follow-ups like "and the other one?" still work before memory catches up).
- **Where memory goes**: into the latest user message, not the system prompt, to preserve prompt
  caching. The injection-resistance instructions are in the system prompt.
- **Markdown format**: entries carry stable ids, so updates are deterministic. The spec's free-form
  examples (e.g. `## Current Environment` in STATE.md) are expressed as entries.
- **Tokens** are estimated at ~3.5 characters per token. This is conservative, so the budget errs small.
- **Not built yet**: Phase 3 (embeddings/vector retrieval: `MemoryRetriever` is the interface to
  implement), Phase 4 (consolidation, automatic conflict resolution beyond update/deactivate,
  context compression), and multi-project namespaces. The schema already has `project_id` columns
  for those.

## 10. Tests

`.\scripts\start-orchestrator.ps1 -Test` runs 86 tests against fake Ollama instances: validator
rejections, atomic writes and recovery, JSONL ordering/Unicode/1 MB messages, budget and priority,
client timeout/connection/malformed handling, worker success/retry/failure/parse errors, proxy
streaming and the tool-call loop, flag stripping split at every character, stepped trimming and
prompt-prefix stability, summary priority and staleness, compression boundary schedule, frozen digest
sets, unusable-digest fallback, and an end-to-end check that a fact learned in one turn is
injected into a later one. Real-GPU behaviour (pinning, VRAM fit, qwen3:8b's JSON quality) can only be
verified on your machine.
