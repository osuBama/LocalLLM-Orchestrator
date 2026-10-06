# DECISIONS

<!-- Managed by the local AI orchestrator. You may edit entry text by hand; keep the ### headings and meta lines intact. -->

## Active

### D-001 — External memory over full-history replay
<!-- meta: status=active created=2026-09-30T17:06:47+00:00 updated=2026-09-30T17:06:47+00:00 source=seed -->
Use maintained external memory instead of resending the full conversation, to reduce context size and repeated prefill. Reason: 12 GB VRAM limits the usable context window.

### D-002 — OpenClaw via Ollama-compatible proxy
<!-- meta: status=active created=2026-09-30T17:06:47+00:00 updated=2026-09-30T17:06:47+00:00 source=seed -->
OpenClaw points its Ollama baseUrl at the orchestrator (127.0.0.1:8000, api: ollama) so memory is injected transparently for every chat.

## Superseded
