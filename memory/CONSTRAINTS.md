# CONSTRAINTS

<!-- Managed by the local AI orchestrator. You may edit entry text by hand; keep the ### headings and meta lines intact. -->

## Active

### C-001 — No cloud inference
<!-- meta: status=active created=2026-09-30T17:06:47+00:00 updated=2026-09-30T17:06:47+00:00 source=seed -->
All inference runs locally through Ollama; no cloud LLM APIs.

### C-002 — Data lives on G:
<!-- meta: status=active created=2026-09-30T17:06:47+00:00 updated=2026-09-30T17:06:47+00:00 source=seed -->
Model weights and AI data live under G:\AI. Do not assume C: has large free space.

### C-003 — Preserve raw history
<!-- meta: status=active created=2026-09-30T17:06:47+00:00 updated=2026-09-30T17:06:47+00:00 source=seed -->
Raw conversation history in conversations\*.jsonl is never deleted; memory is derived from it.

### C-004 — Memory model has no authority
<!-- meta: status=active created=2026-09-30T17:06:47+00:00 updated=2026-09-30T17:06:47+00:00 source=seed -->
The memory model only proposes structured changes; the orchestrator validates and applies them.

## Inactive
