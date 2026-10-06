# ENVIRONMENT

<!-- Managed by the local AI orchestrator. You may edit entry text by hand; keep the ### headings and meta lines intact. -->

## Active

### E-001 — Primary GPU
<!-- meta: status=active created=2026-09-30T17:06:47+00:00 updated=2026-09-30T17:06:47+00:00 source=seed -->
NVIDIA RTX 5070 (12 GB), Ollama instance A on 127.0.0.1:11434.

### E-002 — Memory GPU
<!-- meta: status=active created=2026-09-30T17:06:47+00:00 updated=2026-09-30T17:06:47+00:00 source=seed -->
NVIDIA RTX 2070 SUPER (8 GB), Ollama instance B on 127.0.0.1:11435.

### E-003 — System RAM
<!-- meta: status=active created=2026-09-30T17:06:47+00:00 updated=2026-09-30T17:06:47+00:00 source=seed -->
16 GB DDR4. Models must stay fully GPU-resident; CPU spill is not viable.

### E-004 — Frontend
<!-- meta: status=active created=2026-09-30T17:06:47+00:00 updated=2026-09-30T17:06:47+00:00 source=seed -->
OpenClaw gateway running in a WSL distro, using the orchestrator as its Ollama provider. OpenClaw's own memory is disabled.

### E-005 — Models
<!-- meta: status=active created=2026-09-30T17:06:47+00:00 updated=2026-09-30T17:06:47+00:00 source=seed -->
Primary model qwen3:14b; memory model qwen3:8b with thinking disabled.

## Superseded
