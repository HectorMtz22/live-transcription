---
description: Choose the issue tracker (default Plane) and write .claude/tracker.md
argument-hint: (none; interactive)
---

# /harness-setup — configure the tracker

One-time, **offline** setup. Asks which tracker the harness should use and writes
`.claude/tracker.md`. Makes no network/MCP calls — run `/harness-bootstrap` after
this to provision the live tracker.

## Steps

1. **Ask which tracker** (default **Plane**):
   `[Plane]` / `Linear` / `GitHub Issues` / `other`.

2. **Map the choice** to defaults:
   | Tracker | `mcp_prefix` | `has_cycles` |
   |---|---|---|
   | Plane | `mcp__plane` | `true` |
   | Linear | `mcp__linear` (or the connected server's prefix) | `true` |
   | GitHub Issues | `gh` CLI / MCP prefix | `false` (uses milestones) |
   | other | ask the user | ask the user |

3. **Ask the tracker-specific fields:**
   - `project_code` — short identifier used in issue ids (e.g. `PROJ`).
   - `project_id` — workspace/project id or slug, if the MCP server needs one
     (Plane does; confirm with the user or look it up if already authenticated).
   - Confirm `cycle_length` (`1w`) and `cycle_anchor` (`monday`); accept defaults
     unless the user overrides.

4. **Write `.claude/tracker.md`** with the chosen values (same key set as the
   committed example). Overwrite if it exists.

5. **Report** the written values and tell the user to run `/harness-bootstrap`
   next to create the project, states, labels, and cycles.

## Guardrails

- No live/MCP calls here — this command only writes the local config file.
- Keep the exact key set so `/harness-bootstrap`, `/task-init`, and
  `/task-implement` can read it.
