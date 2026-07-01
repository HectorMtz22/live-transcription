---
description: Brainstorm a new task, write a local spec, and file issue(s) in the tracker
argument-hint: [short description of the task]
---

# /task-init — start a new task

Turn an idea into a local spec and one or more issues in the configured tracker, ready for
`/task-implement`. This is the front half of the harness (see `HARNESS.md`):
`brainstorm → spec → issue(s)`. Issues live in the tracker; specs/plans stay
local and gitignored.

Task description (may be empty — ask if so): **$ARGUMENTS**

## Tracker coordinates

Read these from `.claude/tracker.md` (written by `/harness-setup`):
`tracker`, `mcp_prefix`, `project_code`, `project_id`. If the file is missing,
tell the user to run `/harness-setup` first.

- Project = `project_code` (pass `project_id` to MCP tools that need it).
- Resolve states and labels **by name at runtime** — don't hardcode UUIDs:
  - `<mcp_prefix>__list_states` → pick the state named **"Todo"**.
  - `<mcp_prefix>__list_labels` → map label names to IDs.

## Steps

1. **Brainstorm.** Invoke `superpowers:brainstorming` and design the change with
   the user. Do not skip this even if the task seems small — the spec can be
   short. The brainstorming skill writes the spec to
   `docs/superpowers/specs/YYYY-MM-DD-<topic>-design.md` (local, gitignored) and
   gets the user's approval. Let it run its full flow.

2. **Plan into PR-sized chunks.** If the spec is bigger than one PR, use
   `superpowers:writing-plans` and split it into independent, PR-sized chunks —
   one issue each (this is what lets `/task-implement` run them in parallel). A
   single-PR task is just one issue.

3. **Determine labels per issue:**
   - **Project label** = the sub-project the work touches. If the right project
     label does not exist yet (new sub-project), create it. If the work spans
     projects, ask the user which project the issue belongs to.
   - **Type label** = the conventional-commit type the chunk will land as
     (`feat`, `fix`, `refactor`, `test`, `docs`, `chore`).

4. **Create the issue(s):**
   - `project`: the configured project (`project_code`).
   - `name`: imperative, conventional-commit-style summary, e.g.
     `feat(<project>): add page-range presets`.
   - `state`: the **Todo** state id.
   - `labels`: `[project label id, type label id]`.
   - `description`: problem, chosen approach, code map (files to touch), test
     list, and the local spec filename (`docs/superpowers/specs/...`). Keep it
     tight — the spec is the contract.

5. **Report** each created issue's identifier (e.g. `project_code-12`) and tell
   the user they can run `/task-implement project_code-12 …` next.

## Guardrails

- Specs/plans are **local-only** (`docs/superpowers/` is gitignored). Never
  commit them and never put them anywhere but the issue description as a pointer.
- One issue ≈ one PR. Prefer several small independent issues over one large one
  so implementation can parallelize.
- Don't write production code here — `/task-init` only plans and files issues.
