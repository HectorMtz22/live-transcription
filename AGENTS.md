# Agent workflow for this repo

This file captures how feature work is done in this repo by automated coding
agents (Claude Code with the `superpowers` plugin and an MCP server for your
issue tracker). It's a recipe, not a rule. Tracker coordinates live in
`.claude/tracker.md` — run `/harness-setup` to create it. **Always use
superpowers** — invoke the named skill at each stage. See [`HARNESS.md`](HARNESS.md)
for the full TDD detail.

Two slash commands wrap the loop: **`/task-init`** (front half) and
**`/task-implement`** (back half).

## TL;DR

1. **`/task-init`** runs `superpowers:brainstorming` → a design at
   `docs/superpowers/specs/YYYY-MM-DD-<topic>-design.md` (gitignored, local-only)
   → files **issue(s)** in the configured project (`project_code`) (state `Todo`, project +
   type labels).
2. **`/task-implement [project_code-…]`** picks up the issue(s) and, for each,
   moves it to `In Progress` and **creates a worktree** under `.worktrees/<topic>/`
   on a new branch (gitignored) via `superpowers:using-git-worktrees`.
3. **Dispatch a subagent** to implement inside that worktree, following TDD.
   Multiple issues → `superpowers:dispatching-parallel-agents`, one agent per
   worktree, in a single message.
4. **Run `superpowers:requesting-code-review`** on each branch.
5. **Report findings, ask before fixing.**
6. **Open a PR automatically** once the branch is verified, green, committed, and
   clean (no Important+ findings remain), then move the issue to **In Review**
   (and to **Done** when the PR merges). Only the fix decision in step 5 waits
   for the user.

## Why worktrees

Subagents touch a lot of files. Doing implementation in a separate worktree on a
feature branch keeps the main checkout clean and lets the parent agent inspect
the diff without context pollution.

When two or more independent tasks are dispatched in parallel, each gets its own
worktree, branch, and PR, so the diffs never tangle. Worktree setup goes through
`superpowers:using-git-worktrees`; implementation **always** happens in a
worktree, never in the main checkout.

```bash
git worktree add -b <type>/<scope>-<topic> .worktrees/<topic> main
```

`.worktrees/` is gitignored.

## Issue tracking

Issues live in the tracker (configured in `.claude/tracker.md`), not in local files. Project `project_code`.
`/task-init` files them; `/task-implement` reads and advances them.

- **States:** `Todo` → `In Progress` → `In Review` (PR open) → `Done` (merged)
  (resolve ids at runtime; create any your tracker lacks — e.g. Plane ships
  without `In Review`).
- **Labels:** one **project** label plus one **type** label (`feat`, `fix`,
  `refactor`, `test`, `docs`, `chore`) per issue.
- One issue ≈ one PR-sized chunk; independent issues enable parallel agents.
- The issue description carries the problem/approach/code-map/test-list and a
  pointer to the local spec filename.

## Why a separate spec workspace

`docs/superpowers/specs/` holds design docs the agent uses to align with the user
before coding. They are local-only (gitignored) because they're scratch
artifacts, not project documentation. The committed PR description and code are
the durable record.

## Subagents

For an implementation task that's bigger than a single edit, use
`superpowers:subagent-driven-development` to dispatch a subagent with:

- A pointer to its **issue** and the linked design doc.
- The exact worktree path and a "do not touch the main checkout or sibling
  worktrees" instruction.
- An explicit code map (files to touch).
- A "report back briefly" instruction (so the parent's context isn't flooded).

When tasks run in parallel, use `superpowers:dispatching-parallel-agents`:
dispatch one subagent per worktree in a single message. Keep them on disjoint
files; sequence anything that overlaps.

The parent verifies the diff and test results
(`superpowers:verification-before-completion`) before moving on.

## Code review

Always run `superpowers:requesting-code-review` before opening a PR. Report
findings to the user grouped by severity. **Do not auto-fix** — the user decides
which items are in scope.

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/):
`<type>(<scope>): <subject>`. Common types: `feat`, `fix`, `refactor`, `test`,
`docs`, `chore`. The scope is the workspace package touched — `core` or `cli`
(omit the scope for repo-wide changes).

## Tests first

Implementation follows TDD: write the failing test, watch it fail, make it pass,
refactor. Subagents dispatched for non-trivial work should be told to
red/green/refactor and to include the failing-test commit in the diff (or at
minimum show the failing run in their report).

## Layout summary

```
.worktrees/                              # gitignored, agent worktrees
.claude/commands/                        # /task-init, /task-implement (committed)
.claude/tracker.md                       # tracker config (committed; written by /harness-setup)
docs/
  superpowers/                           # gitignored
    specs/YYYY-MM-DD-<topic>-design.md   # design docs (issues live in the tracker, not on disk)
packages/
  core/  src/live_transcribe_core/       # engine, VAD, whisper, translators
  cli/   src/live_transcribe_cli/        # argparse, audio, displays, transcript
tests/
  core/  cli/                            # mirror the packages above
```
