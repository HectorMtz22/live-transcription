# Development Harness (TDD)

The repeatable loop for shipping a change in this repo. It assumes Claude Code
with the `superpowers` plugin and your tracker (configured in `.claude/tracker.md`,
reached via an MCP server). It complements [`AGENTS.md`](AGENTS.md) (the *why* of
worktrees/specs) with the *how* of TDD and issue tracking. Coordinates live in
`.claude/tracker.md` — run `/harness-setup` to create it.

Optimize for the **simplest approach that passes a test**, and **verify every
step with real output** before moving on. **Always use superpowers** — invoke
the relevant skill at each stage rather than improvising.

---

## The loop at a glance

```
brainstorm ─▶ spec ─▶ issue(s) ─▶ worktree ─▶ TDD ─▶ verify ─▶ review ─▶ PR
  (skill)    (local)  (tracker)  (gitignored) (R/G/R) (skill)  (skill)
└─────────── /task-init ──────┘  └──────────── /task-implement ───────────┘
```

Two slash commands drive the loop:

- **`/task-init [description]`** — brainstorm → local spec → issue(s).
- **`/task-implement [project_code-12 …]`** — worktree → TDD → verify → review
  → PR, with parallel agents when there are multiple issues.

Issues live in the tracker (project `project_code`). Specs and plans
stay *local* and *gitignored* under `docs/superpowers/`; worktrees live under
`.worktrees/` (also gitignored). **Never commit either.** The durable record is
the code, the tracked issue, and the PR.

---

## Set up the tracker (once per repo)

Before the loop, configure and provision the tracker:

1. **`/harness-setup`** — choose the tracker (default Plane); writes
   `.claude/tracker.md`.
2. **`/harness-bootstrap`** — create the project (if missing), the
   `Todo → In Progress → In Review → Done` states, the type labels, and the
   weekly (Mon→Sun) cycles. Idempotent — re-run to top up future cycles.

---

## 0. Decide the size

- **Trivial** (one-line fix, typo, obvious tweak): skip straight to TDD on a
  worktree branch. No spec, no issue.
- **Non-trivial** (new feature, behavior change, multi-file): run the full loop
  via `/task-init` then `/task-implement`.

When unsure, treat it as non-trivial — a 10-line spec is cheap.

---

## 1. Brainstorm (non-trivial only) — `superpowers:brainstorming`

Run `/task-init`, which invokes `superpowers:brainstorming` to pressure-test the
idea before any code. Goal: agree on the **simplest** approach and surface
unknowns. The skill writes the spec and gets your approval.

## 2. The spec — local only

The brainstorming skill writes to:

```
docs/superpowers/specs/YYYY-MM-DD-<topic>-design.md
```

Gitignored. Keep it short: problem, chosen approach, the code map (files to
touch), and the test list (what proves it works). This is the contract the
implementation agent works against.

## 3. Issue(s) — the tracker

> **Tracker setup (one-time):** `/harness-bootstrap` creates the four required
> states — `Todo`, `In Progress`, `In Review`, `Done` — along with the type
> labels and weekly cycles. Run it once before starting the loop (see "Set up
> the tracker" above). If you're on a tracker without bootstrap support, create
> any missing states by hand — put `In Review` in the "started" group, ordered
> just before `Done` — so `/task-implement` can move an issue there when its PR
> opens.

`/task-init` files the work in the tracker, project `project_code`. One
issue ≈ one PR-sized chunk. Each issue gets:

- **State `Todo`** (resolve state ids at runtime).
- A **project label** (the sub-project the work touches) and a **type label**
  (`feat`, `fix`, `refactor`, `test`, `docs`, `chore`) — resolve label ids at
  runtime.
- A description carrying the problem, approach, code map, test list, and the
  local spec filename.

Use multiple independent issues to coordinate **parallel agents** — each agent
owns one issue in its own worktree.

## 4. Worktree — `superpowers:using-git-worktrees`

Implementation always happens in an isolated worktree so the main checkout stays
clean. `/task-implement` uses `superpowers:using-git-worktrees`:

```bash
git worktree add -b <type>/<scope>-<topic> .worktrees/<topic> main
```

`.worktrees/` is gitignored. One worktree per issue/branch. For independent
issues, create multiple worktrees and dispatch one agent each — they won't
collide.

## 5. TDD: Red → Green → Refactor — `superpowers:test-driven-development`

This is the core. **Never write production code without a failing test first.**

1. **Red** — write the smallest test that captures the next behavior. Run it,
   **watch it fail for the right reason** (assertion, not import error).
   ```bash
   just test tests/core/test_<name>.py::test_<behavior>   # or: uv run pytest <path>::<test> -x
   ```
2. **Green** — write the *minimum* code to pass. No extra cases, no speculative
   options. Run the test, watch it pass.
3. **Refactor** — clean up names/duplication with the test green. Re-run.
4. **Widen** — run the **full project suite** before considering the step done:
   ```bash
   just test        # == uv run pytest (whole tests/ tree)
   ```

Repeat per behavior. Commit at green points using conventional-commit messages —
the back half (verify → review → PR) needs the work committed and the tree clean.

### Test conventions

- Tests live under `tests/` (repo root) mirroring the package they cover:
  `packages/core/src/live_transcribe_core/whisper.py` → `tests/core/test_whisper.py`;
  `packages/cli/src/live_transcribe_cli/transcript.py` → `tests/cli/test_transcript.py`.
- Pure helpers get happy-path + error tests; use `pytest.raises` for typed errors.
- Shared fixtures live in `tests/conftest.py`.
- One assert-able behavior per test; name it `test_<does_what>`.
- `filterwarnings = ["error"]` is set in `pyproject.toml` — a new
  DeprecationWarning fails the suite; scope an `ignore` there only for
  transitive warnings you don't own.

### Code you change that isn't under test yet

The live runtime (`packages/core`, `packages/cli`) has a suite; the `training/`
scripts do not (they run in a separate venv, off the runtime path). If you change
runtime behavior, add or extend a `tests/{core,cli}/` test first. Don't grow an
untested `training/` script's behavior without extracting the logic into an
importable, tested function.

## 6. Verify for real — `superpowers:verification-before-completion`

Beyond green tests, run the actual command once to confirm end-to-end behavior.
Report what you observed.

## 7. Code review — always — `superpowers:requesting-code-review`

Run `superpowers:requesting-code-review` on the branch before any PR.

- Report findings to the user **grouped by severity** (Critical / Important /
  Minor).
- **Do not auto-fix.** The user decides scope. Re-review after agreed fixes.

## 8. PR & close out — automatic

Open a PR **automatically** as soon as a branch is **verified, green, committed,
and clean**: no Important+ findings outstanding, the full suite passing, and the
working tree clean. No need to ask first. Conventional-commit title with scope
(`feat(core): …` / `feat(cli): …`). One feature per PR. Then move the issue to
**In Review**; set it to **Done** when the PR merges.

Only the PR-open step is automatic — the review *fix* decision still waits for
the user (§7).

---

## Parallel agents — `superpowers:dispatching-parallel-agents`

For work that splits cleanly, `/task-implement` runs issues concurrently:

1. `/task-init` files N independent issues (disjoint files).
2. Create N worktrees (one per issue) via `superpowers:using-git-worktrees`.
3. Dispatch one subagent per worktree in a **single message** so they run
   concurrently (`superpowers:dispatching-parallel-agents` +
   `subagent-driven-development`). Each agent gets: its issue + spec, its
   worktree path, a "don't touch the main checkout or other worktrees"
   instruction, the code map, and a "report back briefly" instruction.
4. The parent verifies each diff + test run, then reviews and PRs them
   independently, moving each issue to `In Review` on PR-open (and to `Done` when
   it merges).

Keep agents on **disjoint files** — if two issues touch the same module,
sequence them instead.

---

## Guardrails

- **Issues in the tracker; specs/plans local.** `docs/superpowers/` and
  `.worktrees/` are gitignored — never `git add` them.
- **Always superpowers.** Use the named skill at each stage; don't improvise the
  workflow.
- **Worktrees always.** Implementation never happens in the main checkout.
- **Branch off the default branch; never commit to it.** Commit at green points;
  open the PR **automatically** once the branch is verified, green, committed,
  and clean (§8). The review fix decision still waits for the user.
- **Conventional commits always.** `<type>(<scope>): <subject>`; scope is the
  sub-project.
- **Simplest first.** If a test passes without a new abstraction or dependency,
  don't add one.
- **No green claim without a run.** Paste/observe real output.
