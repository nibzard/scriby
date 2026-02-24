## Core principles for AI-native agents (and their tools)

### 1) Treat interfaces as contracts, not UX

**Principle:** The “API” an agent uses is the *command surface + help text + output shapes + exit codes*.
**Recommendations:**

* Make the contract **explicit and complete** in `--help`: usage, args, flags, examples, output modes, exit codes.
* Version the contract (schema version / CLI contract version) and keep it stable.

### 2) Default to structured, machine-parseable output

**Principle:** Agents shouldn’t have to infer state from prose (“Hooray!”).
**Recommendations:**

* Make output **JSON-first** (or at least `--json` everywhere).
* Use a **single envelope** shape across commands so agents can write one parser:

  * `schema_version`, `command`, `status`, `run_id`, `data`, `errors[]`, `warnings[]`, `metrics`.
* Keep “human pretty text” as an opt-in or alternative mode (`--output text`).

### 3) Make success/failure unambiguous and deterministic

**Principle:** Agents need a reliable stopping condition and branching logic.
**Recommendations:**

* Emit a **clear status field** (`succeeded|failed|partial`) and a **deterministic exit code**.
* Ensure every failure includes:

  * `errors[0].class` (input/auth/network/session/…)
  * `errors[0].code`
  * `retryable: true/false`
  * a bounded `hint` (what to try next, not a novel)

### 4) Design for recovery, not perfection

**Principle:** Agents are iterative systems; your tool should make retries cheap and safe.
**Recommendations:**

* Add **idempotency keys** and **bounded retries** (`--max-retries`, `--timeout-ms`).
* Separate **validate** from **run** (`task validate` vs `task run`) to reduce churn.
* Provide “diagnose and replay” primitives:

  * `doctor` → deterministic remediation suggestions
  * `replay` → regression triage at a specific step

### 5) Make state (sessions) explicit and policy-driven

**Principle:** Hidden state causes agent confusion and accidental coupling.
**Recommendations:**

* Support clear session policies (`ephemeral|sticky|resume`) and always emit `session_id` when used.
* Make lifecycle operations **idempotent** (e.g., session close).
* Surface expiry/TTL and conflicts as first-class, typed errors.

### 6) Provide strictness and escape hatches

**Principle:** Agents need “guarantees” in production and flexibility in exploration.
**Recommendations:**

* Offer `--strict` to prevent silent fallbacks and enforce schema completeness.
* Keep a low-level escape hatch for power users and complex edge cases, but ensure the agent path is still contract-driven.

### 7) Minimize context pollution

**Principle:** Every unnecessary token in help/output competes with task reasoning.
**Recommendations:**

* Keep `--help` concise but complete: list flags, formats, exit codes, and a few canonical examples.
* Avoid spinners, progress bars, and chatty narratives in machine modes; prefer line-delimited events (`--output jsonl`) or metrics.

### 8) Avoid interaction traps

**Principle:** Agents break on anything that assumes a human at a terminal.
**Recommendations:**

* No mandatory prompts; provide `--yes`, `--non-interactive`.
* Avoid browser/OAuth redirects as the primary auth path; offer token/key flows.
* Don’t make help/context vary based on environment in surprising ways.

### 9) Measure the right outcomes

**Principle:** “AI-native” should be validated with agent benchmarks, not vibes.
**Recommendations:**

* Track: commands per successful task, schema-valid output rate, session churn, automatic recovery rate on retryables (exactly like Steel’s acceptance criteria).

---

## A practical checklist for “AI-native agent interfaces”

If you implement only these, you’ll get most of the benefit:

1. `--help` includes **Usage + args/flags + examples + output modes + exit codes**
2. `--output json` (or default JSON) with a **versioned envelope**
3. **Deterministic exit codes** + `retryable` + bounded `hint`
4. Split **validate/run**, and add **doctor/replay** equivalents
5. Explicit **session policy** + idempotency + timeouts
6. Non-interactive by default in agent mode (`--yes`, no spinners)

That’s the “core principles” synthesis: **clarity + structure + determinism + recovery**.

