# Product / design summary

## States (simulator flow)

- **Terminal states:** `finished`, `abandoned` — trajectory ends, outcome set.
- **Non-terminal states:** `start`; all **goals** (goal_1 … goal_N); **publish**; **subscribe**. From these the chain can move to other goals, publish/subscribe, or eventually to finished/abandoned.

**Nested (goals only):** When in a goal, we sample a nested outcome: **succeeded** / **continue** / **failed**. Formula: `p_success = 0.25 + 0.50*persona.determined + 0.25*tool_quality` (range 0.25–1.0 for system variance); `p_failed = 0.3`; `p_continue` capped at 0.12. These drive allowed next top-level states (e.g. succeeded → goals, publish, subscribe, finished; failed → goals, abandoned). **Continue** is useful: it models “goal not yet resolved” and adds step variance; the chain and E[T] depend on it; it’s kept rare so trajectories stay ~10 steps.

**Top-level:** From start → goals (uniform). From goal (by nested) → see doc/simulator_flow.md §4. From publish → goals, subscribe, finished, abandoned. From subscribe → goals, publish, finished, abandoned. Sampling weights: finished/abandoned = 2, others = 1. Transition matrix is in `trajectory/transition_matrix.py`; runtime sampling in `trajectory/transitions.py` (TransitionSampler); one trajectory loop in `trajectory/generator.py`; N samples in `simulator/experiment_simulator.py`.

---

## Theory

- **Inputs:** `PersonaSet` + `GoalSet` (same as simulator). Builds transition matrix once.
- **Outputs (TheoreticalValues):** E[T], E[T²]; P(finished), P(abandoned), P(publish), P(subscribe); correlation ρ(publish, subscribe); P(subscribe rate q₂ > q₁); sample size N for subscribe-rate comparison and for publish-as-proxy; ratio N_sub/N_pub.
- **Modules:** chain (Q, R, α, fundamental matrix), absorption (P(fin/aban), hitting probs), moments (length), correlation (ρ), sample_size (N, power), values (single `TheoreticalValues.compute()`).

---

## Verifier

- **Purpose:** Check that the simulator (fast mode, no LLM) matches theory.
- **Flow:** Load experiment (persona, goals); compute theory once; run simulator at one or more sample sizes n; compute empirical means (length, p_fin, p_aban, p_pub, p_sub, ρ); check empirical ∈ theory CI (or within tolerance for ρ); report pass/fail per metric per n.
- **Features:** Single run (`run_verification`) or N experiments with subsampling (`verify`); multi-seed with pass-rate threshold; report tables (Markdown/LaTeX) and figures (theory vs empirical vs n). Can auto-create experiment (persona+goals) if missing.
- **Modules:** io (load/copy experiment, ensure_experiment_exists), empirical (from trajectories), checks (append_checks, z), run (VerificationResult, run_verification, run_n_verifications), multi_seed, report (summary_table, figures).

---

## Persona

- **PersonaSpec:** id, name, weight, traits: `technical`, `determined`, `swearing`, `baseline_sentiment` (0–1). Used in nested p_success (determined) and optionally for LLM/name generation.
- **PersonaSet:** experiment_id + list of PersonaSpec; weights sum to 1. One persona is drawn per trajectory (weighted random).
- **Sources:** `persona.yaml` (load/save); generator (LLM or synthetic) produces PersonaSet; namegen (optional) uses traits to get a short label via Gemini.

---

## Goal

- **GoalSpec:** id, name, tools (list of tool ids).
- **Tool:** id, quality (0–1), name. Quality feeds `tool_quality_for_goal()` and thus nested p_success.
- **GoalSet:** experiment_id, goals, tools; `terminal_states` = [finished, abandoned]; `outcome_states` = [publish, subscribe]; optional `required_goals_for_finish` (disabled in verification so theory matches).
- **Sources:** `goals.yaml` (load/save); generator produces GoalSet. Transition matrix is built from PersonaSet + GoalSet (nested probs per persona×goal; top_level from_start, from_goal_*, from_publish, from_subscribe).
