# Product Doc: `goldfish` — Context Compaction Benchmark

Status: v2, M0 built
Owner: Sreshta
Date: 2026-08-27

---

## 0. Read this first: what the research changes

The original pitch says context engineering is "the most discussed and least measured topic in agents." Half right, and the half that is wrong matters.

**Heavily measured already:** conversational memory recall. LoCoMo, LongMemEval and BEAM are the standard trio, and every memory vendor publishes numbers against them. LongMemEval-S alone is 500 questions over roughly 115k token histories with about 40 sessions each, testing extraction, multi session reasoning, temporal reasoning, knowledge updates and abstention. There is a successor, LongMemEval-V2, extending the format into web agent trajectories. There are dozens of 2026 papers proposing memory architectures against these. Do not touch this.

**Genuinely thin:** comparing condensation *strategies* under controlled conditions during *agentic task execution*. The closest work is a May 2026 paper on memory condensation for coding agents in scientific discovery, and its own related work section states that prior work covers memory quality, cost aware planning, framework efficiency, prompt caching and performance under growing context, but that none of them compare multiple condensation strategies while controlling for task type and model. So one paper now does this, in one domain. That is a much better position than the MCP harness idea was in: the ground is thin, not empty.

**The best single data point I found for you:** Factory.ai evaluated 36,611 production messages and found every compaction method scored roughly 2.2 to 2.5 out of 5 on artifact tracking, meaning remembering which files were modified is uniformly weak across strategies. That is the shape of the real finding. Not "summarization gets 71 percent." Rather, "every strategy loses the same specific class of information, and here is which class."

**Reframe:**

> Original: measure task success under five strategies.
> Revised: measure **what class of information each strategy destroys, and how fast**, and report a per class forgetting curve rather than a single pass rate.

The headline metric becomes **context half life**: the number of turns after which a planted fact of class C, under strategy S, drops below 50 percent recall. That is a quotable number, it is per class, and no one currently publishes it. It is also the reason the repo is called goldfish.

---

## 0.5 What the thing actually is

Three products were hiding in the original description. Only one is v0.

| | What it is | Value | Verdict |
|---|---|---|---|
| **A study** | Run it once, publish results | The finding, plus what you learn building it | **v0** |
| **A package** | Others install and measure their own strategy | Adoption | Build the seam, do not chase it |
| **A continuous check** | Runs on every commit, fails the build on regression | The only version that outlives the writeup | v1, cheap once v0 exists |

**Delivery shape:** an installable Python package with a config driven CLI
(`goldfish run --config exp.yaml`), not a daemon and not a library you have to
wire into your agent.

**The reuse seam is one method.** `Strategy.reduce(messages, budget) -> messages`.
If someone can implement that, they get the full report for their own logic.
Everything else in the repo is an implementation detail. Do not let the
interface grow.

**Why not a production monitor.** Online probing means injecting synthetic
content into real user sessions to see whether the agent still remembers. That
is invasive, it pollutes user transcripts, and nobody will run it. The version
that has legs is `goldfish check` in CI: a small fixed probe suite that fails
the build when context half life regresses by more than a threshold. Pytest for
context strategy. That is v1 and costs almost nothing once the runner exists.

## 0.6 What building this teaches

Worth being precise, because the honest list is short and one item on it is rare.

- **Writing an agent loop from scratch.** No LangGraph. Message state, tool
  dispatch, compaction mutating persistent history, token accounting per
  message. Directly relevant to the AgentMail AXE role.
- **Experimental design under nondeterminism.** Controlling confounds, seeds,
  confidence intervals, deciding required n. The rarest skill among application
  level AI engineers, and an extension of the eval layer you already own at
  work rather than a repeat of it.
- **Prompt caching mechanics and real cost accounting.** Undertaught, and the
  likely source of the one counterintuitive finding.
- **Building an instrument you can defend under attack.** Different muscle from
  shipping a feature. Section 11 of this doc is where that shows up.

What it does **not** teach: distributed systems, scale, infrastructure, product
judgment. If the founding engineer track is the goal, pick the next project to
cover that gap deliberately.

## 1. Problem

Every agent framework compacts. Claude Code, Codex CLI, Aider, OpenHands and Gemini CLI all trigger LLM summarization at a token threshold. A 2026 survey of eight frameworks found six of eight use summarization as the primary strategy, each agent compacting independently with essentially no coordination. Thresholds are folklore: the commonly cited advice is to compact around 70 to 75 percent of window capacity because degradation is believed to start around 70 to 80 percent.

Nobody publishes the evidence for those numbers on their own workload. A team choosing between sliding window, summarization and a scratchpad has no way to answer:

- Which strategy loses my task's critical information first?
- How much does compaction cost once prompt caching is accounted for?
- What happens after the third consecutive compaction, when I am summarizing summaries?
- Where is the actual threshold for my task, as opposed to the folklore number?

`goldfish` answers those four questions for a given agent loop.

## 2. Users

**Primary: engineers building agent loops** who have to pick or tune a context strategy and currently pick by vibes.

**Secondary: framework and infra teams** who ship a compaction default and would like evidence for it.

**Non user:** memory system vendors benchmarking retrieval quality. LoCoMo and LongMemEval own that. Do not compete.

## 3. Goals and non goals

**Goals**
- A controlled environment where the required information is known, planted, and programmatically verifiable.
- Per information class degradation curves, not a single aggregate score.
- Cost reported cache aware, because uncached token math is misleading by a large factor.
- Compaction generation analysis: what compounds when you compact a compaction.
- Reusable: someone can plug their own agent loop or strategy in and get the same report.

**Non goals for v0**
- Proposing a new memory architecture. This is a measurement project. Resist.
- Conversational memory QA. Solved space.
- Real world environments (SWE-bench style repos). One synthetic controllable environment first. Real environments are v1 and only for external validity.
- A leaderboard. You do not have the credibility budget to run one yet.

## 4. Design of the environment: the part that decides whether this works

The failure mode for this project is a benchmark where results are obvious or unfalsifiable. Both come from a badly designed task.

**Requirement 1: long range dependency must be forced.** If information from turn 5 is not required at turn 90, sliding window wins trivially and the whole study is a null result. Plant dependencies deliberately.

**Requirement 2: verification must be programmatic.** No LLM judge on the primary metric. An LLM judge introduces exactly the variance you are trying to measure.

**Requirement 3: the environment must be cheap to run many times.** You need N seeds across five strategies across three models. Anything expensive per run kills the study.

**Proposed environment: a synthetic state tracking workload with planted probes.**

A deterministic tool environment (a small ledger, an inventory, or a simulated file tree) where the agent performs a long sequence of operations. Interleaved into the trajectory are **probes**: pieces of information of a known class, introduced at turn k, required at turn k+n.

The six probe classes, chosen because they are the ones practitioners actually lose:

| Class | Example planted at turn k | Tested at turn k+n |
|---|---|---|
| Goal | The user's actual objective and its acceptance criteria | Does the agent still optimize for it |
| Constraint | "Never modify files under `vendor/`" | Does it violate the constraint later |
| Negative knowledge | "Approach A was tried at step 12 and failed because X" | Does it retry A |
| Artifact state | Which files or records were created or modified and their current values | Can it report or reuse them correctly |
| Numeric or identifier fact | An ID, a threshold, an account number returned by a tool | Is it reproduced exactly or hallucinated |
| Provenance | Which tool call produced a given result | Can it attribute or re verify |

Negative knowledge and artifact state are the two I would bet on being worst, and they are the two least discussed. Factory's artifact tracking result supports the second.

**Why this design is better than "run a long task and see if it passes."** A binary pass rate on a long task tells you a strategy is worse without telling you why, and takes far more compute to reach significance. Probes give you a graded, per class, cheap signal, and the aggregate task success can still be reported alongside as a sanity check.

**External validity check (small, but do it):** run one real workload, ideally a multi file refactor in a repo you control, and confirm the ranking of strategies matches the synthetic result. If it does not, that is itself the most interesting finding in the project, and you should report it rather than bury it.

## 5. Strategies to compare

The original five conflate two independent axes. Separate them.

**Axis A, what is evicted:** oldest first, largest tool outputs first, or model chosen.
**Axis B, where evicted content goes:** discarded, into an LLM summary, into a retrievable store, or into an agent written file.

The strategy set for v0, holding axis A at oldest first except where noted:

1. **Full history.** Control. Also the cost baseline that caching makes interesting.
2. **Sliding window.** Keep last N tokens. The cheap default.
3. **LLM summarization.** The industry default, six of eight frameworks.
4. **Retrieval over prior turns.** Evicted turns embedded and retrieved on demand.
5. **External scratchpad.** Agent maintains a file it can read and write.
6. **Structured note taking, added.** The agent annotates the transcript as it goes, and eviction preserves the annotations rather than re reading the raw transcript. This is what Manus style architectures and recent eviction papers argue for, and leaving it out means your benchmark tests only the strategies that are already known to be weak. If it wins, that is a real result. If it loses, that is a bigger one.
7. **Tool output masking, optional.** Keep the call and its metadata, drop the payload. Cheap and surprisingly strong in some scaffolds.

Hold everything else fixed: same model, same system prompt, same tools, same trigger threshold, same recency window. One free variable per run.

## 6. Metrics

**Fidelity**
- Probe recall by class, by turn distance. The core table.
- **Context half life** by class and strategy: turns until recall crosses 50 percent. The headline number.
- Constraint violation rate. A separate hard failure, not a recall miss.
- Hallucinated substitution rate: how often a lost fact is replaced by a confident wrong one rather than an admission of not knowing. Silent substitution is far more dangerous than a blank, and no existing comparison separates the two.

**Compaction dynamics**
- Fidelity as a function of compaction generation, 1 through 5. Does loss compound linearly or does it cliff.
- Cost and latency of the compaction step itself. It is a blocking LLM call over most of the window, so it is not free and papers routinely ignore it.

**Cost, cache aware**
- Total cost per completed task with and without prompt caching enabled.
- This is where a genuinely counterintuitive result may live. Compaction invalidates the cache prefix by rewriting history, so a strategy that reduces token count can increase dollar cost. If that holds, it is the most quotable thing in the project and it directly contradicts the standard justification for compacting.

**Task level**
- Success rate, turns to completion, redundant work rate (repeating an already completed operation, a direct symptom of forgetting).

**Everything reported with n, seeds, and confidence intervals.** Five strategies times three models times one aggregate number with no variance is a blog post that gets dismantled in the comments.

## 7. Architecture

```
goldfish/
  env/            deterministic task environment + tool surface
  probes/         probe definitions, planting schedule, programmatic checkers
  strategies/     one class per strategy, single interface: reduce(history, budget) -> history
  runner/         agent loop, model adapters, seeds, caching control
  metrics/        recall curves, half life fit, cost accounting, violation detection
  report/         markdown + JSON, curve plots
  configs/        experiment definitions, fully declarative and checked in
```

The strategy interface is the reuse surface. If a user can implement `reduce()` and get the full report, the repo becomes a place people test their own compaction logic, which is the only path to it outliving the blog post.

Model adapters: one Anthropic, one OpenAI, one open weight. Pin versions in every report. Run caching on and off as a flag, not a config.

## 8. Milestones

Two to three weeks is achievable only if you do not build a real world environment in v0. The environment is the entire cost.

**M0, days 1 to 3. Built.** Environment, tools, probe framework, checkers,
strategy interface, runner, and an offline context bound simulator. Five model
free strategies run end to end across a budget sweep.

What M0 already surfaced, all of it before spending a cent on tokens:

1. **A scoring bug that would have inflated artifact recall.** The simulated
   agent counted transaction IDs appearing anywhere in context, including in
   the arguments of *failed* calls, so it credited itself for work it had not
   done. Every artifact state number would have been wrong.
2. **A confound in tool masking.** It masked unconditionally rather than under
   context pressure, so at an unbounded budget it did not match the control.
   Every comparison against it would have been invalid. Caught by an invariant
   test, not by inspection.
3. **The environment is too small to force compaction at realistic budgets.**
   Peak context is roughly 1.3k approximate tokens, so the budget sweep has to
   run at 150 to 700 to produce any eviction at all. Before real runs, the
   episode has to grow by roughly two orders of magnitude, most cheaply by
   inflating tool payloads rather than lengthening the trajectory.

Two invariants now gate every change and belong in CI:

- **No leak.** `full_history` scores a perfect ceiling at every budget.
  Anything less means the environment or probes lose information the strategy
  did not.
- **Control invariance.** At an unbounded budget, every strategy is identical
  to the control.

**M1, days 4 to 6.** Strategies 1 through 5 behind the common interface. Single model. First recall curves.

**M2, days 7 to 9.** Seeds, confidence intervals, half life fitting, cost accounting with and without caching. This is where the results become defensible rather than suggestive.

**M3, days 10 to 12.** Compaction generation study and the two added strategies (structured notes, masking). Multi model.

**M4, days 13 to 14.** Report, README, plots, writeup. Ship.

**M5, optional.** One real workload for external validity. Only after the synthetic result is solid.

## 9. Risks

| Risk | Severity | Mitigation |
|---|---|---|
| Results are obvious (more context is better, summarization loses detail) | High | Probe classes and the caching cost angle are where non obvious findings live. If the per class table is flat, the environment is too easy. Test this at M1 before investing further. |
| Synthetic environment does not generalize | High | State it as a limitation up front, and run the M5 validity check. Overclaiming generality is what gets a benchmark dismissed. |
| Confounded comparisons | High | One free variable per run. Same trigger threshold, same recency window, same prompts across strategies. Document every held constant in the report. |
| Cost of the study itself | Medium | Estimate before starting: strategies times models times seeds times tasks times average tokens. Cap the window at a size where full history is affordable. Use a small model for pilot runs, large models only for the final matrix. |
| Academic overlap | Medium | The May 2026 condensation paper is adjacent. Cite it, and differentiate on: probe class taxonomy, cache aware cost, compaction generation compounding, and being a runnable tool rather than a paper. |
| Scope creep into proposing a better strategy | Medium | Explicit non goal. If a finding suggests one, that is the *next* project, and a much stronger one. |

## 10. Career and positioning

Be honest with yourself about what this is worth relative to the MCP harness.

**Weaker on reuse.** The MCP harness produced one artifact per target company. This produces one artifact total: a repo and a writeup. There is no per company version of it.

**Stronger on interview leverage.** Context strategy comes up in every agent infrastructure interview. Having measured it yourself gives you a specific, defensible opinion instead of the same secondhand takes everyone else repeats. "I ran this and here is where the standard advice breaks" is a different conversation from "I read the Anthropic context engineering post."

**Be careful with the Albertsons bridge.** The pitch says this connects directly to work you already did. Check that claim before you make it in an interview. Your production work is the eval and observability layer and the context engine and knowledge graph. That is a real and relevant bridge: you have built the measurement apparatus for agents in production, and you have worked on what goes into the window. It is not the same as having run a compaction study at work. State the former, and let the project be the latter. An interviewer who probes an overstated link will find the seam.

**Sequencing, revised.** Previous recommendation was the MCP harness first.
Reversing it, on the research rather than on preference. The MCP harness has
four shipped competitors and one unresolved external dependency: if the target
companies' servers are not reachable without a paid account, its entire reuse
argument collapses, and that is only discoverable after building. Goldfish has
thin competition, no external dependency, and is finishable alone in a fixed
window. Build this one first.

**Opportunity cost, stated plainly.** This would be the sixth open project
alongside a full time job and an active search, and one of the existing five is
already parked pending a rebuild. The binding constraint is not idea quality,
it is finishing. If goldfish is not shippable in fourteen days of real elapsed
time, the honest move is to cut M3 and ship M2.

## 11. Success criteria

- At least one finding that contradicts common practice. The cache aware cost inversion is the most likely candidate. If nothing contradicts anything, the study was not designed to be surprising enough.
- The per class table shows real separation between strategies, not noise.
- One external person reproduces a result or plugs in their own strategy.
- The writeup is cited or shared by someone building an agent framework.

## 12. Names

`goldfish` is good. It is memorable, the metaphor is instant, and it pairs perfectly with a context half life metric. Keep it.

| Name | Read | Notes |
|---|---|---|
| **goldfish** | How fast does it forget | Top pick. Fun, memorable, and the half life metric fits it exactly. |
| **halflife** | The metric as the name | Strong, but collides with a very famous game in search results. |
| **ebbinghaus** | The forgetting curve, named for its discoverer | Precise and clever, hard to spell and to say. Nice as a module name inside goldfish. |
| **longhaul** | Straight faced version | Safe, generic, forgettable. Use only if a serious audience is the priority, which it is not. |
| **decay** | What is being measured | Clean, likely heavily contested as a package name. |
| **sieve** | What compaction actually is | Good metaphor, less specific to forgetting. |

Recommendation: repo `goldfish`, CLI `goldfish`, tagline "how fast does your agent forget, and what does remembering cost." Keep the tagline, it is the best line in the original pitch. Publish `context half life` as the metric name and use it consistently everywhere; a named metric is what gets quoted.
