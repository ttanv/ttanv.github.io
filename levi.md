---
layout: blog
title: "LEVI: LLM-Guided Evolution for the Price of a Cup of Coffee"
subtitle: "An open-source evolutionary framework for algorithmic discovery."
date: 2026-02-01
permalink: /levi
---

<details>
<summary><strong>Example: LEVI-evolved scheduling strategy (Spot Multi-Region, 72.4%)</strong></summary>
<p>One of the programs LEVI discovered for the Can't Be Late Multi-Region problem -- scheduling deadline-driven jobs across multiple cloud regions to minimize cost using spot instances. The strategy checks deadline safety first, exploits spot availability across all regions via arbitrage, and falls back to on-demand only when behind schedule.</p>
<pre><code class="language-python">import enum

class ClusterType(str, enum.Enum):
    NONE = "NONE"
    SPOT = "SPOT"
    ON_DEMAND = "ON_DEMAND"

def strategy_step(ctx, last_cluster_type: ClusterType, has_spot: bool) -&gt; ClusterType:
    """
    Multi-Region Cloud Instance Scheduling Strategy v2

    Optimized for:
    - Deadline compliance (0 score = failure)
    - Cost minimization via spot instance exploitation
    - Efficient cross-region switching with validation
    - Avoiding over-reliance on correlated regions

    Logic Flow:
    1. Deadline Safety Check (Critical)
    2. Current Region Spot Available? -&gt; Use SPOT
    3. Else: Search ALL regions for SPOT availability -&gt; Switch to first available
    4. If no spot anywhere: use ON_DEMAND if behind schedule, else wait (NONE) if safe
    """

    env = ctx.env
    now = env.elapsed_seconds
    deadline = ctx.deadline
    total_work = ctx.task_duration
    done_work = sum(ctx.task_done_time)
    remaining_work = total_work - done_work
    time_left = deadline - now

    # Task already finished
    if remaining_work &lt;= 0:
        return ClusterType.NONE

    # --- 1. DEADLINE SAFETY: Point of No Return (PNR) ---
    required_time = remaining_work + ctx.restart_overhead
    safety_threshold = required_time * 1.05

    if time_left &lt;= safety_threshold:
        return ClusterType.SPOT if has_spot else ClusterType.ON_DEMAND

    # --- 2. Check Current Region Spot Availability ---
    if has_spot:
        return ClusterType.SPOT

    # --- 3. Multi-Region Spot Arbitrage: Find Any Available Spot ---
    all_spots = env.get_all_regions_spot_available()
    num_regions = env.get_num_regions()
    current_region = env.get_current_region()

    for idx in range(num_regions):
        if all_spots[idx]:
            if env.switch_region(idx):
                return ClusterType.SPOT

    # --- 4. No Spot Available Anywhere ---
    ideal_rate = total_work / deadline
    expected_progress = ideal_rate * now
    progress_deviation = done_work - expected_progress

    if progress_deviation &lt; 0:
        return ClusterType.ON_DEMAND

    slack = time_left - remaining_work
    min_slack = max(3600.0, deadline * 0.05)

    if slack &gt; min_slack:
        tick_index = int(now // env.gap_seconds)
        if tick_index % 10 == 0:
            next_region = (current_region + 1) % num_regions
            env.switch_region(next_region)
        return ClusterType.NONE

    return ClusterType.ON_DEMAND
</code></pre>
</details>

## TLDR

Existing LLM-based evolutionary systems (OpenEvolve, ShinkaEvolve) have **weak diversity mechanisms**, converge early, and compensate by throwing expensive models at the problem. This leads to bloated costs and unnecessary complexity.

LEVI fixes the root cause: **CVT-MAP-Elites** with **AST-based behavioral fingerprinting** keeps a diverse archive where each cell holds the best solution for its behavioral niche. Different algorithmic approaches naturally land in different cells, so the system never collapses onto one strategy. We split mutations into two tiers: **cheap small models** (e.g. Qwen-30B) for hundreds of narrow mutations, and a **larger model** (Gemini Flash) used sparingly for paradigm shifts.

Result: ***better*** scores than OpenEvolve, ShinkaEvolve, and GEPA on ADRS benchmarks at **1.5-6.7x lower cost**.

**LEVI will be open-sourced on GitHub soon.** Point it at a scoring function and a seed program and it runs until the budget is spent.

### ADRS Benchmark Results (% score)

[ADRS](https://ucbskyadrs.github.io/) is a benchmark suite from UC Berkeley for evaluating LLM-guided optimization on real-world systems problems -- cloud scheduling, load balancing, congestion control, SQL optimization, and more.

| Framework | Average | Cloudcast | EPLB | LLM-SQL | Prism | Spot Multi-Reg | Spot Single-Reg | Txn Scheduling |
|---|---|---|---|---|---|---|---|---|
| Human SOTA | 59.4 | 100.0 | 45.8 | 67.7 | 60.8 | 54.5 | 45.1 | 41.9 |
| AutoEvolve | 74.1 | 97.8 | 70.2 | 76.4 | 87.4 | 70.0 | 46.3 | 70.6 |
| GEPA | 71.9 | 96.6 | 70.2 | 67.7 | 87.4 | 62.2 | 51.4 | 67.7 |
| OpenEvolve | 70.6 | 92.9 | 62.0 | 72.5 | 87.4 | 66.7 | 42.5 | 70.0 |
| ShinkaEvolve | 67.4 | 72.0 | 66.4 | 68.5 | 87.4 | 63.6 | 45.6 | 68.2 |
| **LEVI** | **76.5** | **100.0** | **74.6** | **78.3** | **87.4** | **72.4** | **51.7** | **71.1** |

*Table 1: ADRS benchmark scores. LEVI achieves the highest average across all frameworks.*

### Cost per Problem

| Problem | ADRS Baseline Cost | LEVI Cost | Reduction |
|---|---|---|---|
| Cloudcast | ≤$15 | $4.50 | 3.3x |
| EPLB | <$15 | $4.50 | <3.3x |
| LLM-SQL | ≤$20 | $4.50 | 4.4x |
| Prism | ≤$15 | $4.50 | 3.3x |
| Spot Multi-Reg | ≤$25 | $4.50 | 5.6x |
| Spot Single-Reg | ≤$30 | $4.50 | 6.7x |
| Txn Scheduling | ≤$20 | $13 | 1.5x |

*Table 2: Cost comparison. LEVI uses $4.50 on most tasks (Txn Scheduling is $13) versus the baselines' $15-$30, yielding 1.5-6.7x reductions.*

## The Problem with Existing Systems

The open-source LLM evolution ecosystem optimizes the **wrong parts of the stack**. Systems use large LLMs (Claude Sonnet 4.5, Gemini 3.0 Pro, GPT 5.2), add LLM-as-judge filters and embedding-based deduplication, and still converge within a few hundred generations. The expensive models aren't the cause -- they're a **symptom of early stagnation**. If your system plateaus quickly, of course you want the strongest model for each generation.

How do we know more generations help? DeepMind's **FunSearch** needed ~1 million generations; **AlphaEvolve** needed thousands. FunSearch even found that larger LLMs didn't help -- it was only AlphaEvolve that could harness them.

LEVI aims to dismiss two notions:

- **"LLM evolution must be expensive"** -- we show 1.5-6.7x budget reductions
- **"You need SOTA models"** -- we mainly use 30B-100B models, with Gemini Flash sparingly

## How LEVI Fixes This

Two core improvements:

1. **Better diversity maintenance** in the solution archive
2. **Smarter model allocation** -- cheap models for refinement, expensive models only for paradigm shifts

## System Design

DSPy optimizes mutation prompts once up front, then producer workers sample parents from the CVT-MAP-Elites archive via LiteLLM, push candidate code through an asyncio queue, and consumer workers evaluate each candidate in a sandboxed subprocess. The archive only accepts improvements per behavioral niche. Punctuated Equilibrium periodically triggers paradigm shifts, and a budget manager shuts everything down when the dollar/eval/time limit is hit.

<img src="/images/levi_overview.png" alt="LEVI System Overview" class="img-light" style="width:100%; height:auto;" />
<img src="/images/levi_overview_dark.png" alt="LEVI System Overview" class="img-dark" style="width:100%; height:auto;" />

*Figure 1: LEVI system overview. Producers generate mutations, consumers evaluate in sandboxed subprocesses, and the CVT-MAP-Elites archive maintains behavioral diversity.*

## Key Components

### CVT-MAP-Elites Archive

**MAP-Elites** is a quality-diversity algorithm: instead of tracking one best solution, it partitions solution *behaviors* into cells and keeps the **single best program per cell**. A greedy approach and a DP approach coexist, each the best of its kind.

We use **Centroidal Voronoi Tessellation** (CVT) to define cells via k-means centroids. Programs are assigned to their nearest centroid. Empty cell? Move in. Occupied? Replace only if the new score is higher.

**Behavior is defined by AST features**: loop depth, branch count, cyclomatic complexity, math operators, etc. These fingerprint the *shape* of an algorithm without running it. A greedy scheduler and a DP approach with nested loops naturally land in different cells.

**Parent selection** uses softmax sampling weighted by fitness, with multiple temperatures running simultaneously -- some greedily exploiting top solutions, others broadly exploring.

### Tiered Model Strategy

**Small models** (e.g. Qwen-30B) handle **90%+ of mutations**: tweaking thresholds, swapping sort keys, adjusting heuristics. Cheap enough to call hundreds of times.

**Large models** (Gemini Flash) are used sparingly for **paradigm shifts**: synthesizing the best solutions from different behavioral regions into something fundamentally new.

```
FunSearch
·→  ·→  ·→
   ↗         ↘
  ·→ ·→ ·→ ·→ ·→ ·→ ·→ ... (millions) ... → ★
   ↘         ↗
    ·→  ·→  ·→


LEVI
·═════►  ·→ ·→  ═════►  ·→ ·→  ═════►  ·→ → ★
              ↘ ·→           ↘ ·→
              ↗ ·→           ↗ ·→
```

*Figure 2: FunSearch uses millions of small mutations. LEVI alternates paradigm shifts (═════►) from a larger model with narrow mutations (·→) from smaller models.*

This is implemented via **Punctuated Equilibrium** (every K evaluations):

1. **Cluster** archive cells into behavioral regions
2. **Pick** the best elite from each cluster
3. **Generate a paradigm shift** with the heavy model (prompt adapts to budget stage: radical early, synthesis mid-run, surgical refinement late)
4. **Generate variants** with lighter models
5. **Insert with noise** on behavior vectors to explore adjacent cells

### Prompt Optimization

A one-time **DSPy MIPROv2** pass tunes mutation prompts per model. The metric rewards compilable, score-improving code and penalizes overly prescriptive prompts.

### Async Pipeline

**N producer workers** sample parents, call LLMs, push code to an asyncio queue. **M consumer workers** evaluate in sandboxed subprocesses (`ResilientProcessPool` with hard SIGKILL timeouts). Archive access is lock-protected but contention stays low since LLM calls and evaluations happen outside the lock.

**Budget enforcement is by-construction**: the pipeline checks dollars, eval count, and wall time before every LLM call and shuts down cleanly when any limit is hit.

## Putting It Together

A typical LEVI run: **$4.50 total**, 12 LLM workers, 50 eval workers, punctuated equilibrium every 10 evaluations. Small models (e.g. Qwen-30B) handle 90%+ of mutations at fractions of a cent each. Gemini Flash handles paradigm shifts at a few cents each.
