---
layout: blog
title: "LEVI: Budget-Efficient LLM-Guided Evolution"
date: 2024-06-01
permalink: /levi
---

## TLDR

Existing LLM-based evolutionary systems (OpenEvolve, ShinkaEvolve) have **weak diversity mechanisms**, converge early, and compensate by throwing expensive models at the problem. This leads to bloated costs and unnecessary complexity.

LEVI fixes the root cause: **CVT-MAP-Elites** with **AST-based behavioral fingerprinting** keeps a diverse archive where each cell holds the best solution for its behavioral niche. Different algorithmic approaches naturally land in different cells, so the system never collapses onto one strategy. We split mutations into two tiers: **cheap small models** (e.g. Qwen-30B) for hundreds of narrow mutations, and a **larger model** (Gemini Flash) used sparingly for paradigm shifts.

Result: ***better*** scores than OpenEvolve, ShinkaEvolve, and GEPA on ADRS benchmarks at **3-7x lower cost**.

**LEVI is open-source at github.com.** Point it at a scoring function and a seed program and it runs until the budget is spent.

### ADRS Benchmark Results (% score)

| Framework | Average | Cloudcast | LLM-SQL | Prism | Spot Multi-Reg | Spot Single-Reg | Txn Scheduling |
|---|---|---|---|---|---|---|---|
| Human SOTA | 61.7 | 100.0 | 67.7 | 60.8 | 54.5 | 45.1 | 41.9 |
| AutoEvolve | 74.9 | 97.8 | 76.4 | 87.4 | 70.0 | 46.3 | 70.6 |
| GEPA | 72.7 | 96.6 | 67.7 | 87.4 | 62.2 | 51.4 | 67.7 |
| OpenEvolve | 72.0 | 92.9 | 72.5 | 87.4 | 66.7 | 42.5 | 70.0 |
| ShinkaEvolve | 67.5 | 72.0 | 68.5 | 87.4 | 63.6 | 45.6 | 68.2 |
| **LEVI** | **76.7** | **100.0** | **78.3** | **87.4** | **72.4** | **51.7** | **70.4** |

*Table 1: ADRS benchmark scores. LEVI achieves the highest average across all frameworks.*

### Cost per Problem

| Problem | ADRS Baseline Cost | LEVI Cost | Reduction |
|---|---|---|---|
| Cloudcast | ≤$15 | $4.50 | 3.3x |
| LLM-SQL | ≤$20 | $4.50 | 4.4x |
| Prism | ≤$15 | $4.50 | 3.3x |
| Spot Multi-Reg | ≤$25 | $4.50 | 5.6x |
| Spot Single-Reg | ≤$30 | $4.50 | 6.7x |
| Txn Scheduling | ≤$20 | $4.50 | 4.4x |

*Table 2: Cost comparison. LEVI uses a flat $4.50 per problem versus the baselines' $15-$30, yielding 3-7x reductions.*

## The Problem with Existing Systems

The open-source LLM evolution ecosystem optimizes the **wrong parts of the stack**. Systems use large LLMs (Claude Sonnet 4.5, Gemini 3.0 Pro, GPT 5.2), add LLM-as-judge filters and embedding-based deduplication, and still converge within a few hundred generations. The expensive models aren't the cause -- they're a **symptom of early stagnation**. If your system plateaus quickly, of course you want the strongest model for each generation.

How do we know more generations help? DeepMind's **FunSearch** needed ~1 million generations; **AlphaEvolve** needed thousands. FunSearch even found that larger LLMs didn't help -- it was only AlphaEvolve that could harness them.

LEVI aims to dismiss two notions:

- **"LLM evolution must be expensive"** -- we show 3-7x budget reductions
- **"You need SOTA models"** -- we mainly use 30B-100B models, with Gemini Flash sparingly

## How LEVI Fixes This

Two core improvements:

1. **Better diversity maintenance** in the solution archive
2. **Smarter model allocation** -- cheap models for refinement, expensive models only for paradigm shifts

## System Design

DSPy optimizes mutation prompts once up front, then producer workers sample parents from the CVT-MAP-Elites archive via LiteLLM, push candidate code through an asyncio queue, and consumer workers evaluate each candidate in a sandboxed subprocess. The archive only accepts improvements per behavioral niche. Punctuated Equilibrium periodically triggers paradigm shifts, and a budget manager shuts everything down when the dollar/eval/time limit is hit.

![LEVI System Overview](/images/levi_overview.png)

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

A one-time **DSPy MIPROv2** pass tunes mutation prompts per model. The metric rewards compilable, score-improving code and penalizes overly prescriptive prompts. Cached as JSON for **~$0.60**.

### Async Pipeline

**N producer workers** sample parents, call LLMs, push code to an asyncio queue. **M consumer workers** evaluate in sandboxed subprocesses (`ResilientProcessPool` with hard SIGKILL timeouts). Archive access is lock-protected but contention stays low since LLM calls and evaluations happen outside the lock.

**Budget enforcement is by-construction**: the pipeline checks dollars, eval count, and wall time before every LLM call and shuts down cleanly when any limit is hit.

## Putting It Together

A typical LEVI run: **$4.50 total** ($0.60 prompt optimization + $3.90 evolution), 12 LLM workers, 50 eval workers, punctuated equilibrium every 10 evaluations. Small models (e.g. Qwen-30B) handle 90%+ of mutations at fractions of a cent each. Gemini Flash handles paradigm shifts at a few cents each.
