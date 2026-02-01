---
layout: blog
title: "LEVI: Budget-Efficient LLM-Guided Evolution"
date: 2024-06-01
permalink: /levi
---

## TLDR

Existing LLM-based evolutionary systems (OpenEvolve, ShinkaEvolve) have weak diversity mechanisms, converge early, and compensate by throwing expensive models at the problem. This leads to bloated costs and unnecessary complexity: rejection sampling, embedding-based deduplication, LLM-as-judge filters.

We fix the root cause. LEVI uses CVT-MAP-Elites with AST-based behavioral fingerprinting (loop depth, branch count, cyclomatic complexity, etc.) to maintain a diverse archive where each cell holds the single best solution for its behavioral niche. Different algorithmic approaches naturally land in different cells, so the system never collapses onto one strategy. On top of this, we split the mutation process into two tiers: cheap small models (MiMo, Qwen-30B, MiniMax) making hundreds of narrow mutations, and a larger model (Gemini Flash) used sparingly for paradigm shifts that jump to entirely new regions of the solution space.

With this setup we get ***better*** results than OpenEvolve, ShinkaEvolve, and GEPA on ADRS benchmarks (Cloudcast, EPLB) at 1/3rd the budget. The improvement holds across prompt optimization, heuristic design, and general coding problems.

**LEVI is open-source at github.com.** Async producer-consumer pipeline, subprocess sandboxing, built-in dollar-budget enforcement. Point it at a scoring function and a seed program and it runs until the budget is spent.

### ADRS Benchmark Results (% score)

| Framework | Average | Cloudcast | LLM-SQL | Prism | Spot Multi-Reg | Spot Single-Reg | Txn Scheduling |
|---|---|---|---|---|---|---|---|
| Human SOTA | 61.7 | 100.0 | 67.7 | 60.8 | 54.5 | 45.1 | 41.9 |
| AutoEvolve | 74.9 | 97.8 | 76.4 | 87.4 | 70.0 | 46.3 | 70.6 |
| GEPA | 72.7 | 96.6 | 67.7 | 87.4 | 62.2 | 51.4 | 67.7 |
| OpenEvolve | 72.0 | 92.9 | 72.5 | 87.4 | 66.7 | 42.5 | 70.0 |
| ShinkaEvolve | 67.5 | 72.0 | 68.5 | 87.4 | 63.6 | 45.6 | 68.2 |
| **LEVI** | **76.7** | **100.0** | **78.3** | **87.4** | **72.4** | **51.7** | **70.4** |

### Cost per Problem

| Problem | ADRS Baseline Cost | LEVI Cost | Reduction |
|---|---|---|---|
| Cloudcast | ≤$15 | $4.50 | 3.3x |
| LLM-SQL | ≤$20 | $4.50 | 4.4x |
| Prism | ≤$15 | $4.50 | 3.3x |
| Spot Multi-Reg | ≤$25 | $4.50 | 5.6x |
| Spot Single-Reg | ≤$30 | $4.50 | 6.7x |
| Txn Scheduling | ≤$20 | $4.50 | 4.4x |

LEVI uses a flat $4.50 per problem versus the ADRS baselines' $15-$30 range, yielding 3-7x cost reductions while achieving higher scores.

## What is lacking in existing systems

The current open-source ecosystem is focused on optimizing fundamentally incorrect parts of the LLM-based evolution stack; leading to increased costs and a higher barrier of entry for researchers with limited resources. They try using unnecessarily large LLMs, add additional costs through using LLM-based judges or extract embeddings, and converge too soon to yield a truly powerful result. Their usage of large LLMs like Claude Sonnet 4.5, Gemini 3.0 Pro, GPT 5.2 is not a problem in of itself, but rather a byproduct of the underlying problems. If your evolutionary system stagnates within a few hundred or so generations, it is no surprise you resort to using the strongest models, you want to make the most out of those few generations.

But you may ask: how do you know a few hundred generations is not enough? Why would a thousand or 10 thousand be enough? That's a very good question. Unfortunately we don't really have some elaborate study on this to clearly understand it. The closest thing we have is Deepmind's FunSearch and AlphaEvolve systems (not many other players can just burn through enough cash to answer these questions), where funsearch needed on the order of a million generations, and AlphaEvolve needed on the order of thousands. Furthermore, FunSearch originally claimed that their system does not even benefit from larger LLMs! It is only around AlphaEvolve that they were able to harness them.

But I am not entirely sure it is the fault of these systems or the users using them. Even if the systems themselves survive stagnation and can run for much longer, works that use these systems are also seem to have this notion of using the strongest model for every generation and burning through cash.

Thus I will try to dismiss the following notions with this work:

- LLM-Based evolution needs to be very expensive
    - We will show it can be done with 1/3rd to 1/10th of the budget of existing works
- It can only be done with SOTA models
    - We will mainly use smaller 30B-100B models mainly and occasionally use models like Gemini Flash 3

## How LEVI aims to fix them

We improve on existing works in two core places:

- The diversity maintenance in the pool of solutions tracked so far.
- How to best extract the most benefit from larger models without ballooning the costs.

## LEVI System Design

![LEVI System Overview](/images/levi_overview.png)

```
                              LEVI System Overview
 ┌──────────────────────────────────────────────────────────────────────────┐
 │                                                                          │
 │   ┌─────────────┐        ┌──────────────────────────────────────────┐   │
 │   │  DSPy Prompt │        │         CVT-MAP-Elites Archive          │   │
 │   │ Optimization │        │                                          │   │
 │   │  (one-time)  │        │  ┌────┬────┬────┬────┬────┬────┬────┐   │   │
 │   └──────┬───────┘        │  │ c0 │ c1 │ c2 │    │ c5 │    │ c7 │   │   │
 │          │                │  │★83 │★71 │★69 │    │★77 │    │★65 │   │   │
 │          │ optimized      │  └────┴────┴────┴────┴────┴────┴────┘   │   │
 │          │ prompts        │   Each cell = 1 centroid, keeps best     │   │
 │          ▼                │   program by score. Empty cells = open.  │   │
 │   ┌──────────────┐       └───────────────┬──────────────────────────┘   │
 │   │              │ sample parents         │ add if score > existing     │
 │   │  N Producer  │◄──────────────────────►│                             │
 │   │   Workers    │                        │                             │
 │   │              │        ┌───────────┐   │                             │
 │   │  MiMo, Qwen, │──────►│ code_queue │──►│  M Consumer Workers        │
 │   │  MiniMax,    │ push   └───────────┘   │                             │
 │   │  Gemini      │ code        pull code  │  Evaluate in subprocess     │
 │   └──────────────┘                        │  (exec + score_fn)          │
 │          ▲                                │  Hard timeout (SIGKILL)     │
 │          │                                │                             │
 │   ┌──────┴───────────────────────────┐    │                             │
 │   │   Punctuated Equilibrium         │    │                             │
 │   │   (every K evals)                │    │                             │
 │   │                                  │    │                             │
 │   │   1. Cluster archive cells       │    │                             │
 │   │   2. Pick best per cluster       │    │                             │
 │   │   3. Heavy model: paradigm shift │    │                             │
 │   │   4. Light models: variants      │    │                             │
 │   │   5. Insert with behavior noise  │    │                             │
 │   └──────────────────────────────────┘    │                             │
 │                                           │                             │
 │   ┌──────────────────────────────────┐    │                             │
 │   │   Budget Manager                 │    │                             │
 │   │   Tracks: $dollars, #evals, time │    │                             │
 │   │   Stops everything when exhaust. │    │                             │
 │   └──────────────────────────────────┘    │                             │
 └──────────────────────────────────────────────────────────────────────────┘
```

## Core Abstractions

### CVT-MAP-Elites

The archive is the heart of LEVI. MAP-Elites is a quality-diversity algorithm: instead of tracking just the single best solution, it partitions the space of possible solution *behaviors* into cells and keeps the single best-scoring program in each cell. This means the archive simultaneously maintains many high-quality solutions that are behaviorally *different* from each other. A greedy approach and a dynamic programming approach can coexist, each the best of its kind.

We use Centroidal Voronoi Tessellation (CVT) to define the cells. The behavior space is partitioned into regions via k-means centroids, and each program is assigned to its nearest centroid. If the cell is empty, the program moves in. If the cell is occupied, the new program replaces the incumbent only if it scores higher. This is the elitism rule that ensures each cell always holds the strongest solution found so far for that behavioral niche.

What defines "behavior"? We use AST-based features extracted directly from the code's structure. Things like loop nesting depth, number of comparisons, branch count, cyclomatic complexity, math operator count. These fingerprint the *shape* of an algorithm without running it. A greedy scheduler that iterates once over transactions and a DP approach with nested loops will naturally land in different cells, even before we evaluate their scores. Each problem selects its own subset of these features as behavioral dimensions, and raw values are normalized to [0,1] so k-means can operate in a consistent space.

To select parents for mutation, we use softmax sampling over the archive cells, weighted by fitness. The temperature parameter controls the exploration/exploitation tradeoff: low temperature (T=0.3) heavily favors high-scoring cells, high temperature (T=1.2) samples nearly uniformly. We run multiple temperature settings simultaneously in the same run, so the archive is being explored from multiple angles at once, some greedily exploiting the best solutions while others broadly exploring underrepresented regions.

### Archive Init

We use a data-driven initialization. A heavier model (e.g., Gemini Flash) generates N fundamentally diverse seed algorithms. The prompt explicitly instructs the model to analyze existing seeds and design something using a *completely different algorithmic paradigm*. Each new seed sees all previously generated seeds to avoid repetition. Then lighter models generate M variants per seed. All programs are evaluated, behaviors extracted, and k-means is run on the actual behavior vectors to create centroids that reflect the real distribution of algorithmic approaches.

Optionally, Gaussian noise is added to behavior vectors during init to prevent centroid overfitting to the specific code shapes seen. Once init finishes, noise is turned off for the evolution phase.

### Utilizing both smaller and larger models effectively

The core insight is that smaller and larger models serve fundamentally different roles in the search process. Smaller models (MiMo-V2-Flash, Qwen3-30B, MiniMax-M2.1) make narrow mutations: tweak a threshold, swap a sorting key, adjust a heuristic. They are cheap enough to call hundreds of times, and their mutations stay close to the parent in behavior space. This is the fine-grained hill climbing that improves solutions within their current algorithmic paradigm.

Larger models (Gemini Flash 3) are used sparingly for *paradigm shifts*: looking at the best solutions from different behavioral regions, analyzing their strategies, and proposing something fundamentally different. This is not a mutation. It's a synthesis step where the model sees the landscape of what has been tried and explicitly designs a new approach that fills the gaps.

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

The `═════►` arrows are paradigm shifts from the larger model. The `·→` arrows are narrow mutations from smaller models. After each paradigm shift, smaller models take the new solution and refine it, exploring its local neighborhood. Then another paradigm shift happens, potentially jumping to an entirely different region of the solution space.

This is implemented through **Punctuated Equilibrium**, which triggers every N evaluations (e.g., every 5 or 10). When triggered, it:

1. **Clusters** the occupied archive cells into behavioral regions using k-means on the centroid vectors.
2. **Selects representatives**: the highest-scoring elite from each cluster.
3. **Generates a paradigm shift**: sends all representatives to the heavy model with a budget-stage-aware prompt. Early in the run (<30% budget), the prompt demands radical departures. Mid-run (30-60%), it asks for synthesis of the best ideas across regions. Late (>60%), it asks for surgical refinement of the best solution's weak spots.
4. **Generates variants**: sends the paradigm shift solution to lighter models for minor improvements.
5. **Inserts with behavior noise**: paradigm shifts are added to the archive with Gaussian noise on their behavior vectors, letting them explore adjacent cells rather than being forced into a single slot.

The result is a system where 90%+ of the budget goes to cheap small-model mutations, and the remaining 5-10% goes to expensive but high-impact paradigm shifts that prevent stagnation.

### Prompt Optimization

Before evolution begins, we optionally run a DSPy-based prompt optimization pass (MIPROv2) that tunes the mutation instruction prompt for each model separately. The metric rewards prompts that produce compilable, runnable, score-improving code and penalizes prompts that are too long or too prescriptive. We want prompts that say *what* to achieve, not *how*. "Use dynamic programming with memoization" locks the model into one paradigm; "find a fundamentally better approach" lets it explore. Optimized prompts are cached as JSON (~$0.60 one-time cost) and reused across runs.

### Async Pipeline (Producer-Consumer)

The evolution runs as an async producer-consumer pipeline. N LLM producer workers sample parents from the archive, build prompts, call LLMs, extract code, and push to an asyncio queue. M evaluation consumer workers pull from the queue, evaluate code in isolated subprocesses, and update the archive.

Evaluation happens in true subprocesses via `ResilientProcessPool`. This gives us hard timeout enforcement (SIGTERM then SIGKILL), memory isolation (a malformed program can't corrupt the parent process), and true parallelism past the GIL. Each evaluation runs the candidate code via `exec()` in a fresh namespace, extracts the target function, and calls the user-provided `score_fn`.

The archive is protected by an asyncio lock. Producers acquire it briefly to sample parents; consumers acquire it briefly to update the archive after evaluation. The actual LLM calls and evaluations happen outside the lock, keeping contention low.

Budget enforcement is by-construction: the pipeline checks `state.budget_exhausted` (tracking dollars, evaluations, and wall time) before every LLM call. When the budget hits the limit, producers stop, consumers drain the queue, and the run ends cleanly with a final snapshot.

## Complete System

Putting it all together, a LEVI run proceeds as:

1. **Setup**: Initialize the LLM client (routing local and cloud models), the behavior extractor (with problem-specific AST features), and the CVT-MAP-Elites pool (with deferred centroids).

2. **Seed evaluation**: Evaluate the user-provided seed program in a subprocess. This is the baseline score.

3. **Init phase** (optional): The Diversifier generates N diverse seeds with the heavy model, then M variants per seed with lighter models. Behaviors are extracted, k-means builds centroids from the real data, and the archive is populated with 30-100+ diverse solutions.

4. **Evolution pipeline**: N producer workers and M consumer workers run concurrently. Producers sample parents via softmax at various temperatures, call LLMs, extract code. Consumers evaluate in subprocesses, update the archive. Punctuated equilibrium triggers paradigm shifts every K evaluations.

5. **Budget exhaustion**: When the dollar budget is hit, producers stop, consumers drain, and a final snapshot is saved with the full archive state, score history, and sampler statistics.

6. **Result**: The best program found across all cells, along with total cost, evaluation count, and the full score progression.

A typical run on transaction scheduling uses $4.50 in total ($0.60 for prompt optimization, $3.90 for evolution), runs 12 LLM workers and 50 eval workers concurrently, and triggers punctuated equilibrium every 10 evaluations. The light models (MiMo, Qwen, MiniMax) handle 90%+ of mutations at fractions of a cent each. Gemini Flash handles paradigm shifts at a few cents each.
