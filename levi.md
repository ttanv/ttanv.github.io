---
layout: blog
title: "LEVI: LLM-Based Optimization at a Fraction of the Cost"
subtitle: "A harness-first framework for LLM-guided evolutionary search. Highest ADRS scores at 1.5--6.5x lower cost."
date: 2026-03-01
permalink: /levi
---

Code: [github.com/ttanv/levi](https://github.com/ttanv/levi)

<nav class="blog-toc" id="blog-toc">
  <ul>
    <li><a href="#key-insight-invest-in-the-harness-not-the-model">Key Insight</a></li>
    <li><a href="#stratified-model-allocation">Stratified Model Allocation</a></li>
    <li><a href="#improved-diversity-maintenance">Improved Diversity Maintenance</a>
      <ul>
        <li><a href="#getting-started-with-levi">Getting Started</a></li>
      </ul>
    </li>
    <li><a href="#adrs-benchmark-results">ADRS Results</a>
      <ul>
        <li><a href="#cost">Cost</a></li>
        <li><a href="#controlled-architecture-comparison">Controlled Comparison</a></li>
      </ul>
    </li>
    <li><a href="#lessons-and-looking-forward">Lessons</a></li>
  </ul>
</nav>

<img src="/results/txn_scheduling.png" alt="Controlled comparison: LEVI vs OpenEvolve vs GEPA on Transaction Scheduling, same model, same budget" style="max-width:600px;width:100%;height:auto;margin:1rem 0;">
<p class="figure-caption">Controlled comparison on Transaction Scheduling. Same model (Qwen3-30B-A3B), same budget (750 evaluations), three seeds.</p>

<style>

.figure-caption {
  color: var(--text-secondary);
  font-size: 14px;
  line-height: 1.6;
  margin-top: -1rem;
  margin-bottom: 2rem;
  font-style: italic;
}
.sidenote-ref {
  font-size: 0.75em;
  vertical-align: super;
  line-height: 0;
  color: var(--text-secondary);
  cursor: default;
}
.sidenote {
  float: right;
  clear: right;
  width: 200px;
  margin-right: -240px;
  margin-top: 0;
  margin-bottom: 1rem;
  font-size: 0.82rem;
  line-height: 1.5;
  color: var(--text-secondary);
}
.sidenote .sn-num {
  font-size: 0.75em;
  vertical-align: super;
  line-height: 0;
  margin-right: 2px;
}
.sidenote a {
  color: var(--text-secondary);
  text-decoration: underline;
  text-underline-offset: 2px;
  font-size: inherit !important;
}
.sidenote a:hover {
  color: var(--heading-color);
}
@media screen and (max-width: 1100px) {
  .sidenote {
    float: none;
    width: 100%;
    margin: 0.5rem 0 1rem 0;
    padding-left: 1rem;
    border-left: 2px solid var(--border-color);
  }
}
</style>

This blog introduces LEVI: an LLM-based evolutionary framework that produces SOTA performances on [ADRS](https://ucbskyadrs.github.io/) problems at a fraction of the cost. It is built on the key insight that too many frameworks assume access to the largest SOTA models, and build their harnesses around them.

## Key Insight: Invest in the Harness, Not the Model

Assuming access to the largest models should not be the default. In fact, the original [FunSearch](https://www.nature.com/articles/s41586-023-06924-6) paper reported being unable to benefit from larger models, and only with [AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) did they succeed. The open-source community often misses this, throwing the strongest models at every step.<span class="sidenote-ref">1,2</span><span class="sidenote"><span class="sn-num">1</span> <a href="https://github.com/algorithmicsuperintelligence/openevolve/blob/main/examples/circle_packing/config_phase_1_anthropic.yaml">OpenEvolve config</a>: uses Claude Opus for mutations.</span><span class="sidenote"><span class="sn-num">2</span> <a href="https://arxiv.org/pdf/2509.19349">ShinkaEvolve</a> (Ye et al, 2025): relies on frontier-scale models throughout the search.</span> LEVI takes a harness-first approach instead, through two key components: **stratified model allocation** and **improved diversity maintenance.**

<div class="diagram">
  <div class="levi-arch-diagram" id="levi-arch-diagram">
    <div class="levi-arch-tooltip" id="levi-arch-tooltip">
      <div class="tt-title"></div>
      <div class="tt-body"></div>
    </div>

    <svg width="880" height="440" viewBox="0 0 880 440" fill="none" xmlns="http://www.w3.org/2000/svg" aria-label="LEVI System Overview Architecture Diagram" role="img">
      <defs>
        <marker id="levi-arch-arr" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
          <polygon points="0 0.8, 7 3, 0 5.2" fill="var(--levi-arch-arrow-head)"/>
        </marker>
        <symbol id="levi-arch-star" viewBox="0 0 20 20">
          <polygon points="10,2 12.4,7.5 18,8 13.8,12 15,18 10,15 5,18 6.2,12 2,8 7.6,7.5" fill="#ffffff"/>
        </symbol>
      </defs>

      <!-- Producer Workers -->
      <rect x="50" y="20" width="340" height="58" rx="7" fill="var(--levi-arch-node-fill)" stroke="var(--levi-arch-node-stroke)" stroke-width="1"/>
      <text x="220" y="44" text-anchor="middle" fill="var(--levi-arch-text-primary)" font-size="14" font-weight="600">N Producer Workers</text>
      <text x="220" y="62" text-anchor="middle" fill="var(--levi-arch-text-secondary)" font-size="11" font-weight="300">(LLMs: LiteLLM)</text>
      <rect class="levi-arch-hover-zone" data-tip="producers" x="50" y="20" width="340" height="58" rx="7"/>

      <line x1="220" y1="78" x2="220" y2="114" stroke="var(--levi-arch-edge)" stroke-width="1.2" marker-end="url(#levi-arch-arr)"/>
      <text x="232" y="100" fill="var(--levi-arch-text-tertiary)" font-size="11" font-weight="300">Push Code</text>

      <!-- Code Queue -->
      <rect x="50" y="116" width="340" height="58" rx="7" fill="var(--levi-arch-node-fill)" stroke="var(--levi-arch-node-stroke)" stroke-width="1"/>
      <text x="220" y="140" text-anchor="middle" fill="var(--levi-arch-text-primary)" font-size="14" font-weight="600">Code Queue</text>
      <text x="220" y="158" text-anchor="middle" fill="var(--levi-arch-text-secondary)" font-size="11" font-weight="300">(Asyncio)</text>
      <rect class="levi-arch-hover-zone" data-tip="queue" x="50" y="116" width="340" height="58" rx="7"/>

      <line x1="220" y1="174" x2="220" y2="210" stroke="var(--levi-arch-edge)" stroke-width="1.2" marker-end="url(#levi-arch-arr)"/>
      <text x="232" y="196" fill="var(--levi-arch-text-tertiary)" font-size="11" font-weight="300">Pull Code</text>

      <!-- Consumer Workers -->
      <rect x="50" y="212" width="340" height="78" rx="7" fill="var(--levi-arch-node-fill)" stroke="var(--levi-arch-node-stroke)" stroke-width="1"/>
      <text x="220" y="238" text-anchor="middle" fill="var(--levi-arch-text-primary)" font-size="14" font-weight="600">M Consumer Workers</text>
      <text x="220" y="256" text-anchor="middle" fill="var(--levi-arch-text-secondary)" font-size="11" font-weight="300">(Evaluation in Subprocess)</text>
      <rect x="110" y="264" width="220" height="20" rx="3" fill="var(--levi-arch-subnode-fill)" stroke="var(--levi-arch-subnode-stroke)" stroke-width="0.7"/>
      <text x="220" y="278" text-anchor="middle" fill="var(--levi-arch-text-tertiary)" font-size="10" font-weight="300">exec + score_fn, Hard Timeout</text>
      <rect class="levi-arch-hover-zone" data-tip="consumers" x="50" y="212" width="340" height="78" rx="7"/>

      <line x1="220" y1="290" x2="220" y2="336" stroke="var(--levi-arch-edge)" stroke-width="1.2" marker-end="url(#levi-arch-arr)"/>

      <!-- Budget Manager -->
      <rect x="70" y="338" width="300" height="58" rx="7" fill="var(--levi-arch-node-fill)" stroke="var(--levi-arch-node-stroke)" stroke-width="1"/>
      <text x="220" y="362" text-anchor="middle" fill="var(--levi-arch-text-primary)" font-size="14" font-weight="600">Budget Manager</text>
      <text x="220" y="380" text-anchor="middle" fill="var(--levi-arch-text-secondary)" font-size="11" font-weight="300">(Monitors $, #evals, time)</text>
      <rect class="levi-arch-hover-zone" data-tip="budget" x="70" y="338" width="300" height="58" rx="7"/>

      <!-- Connection: Producers → Archive (Sample Parents) -->
      <line x1="390" y1="49" x2="538" y2="49" stroke="var(--levi-arch-edge)" stroke-width="1.2" marker-end="url(#levi-arch-arr)"/>
      <text x="420" y="41" fill="var(--levi-arch-text-tertiary)" font-size="11" font-weight="300">Sample Parents</text>

      <!-- Connection: Consumers → Archive (Update) -->
      <polyline points="390,245 490,245 490,180 538,180" stroke="var(--levi-arch-edge)" stroke-width="1.2" marker-end="url(#levi-arch-arr)" fill="none"/>
      <text x="478" y="206" text-anchor="end" fill="var(--levi-arch-text-tertiary)" font-size="11" font-weight="300">Update if</text>
      <text x="478" y="220" text-anchor="end" fill="var(--levi-arch-text-tertiary)" font-size="11" font-weight="300">Score &gt; Existing</text>

      <!-- Connection: Budget → Paradigm Shifts (STOP) -->
      <line x1="370" y1="367" x2="538" y2="367" stroke="var(--levi-arch-edge)" stroke-width="1.2" marker-end="url(#levi-arch-arr)"/>
      <text x="392" y="359" fill="var(--levi-arch-text-tertiary)" font-size="11" font-weight="300">STOP signal on exhaustion</text>

      <!-- Connection: Paradigm Shifts → Archive -->
      <line x1="690" y1="340" x2="690" y2="310" stroke="var(--levi-arch-edge)" stroke-width="1.2" marker-end="url(#levi-arch-arr)"/>

      <!-- CVT-MAP-Elites Archive -->
      <rect x="540" y="10" width="320" height="298" rx="9" fill="var(--levi-arch-node-fill)" stroke="var(--levi-arch-node-stroke)" stroke-width="1"/>
      <text x="700" y="38" text-anchor="middle" fill="var(--levi-arch-text-primary)" font-size="14" font-weight="600">CVT-MAP-Elites Archive</text>

      <g transform="translate(565, 50)">
        <rect width="270" height="210" rx="5" fill="var(--levi-arch-voronoi-bg)" stroke="var(--levi-arch-subnode-stroke)" stroke-width="0.7"/>

        <polygon points="0,0 70,0 59,53 43,74 0,63" fill="#6366f1" stroke="var(--levi-arch-voronoi-bg)" stroke-width="2"/>
        <polygon points="70,0 140,0 151,42 119,79 59,53" fill="#10b981" stroke="var(--levi-arch-voronoi-bg)" stroke-width="2"/>
        <polygon points="140,0 210,0 216,58 173,74 151,42" fill="#10b981" stroke="var(--levi-arch-voronoi-bg)" stroke-width="2"/>
        <polygon points="210,0 270,0 270,63 238,79 216,58" fill="#ef4444" stroke="var(--levi-arch-voronoi-bg)" stroke-width="2"/>

        <polygon points="0,63 43,74 54,126 32,147 0,137" fill="#6366f1" stroke="var(--levi-arch-voronoi-bg)" stroke-width="2"/>
        <polygon points="43,74 59,53 119,79 130,116 54,126" fill="#f59e0b" stroke="var(--levi-arch-voronoi-bg)" stroke-width="2"/>
        <polygon points="119,79 151,42 173,74 184,126 130,116" fill="#6366f1" stroke="var(--levi-arch-voronoi-bg)" stroke-width="2"/>
        <polygon points="173,74 216,58 238,79 270,63 270,137 227,131 184,126" fill="#ef4444" stroke="var(--levi-arch-voronoi-bg)" stroke-width="2"/>

        <polygon points="0,137 32,147 54,126 65,168 43,210 0,210" fill="#10b981" stroke="var(--levi-arch-voronoi-bg)" stroke-width="2"/>
        <polygon points="54,126 130,116 140,168 65,168" fill="#10b981" stroke="var(--levi-arch-voronoi-bg)" stroke-width="2"/>
        <polygon points="130,116 184,126 194,174 140,168" fill="#f59e0b" stroke="var(--levi-arch-voronoi-bg)" stroke-width="2"/>
        <polygon points="184,126 227,131 270,137 270,210 216,210 194,174" fill="#10b981" stroke="var(--levi-arch-voronoi-bg)" stroke-width="2"/>

        <polygon points="65,168 140,168 130,210 43,210" fill="#6366f1" stroke="var(--levi-arch-voronoi-bg)" stroke-width="2"/>
        <polygon points="140,168 194,174 216,210 130,210" fill="#ef4444" stroke="var(--levi-arch-voronoi-bg)" stroke-width="2"/>

        <use href="#levi-arch-star" x="19" y="17" width="18" height="18"/>
        <use href="#levi-arch-star" x="83" y="19" width="18" height="18"/>
        <use href="#levi-arch-star" x="162" y="13" width="18" height="18"/>
        <use href="#levi-arch-star" x="229" y="17" width="18" height="18"/>
        <use href="#levi-arch-star" x="14" y="89" width="18" height="18"/>
        <use href="#levi-arch-star" x="77" y="72" width="18" height="18"/>
        <use href="#levi-arch-star" x="147" y="72" width="18" height="18"/>
        <use href="#levi-arch-star" x="217" y="87" width="18" height="18"/>
        <use href="#levi-arch-star" x="19" y="152" width="18" height="18"/>
        <use href="#levi-arch-star" x="82" y="131" width="18" height="18"/>
        <use href="#levi-arch-star" x="151" y="135" width="18" height="18"/>
        <use href="#levi-arch-star" x="217" y="156" width="18" height="18"/>
        <use href="#levi-arch-star" x="76" y="174" width="18" height="18"/>
        <use href="#levi-arch-star" x="160" y="172" width="18" height="18"/>

        <text x="26" y="65" fill="var(--levi-arch-text-cell)" font-size="11" font-weight="400">c0</text>
        <text x="89" y="60" fill="var(--levi-arch-text-cell)" font-size="11" font-weight="400">c1</text>
        <text x="162" y="60" fill="var(--levi-arch-text-cell)" font-size="11" font-weight="400">c2</text>
        <text x="212" y="51" fill="var(--levi-arch-text-cell-muted)" font-size="12" font-weight="400">···</text>
        <text x="234" y="65" fill="var(--levi-arch-text-cell)" font-size="11" font-weight="400">c7</text>
      </g>

      <text x="700" y="280" text-anchor="middle" fill="var(--levi-arch-text-secondary)" font-size="11" font-weight="300">Grid of Behavioral Niches</text>
      <text x="700" y="295" text-anchor="middle" fill="var(--levi-arch-text-secondary)" font-size="11" font-weight="300">(K-means Centroids)</text>

      <rect class="levi-arch-hover-zone" data-tip="archive" x="540" y="10" width="320" height="298" rx="9"/>

      <!-- Paradigm Shifts -->
      <rect x="540" y="330" width="300" height="78" rx="7" fill="var(--levi-arch-node-fill)" stroke="var(--levi-arch-node-stroke)" stroke-width="1"/>
      <text x="690" y="358" text-anchor="middle" fill="var(--levi-arch-text-primary)" font-size="14" font-weight="600">Paradigm Shifts</text>
      <text x="690" y="376" text-anchor="middle" fill="var(--levi-arch-text-secondary)" font-size="11" font-weight="300">(Every K evals:</text>
      <text x="690" y="392" text-anchor="middle" fill="var(--levi-arch-text-secondary)" font-size="11" font-weight="300">New Strategies &amp; Variants)</text>
      <rect class="levi-arch-hover-zone" data-tip="pe" x="540" y="330" width="300" height="78" rx="7"/>
    </svg>
  </div>
</div>

<script>
(function () {
  const root = document.getElementById('levi-arch-diagram');
  if (!root) return;

  const descriptions = {
    producers: {
      title: "N Producer Workers (LLMs: LiteLLM)",
      body: "N concurrent workers that sample parent solutions from the archive and call LLMs (via LiteLLM) to generate new or mutated code. Running multiple producers in parallel maximizes throughput and exploration."
    },
    queue: {
      title: "Code Queue (Asyncio)",
      body: "An async buffer decoupling producers from consumers. Production and evaluation run at independent rates, preventing either side from blocking the other."
    },
    consumers: {
      title: "M Consumer Workers (Evaluation in Subprocess)",
      body: "M workers that pull code from the queue, execute it in isolated subprocesses with hard timeouts, and run scoring functions. Sandboxed execution ensures buggy generated code cannot crash the system."
    },
    budget: {
      title: "Budget Manager (Monitors $, #evals, time)",
      body: "Tracks LLM API cost, evaluation count, and wall-clock time. When any budget is exhausted, it sends a STOP signal to gracefully shut down the search loop."
    },
    archive: {
      title: "CVT-MAP-Elites Archive",
      body: "The solution space is partitioned into behavioral niches via Centroidal Voronoi Tessellation. Each niche keeps only its highest-scoring solution, maintaining diversity rather than collapsing to a single optimum."
    },
    pe: {
      title: "Paradigm Shifts",
      body: "Every K evaluations, triggers large-scale mutations or entirely new strategies to escape local optima, alternating incremental refinement with radical changes to avoid stagnation."
    }
  };

  const tooltip = document.getElementById('levi-arch-tooltip');
  if (!tooltip) return;
  const ttTitle = tooltip.querySelector('.tt-title');
  const ttBody = tooltip.querySelector('.tt-body');

  root.querySelectorAll('.levi-arch-hover-zone').forEach((zone) => {
    zone.addEventListener('mouseenter', () => {
      const key = zone.getAttribute('data-tip');
      const desc = descriptions[key];
      if (!desc) return;
      ttTitle.textContent = desc.title;
      ttBody.textContent = desc.body;
      tooltip.classList.add('visible');
    });

    zone.addEventListener('mousemove', (event) => {
      const pad = 16;
      const ttW = tooltip.offsetWidth;
      const ttH = tooltip.offsetHeight;
      let x = event.clientX + pad;
      let y = event.clientY + pad;

      if (x + ttW > window.innerWidth - pad) {
        x = event.clientX - ttW - pad;
      }
      if (y + ttH > window.innerHeight - pad) {
        y = event.clientY - ttH - pad;
      }

      tooltip.style.left = x + 'px';
      tooltip.style.top = y + 'px';
    });

    zone.addEventListener('mouseleave', () => {
      tooltip.classList.remove('visible');
    });
  });
})();
</script>

<p class="figure-caption">LEVI's architecture: diverse seeds initialize a CVT-MAP-Elites archive; smaller models handle most mutations; a frontier model injects paradigm shifts every K evaluations. Hover over each component for details.</p>

## Stratified Model Allocation

Frontier models help, but **they are a waste if used for every mutation**. Smaller LLMs may actually be preferred under tight budgets, since the sheer quantity of solutions they produce can outweigh the quality advantage of larger models. However, smaller models have a narrower pretraining distribution, limiting their range of ideas and ability to propose fundamentally different approaches. Neither model class is strictly better; they just have different strengths.

Some existing frameworks already support multiple models, but treat them as interchangeable, sampling from an ensemble uniformly or routing calls without regard to what the mutation actually demands. This ignores a natural asymmetry: proposing an entirely new algorithmic direction requires broad knowledge and creative reasoning, while refining an existing approach (adjusting constants, reordering operations, tuning edge cases) requires far less. The harness should be aware of this distinction and allocate accordingly.

LEVI introduces stratified model allocation, which matches model capacity to task demand. Smaller, cheaper models handle the majority of the search: local refinements and incremental improvements within an established algorithmic family. Larger models are reserved for infrequent *paradigm shifts*: mutations that aim to propose structurally different approaches rather than polish existing ones. The principle is straightforward: allocate each model toward its strength. Small models for breadth and throughput, large models for creative leaps.

However, this raises two questions. First, how do we select representative solutions from each algorithmic family to give the larger model meaningful context for paradigm shifts? Second, since we now rely more heavily on smaller models and their volume of output, we need a more robust mechanism to prevent the archive from converging.

<div class="callout">
<div class="callout-label">Key idea</div>
<p>LEVI matches model capacity to task demand: cheap models (e.g. a local Qwen 30B) for refinement, expensive models for paradigm shifts.</p>
</div>

## Improved Diversity Maintenance

**Unifying Structural and Behavioral Diversity.** A less obvious reason existing frameworks require frontier models is that those models are doing double duty. Their larger output space implicitly maintains diversity: a GPT-5 or Claude Opus naturally produces a wider spread of solutions than a 30B model, ignoring the fact that the archive itself has no strong mechanism to prevent convergence. When diversity does collapse, the response has been to add complexity on top: ranging from rejection sampling using even more LLM calls to using embedding models. These are compensations for a weak foundation, not solutions to the underlying issue.

The underlying issue is that existing frameworks maintain diversity along **only one axis**, and a narrow one at that. OpenEvolve considers structural features like code length; GEPA considers per-instance performance trade-offs through Pareto fronts (in practice often more powerful than the former mechanism). Both capture something real, but neither captures the full picture. Structure alone misses behavioral differences: two programs with different loop counts might solve the problem identically. And per-instance scores alone miss solutions that perform similarly on individual instances but work in fundamentally different ways.

Rather than choosing one axis, LEVI uses both as dimensions of a single behavioral descriptor. Each solution is mapped to a *fingerprint*: a vector combining code-structural features (going beyond simple dimensions like code length to measures such as loop count, cyclomatic complexity) alongside per-instance behavioral results, all normalized and projected to [0, 1]. The framework is also flexible here: users can define their own dimensions when the defaults do not fit their problem.

This fingerprint lives in a CVT-MAP-Elites archive, where a Voronoi tessellation over the combined space maintains geometric structure that neither axis provides alone. The archive holds a diverse set of solutions with different values along the different dimensions. This also directly answers the first question from the previous section: the Voronoi regions naturally cluster solutions into algorithmic families, giving us representative solutions for paradigm shifts.

**Archive Initialization.** Traditional CVT-MAP-Elites initializes centroids uniformly across the descriptor space. With the higher dimensionality we use (6 to 10 dims), this leads to an extremely sparse tessellation where most regions will never be visited. LEVI adopts a **data-driven approach**: it creates a set of deliberately unique approaches (regardless of scores) through sequential generation, and then uses those to create the initial centroids. This ensures that the archive is based on solutions that are known to be different.

<div class="callout">
<div class="callout-label">Key idea</div>
<p>LEVI maintains diversity through a shared fingerprint space over both structure and behavior, so the archive itself carries more of the diversity burden instead of relying on ever-stronger models or auxiliary heuristics.</p>
</div>

### Getting Started with LEVI

Below is an example LEVI program, optimizing a dummy bin packing problem. All of the framework details are abstracted away, and the user can focus on defining the problem.

```python
import levi

def score_fn(pack):
    bins = pack([4, 8, 1, 4, 2, 1], 10)
    wasted = sum(10 - sum(b) for b in bins)
    return {"score": max(0.0, 100.0 - wasted)}

result = levi.evolve_code(
    "Optimize bin packing to minimize wasted space",
    function_signature="def pack(items, bin_capacity):",
    score_fn=score_fn,
    model="openai/gpt-4o-mini",
    budget_dollars=2.0,
)
```

Try it out at [github.com/ttanv/levi](https://github.com/ttanv/levi)!



## ADRS Benchmark Results

We evaluate on the ADRS benchmark suite,<span class="sidenote-ref">3</span><span class="sidenote"><span class="sn-num">3</span> <a href="https://ucbskyadrs.github.io/">ADRS</a> (Cheng et al, 2025): benchmark suite from UC Berkeley for LLM-guided optimization on real-world systems problems.</span> a set of real-world systems problems spanning cloud scheduling, load balancing, SQL optimization, and transaction scheduling.

<div class="adrs-table-card">
  <table class="adrs-table" id="adrs-table">
  <thead>
    <tr>
      <th class="adrs-sticky">Framework</th>
      <th class="adrs-average">Avg</th>
      <th>Cloud</th>
      <th>EPLB</th>
      <th>SQL</th>
      <th>Prism</th>
      <th>Spot-M</th>
      <th>Spot-S</th>
      <th>Txn</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="adrs-sticky">
        <div class="adrs-fw">
          <span class="adrs-swatch swatch-gepa"></span>
          <span class="adrs-fw-name">GEPA</span>
        </div>
      </td>
      <td class="adrs-average"><span class="adrs-val">71.9</span></td>
      <td><span class="adrs-val">96.6</span></td>
      <td><span class="adrs-val">70.2</span></td>
      <td><span class="adrs-val">67.7</span></td>
      <td><span class="adrs-val is-best">87.4</span></td>
      <td><span class="adrs-val">62.2</span></td>
      <td><span class="adrs-val">51.4</span></td>
      <td><span class="adrs-val">67.7</span></td>
    </tr>
    <tr>
      <td class="adrs-sticky">
        <div class="adrs-fw">
          <span class="adrs-swatch swatch-openevolve"></span>
          <span class="adrs-fw-name">OpenEvolve</span>
        </div>
      </td>
      <td class="adrs-average"><span class="adrs-val">70.6</span></td>
      <td><span class="adrs-val">92.9</span></td>
      <td><span class="adrs-val">62.0</span></td>
      <td><span class="adrs-val">72.5</span></td>
      <td><span class="adrs-val is-best">87.4</span></td>
      <td><span class="adrs-val">66.7</span></td>
      <td><span class="adrs-val">42.5</span></td>
      <td><span class="adrs-val">70.0</span></td>
    </tr>
    <tr>
      <td class="adrs-sticky">
        <div class="adrs-fw">
          <span class="adrs-swatch swatch-shinka"></span>
          <span class="adrs-fw-name">ShinkaEvolve</span>
        </div>
      </td>
      <td class="adrs-average"><span class="adrs-val">67.4</span></td>
      <td><span class="adrs-val">72.0</span></td>
      <td><span class="adrs-val">66.4</span></td>
      <td><span class="adrs-val">68.5</span></td>
      <td><span class="adrs-val is-best">87.4</span></td>
      <td><span class="adrs-val">63.6</span></td>
      <td><span class="adrs-val">45.6</span></td>
      <td><span class="adrs-val">68.2</span></td>
    </tr>
    <tr class="is-levi">
      <td class="adrs-sticky">
        <div class="adrs-fw">
          <span class="adrs-swatch swatch-levi"></span>
          <span class="adrs-fw-name">LEVI</span>
        </div>
      </td>
      <td class="adrs-average"><span class="adrs-val is-best is-levi">76.5</span></td>
      <td><span class="adrs-val is-best is-levi">100.0</span></td>
      <td><span class="adrs-val is-best is-levi">74.6</span></td>
      <td><span class="adrs-val is-best is-levi">78.3</span></td>
      <td><span class="adrs-val is-best is-levi">87.4</span></td>
      <td><span class="adrs-val is-best is-levi">72.4</span></td>
      <td><span class="adrs-val is-best is-levi">51.7</span></td>
      <td><span class="adrs-val is-best is-levi">71.1</span></td>
    </tr>
  </tbody>
  </table>
</div>
<div class="adrs-note">Bold values indicate best per benchmark. Cloud = Cloudcast, SQL = LLM-SQL, Spot-M/S = Spot Multi/Single-Reg, Txn = Txn Scheduling.</div>

LEVI achieves the highest score on every problem where improvement is possible, with an average of **76.5** compared to 71.9 for the next-best framework (GEPA), a **+4.6 point improvement** over the prior state of the art. On Cloudcast, LEVI reaches a perfect 100.0, indicating the problem is fully solved under the benchmark's scoring function. The largest gains appear on LLM-SQL (+5.8) and Spot Multi (+5.7), while more modest improvements on Spot Single (+0.3) and Transaction Scheduling (+1.1) reflect problems with smaller decision spaces or harder optimization landscapes. Prism remains tied at 87.4 across all frameworks, confirming that the current problem formulation admits a single dominant solution.

### Cost

LEVI's stratified allocation is the primary driver of cost reduction. By routing the majority of mutations through lightweight models, the per-generation cost drops by roughly an order of magnitude compared to baselines that use GPT-5 or Gemini-3.0-Pro for every call. This allows LEVI to run substantially more generations while still spending less in total: $4.50 per problem on most tasks (Transaction Scheduling: $13), versus $15 to $30 for baselines.

<div class="adrs-table-card">
  <table class="adrs-table">
  <thead>
    <tr>
      <th class="adrs-sticky">Problem</th>
      <th>Baseline</th>
      <th>LEVI</th>
      <th>Savings</th>
    </tr>
  </thead>
  <tbody>
    <tr><td class="adrs-sticky"><span class="adrs-fw-name">Spot Single-Reg</span></td><td><span class="adrs-val">$30</span></td><td><span class="adrs-val is-best is-levi">$4.50</span></td><td><span class="adrs-val is-best is-levi">6.7x</span></td></tr>
    <tr><td class="adrs-sticky"><span class="adrs-fw-name">Spot Multi-Reg</span></td><td><span class="adrs-val">$25</span></td><td><span class="adrs-val is-best is-levi">$4.50</span></td><td><span class="adrs-val is-best is-levi">5.6x</span></td></tr>
    <tr><td class="adrs-sticky"><span class="adrs-fw-name">LLM-SQL</span></td><td><span class="adrs-val">$20</span></td><td><span class="adrs-val is-best is-levi">$4.50</span></td><td><span class="adrs-val is-best is-levi">4.4x</span></td></tr>
    <tr><td class="adrs-sticky"><span class="adrs-fw-name">Cloudcast</span></td><td><span class="adrs-val">$15</span></td><td><span class="adrs-val is-best is-levi">$4.50</span></td><td><span class="adrs-val is-best is-levi">3.3x</span></td></tr>
    <tr><td class="adrs-sticky"><span class="adrs-fw-name">Prism</span></td><td><span class="adrs-val">$15</span></td><td><span class="adrs-val is-best is-levi">$4.50</span></td><td><span class="adrs-val is-best is-levi">3.3x</span></td></tr>
    <tr><td class="adrs-sticky"><span class="adrs-fw-name">EPLB</span></td><td><span class="adrs-val">$15</span></td><td><span class="adrs-val is-best is-levi">$4.50</span></td><td><span class="adrs-val is-best is-levi">3.3x</span></td></tr>
    <tr><td class="adrs-sticky"><span class="adrs-fw-name">Txn Scheduling</span></td><td><span class="adrs-val">$20</span></td><td><span class="adrs-val is-best is-levi">$13</span></td><td><span class="adrs-val is-best is-levi">1.5x</span></td></tr>
  </tbody>
  </table>
</div>

The cost reduction is evidence that the harness-first approach works. **When the archive maintains diversity, cheap models suffice for most of the search.**

### Controlled Architecture Comparison

Same model, same budget, three seeds: isolating the search architecture's contribution.

The main results compare frameworks that differ simultaneously in model choice, budget, and architecture. To isolate the contribution of the search architecture, we run LEVI, OpenEvolve, and GEPA under identical conditions: a single locally-served Qwen3-30B-A3B model, 750 successful evaluations,<span class="sidenote-ref">4</span><span class="sidenote"><span class="sn-num">4</span> OpenEvolve required reducing parent count from 5 to 2 for the smaller model and still produced many failures. We report successful evaluations rather than total to give OpenEvolve a fair comparison.</span> and three random seeds on two representative problems.

**Transaction Scheduling** is a variant of an NP-hard ordering problem where multiple algorithmic families (greedy, simulated annealing, genetic) are viable but performance is measured on a single instance, giving Pareto-based diversity no trade-off to exploit. LEVI reaches a score of 62 within the first 100 evaluations, a level neither baseline achieves at any point. Final scores: LEVI 64.9, OpenEvolve 59.9, GEPA 54.4. Both baselines plateau sharply, consistent with early convergence onto a single algorithmic family; LEVI's curve continues rising past evaluation 500.

<img src="/results/txn_scheduling.png" alt="Transaction Scheduling controlled comparison plot" style="max-width:100%;height:auto;margin:1.5rem 0;">

<p class="figure-caption">Transaction Scheduling, controlled (Qwen3-30B, 750 evals, 3 seeds). LEVI hits 62 within 100 evals; neither baseline reaches that level at any point.</p>

**Can't Be Late** is scored across 1,080 simulations that give Pareto-based approaches a richer signal. The final-score gap narrows (LEVI 44.9, OpenEvolve 43.2, GEPA 37.5), but the efficiency gap widens dramatically. LEVI reaches near-peak performance by roughly evaluation 50, while OpenEvolve requires over 600 evaluations to approach the same level, a roughly 12x advantage in sample efficiency.

<img src="/results/cant_be_late.png" alt="Can't Be Late controlled comparison plot" style="max-width:100%;height:auto;margin:1.5rem 0;">

<p class="figure-caption">Can't Be Late, same controlled setup. LEVI reaches near-peak by eval 50, roughly 12x faster than OpenEvolve.</p>

<div class="callout">
<div class="callout-label">Key finding</div>
<p>Same model, same budget: the performance gains come from the search architecture, not model choice. A 30B model under LEVI matches or exceeds what the same model achieves under alternative selection mechanisms.</p>
</div>

## Lessons and Looking Forward

Working with smaller models surfaces real tradeoffs that frameworks built around frontier models never have to confront:

- **Higher error rates, but cheaper retries.** Smaller models produce broken code more often, but the calls are so cheap that you can afford many more attempts and still come out ahead on total spend.
- **Reward hacking.** Smaller models are more susceptible to exploiting evaluator weaknesses rather than genuinely solving the problem. But this is an evaluator problem as much as a model problem, and fixing evaluators benefits every framework.
- **Code over text.** When expressing a useful idea for a smaller model to work with, code beats natural language. A prompt saying "try simulated annealing" leaves enormous room for interpretation; a code skeleton implementing the acceptance criterion and cooling schedule gives the model something concrete. This is why LEVI's paradigm shift step generates code skeletons rather than text suggestions.
- **Quantity vs. eval time.** The core advantage of smaller models is volume: as shown above, more cheap calls can outperform fewer expensive ones. But this advantage depends on evaluations being fast. For problems where a single eval takes an hour, every call is precious and larger models become more sensible. LEVI mitigates this for most problems through an async distributed producer-consumer model, but for long-eval domains this is a different dimension of tradeoff worth considering.

**More benchmarks and domains are in progress.** ADRS is a first validation, not the full story. Try LEVI at [github.com/ttanv/levi](https://github.com/ttanv/levi).
