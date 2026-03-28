---
layout: blog
title: "LEVI: AlphaEvolve Performance for the Price of a Cup of Coffee"
subtitle: "A harness-first framework for LLM-guided evolutionary search. Highest ADRS scores at 1.5--6.5x lower cost."
date: 2026-03-01
permalink: /levi
---



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
.blog-footnotes {
  display: none;
  margin-top: 3rem;
  padding-top: 1.5rem;
  border-top: 1px solid var(--border-color);
  font-size: 0.85rem;
  color: var(--text-secondary);
  line-height: 1.6;
}
.blog-footnotes ol {
  padding-left: 1.25rem;
  margin: 0.5rem 0;
}
.blog-footnotes li {
  margin: 0.4rem 0;
}
.blog-footnotes a {
  color: var(--text-secondary);
  text-decoration: underline;
  text-underline-offset: 2px;
}
.blog-footnotes a:hover {
  color: var(--heading-color);
}
@media screen and (max-width: 1100px) {
  .sidenote {
    display: none;
  }
  .sidenote-ref a {
    color: var(--blog-link);
    text-decoration: none;
  }
  .blog-footnotes {
    display: block;
  }
}
</style>

<div class="callout">
<p><strong>LEVI</strong> outperforms leading algorithmic discovery frameworks (OpenEvolve, GEPA, ShinkaEvolve) at up to 1/6th the cost, saving over $100 per problem on a systems benchmark suite. It does this while routing 90%+ of mutations through a local Qwen3-30B model. In controlled comparisons with the same model and budget, LEVI reaches peak performance 12x faster than alternatives.</p>
<p>Code: <a href="https://github.com/ttanv/levi">github.com/ttanv/levi</a></p>
</div>

This blog introduces LEVI: an LLM-based evolutionary framework that produces SOTA performances on ADRS problems at a fraction of the cost. It is built on the key insight that too many frameworks assume access to the largest SOTA models, and build their harnesses around them.

## Key Insight: Invest in the Harness, Not the Model

Assuming access to the largest models should not be the default. In fact, the original [FunSearch](https://www.nature.com/articles/s41586-023-06924-6) paper reported being unable to benefit from larger models, and only with [AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) did they succeed. The open-source community often misses this, throwing the strongest models at every step.<span class="sidenote-ref"><a href="#fn1">1</a>,<a href="#fn2">2</a></span><span class="sidenote"><span class="sn-num">1</span> <a href="https://github.com/algorithmicsuperintelligence/openevolve/blob/main/examples/circle_packing/config_phase_1_anthropic.yaml">OpenEvolve config</a>: uses Claude Opus for mutations.</span><span class="sidenote"><span class="sn-num">2</span> <a href="https://arxiv.org/pdf/2509.19349">ShinkaEvolve</a> (Ye et al, 2025): relies on frontier-scale models throughout the search.</span> LEVI takes a harness-first approach instead, through two key components: **stratified model allocation** and **improved diversity maintenance.**

<div class="diagram">
  <div class="levi-arch-diagram" id="levi-arch-diagram">
    <div class="levi-arch-tooltip" id="levi-arch-tooltip">
      <div class="tt-title"></div>
      <div class="tt-body"></div>
    </div>

    <svg width="880" height="570" viewBox="0 0 880 570" fill="none" xmlns="http://www.w3.org/2000/svg" aria-label="LEVI System Overview Architecture Diagram" role="img">
      <defs>
        <marker id="levi-arch-arr" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
          <polygon points="0 0.8, 7 3, 0 5.2" fill="var(--levi-arch-arrow-head)"/>
        </marker>
      </defs>

      <!-- Init: Diverse Seed Generation -->
      <rect x="200" y="10" width="480" height="54" rx="7" fill="var(--levi-arch-node-fill)" stroke="var(--levi-arch-node-stroke)" stroke-width="1"/>
      <text x="440" y="34" text-anchor="middle" fill="var(--levi-arch-text-primary)" font-size="14" font-weight="600">Init: Diverse seed generation</text>
      <text x="440" y="52" text-anchor="middle" fill="var(--levi-arch-text-secondary)" font-size="11">LLM generates algorithmically diverse initial population</text>
      <rect class="levi-arch-hover-zone" data-tip="init" x="200" y="10" width="480" height="54" rx="7"/>

      <!-- Arrow: Init → Budget container -->
      <line x1="440" y1="64" x2="440" y2="88" stroke="var(--levi-arch-edge)" stroke-width="1.2" marker-end="url(#levi-arch-arr)"/>

      <!-- Budget Manager (top-level container) -->
      <rect x="20" y="80" width="840" height="440" rx="10" fill="none" stroke="var(--levi-arch-node-stroke)" stroke-width="1.5" stroke-dasharray="6 4"/>
      <text x="40" y="106" fill="var(--levi-arch-text-primary)" font-size="15" font-weight="600">Budget manager</text>
      <text x="40" y="124" fill="var(--levi-arch-text-secondary)" font-size="11">Gates all paths. STOP on exhaustion.</text>
      <rect class="levi-arch-hover-zone" data-tip="budget" x="20" y="80" width="840" height="50" rx="10"/>

      <!-- Mutation Producers -->
      <rect x="50" y="150" width="240" height="68" rx="7" fill="var(--levi-arch-node-fill)" stroke="var(--levi-arch-node-stroke)" stroke-width="1"/>
      <text x="170" y="178" text-anchor="middle" fill="var(--levi-arch-text-primary)" font-size="14" font-weight="600">Mutation producers</text>
      <text x="170" y="198" text-anchor="middle" fill="var(--levi-arch-text-secondary)" font-size="11">Smaller models</text>
      <rect class="levi-arch-hover-zone" data-tip="producers" x="50" y="150" width="240" height="68" rx="7"/>

      <!-- Arrow: Producers → Queue (Push) -->
      <line x1="170" y1="218" x2="170" y2="266" stroke="var(--levi-arch-edge)" stroke-width="1.2" marker-end="url(#levi-arch-arr)"/>
      <text x="182" y="248" fill="var(--levi-arch-text-tertiary)" font-size="11">Push</text>

      <!-- Candidate Queue -->
      <rect x="50" y="268" width="240" height="58" rx="7" fill="var(--levi-arch-node-fill)" stroke="var(--levi-arch-node-stroke)" stroke-width="1"/>
      <text x="170" y="302" text-anchor="middle" fill="var(--levi-arch-text-primary)" font-size="14" font-weight="600">Candidate queue</text>
      <rect class="levi-arch-hover-zone" data-tip="queue" x="50" y="268" width="240" height="58" rx="7"/>

      <!-- Arrow: Queue → Eval (Pull) -->
      <line x1="170" y1="326" x2="170" y2="374" stroke="var(--levi-arch-edge)" stroke-width="1.2" marker-end="url(#levi-arch-arr)"/>
      <text x="182" y="356" fill="var(--levi-arch-text-tertiary)" font-size="11">Pull</text>

      <!-- Eval Workers -->
      <rect x="50" y="376" width="240" height="68" rx="7" fill="var(--levi-arch-node-fill)" stroke="var(--levi-arch-node-stroke)" stroke-width="1"/>
      <text x="170" y="404" text-anchor="middle" fill="var(--levi-arch-text-primary)" font-size="14" font-weight="600">Eval workers</text>
      <text x="170" y="424" text-anchor="middle" fill="var(--levi-arch-text-secondary)" font-size="11">Scoring function</text>
      <rect class="levi-arch-hover-zone" data-tip="consumers" x="50" y="376" width="240" height="68" rx="7"/>

      <!-- Arrow: Producers ← Archive (Sample) -->
      <line x1="290" y1="184" x2="400" y2="184" stroke="var(--levi-arch-edge)" stroke-width="1.2" marker-end="url(#levi-arch-arr)"/>
      <text x="320" y="176" fill="var(--levi-arch-text-tertiary)" font-size="11">Sample</text>

      <!-- Arrow: Eval → Archive (Update if better) -->
      <polyline points="290,410 370,410 370,330 400,330" stroke="var(--levi-arch-edge)" stroke-width="1.2" marker-end="url(#levi-arch-arr)" fill="none"/>
      <text x="336" y="370" fill="var(--levi-arch-text-tertiary)" font-size="11" text-anchor="middle">Update if</text>
      <text x="336" y="384" fill="var(--levi-arch-text-tertiary)" font-size="11" text-anchor="middle">better</text>

      <!-- CVT-MAP-Elites Archive -->
      <rect x="402" y="142" width="340" height="290" rx="9" fill="var(--levi-arch-node-fill)" stroke="var(--levi-arch-node-stroke)" stroke-width="1"/>
      <text x="572" y="170" text-anchor="middle" fill="var(--levi-arch-text-primary)" font-size="14" font-weight="600">CVT-MAP-Elites archive</text>

      <!-- Voronoi cells -->
      <g transform="translate(425, 182)">
        <rect width="294" height="195" rx="5" fill="var(--levi-arch-voronoi-bg)" stroke="var(--levi-arch-subnode-stroke)" stroke-width="0.7"/>

        <!-- Row labels -->
        <text x="48" y="20" fill="rgba(255,255,255,0.6)" font-size="12" font-weight="500" text-anchor="middle">c0</text>
        <text x="147" y="20" fill="rgba(255,255,255,0.6)" font-size="12" font-weight="500" text-anchor="middle">c1</text>
        <text x="246" y="20" fill="rgba(255,255,255,0.6)" font-size="12" font-weight="500" text-anchor="middle">c2</text>

        <!-- Voronoi-style regions -->
        <polygon points="0,28 98,28 85,75 50,90 0,80" fill="#4a8a97" stroke="var(--levi-arch-voronoi-bg)" stroke-width="2"/>
        <polygon points="98,28 196,28 185,68 140,88 85,75" fill="#8b6d45" stroke="var(--levi-arch-voronoi-bg)" stroke-width="2"/>
        <polygon points="196,28 294,28 294,80 250,90 185,68" fill="#6aadba" stroke="var(--levi-arch-voronoi-bg)" stroke-width="2"/>

        <polygon points="0,80 50,90 60,130 35,155 0,145" fill="#8b6d45" stroke="var(--levi-arch-voronoi-bg)" stroke-width="2"/>
        <polygon points="50,90 85,75 140,88 150,130 60,130" fill="#3d7a6a" stroke="var(--levi-arch-voronoi-bg)" stroke-width="2"/>
        <polygon points="140,88 185,68 250,90 240,135 150,130" fill="#4a8a97" stroke="var(--levi-arch-voronoi-bg)" stroke-width="2"/>
        <polygon points="250,90 294,80 294,145 260,150 240,135" fill="#8b6d45" stroke="var(--levi-arch-voronoi-bg)" stroke-width="2"/>

        <polygon points="0,145 35,155 60,130 75,170 45,195 0,195" fill="#6aadba" stroke="var(--levi-arch-voronoi-bg)" stroke-width="2"/>
        <polygon points="60,130 150,130 155,170 75,170" fill="#4a8a97" stroke="var(--levi-arch-voronoi-bg)" stroke-width="2"/>
        <polygon points="150,130 240,135 245,175 155,170" fill="#3d7a6a" stroke="var(--levi-arch-voronoi-bg)" stroke-width="2"/>
        <polygon points="240,135 260,150 294,145 294,195 225,195 245,175" fill="#6aadba" stroke="var(--levi-arch-voronoi-bg)" stroke-width="2"/>

        <polygon points="75,170 155,170 140,195 45,195" fill="#8b6d45" stroke="var(--levi-arch-voronoi-bg)" stroke-width="2"/>
        <polygon points="155,170 245,175 225,195 140,195" fill="#4a8a97" stroke="var(--levi-arch-voronoi-bg)" stroke-width="2"/>

        <!-- Centroids (white dots) -->
        <circle cx="40" cy="58" r="4" fill="white"/>
        <circle cx="130" cy="55" r="4" fill="white"/>
        <circle cx="235" cy="58" r="4" fill="white"/>
        <circle cx="25" cy="118" r="4" fill="white"/>
        <circle cx="105" cy="105" r="4" fill="white"/>
        <circle cx="195" cy="108" r="4" fill="white"/>
        <circle cx="268" cy="118" r="4" fill="white"/>
        <circle cx="40" cy="165" r="4" fill="white"/>
        <circle cx="110" cy="152" r="4" fill="white"/>
        <circle cx="200" cy="155" r="4" fill="white"/>
        <circle cx="260" cy="168" r="4" fill="white"/>
        <circle cx="95" cy="182" r="4" fill="white"/>
        <circle cx="190" cy="182" r="4" fill="white"/>
      </g>

      <text x="572" y="398" text-anchor="middle" fill="var(--levi-arch-text-secondary)" font-size="11">Behavioral niches (k-means centroids)</text>
      <rect class="levi-arch-hover-zone" data-tip="archive" x="402" y="142" width="340" height="290" rx="9"/>

      <!-- Paradigm Shifts -->
      <rect x="462" y="458" width="260" height="54" rx="7" fill="var(--levi-arch-node-fill)" stroke="var(--levi-arch-node-stroke)" stroke-width="1"/>
      <text x="592" y="482" text-anchor="middle" fill="var(--levi-arch-text-primary)" font-size="14" font-weight="600">Paradigm shifts</text>
      <text x="592" y="500" text-anchor="middle" fill="var(--levi-arch-text-secondary)" font-size="11">Every K evals: frontier model</text>
      <rect class="levi-arch-hover-zone" data-tip="pe" x="462" y="458" width="260" height="54" rx="7"/>

      <!-- Arrow: Archive → Paradigm (Inject) -->
      <line x1="530" y1="432" x2="530" y2="456" stroke="var(--levi-arch-edge)" stroke-width="1.2" marker-end="url(#levi-arch-arr)"/>
      <text x="500" y="450" fill="var(--levi-arch-text-tertiary)" font-size="11" text-anchor="end">Inject</text>

      <!-- Arrow: Paradigm → Archive (Cluster + select) -->
      <line x1="654" y1="458" x2="654" y2="434" stroke="var(--levi-arch-edge)" stroke-width="1.2" marker-end="url(#levi-arch-arr)"/>
      <text x="666" y="450" fill="var(--levi-arch-text-tertiary)" font-size="11">Cluster + select</text>

      <!-- Footer -->
      <text x="440" y="550" text-anchor="middle" fill="var(--levi-arch-text-secondary)" font-size="12">$ budget  ·  # evals  ·  wall time</text>
    </svg>
  </div>
</div>

<script>
(function () {
  const root = document.getElementById('levi-arch-diagram');
  if (!root) return;

  const descriptions = {
    init: {
      title: "Init: Diverse Seed Generation",
      body: "Before the main loop, an LLM generates algorithmically diverse initial solutions. Each seed is prompted to use a fundamentally different algorithmic paradigm, ensuring the archive starts with broad coverage of the design space."
    },
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

Try it out at [github.com/ttanv/levi](https://github.com/ttanv/levi)! For full installation, configuration, and usage details, see the [LEVI Documentation](/levi/docs).



## ADRS Benchmark Results

We evaluate on the ADRS benchmark suite,<span class="sidenote-ref"><a href="#fn3">3</a></span><span class="sidenote"><span class="sn-num">3</span> <a href="https://ucbskyadrs.github.io/">ADRS</a> (Cheng et al, 2025): benchmark suite from UC Berkeley for LLM-guided optimization on real-world systems problems.</span> a set of real-world systems problems spanning cloud scheduling, load balancing, SQL optimization, and transaction scheduling.

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

The main results compare frameworks that differ simultaneously in model choice, budget, and architecture. To isolate the contribution of the search architecture, we run LEVI, OpenEvolve, and GEPA under identical conditions: a single locally-served Qwen3-30B-A3B model, 750 successful evaluations,<span class="sidenote-ref"><a href="#fn4">4</a></span><span class="sidenote"><span class="sn-num">4</span> OpenEvolve required reducing parent count from 5 to 2 for the smaller model and still produced many failures. We report successful evaluations rather than total to give OpenEvolve a fair comparison.</span> and three random seeds on two representative problems.

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

<div class="blog-footnotes">
<strong>Notes</strong>
<ol>
<li id="fn1"><a href="https://github.com/algorithmicsuperintelligence/openevolve/blob/main/examples/circle_packing/config_phase_1_anthropic.yaml">OpenEvolve config</a>: uses Claude Opus for mutations.</li>
<li id="fn2"><a href="https://arxiv.org/pdf/2509.19349">ShinkaEvolve</a> (Ye et al, 2025): relies on frontier-scale models throughout the search.</li>
<li id="fn3"><a href="https://ucbskyadrs.github.io/">ADRS</a> (Cheng et al, 2025): benchmark suite from UC Berkeley for LLM-guided optimization on real-world systems problems.</li>
<li id="fn4">OpenEvolve required reducing parent count from 5 to 2 for the smaller model and still produced many failures. We report successful evaluations rather than total to give OpenEvolve a fair comparison.</li>
</ol>
</div>
