---
layout: blog
title: "LEVI: Better LLM Optimization for the Price of a Cup of Coffee"
subtitle: "A harness-first framework for LLM-guided evolutionary search."
date: 2026-02-01
permalink: /levi
---

<style>
.figure-caption {
  color: var(--text-secondary);
  font-size: 14px;
  line-height: 1.6;
  margin-top: -1rem;
  margin-bottom: 2rem;
  font-style: italic;
}
.section-desc {
  color: var(--text-primary);
  font-size: 15px;
  line-height: 1.7;
  margin-top: -0.5rem;
  margin-bottom: 1.5rem;
  opacity: 0.75;
}
</style>

## TLDR

Existing LLM-guided evolutionary frameworks have weak diversity mechanisms that cause early convergence, then compensate by throwing expensive frontier models at the problem. LEVI takes a harness-first approach: fix the search architecture so the archive preserves structurally diverse solutions throughout the run, and strong performance follows even with cheap models. The result is better scores than OpenEvolve, ShinkaEvolve, and GEPA on the ADRS benchmark at 1.5--6.7× lower cost. LEVI will be open-sourced on GitHub soon.

<img src="/results/txn_scheduling.png" alt="Controlled comparison: LEVI vs OpenEvolve vs GEPA on Transaction Scheduling, same model, same budget" style="max-width:100%;height:auto;margin:1.5rem 0;">

<p class="figure-caption">Figure 1: Controlled comparison on Transaction Scheduling. Same model (Qwen3-30B-A3B), same budget (750 evaluations), three seeds. LEVI's archive sustains exploration well past the point where baselines converge.</p>


## Background and Motivation

<p class="section-desc">Why existing frameworks couple strong performance with large budgets, and why that coupling is a design choice rather than a fundamental requirement.</p>

The idea of pairing large language models with evolutionary search over programs was introduced by FunSearch, which used an island-based method to discover solutions to problems that are easy to verify but hard to solve. AlphaEvolve scaled the paradigm to stronger LLMs and larger codebases, and subsequent work extended it to mathematical constructions, heuristic design, prompt optimization, and systems research. The core loop is simple: an LLM proposes candidate programs, an evaluator scores them, and a selection mechanism guides the population toward better solutions.

Several open-source frameworks now implement this loop---OpenEvolve, ShinkaEvolve, and GEPA being the most widely used. These have demonstrated strong results, but they share a common characteristic: strong performance is tightly coupled with large budgets and frontier-scale models. Most published runs cost \$15--30 per problem and assume access to models like GPT-5 or Gemini-3, making the paradigm expensive to use and difficult to iterate on.

We believe this coupling reflects a design assumption more than a fundamental requirement. Existing frameworks were built with frontier models as the default, and their search architectures reflect this: when diversity stalls, the response tends to be additional layers of mechanism---islands, embedding-based novelty filters, LLM judges---each patching over convergence that still occurs, rather than preventing it at the archive level. GEPA takes a cleaner approach through per-instance Pareto fronts, but its diversity signal weakens when performance across instances is highly correlated. The result across the board is that capable models end up doing double duty---both proposing new solutions and compensating for a selection layer that lets the population narrow too quickly.

LEVI takes a different starting point. Rather than building the harness around the assumption of a strong model, we ask what the search architecture should look like if model calls are expensive and limited. By improving the archive's ability to maintain structurally diverse solutions throughout the search, we reduce the burden on the model---making it possible to get strong results with cheaper models and smaller budgets. The goal is not to eliminate the need for capable models, but to ensure they are used where they matter most, and that researchers without frontier-model budgets can still push the state of the art.


## LEVI

<p class="section-desc">LEVI is built on two core ideas: <strong>stratified model allocation</strong> and <strong>improved diversity maintenance</strong>. While explained separately, they are best understood as extensions of each other---the archive provides the structure that makes principled allocation possible, and principled allocation is what makes a diversity-preserving archive practical under tight budgets.</p>

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

<p class="figure-caption">Hover over each component for a detailed description.</p>

### Stratified Model Allocation

<p class="section-desc">Match model capacity to task demand: cheap models for refinement, expensive models for paradigm shifts.</p>

Frontier models help---but they are a waste if used for every mutation. Smaller LLMs may actually be preferred under tight budgets, since the sheer quantity of solutions they produce can outweigh the quality advantage of larger models. However, smaller models have a narrower pretraining distribution, limiting their range of ideas and ability to propose fundamentally different approaches. Neither model class is strictly better; they have different strengths.

Some existing frameworks already support multiple models, but treat them as interchangeable---sampling from an ensemble uniformly or routing calls without regard to what the mutation actually demands. This ignores a natural asymmetry: proposing an entirely new algorithmic direction requires broad knowledge and creative reasoning, while refining an existing approach---adjusting constants, reordering operations, tuning edge cases---requires far less. The harness should be aware of this distinction and allocate accordingly.

LEVI introduces stratified model allocation, which matches model capacity to task demand. Smaller, cheaper models handle the majority of the search: local refinements and incremental improvements within an established algorithmic family. Larger models are reserved for infrequent *paradigm shifts*---mutations that aim to propose structurally different approaches rather than polish existing ones. The principle is straightforward: allocate each model toward its strength. Small models for breadth and throughput, large models for creative leaps.

However, this raises two questions. First, how do we select representative solutions from each algorithmic family to give the larger model meaningful context for paradigm shifts? Second, since we now rely more heavily on smaller models and their volume of output, we need a more robust mechanism to prevent the archive from converging---quantity without diversity is just noise.

### Improved Diversity Maintenance

<p class="section-desc">A unified fingerprint space with noise-perturbed initialization keeps the archive structurally diverse throughout the search.</p>

The diversity mechanism must address two things: how to represent solutions and how to initialize the archive.

**Unifying structural and performance diversity.** Existing frameworks maintain diversity along different axes. OpenEvolve considers structural features like code length; GEPA considers per-instance performance trade-offs through Pareto fronts. Both capture something real, but neither captures the full picture---structure alone misses behavioral differences, and per-instance scores alone miss solutions that perform similarly but work in fundamentally different ways. Rather than choosing one, we use both as dimensions of a single behavioral descriptor. Each solution is mapped to a *fingerprint*: a vector of AST-based structural features (depth, loop count, cyclomatic complexity, etc.) alongside per-instance performance scores, normalized and projected to [0, 1]. This fingerprint lives in a CVT-MAP-Elites archive, where a Voronoi tessellation over the combined space maintains geometric structure that neither axis provides alone. This also directly answers the first question from the previous section: the Voronoi regions naturally cluster solutions into algorithmic families, giving us representative solutions for paradigm shifts.

**Initializing between two extremes.** Traditional CVT-MAP-Elites initializes centroids uniformly across the descriptor space. With the higher dimensionality we use (6--10 dims), this leads to an extremely sparse tessellation---most regions will never be visited. A purely data-driven approach---fitting centroids to the first observed solutions---solves sparsity but creates the opposite problem: the archive's geometry overfits to whatever strategies appear early, leaving little room for novel approaches that emerge later. We take a middle path: *data-driven initialization with noise*. We generate a small set of structurally distinct seed programs (fewer than 10), expand them into variants, fingerprint them all, and then add Gaussian noise before fitting centroids. The seeds anchor the tessellation in regions of the space that viable programs actually occupy, while the noise broadens each family's footprint, ensuring the archive can accept innovations that fall between or outside the initial seed families. In practice, this is much more effective than either extreme.



## Preliminary Results: ADRS Benchmark

<p class="section-desc">We evaluate on the <a href="https://ucbskyadrs.github.io/">ADRS benchmark suite</a> introduced by Cheng et al.---a set of real-world systems problems spanning cloud scheduling, load balancing, SQL optimization, and transaction scheduling. We evaluate on seven of the nine problems. Our archive uses 50 centroids initialized via the fingerprint-then-perturb procedure with 5 structurally distinct seeds. Approximately 90% of LLM calls are routed to lightweight models (Qwen3-30B-A3B and MiMo-v2-Flash), with the remaining 10% reserved for paradigm shifts via Gemini Flash 3. We run 600--2,000 generations per problem.</p>

### Benchmark Scores

<div class="adrs-dashboard">
  <div class="adrs-controls">
    <!-- Legend -->
    <div class="adrs-legend" id="adrs-legend">
      <div class="adrs-legend-item" data-framework="GEPA">
        <span class="adrs-swatch swatch-gepa"></span>
        <span class="adrs-legend-name">GEPA</span>
      </div>
      <div class="adrs-legend-item" data-framework="OpenEvolve">
        <span class="adrs-swatch swatch-openevolve"></span>
        <span class="adrs-legend-name">OpenEvolve</span>
      </div>
      <div class="adrs-legend-item" data-framework="ShinkaEvolve">
        <span class="adrs-swatch swatch-shinka"></span>
        <span class="adrs-legend-name">ShinkaEvolve</span>
      </div>
      <div class="adrs-legend-item" data-framework="LEVI">
        <span class="adrs-swatch swatch-levi"></span>
        <span class="adrs-legend-name adrs-legend-levi">LEVI</span>
      </div>
    </div>
    <div class="adrs-view-toggle" id="adrs-view-toggle" role="tablist" aria-label="ADRS view toggle">
      <button type="button" class="adrs-toggle-btn is-active" data-view="chart" role="tab" aria-selected="true" aria-controls="adrs-chart-view">Diagram</button>
      <button type="button" class="adrs-toggle-btn" data-view="table" role="tab" aria-selected="false" aria-controls="adrs-table-view">Table</button>
    </div>
  </div>

  <!-- Chart -->
  <div class="adrs-view" id="adrs-chart-view" data-adrs-view="chart">
    <div class="adrs-chart-container">
      <canvas id="adrs-chart"></canvas>
    </div>
  </div>

  <!-- Table -->
  <div class="adrs-view" id="adrs-table-view" data-adrs-view="table" hidden>
    <div class="adrs-table-card">
      <div class="adrs-table-wrap">
        <table class="adrs-table" id="adrs-table">
        <thead>
          <tr>
            <th class="adrs-sticky">Framework</th>
            <th class="adrs-average">Average</th>
            <th>Cloudcast</th>
            <th>EPLB</th>
            <th>LLM-SQL</th>
            <th>Prism</th>
            <th>Spot Multi-Reg</th>
            <th>Spot Single-Reg</th>
            <th>Txn Scheduling</th>
          </tr>
        </thead>
        <tbody>
          <tr data-framework="GEPA">
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
          <tr data-framework="OpenEvolve">
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
          <tr data-framework="ShinkaEvolve">
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
          <tr class="is-levi" data-framework="LEVI">
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
    </div>
    <div class="adrs-note">Bold values indicate best performance per benchmark.</div>
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<script>
(function() {
  const FRAMEWORKS = ["GEPA", "OpenEvolve", "ShinkaEvolve", "LEVI"];
  const BENCHMARKS = ["Average", "Cloudcast", "EPLB", "LLM-SQL", "Prism", "Spot Multi-Reg", "Spot Single-Reg", "Txn Scheduling"];
  const DATA = {
    "GEPA":         [71.9, 96.6, 70.2, 67.7, 87.4, 62.2, 51.4, 67.7],
    "OpenEvolve":   [70.6, 92.9, 62.0, 72.5, 87.4, 66.7, 42.5, 70.0],
    "ShinkaEvolve": [67.4, 72.0, 66.4, 68.5, 87.4, 63.6, 45.6, 68.2],
    "LEVI":         [76.5, 100.0, 74.6, 78.3, 87.4, 72.4, 51.7, 71.1],
  };

  function getColors() {
    const style = getComputedStyle(document.documentElement);
    return {
      "GEPA": style.getPropertyValue('--adrs-gepa').trim() || '#7a9e8a',
      "OpenEvolve": style.getPropertyValue('--adrs-openevolve').trim() || '#8b96b3',
      "ShinkaEvolve": style.getPropertyValue('--adrs-shinka').trim() || '#b0a07a',
      "LEVI": style.getPropertyValue('--adrs-levi').trim() || '#c0392b',
    };
  }

  function getLeviDelta(benchIdx) {
    const leviVal = DATA["LEVI"][benchIdx];
    let secondBest = 0;
    FRAMEWORKS.forEach(fw => {
      if (fw !== "LEVI" && DATA[fw][benchIdx] > secondBest) secondBest = DATA[fw][benchIdx];
    });
    return +(leviVal - secondBest).toFixed(1);
  }

  let chart = null;
  let hoveredFramework = null;
  let currentView = 'chart';

  function createChart() {
    const ctx = document.getElementById('adrs-chart');
    if (!ctx) return;

    const colors = getColors();
    const style = getComputedStyle(document.documentElement);
    const textColor = style.getPropertyValue('--text-secondary').trim() || '#888';
    const gridColor = style.getPropertyValue('--border-color').trim() || '#2a2a2a';

    const datasets = FRAMEWORKS.map(fw => ({
      label: fw,
      data: DATA[fw],
      backgroundColor: colors[fw],
      borderRadius: 2,
      barPercentage: 0.85,
      categoryPercentage: 0.8,
    }));

    if (chart) chart.destroy();

    chart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: BENCHMARKS,
        datasets: datasets
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
          mode: 'index',
          intersect: false,
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: style.getPropertyValue('--card-bg').trim() || '#1a1a1a',
            titleColor: style.getPropertyValue('--heading-color').trim() || '#fff',
            bodyColor: textColor,
            borderColor: gridColor,
            borderWidth: 1,
            padding: 12,
            callbacks: {
              afterBody: function(context) {
                const benchIdx = context[0].dataIndex;
                const delta = getLeviDelta(benchIdx);
                return delta > 0 ? `\nLEVI vs 2nd best: +${delta}` : '\nLEVI vs 2nd best: tied';
              }
            }
          }
        },
        scales: {
          x: {
            grid: { display: false },
            ticks: {
              color: textColor,
              font: { size: 11 },
              maxRotation: 25,
              minRotation: 25
            }
          },
          y: {
            min: 40,
            max: 105,
            grid: {
              color: gridColor,
              drawBorder: false
            },
            ticks: {
              color: textColor,
              font: { size: 11, family: 'ui-monospace, monospace' }
            }
          }
        }
      }
    });

    updateChartOpacity();
  }

  function updateChartOpacity() {
    if (!chart) return;
    const colors = getColors();
    chart.data.datasets.forEach((dataset, i) => {
      const fw = FRAMEWORKS[i];
      const isLevi = fw === "LEVI";
      let opacity;
      if (hoveredFramework) {
        opacity = hoveredFramework === fw ? 1 : 0.15;
      } else {
        opacity = isLevi ? 1 : 0.55;
      }
      const baseColor = colors[fw];
      dataset.backgroundColor = hexToRgba(baseColor, opacity);
    });
    chart.update('none');
  }

  function hexToRgba(hex, alpha) {
    hex = hex.replace('#', '');
    if (hex.length === 3) {
      hex = hex.split('').map(c => c + c).join('');
    }
    const r = parseInt(hex.substring(0, 2), 16);
    const g = parseInt(hex.substring(2, 4), 16);
    const b = parseInt(hex.substring(4, 6), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
  }

  function handleHover(framework) {
    hoveredFramework = framework;
    updateChartOpacity();

    document.querySelectorAll('.adrs-legend-item').forEach(item => {
      const fw = item.dataset.framework;
      if (hoveredFramework && hoveredFramework !== fw) {
        item.classList.add('dimmed');
      } else {
        item.classList.remove('dimmed');
      }
    });

    document.querySelectorAll('#adrs-table tbody tr').forEach(row => {
      const fw = row.dataset.framework;
      if (hoveredFramework && hoveredFramework !== fw) {
        row.classList.add('dimmed');
      } else {
        row.classList.remove('dimmed');
      }
    });
  }

  function setView(view) {
    currentView = view;
    const chartView = document.getElementById('adrs-chart-view');
    const tableView = document.getElementById('adrs-table-view');
    const toggle = document.getElementById('adrs-view-toggle');
    if (chartView) chartView.hidden = view !== 'chart';
    if (tableView) tableView.hidden = view !== 'table';
    if (toggle) {
      toggle.querySelectorAll('.adrs-toggle-btn').forEach(btn => {
        const isActive = btn.dataset.view === view;
        btn.classList.toggle('is-active', isActive);
        btn.setAttribute('aria-selected', isActive ? 'true' : 'false');
      });
    }
    if (view === 'chart' && chart) {
      setTimeout(() => chart.resize(), 0);
    }
  }

  function initInteractivity() {
    document.querySelectorAll('.adrs-legend-item').forEach(item => {
      item.addEventListener('mouseenter', () => handleHover(item.dataset.framework));
      item.addEventListener('mouseleave', () => handleHover(null));
    });

    document.querySelectorAll('#adrs-table tbody tr').forEach(row => {
      row.addEventListener('mouseenter', () => handleHover(row.dataset.framework));
      row.addEventListener('mouseleave', () => handleHover(null));
    });

    const toggle = document.getElementById('adrs-view-toggle');
    if (toggle) {
      toggle.querySelectorAll('.adrs-toggle-btn').forEach(btn => {
        btn.addEventListener('click', () => {
          const view = btn.dataset.view;
          if (view && view !== currentView) setView(view);
        });
      });
    }
  }

  const observer = new MutationObserver(() => {
    setTimeout(createChart, 50);
  });
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

  if (window.matchMedia) {
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
      setTimeout(createChart, 50);
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', () => {
      createChart();
      initInteractivity();
      setView(currentView);
    });
  } else {
    createChart();
    initInteractivity();
    setView(currentView);
  }
})();
</script>

<p class="figure-caption">Figure 3: ADRS benchmark scores (%). Bold indicates best per problem. LEVI achieves the highest score on every problem where improvement is possible.</p>

LEVI achieves the highest score on every problem where improvement is possible, with an average of 76.5 compared to 71.9 for the next-best framework (GEPA)---a +4.6 point improvement over the prior state of the art. On Cloudcast, LEVI reaches a perfect 100.0, indicating the problem is fully solved under the benchmark's scoring function. The largest gains appear on LLM-SQL (+5.8) and Spot Multi (+5.7), while more modest improvements on Spot Single (+0.3) and Transaction Scheduling (+1.1) reflect problems with smaller decision spaces or harder optimization landscapes. Prism remains tied at 87.4 across all frameworks, confirming that the current problem formulation admits a single dominant solution.

An additional observation: no single baseline is consistently second-best across problems, reflecting how the different diversity mechanisms each framework employs interact unevenly with different problem structures. LEVI's consistent first-place performance suggests that CVT-MAP-Elites with fingerprint-initialized centroids provides a more robust diversity mechanism regardless of problem characteristics.

<img src="/results/best_score_progression.png" alt="LEVI best score progression over time across problems" style="max-width:100%;height:auto;margin:1.5rem 0;">

<p class="figure-caption">Figure 4: LEVI best score progression over generations. The archive sustains steady improvement throughout the search rather than plateauing early.</p>

### Cost

<p class="section-desc">Stratified allocation drops per-generation cost by ~10x, enabling more generations at lower total spend.</p>

LEVI's stratified allocation is the primary driver of cost reduction. By routing the majority of mutations through lightweight models, the per-generation cost drops by roughly an order of magnitude compared to baselines that use GPT-5 or Gemini-3.0-Pro for every call. This allows LEVI to run substantially more generations while still spending less in total: \$4.50 per problem on most tasks (Transaction Scheduling: \$13), versus \$15--30 for baselines.

<div class="adrs-dashboard">
  <div class="adrs-chart-container" style="height: 320px;">
    <canvas id="cost-chart"></canvas>
  </div>
  <div class="adrs-note">LEVI uses $4.50 on most tasks (Txn Scheduling: $13) versus baselines' $15–$30.</div>
</div>

<script>
(function() {
  const PROBLEMS   = ["Spot Single-Reg", "Spot Multi-Reg", "LLM-SQL", "Txn Scheduling", "Cloudcast", "Prism", "EPLB"];
  const BASELINE   = [30, 25, 20, 20, 15, 15, 15];
  const LEVI_COST  = [4.5, 4.5, 4.5, 13, 4.5, 4.5, 4.5];
  const REDUCTIONS = ["6.7x", "5.6x", "4.4x", "1.5x", "3.3x", "3.3x", "3.3x"];

  let costChart = null;

  function createCostChart() {
    const ctx = document.getElementById('cost-chart');
    if (!ctx) return;

    const style = getComputedStyle(document.documentElement);
    const textColor = style.getPropertyValue('--text-secondary').trim() || '#888';
    const gridColor = style.getPropertyValue('--border-color').trim() || '#2a2a2a';
    const leviColor = style.getPropertyValue('--adrs-levi').trim() || '#ef4444';
    const headingColor = style.getPropertyValue('--heading-color').trim() || '#fff';

    if (costChart) costChart.destroy();

    costChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: PROBLEMS,
        datasets: [
          {
            label: 'Baseline Cost',
            data: BASELINE,
            backgroundColor: 'rgba(100, 116, 139, 0.35)',
            borderRadius: 3,
            barPercentage: 0.7,
            categoryPercentage: 0.75,
          },
          {
            label: 'LEVI Cost',
            data: LEVI_COST,
            backgroundColor: leviColor,
            borderRadius: 3,
            barPercentage: 0.7,
            categoryPercentage: 0.75,
          }
        ]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            display: true,
            position: 'top',
            align: 'end',
            labels: {
              color: textColor,
              font: { size: 11 },
              boxWidth: 12,
              boxHeight: 4,
              padding: 16,
              usePointStyle: false,
            }
          },
          tooltip: {
            backgroundColor: style.getPropertyValue('--card-bg').trim() || '#1a1a1a',
            titleColor: headingColor,
            bodyColor: textColor,
            borderColor: gridColor,
            borderWidth: 1,
            padding: 10,
            callbacks: {
              afterBody: function(context) {
                const idx = context[0].dataIndex;
                return 'Reduction: ' + REDUCTIONS[idx];
              }
            }
          }
        },
        scales: {
          x: {
            grid: { color: gridColor, drawBorder: false },
            ticks: {
              color: textColor,
              font: { size: 11, family: 'ui-monospace, monospace' },
              callback: function(v) { return '$' + v; }
            },
            max: 35,
          },
          y: {
            grid: { display: false },
            ticks: {
              color: textColor,
              font: { size: 12 },
            }
          }
        }
      },
      plugins: [{
        id: 'reductionLabels',
        afterDatasetsDraw: function(chart) {
          const meta = chart.getDatasetMeta(1);
          const ctx2 = chart.ctx;
          meta.data.forEach(function(bar, i) {
            ctx2.save();
            ctx2.font = 'bold 11px ui-monospace, SFMono-Regular, monospace';
            ctx2.fillStyle = leviColor;
            ctx2.textAlign = 'left';
            ctx2.textBaseline = 'middle';
            ctx2.fillText(REDUCTIONS[i], bar.x + 6, bar.y);
            ctx2.restore();
          });
        }
      }]
    });
  }

  const obs = new MutationObserver(function() { setTimeout(createCostChart, 50); });
  obs.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  if (window.matchMedia) {
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function() {
      setTimeout(createCostChart, 50);
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', createCostChart);
  } else {
    createCostChart();
  }
})();
</script>

The cost reduction is not the point---it is evidence that the harness-first approach works. When the archive maintains diversity, cheap models suffice for most of the search.

### Controlled Architecture Comparison

<p class="section-desc">Same model, same budget, three seeds---isolating the search architecture's contribution.</p>

The main results compare frameworks that differ simultaneously in model choice, budget, and architecture. To isolate the contribution of the search architecture, we run LEVI, OpenEvolve, and GEPA under identical conditions: a single locally-served Qwen3-30B-A3B model, 750 successful evaluations, and three random seeds on two representative problems.

**Transaction Scheduling** is a variant of an NP-hard ordering problem where multiple algorithmic families (greedy, simulated annealing, genetic) are viable but performance is measured on a single instance, giving Pareto-based diversity no trade-off to exploit. LEVI reaches a score of 62 within the first 100 evaluations, a level neither baseline achieves at any point. Final scores: LEVI 64.9, OpenEvolve 59.9, GEPA 54.4. Both baselines plateau sharply, consistent with early convergence onto a single algorithmic family; LEVI's curve continues rising past evaluation 500.

<img src="/results/txn_scheduling.png" alt="Transaction Scheduling controlled comparison plot" style="max-width:100%;height:auto;margin:1.5rem 0;">

<p class="figure-caption">Figure 5: Controlled comparison on Transaction Scheduling. Same model, same budget. LEVI reaches 62 within 100 evaluations---a level neither baseline achieves at any point during the run.</p>

**Can't Be Late** is scored across 1,080 simulations that give Pareto-based approaches a richer signal. The final-score gap narrows (LEVI 44.9, OpenEvolve 43.2, GEPA 32.6), but the efficiency gap widens dramatically. LEVI reaches near-peak performance by roughly evaluation 50, while OpenEvolve requires over 600 evaluations to approach the same level---a roughly 12× advantage in sample efficiency.

<img src="/results/cant_be_late.png" alt="Can't Be Late controlled comparison plot" style="max-width:100%;height:auto;margin:1.5rem 0;">

<p class="figure-caption">Figure 6: Controlled comparison on Can't Be Late. LEVI reaches near-peak by evaluation 50; OpenEvolve requires 600+ evaluations to approach the same level.</p>

These controlled results confirm that the performance gains are attributable to the search architecture, not to model choice or budget. A 30B model under LEVI's search regime matches or exceeds what the same model achieves under alternative selection mechanisms.

**More benchmarks and domains are in progress---ADRS is a first validation, not the full story.**

LEVI will be open-sourced on GitHub soon. Point it at a scoring function and a seed program and it runs until the budget is spent.
