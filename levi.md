---
layout: blog
title: "LEVI: LLM-Guided Evolution for the Price of a Cup of Coffee"
subtitle: "An open-source evolutionary framework for algorithmic discovery."
date: 2026-02-01
permalink: /levi
---

## TLDR

Existing LLM-based evolutionary systems have weak diversity mechanisms that cause early convergence, compensated by throwing expensive frontier models at the problem. LEVI fixes this at the root: a CVT-MAP-Elites archive with fingerprint-initialized centroids keeps structurally distinct algorithmic families alive throughout the search, while a stratified model allocation routes cheap small models for refinement and reserves larger models for infrequent paradigm shifts. The result is better scores than OpenEvolve, ShinkaEvolve, and GEPA on the ADRS benchmark at 1.5--6.7x lower cost. LEVI will be open-sourced on GitHub soon.

<img src="/results/comparison_plot.png" alt="Controlled comparison: LEVI vs OpenEvolve vs GEPA on Transaction Scheduling, same model, same budget" style="max-width:100%;height:auto;border-radius:8px;margin:1.5rem 0;">

## Background and Motivation

The idea of pairing large language models with evolutionary search over programs was introduced by FunSearch, which used an island-based method to discover solutions to problems that are easy to verify but hard to solve. AlphaEvolve then scaled the paradigm, allowing the system to use stronger LLMs, operate on larger codebases, and tackle a broader set of problems. Later works expanded the set of promising applications to include mathematical constructions, heuristic design, prompt optimization, and systems research.

Cheng et al. formalize systems research through their AI-Driven Research for Systems framework (ADRS), defining it as an iterative loop in which LLMs generate candidate solutions (scheduling policies, load balancers, congestion-control algorithms), an evaluator scores them against real systems or simulators, and a selection mechanism guides the population toward better solutions. ADRS is well-suited to systems performance problems because candidates can be verified robustly and cheaply by executing them against representative workloads. Across ten case studies spanning cloud scheduling, mixture-of-experts load balancing, LLM-based SQL optimization, and transaction scheduling, ADRS-generated solutions match or outperform human state-of-the-art designs.

Multiple open-source frameworks implement this paradigm. OpenEvolve is an open-source implementation of AlphaEvolve's core pipeline with island-based evolution, MAP-Elites archiving along a small number of feature dimensions, and heavy reliance on frontier models for every mutation. ShinkaEvolve adds structured introspection through weighted archive sampling, embedding-based novelty filtering, LLM novelty judges, and periodic meta-prompt evolution. GEPA, while mainly targeting prompt optimization, also generalizes to code; it takes a more minimal approach, using per-instance Pareto fronts as an implicit diversity mechanism with natural-language reflection to mutate prompts.

While all three have demonstrated strong results, a pattern is worth noting: diversity tends to be maintained through multiple overlapping mechanisms---islands to separate populations, embeddings to detect similarity, LLM judges to filter trivial rewrites---each compensating where others fall short. This leads to extra complexity without solving the underlying diversity problem. GEPA avoids this complexity through its minimal approach, but its Pareto-based diversity only works well when there is a clear trade-off across the validation set and weakens when performance across instances is highly correlated. Approaches with weaker diversity mechanisms create pressure to use larger models to overcome the convergence that still occurs, thereby increasing the required budget.

The result is that strong performance becomes tightly coupled with large budgets: most ADRS experiments cost \$15--30 per problem and assume access to frontier models like GPT-5 or Gemini-3. We argue that this coupling is not inherent to the paradigm but rather a symptom of inadequate diversity maintenance at the selection level. When the archive fails to preserve behavioral diversity, the search collapses into narrow basins, compensated by throwing stronger models at an increasingly exploitation-heavy loop. In the next section, we describe LEVI, a framework that addresses diversity directly and, as a consequence, substantially reduces the dependence on model scale.

## LEVI: LLM-Evolution through Voronoi Initialization

LEVI is motivated by two core principles. First, we improve diversity maintenance through a CVT-MAP-Elites archive coupled with a novel fingerprint-then-perturb approach for centroid initialization. Second, we introduce a more principled way of utilizing larger models: using them only for infrequent paradigm shifts, and using smaller models for the majority of mutations.

### Improved Archive Diversity

LLM-guided search tends toward mode collapse: the solution archive gets dominated by a single algorithmic strategy and the system spends subsequent iterations polishing it, abandoning structurally distinct alternatives that may ultimately reach higher performance. LEVI uses a CVT-MAP-Elites archive with AST-based and performance-based features as dimensions. We initialize the archive through a novel fingerprint-then-perturb approach, avoiding the sparsity of uniform initialization and the convergence of data-driven approaches.

The process begins by generating a small set of structurally distinct seed programs (usually fewer than 10). Each seed is produced sequentially with full visibility into all prior seeds, and the LLM is explicitly instructed to propose a fundamentally different algorithmic paradigm. This context accumulation ensures successive seeds differentiate from the full set, producing starting points that span distinct regions of the algorithmic landscape. We then produce more variants using these seeds (usually on the order of tens of variants), giving us a pool of diverse approaches. These seeds and variants may score poorly, but for initialization purposes that is fine.

Each candidate is then mapped to a behavioral descriptor that captures its algorithmic identity. Raw structural features (AST depth, loop count, cyclomatic complexity) and per-instance performance scores are extracted, normalized via online z-scores, and mapped to [0, 1] through a sigmoid transform. This formulation unifies two diversity strategies that prior work treats as separate: structural features, as used by OpenEvolve, and per-instance performance vectors, as used by GEPA via Pareto fronts, coexist naturally as dimensions of the same fingerprint. The Voronoi tessellation provides geometric structure over this combined space that Pareto-based approaches lack.

Gaussian noise is added to the fingerprints during initialization. Without perturbation, the archive's geometry would overfit to the specific algorithmic families observed during seeding, leaving little room for novel strategies that emerge later. The noise broadens each family's footprint in descriptor space, allowing the archive to accept future innovations that fall between or outside the seed families. Centroids are then fit to this perturbed distribution via k-means, yielding a Voronoi tessellation whose regions concentrate around parts of the fingerprint space that viable programs actually occupy. In practice this ends up being much more powerful than a uniform approach that is too sparse, or a data-driven approach that converges too fast.

### Principled Model Allocation

A second source of inefficiency in existing frameworks is the treatment of LLM calls as interchangeable. Prior systems either route all mutations through a single model, or maintain an ensemble from which models are sampled uniformly at random. Both approaches ignore a fundamental asymmetry: the cognitive demands of different mutation tasks vary dramatically.

Proposing an entirely new algorithmic paradigm---replacing a greedy strategy with dynamic programming---requires broad knowledge and deep reasoning. Refining an existing approach---adjusting constants, reordering operations, tuning loop bounds---requires far less. Allocating a frontier-scale model to every incremental tweak is wasteful; asking a lightweight model to rethink an algorithm's foundations is unlikely to succeed.

LEVI introduces stratified model allocation, which matches model capacity to task demand. The diversity-preserving archive provides a natural basis for this stratification: the Voronoi regions cluster solutions into algorithmic families, allowing us to select the best approach from each region. Large-capacity models are reserved for high-level exploration: generating candidates that differ from given representatives, proposing paradigm shifts, or bridging between distant algorithmic families. These calls are infrequent but high-impact. Smaller, cheaper models handle the bulk of the search budget: local refinements, parameter sweeps, and incremental improvements within an established family. These calls are frequent and individually low-cost, but collectively responsible for polishing each family's best representative.

The principle is that the expected return on a large-model call is highest when the task demands creative breadth, and the expected return on a small-model call is highest when the task demands efficient local search. By aligning allocation with this asymmetry, LEVI achieves the creative range of frontier models and the throughput of lightweight models simultaneously, at a fraction of the cost of using a single large model throughout.

## Results

We evaluate on the ADRS benchmark suite introduced by Cheng et al., using the problem implementations and evaluation scripts from the Frontier-CS repository. Each problem provides a simulator or evaluator against which candidate programs are executed. Raw metrics are normalized to a 0--100 scale via problem-specific scoring functions. We evaluate on seven of the nine problems: Cloudcast, EPLB, LLM-SQL, Prism, Spot Single-Reg, Spot Multi-Reg, and Transaction Scheduling. Baseline scores for GEPA, OpenEvolve, and ShinkaEvolve are taken directly from the ADRS Leaderboard, where each framework was evaluated using frontier models (GPT-5 and Gemini-3.0-Pro) at costs of \$15--30 per problem.

Our CVT-MAP-Elites archive uses 50 centroids initialized via the fingerprint-then-perturb procedure. We seed evolution with 5 structurally distinct programs generated sequentially with context accumulation. Approximately 90% of LLM calls are routed to lightweight models (Qwen3-30B-A3B and MiMo-v2-Flash) for local refinement, while the remaining 10% use Gemini Flash 3 for paradigm-shift mutations. We run between 600 and 2,000 generations per problem at a total cost of \$4.50 per problem (with the exception of Transaction Scheduling at \$13).

### ADRS Benchmark (% score)

[ADRS](https://ucbskyadrs.github.io/) is a benchmark suite from UC Berkeley for evaluating LLM-guided optimization on real-world systems problems -- cloud scheduling, load balancing, congestion control, SQL optimization, and more.

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

    // Update legend
    document.querySelectorAll('.adrs-legend-item').forEach(item => {
      const fw = item.dataset.framework;
      if (hoveredFramework && hoveredFramework !== fw) {
        item.classList.add('dimmed');
      } else {
        item.classList.remove('dimmed');
      }
    });

    // Update table rows
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
    // Legend hover
    document.querySelectorAll('.adrs-legend-item').forEach(item => {
      item.addEventListener('mouseenter', () => handleHover(item.dataset.framework));
      item.addEventListener('mouseleave', () => handleHover(null));
    });

    // Table row hover
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

  // Theme change observer
  const observer = new MutationObserver(() => {
    setTimeout(createChart, 50);
  });
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });

  // Also listen for system theme changes
  if (window.matchMedia) {
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => {
      setTimeout(createChart, 50);
    });
  }

  // Initialize
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

LEVI achieves the highest score on every problem where improvement is possible, with an average of 76.5 compared to 71.9 for the next-best framework (GEPA), representing a +4.6 point improvement over the prior state of the art. The largest improvements appear on LLM-SQL (+5.8) and Spot Multi (+5.7). For LLM-SQL, the gain is partly attributable to LEVI's higher iteration count: the cheaper per-generation cost enables substantially more evaluations, and the winning solution emerged from a minor modification that produced a disproportionate score jump---the kind of incremental discovery that requires many trials to surface. Spot Multi and EPLB (+4.4) show similarly strong gains, with the EPLB run resulting in entirely different paradigms with high scores. On Cloudcast, LEVI achieves a perfect score of 100.0 (+3.4 over GEPA), indicating that the problem is now fully solved under the benchmark's scoring function.

Improvements on Spot Single (+0.3) and Transaction Scheduling (+1.1) are more modest. We attribute this to problem structure: Spot Single has a smaller decision space than its multi-region counterpart, limiting the benefit of maintaining diverse algorithmic families. Transaction Scheduling is also the most expensive LEVI run (\$13 vs. \$4.50 for other problems), suggesting that this problem's optimization landscape requires more exploration to navigate. Prism remains tied at 87.4 across all frameworks, confirming that the current problem abstraction admits a single dominant solution.

<img src="/results/best_score_progression.png" alt="LEVI best score progression over time" style="max-width:100%;height:auto;border-radius:8px;margin:1.5rem 0;">

An additional observation is that no single baseline is consistently second-best: GEPA leads on Cloudcast and Spot Single, OpenEvolve on LLM-SQL and Transaction Scheduling, and ShinkaEvolve on EPLB. This variance reflects the different diversity mechanisms each framework employs and their uneven effectiveness across problem structures. LEVI's consistent first-place performance across all seven tasks suggests that CVT-MAP-Elites with fingerprint-initialized centroids provides a more robust diversity mechanism than the alternatives, regardless of problem characteristics.

### Cost per Problem

LEVI's stratified model allocation is the primary driver of cost reduction. By routing approximately 90% of mutations through lightweight models and reserving Gemini Flash 3 for infrequent paradigm shifts, the per-generation cost drops by roughly an order of magnitude compared to baselines that use GPT-5 or Gemini-3.0-Pro for every call. This allows LEVI to run 600--2,000 generations---substantially more than the baselines' typical 100 iterations---while still spending less in total.

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

  // Theme observers
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

The result supports our central thesis: the coupling between strong performance and large budgets in prior work is a symptom of inadequate diversity maintenance, not an inherent property of the paradigm. When the archive preserves behavioral diversity, small models suffice for the majority of the search, and expensive models need only be invoked sparingly for creative leaps.

### Controlled Architecture Comparison

The main results compare frameworks that differ simultaneously in search architecture, model choice, and budget. To isolate the contribution of the search architecture alone, we run LEVI, OpenEvolve, and GEPA under identical conditions: a single locally-served Qwen3-30B-A3B model, 750 successful evaluations, and three random seeds per framework on two problems.

Transaction Scheduling is a variant of an NP-hard ordering problem that produced the highest cost in our main evaluation. Multiple algorithmic families (greedy, simulated annealing, genetic) yield viable solutions, but performance is measured on a single instance, providing no per-instance trade-off for Pareto-based diversity to exploit. LEVI reaches a score of 62 within the first 100 evaluations---a level that neither baseline achieves at any point during the run. The final scores reflect this gap: LEVI attains 64.9, compared to 59.9 for OpenEvolve and 54.4 for GEPA. OpenEvolve's curve flattens sharply after evaluation 300 and GEPA plateaus even earlier, consistent with early convergence onto a single algorithmic family. LEVI's curve continues rising through evaluation 500, indicating that the archive sustains exploration of distinct strategies well into the search.

<img src="/results/cbl_plot3.png" alt="Framework comparison on Can't Be Late" style="max-width:100%;height:auto;border-radius:8px;margin:1.5rem 0;">

On Can't Be Late, scored across 1,080 simulations that give Pareto-based approaches a richer signal, the final-score gap narrows---LEVI scores 44.9 versus OpenEvolve's 43.2 and GEPA's 32.6---but the efficiency gap widens dramatically. LEVI reaches its near-peak score by approximately evaluation 50, while OpenEvolve requires over 600 evaluations to approach the same level, representing a roughly 12x advantage in sample efficiency. Even with favorable problem structure for Pareto-based approaches, GEPA still trails significantly, suggesting that Pareto diversity alone underperforms archive-level diversity maintenance.

These results confirm that the performance--cost coupling observed in prior work arises from search architecture limitations, not from an inherent need for frontier-scale models. A 30B model under LEVI's search regime matches or exceeds what the same model achieves under OpenEvolve's island-based evolution or GEPA's Pareto selection, with the gap attributable entirely to how each framework maintains diversity.

## System Design

The architecture follows a producer-consumer pattern. DSPy optimizes mutation prompts once up front, then N producer workers sample parents from the CVT-MAP-Elites archive via LiteLLM, push candidate code through an asyncio queue, and M consumer workers evaluate each candidate in a sandboxed subprocess with hard timeouts. The archive only accepts improvements per behavioral niche. Punctuated equilibrium periodically triggers paradigm shifts using the larger model, and a budget manager shuts everything down when the dollar, eval, or time limit is hit.

<div class="diagram">
  <div class="levi-arch-diagram" id="levi-arch-diagram">
    <div class="levi-arch-tooltip" id="levi-arch-tooltip">
      <div class="tt-title"></div>
      <div class="tt-body"></div>
    </div>

    <svg width="880" height="520" viewBox="0 0 880 520" fill="none" xmlns="http://www.w3.org/2000/svg" aria-label="LEVI System Overview Architecture Diagram" role="img">
      <defs>
        <marker id="levi-arch-arr" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
          <polygon points="0 0.8, 7 3, 0 5.2" fill="var(--levi-arch-arrow-head)"/>
        </marker>
        <symbol id="levi-arch-star" viewBox="0 0 20 20">
          <polygon points="10,2 12.4,7.5 18,8 13.8,12 15,18 10,15 5,18 6.2,12 2,8 7.6,7.5" fill="#ffffff"/>
        </symbol>
      </defs>

      <rect x="60" y="20" width="320" height="52" rx="7" fill="var(--levi-arch-node-fill)" stroke="var(--levi-arch-node-stroke)" stroke-width="1"/>
      <text x="220" y="42" text-anchor="middle" fill="var(--levi-arch-text-primary)" font-size="13" font-weight="600">DSPy Prompt Optimization</text>
      <text x="220" y="58" text-anchor="middle" fill="var(--levi-arch-text-secondary)" font-size="10" font-weight="300">(One-time)</text>
      <rect class="levi-arch-hover-zone" data-tip="dspy" x="60" y="20" width="320" height="52" rx="7"/>

      <line x1="220" y1="72" x2="220" y2="98" stroke="var(--levi-arch-edge)" stroke-width="1.2" marker-end="url(#levi-arch-arr)"/>

      <rect x="60" y="100" width="320" height="52" rx="7" fill="var(--levi-arch-node-fill)" stroke="var(--levi-arch-node-stroke)" stroke-width="1"/>
      <text x="220" y="122" text-anchor="middle" fill="var(--levi-arch-text-primary)" font-size="13" font-weight="600">N Producer Workers</text>
      <text x="220" y="138" text-anchor="middle" fill="var(--levi-arch-text-secondary)" font-size="10" font-weight="300">(LLMs: LiteLLM)</text>
      <rect class="levi-arch-hover-zone" data-tip="producers" x="60" y="100" width="320" height="52" rx="7"/>

      <line x1="220" y1="152" x2="220" y2="196" stroke="var(--levi-arch-edge)" stroke-width="1.2" marker-end="url(#levi-arch-arr)"/>
      <text x="232" y="178" fill="var(--levi-arch-text-tertiary)" font-size="10" font-weight="300">Push Code</text>

      <rect x="60" y="198" width="320" height="52" rx="7" fill="var(--levi-arch-node-fill)" stroke="var(--levi-arch-node-stroke)" stroke-width="1"/>
      <text x="220" y="220" text-anchor="middle" fill="var(--levi-arch-text-primary)" font-size="13" font-weight="600">Code Queue</text>
      <text x="220" y="236" text-anchor="middle" fill="var(--levi-arch-text-secondary)" font-size="10" font-weight="300">(Asyncio)</text>
      <rect class="levi-arch-hover-zone" data-tip="queue" x="60" y="198" width="320" height="52" rx="7"/>

      <line x1="220" y1="250" x2="220" y2="294" stroke="var(--levi-arch-edge)" stroke-width="1.2" marker-end="url(#levi-arch-arr)"/>
      <text x="232" y="276" fill="var(--levi-arch-text-tertiary)" font-size="10" font-weight="300">Pull Code</text>

      <rect x="60" y="296" width="320" height="72" rx="7" fill="var(--levi-arch-node-fill)" stroke="var(--levi-arch-node-stroke)" stroke-width="1"/>
      <text x="220" y="318" text-anchor="middle" fill="var(--levi-arch-text-primary)" font-size="13" font-weight="600">M Consumer Workers</text>
      <text x="220" y="334" text-anchor="middle" fill="var(--levi-arch-text-secondary)" font-size="10" font-weight="300">(Evaluation in Subprocess)</text>
      <rect x="120" y="344" width="200" height="18" rx="3" fill="var(--levi-arch-subnode-fill)" stroke="var(--levi-arch-subnode-stroke)" stroke-width="0.7"/>
      <text x="220" y="357" text-anchor="middle" fill="var(--levi-arch-text-tertiary)" font-size="9" font-weight="300">exec + score_fn, Hard Timeout</text>
      <rect class="levi-arch-hover-zone" data-tip="consumers" x="60" y="296" width="320" height="72" rx="7"/>

      <line x1="220" y1="368" x2="220" y2="418" stroke="var(--levi-arch-edge)" stroke-width="1.2" marker-end="url(#levi-arch-arr)"/>

      <rect x="80" y="420" width="280" height="52" rx="7" fill="var(--levi-arch-node-fill)" stroke="var(--levi-arch-node-stroke)" stroke-width="1"/>
      <text x="220" y="442" text-anchor="middle" fill="var(--levi-arch-text-primary)" font-size="13" font-weight="600">Budget Manager</text>
      <text x="220" y="458" text-anchor="middle" fill="var(--levi-arch-text-secondary)" font-size="10" font-weight="300">(Monitors $, #evals, time)</text>
      <rect class="levi-arch-hover-zone" data-tip="budget" x="80" y="420" width="280" height="52" rx="7"/>

      <line x1="380" y1="126" x2="548" y2="126" stroke="var(--levi-arch-edge)" stroke-width="1.2" marker-end="url(#levi-arch-arr)"/>
      <text x="420" y="118" fill="var(--levi-arch-text-tertiary)" font-size="10" font-weight="300">Sample Parents</text>

      <polyline points="380,320 500,320 500,250 548,250" stroke="var(--levi-arch-edge)" stroke-width="1.2" marker-end="url(#levi-arch-arr)" fill="none"/>
      <text x="488" y="278" text-anchor="end" fill="var(--levi-arch-text-tertiary)" font-size="10" font-weight="300">Update if</text>
      <text x="488" y="291" text-anchor="end" fill="var(--levi-arch-text-tertiary)" font-size="10" font-weight="300">Score &gt; Existing</text>

      <line x1="360" y1="446" x2="548" y2="446" stroke="var(--levi-arch-edge)" stroke-width="1.2" marker-end="url(#levi-arch-arr)"/>
      <text x="382" y="438" fill="var(--levi-arch-text-tertiary)" font-size="10" font-weight="300">STOP signal on exhaustion</text>

      <line x1="690" y1="420" x2="690" y2="345" stroke="var(--levi-arch-edge)" stroke-width="1.2" marker-end="url(#levi-arch-arr)"/>

      <rect x="550" y="30" width="300" height="312" rx="9" fill="var(--levi-arch-node-fill)" stroke="var(--levi-arch-node-stroke)" stroke-width="1"/>
      <text x="700" y="56" text-anchor="middle" fill="var(--levi-arch-text-primary)" font-size="13" font-weight="600">CVT-MAP-Elites Archive</text>

      <g transform="translate(575, 66)">
        <rect width="250" height="200" rx="5" fill="var(--levi-arch-voronoi-bg)" stroke="var(--levi-arch-subnode-stroke)" stroke-width="0.7"/>

        <polygon points="0,0 65,0 55,50 40,70 0,60" fill="#6366f1" stroke="var(--levi-arch-voronoi-bg)" stroke-width="2"/>
        <polygon points="65,0 130,0 140,40 110,75 55,50" fill="#10b981" stroke="var(--levi-arch-voronoi-bg)" stroke-width="2"/>
        <polygon points="130,0 195,0 200,55 160,70 140,40" fill="#10b981" stroke="var(--levi-arch-voronoi-bg)" stroke-width="2"/>
        <polygon points="195,0 250,0 250,60 220,75 200,55" fill="#ef4444" stroke="var(--levi-arch-voronoi-bg)" stroke-width="2"/>

        <polygon points="0,60 40,70 50,120 30,140 0,130" fill="#6366f1" stroke="var(--levi-arch-voronoi-bg)" stroke-width="2"/>
        <polygon points="40,70 55,50 110,75 120,110 50,120" fill="#f59e0b" stroke="var(--levi-arch-voronoi-bg)" stroke-width="2"/>
        <polygon points="110,75 140,40 160,70 170,120 120,110" fill="#6366f1" stroke="var(--levi-arch-voronoi-bg)" stroke-width="2"/>
        <polygon points="160,70 200,55 220,75 250,60 250,130 210,125 170,120" fill="#ef4444" stroke="var(--levi-arch-voronoi-bg)" stroke-width="2"/>

        <polygon points="0,130 30,140 50,120 60,160 40,200 0,200" fill="#10b981" stroke="var(--levi-arch-voronoi-bg)" stroke-width="2"/>
        <polygon points="50,120 120,110 130,160 60,160" fill="#10b981" stroke="var(--levi-arch-voronoi-bg)" stroke-width="2"/>
        <polygon points="120,110 170,120 180,165 130,160" fill="#f59e0b" stroke="var(--levi-arch-voronoi-bg)" stroke-width="2"/>
        <polygon points="170,120 210,125 250,130 250,200 200,200 180,165" fill="#10b981" stroke="var(--levi-arch-voronoi-bg)" stroke-width="2"/>

        <polygon points="60,160 130,160 120,200 40,200" fill="#6366f1" stroke="var(--levi-arch-voronoi-bg)" stroke-width="2"/>
        <polygon points="130,160 180,165 200,200 120,200" fill="#ef4444" stroke="var(--levi-arch-voronoi-bg)" stroke-width="2"/>

        <use href="#levi-arch-star" x="18" y="16" width="18" height="18"/>
        <use href="#levi-arch-star" x="77" y="18" width="18" height="18"/>
        <use href="#levi-arch-star" x="150" y="12" width="18" height="18"/>
        <use href="#levi-arch-star" x="212" y="16" width="18" height="18"/>
        <use href="#levi-arch-star" x="13" y="84" width="18" height="18"/>
        <use href="#levi-arch-star" x="71" y="68" width="18" height="18"/>
        <use href="#levi-arch-star" x="136" y="68" width="18" height="18"/>
        <use href="#levi-arch-star" x="201" y="82" width="18" height="18"/>
        <use href="#levi-arch-star" x="18" y="144" width="18" height="18"/>
        <use href="#levi-arch-star" x="76" y="124" width="18" height="18"/>
        <use href="#levi-arch-star" x="140" y="128" width="18" height="18"/>
        <use href="#levi-arch-star" x="201" y="148" width="18" height="18"/>
        <use href="#levi-arch-star" x="70" y="164" width="18" height="18"/>
        <use href="#levi-arch-star" x="148" y="162" width="18" height="18"/>

        <text x="24" y="62" fill="var(--levi-arch-text-cell)" font-size="10" font-weight="400">c0</text>
        <text x="82" y="57" fill="var(--levi-arch-text-cell)" font-size="10" font-weight="400">c1</text>
        <text x="150" y="57" fill="var(--levi-arch-text-cell)" font-size="10" font-weight="400">c2</text>
        <text x="196" y="48" fill="var(--levi-arch-text-cell-muted)" font-size="11" font-weight="400">···</text>
        <text x="216" y="62" fill="var(--levi-arch-text-cell)" font-size="10" font-weight="400">c7</text>
      </g>

      <text x="700" y="286" text-anchor="middle" fill="var(--levi-arch-text-secondary)" font-size="10" font-weight="300">Grid of Behavioral Niches</text>
      <text x="700" y="300" text-anchor="middle" fill="var(--levi-arch-text-secondary)" font-size="10" font-weight="300">(K-means Centroids)</text>

      <rect class="levi-arch-hover-zone" data-tip="archive" x="550" y="30" width="300" height="312" rx="9"/>

      <rect x="550" y="410" width="280" height="72" rx="7" fill="var(--levi-arch-node-fill)" stroke="var(--levi-arch-node-stroke)" stroke-width="1"/>
      <text x="690" y="438" text-anchor="middle" fill="var(--levi-arch-text-primary)" font-size="13" font-weight="600">Punctuated Equilibrium</text>
      <text x="690" y="454" text-anchor="middle" fill="var(--levi-arch-text-secondary)" font-size="10" font-weight="300">(Every K evals:</text>
      <text x="690" y="468" text-anchor="middle" fill="var(--levi-arch-text-secondary)" font-size="10" font-weight="300">Paradigm Shifts &amp; Variants)</text>
      <rect class="levi-arch-hover-zone" data-tip="pe" x="550" y="410" width="280" height="72" rx="7"/>
    </svg>
  </div>
</div>

<script>
(function () {
  const root = document.getElementById('levi-arch-diagram');
  if (!root) return;

  const descriptions = {
    dspy: {
      title: "DSPy Prompt Optimization (One-time)",
      body: "A one-time bootstrap step using DSPy's MIPROv2 optimizer to automatically tune LLM prompts. This removes prompt sensitivity as a variable, making results more robust when switching models or problems."
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
      title: "Punctuated Equilibrium",
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

Budget enforcement is by construction: the pipeline checks dollars, eval count, and wall time before every LLM call and shuts down cleanly when any limit is hit. Archive access is lock-protected but contention stays low since LLM calls and evaluations happen outside the lock. A one-time DSPy MIPROv2 pass tunes mutation prompts per model before the main loop begins, removing prompt sensitivity as a variable.

<details markdown="1">
<summary>Example: LEVI-evolved EPLB strategy (74.6%)</summary>

One of the programs LEVI discovered for EPLB (Expert Parallel Load Balancing). Replicating and assigning MoE experts across GPUs to minimize imbalance. This strategy uses greedy apportionment to decide replica counts, then a snake-mapping placement to spread hot experts evenly across devices.

```python
import torch

def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    '''
    Rearrange and replicate logical experts across physical GPU slots.
    Optimized for load balance using Greedy Apportionment and Snake Mapping.

    Parameters:
        weight: [layers, 64] load statistics
        num_replicas: 288 (physical experts)
        num_groups: 8
        num_nodes: 4
        num_gpus: 32

    Returns:
        physical_to_logical_map: [layers, 288] (values 0-63)
        logical_to_physical_map: [layers, 64, X] (physical indices or -1)
        expert_count: [layers, 64] (number of replicas per expert)
    '''
    num_layers, num_logical = weight.shape
    device = weight.device
    slots_per_gpu = num_replicas // num_gpus # 9

    # --- 1. Expert Apportionment (Greedy) ---
    # Every expert must have at least 1 replica
    expert_count = torch.ones((num_layers, num_logical), dtype=torch.int64, device=device)

    # Remaining 224 slots per layer
    remaining = num_replicas - num_logical

    # Work with float64 for precision
    w_float = weight.to(torch.float64) + 1e-12
    current_counts = expert_count.clone().to(torch.float64)

    # Assign remaining slots to experts with highest load-per-replica
    # Vectorized across layers
    for _ in range(remaining):
        priority = w_float / current_counts
        best_expert = torch.argmax(priority, dim=1)
        row_indices = torch.arange(num_layers, device=device)
        expert_count[row_indices, best_expert] += 1
        current_counts[row_indices, best_expert] += 1.0

    # --- 2. Map Generation ---
    # Logical IDs and their replica ranks (0, 1, 2...) for every physical slot
    # sorted by load to allow balanced assignment
    rep_load_per_expert = w_float / current_counts

    # Expand logical experts into a flat list of items per layer
    # expert_offsets: [layers, 65]
    expert_offsets = torch.zeros((num_layers, num_logical + 1), dtype=torch.int64, device=device)
    expert_offsets[:, 1:] = torch.cumsum(expert_count, dim=1)

    # seq: [layers, 288] -> maps flat index to logical expert id
    seq = torch.arange(num_replicas, device=device).expand(num_layers, -1)
    logical_ids = torch.searchsorted(expert_offsets, seq, right=True) - 1
    logical_ids = torch.clamp(logical_ids, 0, num_logical - 1)

    # Calculate the instance rank (which replica it is for that expert)
    # rank: [layers, 288]
    ranks = seq - torch.gather(expert_offsets, 1, logical_ids)

    # Get load for each individual replica
    replica_loads = torch.gather(rep_load_per_expert, 1, logical_ids)

    # Sort replicas by load descending
    sort_idx = torch.argsort(replica_loads, dim=1, descending=True)
    sorted_logical = torch.gather(logical_ids, 1, sort_idx)
    sorted_ranks = torch.gather(ranks, 1, sort_idx)

    # --- 3. Snake Mapping for Load Balancing ---
    # We map sorted replicas to GPUs using a zig-zag pattern
    # GPU 0..31, then 31..0, then 0..31...
    gpu_indices = torch.arange(num_gpus, device=device)
    placement_list = []
    for s in range(slots_per_gpu):
        if s % 2 == 0:
            order = gpu_indices
        else:
            order = gpu_indices.flip(0)
        # Physical index = gpu_id * 9 + slot_in_gpu
        placement_list.append(order * slots_per_gpu + s)

    # placement_order: [288]
    placement_order = torch.cat(placement_list)

    physical_to_logical_map = torch.zeros((num_layers, num_replicas), dtype=torch.int64, device=device)
    physical_rank_map = torch.zeros((num_layers, num_replicas), dtype=torch.int64, device=device)

    # Scatter the sorted items into the physical slots defined by snake order
    dest = placement_order.expand(num_layers, -1)
    physical_to_logical_map.scatter_(1, dest, sorted_logical)
    physical_rank_map.scatter_(1, dest, sorted_ranks)

    # --- 4. Inverse Map (Logical to Physical) ---
    max_reps = int(expert_count.max().item())
    logical_to_physical_map = torch.full((num_layers, num_logical, max_reps), -1, dtype=torch.int64, device=device)

    # Prepare indices for scatter
    layer_idx = torch.arange(num_layers, device=device).unsqueeze(1).expand(-1, num_replicas)
    phys_idx = torch.arange(num_replicas, device=device).expand(num_layers, -1)

    # Flattened index for 3D tensor: [layer, expert, rank]
    flat_dest_indices = (
        layer_idx * (num_logical * max_reps) +
        physical_to_logical_map * max_reps +
        physical_rank_map
    ).reshape(-1)

    logical_to_physical_map.view(-1).scatter_(0, flat_dest_indices, phys_idx.reshape(-1))

    return physical_to_logical_map, logical_to_physical_map, expert_count
```
</details>

LEVI will be open-sourced on GitHub soon. Point it at a scoring function and a seed program and it runs until the budget is spent.
