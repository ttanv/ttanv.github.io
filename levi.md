---
layout: blog
title: "LEVI: LLM-Guided Evolution for the Price of a Cup of Coffee"
subtitle: "An open-source evolutionary framework for algorithmic discovery."
date: 2026-02-01
permalink: /levi
---

<details markdown="1">
<summary><strong>Example: LEVI-evolved EPLB strategy (74.6%)</strong></summary>

One of the programs LEVI discovered for EPLB (Expert Parallel Load Balancing) -- replicating and assigning MoE experts across GPUs to minimize imbalance. This strategy uses greedy apportionment to decide replica counts, then a snake-mapping placement to spread hot experts evenly across devices.

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

## TLDR

Existing LLM-based evolutionary systems (OpenEvolve, ShinkaEvolve) have **weak diversity mechanisms**, converge early, and compensate by throwing expensive models at the problem. This leads to bloated costs and unnecessary complexity.

LEVI fixes the root cause: **CVT-MAP-Elites** with **AST-based behavioral fingerprinting** keeps a diverse archive where each cell holds the best solution for its behavioral niche. Different algorithmic approaches naturally land in different cells, so the system never collapses onto one strategy. We split mutations into two tiers: **cheap small models** (e.g. Qwen-30B) for hundreds of narrow mutations, and a **larger model** (Gemini Flash) used sparingly for paradigm shifts.

Result: ***better*** scores than OpenEvolve, ShinkaEvolve, and GEPA on ADRS benchmarks at **1.5-6.7x lower cost**.

**LEVI will be open-sourced on GitHub soon.** Point it at a scoring function and a seed program and it runs until the budget is spent.

### ADRS Benchmark Results (% score)

[ADRS](https://ucbskyadrs.github.io/) is a benchmark suite from UC Berkeley for evaluating LLM-guided optimization on real-world systems problems -- cloud scheduling, load balancing, congestion control, SQL optimization, and more.

<div class="adrs-dashboard">
  <!-- Summary Stats Row -->
  <div class="adrs-summary">
    <div class="adrs-stat adrs-stat-highlight">
      <div class="adrs-stat-label">LEVI Average</div>
      <div class="adrs-stat-value">76.5</div>
    </div>
    <div class="adrs-stat">
      <div class="adrs-stat-label">2nd Best (GEPA)</div>
      <div class="adrs-stat-value">71.9</div>
    </div>
    <div class="adrs-stat adrs-stat-highlight">
      <div class="adrs-stat-label">LEVI Lead</div>
      <div class="adrs-stat-value">+4.6</div>
    </div>
    <div class="adrs-stat adrs-stat-highlight">
      <div class="adrs-stat-label">Best on Benchmarks</div>
      <div class="adrs-stat-value">8 / 8</div>
    </div>
  </div>

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

  <!-- Chart -->
  <div class="adrs-chart-container">
    <canvas id="adrs-chart"></canvas>
  </div>

  <!-- Table -->
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
    });
  } else {
    createChart();
    initInteractivity();
  }
})();
</script>

### Cost per Problem

Because LEVI's diversity mechanism prevents early stagnation, we don't need expensive models to compensate. Most mutations use small models (Qwen-30B) at fractions of a cent each, and the system runs for more generations instead. The result: **1.5-6.7x cheaper** per problem while matching or beating systems that rely on Gemini 3.0 Pro and GPT 5.2.

<div class="adrs-dashboard">
  <div class="adrs-summary">
    <div class="adrs-stat adrs-stat-highlight">
      <div class="adrs-stat-label">LEVI Typical Cost</div>
      <div class="adrs-stat-value">$4.50</div>
    </div>
    <div class="adrs-stat">
      <div class="adrs-stat-label">Baseline Range</div>
      <div class="adrs-stat-value">$15–$30</div>
    </div>
    <div class="adrs-stat adrs-stat-highlight">
      <div class="adrs-stat-label">Cost Reduction</div>
      <div class="adrs-stat-value">1.5–6.7x</div>
    </div>
  </div>

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

<div class="diagram">
  <img src="/images/levi_overview.png" alt="LEVI System Overview" class="img-light" />
  <img src="/images/levi_overview_dark.png" alt="LEVI System Overview" class="img-dark" />
</div>

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
