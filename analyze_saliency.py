"""
Phase 2: Quantitative analysis of saliency differences
between hallucinated and correct tokens.

Run after step2 has generated .npy files in the npy/ directory.
Does not require GPU.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

NPY_DIR = './npy'
SAMPLES = [
    '011987', '124952', '137612', '279774',
    '296775', '406451', '428231', '499775',
    '521509', '541223'
]

def load_saliency(hmode, name):
    path = os.path.join(NPY_DIR, f'onlytext_{hmode}_{name}_saliency.npy')
    if not os.path.exists(path):
        return None
    return np.load(path)

def compute_scores(saliency_map):
    """
    From the saliency matrix, extract the last row —
    this is the saliency of the generated (target) token
    attending to all prior tokens.
    Returns three scores:
      - mean_saliency : average saliency of the target token
      - max_saliency  : peak saliency of the target token
      - entropy       : how spread out the attention is (lower = more focused)
    """
    last_row = saliency_map[-1, :]  # generated token's saliency to context
    last_row = last_row[last_row > 0]  # ignore padded zeros
    if len(last_row) == 0:
        return None, None, None
    mean_s = float(np.mean(last_row))
    max_s  = float(np.max(last_row))
    # Normalized entropy: higher = more diffuse attention
    p = last_row / (last_row.sum() + 1e-9)
    entropy = float(-np.sum(p * np.log(p + 1e-9)) / np.log(len(p) + 1))
    return mean_s, max_s, entropy

def run_analysis():
    hall_means, real_means = [], []
    hall_maxs,  real_maxs  = [], []
    hall_entropies, real_entropies = [], []
    valid_samples = []

    for name in SAMPLES:
        hall_map = load_saliency('hall', name)
        real_map = load_saliency('real', name)
        if hall_map is None or real_map is None:
            print(f"[skip] {name}: missing npy file")
            continue

        hm, hx, he = compute_scores(hall_map)
        rm, rx, re = compute_scores(real_map)
        if None in (hm, rm):
            continue

        hall_means.append(hm); real_means.append(rm)
        hall_maxs.append(hx);  real_maxs.append(rx)
        hall_entropies.append(he); real_entropies.append(re)
        valid_samples.append(name)

    if len(valid_samples) == 0:
        print("No npy files found. Run step2 first to generate them.")
        return

    print(f"\n{'='*50}")
    print(f"Analysis on {len(valid_samples)} samples: {valid_samples}")
    print(f"{'='*50}")

    for metric, hall_vals, real_vals, label in [
        ('Mean Saliency',    hall_means,    real_means,    'higher = more grounded'),
        ('Max Saliency',     hall_maxs,     real_maxs,     'higher = sharper attention'),
        ('Saliency Entropy', hall_entropies, real_entropies,'higher = more diffuse'),
    ]:
        t_stat, p_val = stats.ttest_rel(hall_vals, real_vals)
        d = (np.mean(real_vals) - np.mean(hall_vals)) / (np.std(hall_vals + real_vals) + 1e-9)
        print(f"\n{metric} ({label})")
        print(f"  Hall : mean={np.mean(hall_vals):.4f}  std={np.std(hall_vals):.4f}")
        print(f"  Real : mean={np.mean(real_vals):.4f}  std={np.std(real_vals):.4f}")
        print(f"  Paired t-test: t={t_stat:.3f}, p={p_val:.4f}")
        print(f"  Cohen's d    : {d:.3f}")

    plot_results(valid_samples, hall_means, real_means,
                 hall_entropies, real_entropies)

def plot_results(samples, hall_means, real_means, hall_ents, real_ents):
    os.makedirs('figures', exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Box plot: mean saliency
    ax = axes[0]
    ax.boxplot([hall_means, real_means], labels=['Hallucinated', 'Correct'],
               patch_artist=True,
               boxprops=dict(facecolor='#d62728', alpha=0.6),
               medianprops=dict(color='black', linewidth=2))
    boxes = ax.patches
    boxes[1].set_facecolor('#1f77b4')
    boxes[1].set_alpha(0.6)
    ax.set_title('Mean Saliency of Generated Token')
    ax.set_ylabel('Normalized Saliency Score')
    ax.set_xlabel('Token Type')

    # Bar plot: per-sample comparison
    ax = axes[1]
    x = np.arange(len(samples))
    width = 0.35
    ax.bar(x - width/2, hall_means, width, label='Hallucinated', color='#d62728', alpha=0.7)
    ax.bar(x + width/2, real_means,  width, label='Correct',      color='#1f77b4', alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(samples, rotation=45, ha='right', fontsize=8)
    ax.set_title('Per-Sample Saliency Comparison')
    ax.set_ylabel('Mean Saliency Score')
    ax.legend()

    plt.tight_layout()
    plt.savefig('figures/saliency_comparison.png', dpi=150, bbox_inches='tight')
    print("\nSaved: figures/saliency_comparison.png")
    plt.close()

if __name__ == '__main__':
    run_analysis()
