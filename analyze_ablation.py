"""
Ablation study: Saliency (|attention × gradient|) vs Attention-only.

Uses existing PNGs in observation/ — no GPU or npy files required.

Two map types per sample:
  saliency_map : |attn_weight × gradient| — the paper's key signal
  weight_map   : attn_weight alone (no gradient)

If saliency discriminates hallucinated vs correct better than weight alone,
that validates the paper's core claim that the gradient component matters.

Metrics reported per map type:
  - Mean score: Hall vs Real, paired t-test, Cohen's d
  - AUC-ROC: probability a random real score > random hallucinated score
              (0.5 = random, 1.0 = perfect)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from PIL import Image
import os

OBS_DIR = '/work/users/z/h/zhenqi/LVLMs-Saliency/observation'
FIG_DIR = '/work/users/z/h/zhenqi/LVLMs-Saliency/figures'

SAMPLES = [
    '011987', '124952', '137612', '279774',
    '296775', '406451', '428231', '499775',
    '521509', '541223'
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def rgb_to_saliency(rgb_array):
    """Viridis luminance as monotonic proxy for mapped saliency value."""
    lum = (0.2126 * rgb_array[:, :, 0] +
           0.7152 * rgb_array[:, :, 1] +
           0.0722 * rgb_array[:, :, 2])
    mn, mx = lum.min(), lum.max()
    if mx == mn:
        return np.zeros_like(lum)
    return (lum - mn) / (mx - mn)


def load_png(map_type, hmode, name):
    """
    Load a map PNG and return a normalized 2-D saliency array.
    map_type : 'saliency_map'  or  'weight_map'
    hmode    : 'hall'          or  'real'
    """
    fname = (f'onlytext_{hmode}_{name}_ALL_layer_avg_lower'
             f'_{map_type}_normalized.png')
    path = os.path.join(OBS_DIR, fname)
    if not os.path.exists(path):
        return None
    img = Image.open(path).convert('RGB')
    arr = np.array(img) / 255.0
    h, w = arr.shape[:2]
    # Crop axes/labels/colorbar margins (empirical)
    crop = arr[int(h * 0.08): h - int(h * 0.15),
               int(w * 0.12): w - int(w * 0.12)]
    return rgb_to_saliency(crop)


def compute_mean_score(saliency_map):
    """
    Focus on the last few rows of the lower-triangular matrix —
    these rows represent generated tokens attending to prior context.
    """
    n = saliency_map.shape[0]
    last_rows = saliency_map[max(0, n - 5):, :]
    values = last_rows[last_rows > 0.05]      # ignore near-zero padding
    if len(values) == 0:
        return None
    return float(np.mean(values))


def compute_auc(hall_scores, real_scores):
    """
    AUC-ROC via the Wilcoxon–Mann–Whitney statistic.
    = P(score_real > score_hall) for a random pair.
    No sklearn needed.
    """
    total = len(hall_scores) * len(real_scores)
    count = sum(
        (1.0 if r > h else 0.5 if r == h else 0.0)
        for h in hall_scores
        for r in real_scores
    )
    return count / total


def print_stats(hall_vals, real_vals):
    t, p = stats.ttest_rel(hall_vals, real_vals)
    d = ((np.mean(real_vals) - np.mean(hall_vals)) /
         (np.std(hall_vals + real_vals) + 1e-9))
    auc = compute_auc(hall_vals, real_vals)
    print(f"    Hall  : {np.mean(hall_vals):.4f} ± {np.std(hall_vals):.4f}")
    print(f"    Real  : {np.mean(real_vals):.4f} ± {np.std(real_vals):.4f}")
    print(f"    t={t:.3f}, p={p:.4f},  Cohen's d={d:.3f},  AUC={auc:.3f}")
    return d, auc


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    sal_hall, sal_real = [], []
    wgt_hall, wgt_real = [], []
    valid = []

    for name in SAMPLES:
        maps = {
            ('saliency_map', 'hall'): load_png('saliency_map', 'hall', name),
            ('saliency_map', 'real'): load_png('saliency_map', 'real', name),
            ('weight_map',   'hall'): load_png('weight_map',   'hall', name),
            ('weight_map',   'real'): load_png('weight_map',   'real', name),
        }

        if any(v is None for v in maps.values()):
            missing = [k for k, v in maps.items() if v is None]
            print(f"[skip] {name}: missing {missing}")
            continue

        scores = {k: compute_mean_score(v) for k, v in maps.items()}
        if any(v is None for v in scores.values()):
            print(f"[skip] {name}: empty heatmap region")
            continue

        sh = scores[('saliency_map', 'hall')]
        sr = scores[('saliency_map', 'real')]
        wh = scores[('weight_map',   'hall')]
        wr = scores[('weight_map',   'real')]

        sal_hall.append(sh); sal_real.append(sr)
        wgt_hall.append(wh); wgt_real.append(wr)
        valid.append(name)

        print(f"  {name}:  sal_diff={sr - sh:+.4f}  wgt_diff={wr - wh:+.4f}")

    if len(valid) < 2:
        print("Not enough samples.")
        return

    n = len(valid)
    print(f"\n{'=' * 55}")
    print(f"Results across {n} samples")
    print(f"{'=' * 55}")

    print("\nSaliency maps  (|attention × gradient|):")
    sal_d, sal_auc = print_stats(sal_hall, sal_real)

    print("\nWeight maps  (attention only — no gradient):")
    wgt_d, wgt_auc = print_stats(wgt_hall, wgt_real)

    print(f"\nAblation — does the gradient component help?")
    print(f"  Saliency  AUC = {sal_auc:.3f}  (Cohen's d = {sal_d:.3f})")
    print(f"  Attention AUC = {wgt_auc:.3f}  (Cohen's d = {wgt_d:.3f})")
    gain = sal_auc - wgt_auc
    verdict = "YES" if gain > 0 else "NO"
    print(f"  AUC gain from gradient: {gain:+.3f}  → gradient helps? {verdict}")

    plot(valid, sal_hall, sal_real, wgt_hall, wgt_real, sal_auc, wgt_auc)


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot(samples, sal_hall, sal_real, wgt_hall, wgt_real, sal_auc, wgt_auc):
    os.makedirs(FIG_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('Saliency vs Attention-Only: Hallucination Discrimination',
                 fontsize=13, fontweight='bold')

    # ── Panel 1: Saliency boxplot ──
    ax = axes[0]
    bp = ax.boxplot([sal_hall, sal_real],
                    labels=['Hallucinated', 'Correct'],
                    patch_artist=True, widths=0.5,
                    medianprops=dict(color='black', linewidth=2))
    bp['boxes'][0].set_facecolor('#d62728'); bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor('#1f77b4'); bp['boxes'][1].set_alpha(0.7)
    for i, vals in enumerate([sal_hall, sal_real], 1):
        ax.scatter([i] * len(vals), vals, color='black', s=30, zorder=3)
    for h, r in zip(sal_hall, sal_real):
        ax.plot([1, 2], [h, r], color='gray', lw=0.8, alpha=0.5)
    ax.set_title(f'Saliency  (|attn × grad|)\nAUC = {sal_auc:.3f}', fontsize=11)
    ax.set_ylabel('Mean Score')

    # ── Panel 2: Attention-only boxplot ──
    ax = axes[1]
    bp = ax.boxplot([wgt_hall, wgt_real],
                    labels=['Hallucinated', 'Correct'],
                    patch_artist=True, widths=0.5,
                    medianprops=dict(color='black', linewidth=2))
    bp['boxes'][0].set_facecolor('#ff7f0e'); bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor('#2ca02c'); bp['boxes'][1].set_alpha(0.7)
    for i, vals in enumerate([wgt_hall, wgt_real], 1):
        ax.scatter([i] * len(vals), vals, color='black', s=30, zorder=3)
    for h, r in zip(wgt_hall, wgt_real):
        ax.plot([1, 2], [h, r], color='gray', lw=0.8, alpha=0.5)
    ax.set_title(f'Attention Only  (no gradient)\nAUC = {wgt_auc:.3f}', fontsize=11)
    ax.set_ylabel('Mean Score')

    # ── Panel 3: AUC bar chart (ablation summary) ──
    ax = axes[2]
    labels = ['Saliency\n(|attn×grad|)', 'Attention\n(no grad)']
    aucs   = [sal_auc, wgt_auc]
    colors = ['#1f77b4', '#ff7f0e']
    bars = ax.bar(labels, aucs, color=colors, alpha=0.85, width=0.45,
                  edgecolor='black', linewidth=0.8)
    ax.axhline(0.5, color='gray', linestyle='--', lw=1.5, label='Random (AUC=0.5)')
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('AUC-ROC')
    ax.set_title('Ablation: Does Gradient Help?\n(higher = better discrimination)',
                 fontsize=11)
    ax.legend(fontsize=9)
    for bar, val in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 0.03,
                f'{val:.3f}', ha='center', va='bottom',
                fontsize=12, fontweight='bold')

    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'ablation_saliency_vs_attention.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out}")
    plt.close()


if __name__ == '__main__':
    run()