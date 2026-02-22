"""
Preliminary analysis using existing PNG saliency maps in observation/.
Extracts approximate saliency intensity by inverting the viridis colormap.
Does NOT require GPU or npy files.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from PIL import Image
import os

OBS_DIR = '/work/users/z/h/zhenqi/LVLMs-Saliency/observation'
SAMPLES = [
    '011987', '124952', '137612', '279774',
    '296775', '406451', '428231', '499775',
    '521509', '541223'
]

def rgb_to_saliency(rgb_array):
    """
    Viridis luminance increases monotonically with saliency value.
    Use perceived luminance as a memory-efficient proxy (no large intermediate arrays).
    """
    lum = (0.2126 * rgb_array[:, :, 0] +
           0.7152 * rgb_array[:, :, 1] +
           0.0722 * rgb_array[:, :, 2])
    mn, mx = lum.min(), lum.max()
    if mx == mn:
        return np.zeros_like(lum)
    return (lum - mn) / (mx - mn)

def load_saliency_from_png(hmode, name):
    path = os.path.join(OBS_DIR, f'onlytext_{hmode}_{name}_ALL_layer_avg_lower_saliency_map_normalized.png')
    if not os.path.exists(path):
        return None
    img = Image.open(path).convert('RGB')
    arr = np.array(img) / 255.0  # (H, W, 3)

    # Crop: remove axes/labels/colorbar (approximately 15% margin on each side)
    h, w = arr.shape[:2]
    margin_top  = int(h * 0.08)
    margin_bot  = int(h * 0.15)
    margin_left = int(w * 0.12)
    margin_right= int(w * 0.12)
    heatmap = arr[margin_top:h-margin_bot, margin_left:w-margin_right]

    saliency = rgb_to_saliency(heatmap)
    return saliency

def compute_scores(saliency_map):
    """
    Aggregate stats from the saliency heatmap.
    Since the map is a lower-triangular token×token matrix,
    the bottom-right corner represents the generated token's saliency.
    """
    # Last few rows = the generated/target token attending to context
    n = saliency_map.shape[0]
    last_rows = saliency_map[max(0, n-5):, :]   # last 5 rows as proxy
    values = last_rows[last_rows > 0.05]         # ignore near-zero padding
    if len(values) == 0:
        return None, None, None
    mean_s   = float(np.mean(values))
    max_s    = float(np.max(values))
    p        = values / (values.sum() + 1e-9)
    entropy  = float(-np.sum(p * np.log(p + 1e-9)) / np.log(len(p) + 1))
    return mean_s, max_s, entropy

def run():
    hall_means, real_means = [], []
    hall_maxs,  real_maxs  = [], []
    valid = []

    for name in SAMPLES:
        hmap = load_saliency_from_png('hall', name)
        rmap = load_saliency_from_png('real', name)
        if hmap is None:
            print(f"[skip] {name}: PNG not found")
            continue

        hm, hx, _ = compute_scores(hmap)
        rm, rx, _ = compute_scores(rmap) if rmap is not None else (None, None, None)
        if None in (hm, rm):
            continue

        hall_means.append(hm); real_means.append(rm)
        hall_maxs.append(hx);  real_maxs.append(rx)
        valid.append(name)
        print(f"{name}:  hall_mean={hm:.4f}  real_mean={rm:.4f}  diff={rm-hm:+.4f}")

    if len(valid) < 2:
        print("Not enough samples.")
        return

    print(f"\n{'='*55}")
    print(f"Results across {len(valid)} samples")
    print(f"{'='*55}")
    for label, hv, rv in [('Mean Saliency', hall_means, real_means),
                           ('Max Saliency',  hall_maxs,  real_maxs)]:
        t, p = stats.ttest_rel(hv, rv)
        d = (np.mean(rv) - np.mean(hv)) / (np.std(hv + rv) + 1e-9)
        print(f"\n{label}:")
        print(f"  Hallucinated: {np.mean(hv):.4f} ± {np.std(hv):.4f}")
        print(f"  Correct     : {np.mean(rv):.4f} ± {np.std(rv):.4f}")
        print(f"  t={t:.3f}, p={p:.4f}, Cohen's d={d:.3f}")

    plot(valid, hall_means, real_means)

def plot(samples, hall_means, real_means):
    os.makedirs('figures', exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Boxplot
    ax = axes[0]
    bp = ax.boxplot([hall_means, real_means],
                    labels=['Hallucinated', 'Correct'],
                    patch_artist=True, widths=0.5,
                    medianprops=dict(color='black', linewidth=2))
    bp['boxes'][0].set_facecolor('#d62728'); bp['boxes'][0].set_alpha(0.7)
    bp['boxes'][1].set_facecolor('#1f77b4'); bp['boxes'][1].set_alpha(0.7)
    # Overlay individual points
    for i, vals in enumerate([hall_means, real_means], 1):
        ax.scatter([i]*len(vals), vals, color='black', s=30, zorder=3)
    for h, r in zip(hall_means, real_means):
        ax.plot([1, 2], [h, r], color='gray', lw=0.8, alpha=0.5)
    ax.set_title('Saliency Score: Hall vs Real', fontsize=13)
    ax.set_ylabel('Mean Saliency (approximate)')

    # Per-sample bar
    ax = axes[1]
    x = np.arange(len(samples))
    ax.bar(x - 0.18, hall_means, 0.35, label='Hallucinated', color='#d62728', alpha=0.75)
    ax.bar(x + 0.18, real_means,  0.35, label='Correct',      color='#1f77b4', alpha=0.75)
    ax.set_xticks(x)
    ax.set_xticklabels(samples, rotation=45, ha='right', fontsize=9)
    ax.set_title('Per-Sample Comparison', fontsize=13)
    ax.set_ylabel('Mean Saliency Score')
    ax.legend()

    plt.tight_layout()
    os.makedirs('/work/users/z/h/zhenqi/LVLMs-Saliency/figures', exist_ok=True)
    out = '/work/users/z/h/zhenqi/LVLMs-Saliency/figures/preliminary_saliency_comparison.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out}")
    plt.close()

if __name__ == '__main__':
    run()
