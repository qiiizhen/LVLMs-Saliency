"""
Analysis 1: Position-Based Saliency Trend

The saliency heatmap is a 2D lower-triangular matrix where:
  row i = query token i attending to all previous tokens (columns)
  value  = saliency (|attn × grad|) or raw attention weight

By taking the mean of each row, we get one saliency score per token
position in the generated sequence. This lets us ask:
  - Does saliency decay as token position increases?
  - Does the decay differ between hallucinated vs. correct tokens?

PNG fallback: divide the cropped heatmap into N_BINS row-segments,
compute mean luminance per segment as a proxy for row-mean saliency.
Normalize positions to [0, 1] so samples with different sequence
lengths are comparable.
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from scipy import stats
import os

OBS_DIR = '/work/users/z/h/zhenqi/LVLMs-Saliency/observation'
NPY_DIR = '/work/users/z/h/zhenqi/LVLMs-Saliency/npy'
FIG_DIR = '/work/users/z/h/zhenqi/LVLMs-Saliency/figures'
RES_DIR = '/work/users/z/h/zhenqi/LVLMs-Saliency/results'

SAMPLES = [
    '011987', '124952', '137612', '279774',
    '296775', '406451', '428231', '499775',
    '521509', '541223'
]
N_BINS = 10   # number of position segments


# ── Helpers ───────────────────────────────────────────────────────────────────

def rgb_to_luminance(rgb_array):
    return (0.2126 * rgb_array[:, :, 0] +
            0.7152 * rgb_array[:, :, 1] +
            0.0722 * rgb_array[:, :, 2])


def crop_heatmap(png_path):
    """Load PNG and crop away axes/labels to isolate the heatmap pixels."""
    img = Image.open(png_path).convert('RGB')
    arr = np.array(img) / 255.0
    h, w = arr.shape[:2]
    return arr[int(h * 0.08): h - int(h * 0.15),
               int(w * 0.12): w - int(w * 0.12)]


def png_row_scores(png_path, n_bins=N_BINS):
    """
    Approximate per-position saliency from a PNG heatmap.
    Divides the heatmap into n_bins horizontal segments and returns
    mean luminance per segment — one value per position bin.
    """
    crop = crop_heatmap(png_path)
    lum  = rgb_to_luminance(crop)
    h    = lum.shape[0]
    scores = []
    for i in range(n_bins):
        row_start = int(i       * h / n_bins)
        row_end   = int((i + 1) * h / n_bins)
        segment   = lum[row_start:row_end, :]
        scores.append(float(np.mean(segment)))
    return np.array(scores)


def npy_row_scores(npy_path, n_bins=N_BINS):
    """
    Per-position saliency from .npy saliency matrix.
    Takes mean of each row (query token attending to context),
    then resamples to n_bins via interpolation.
    """
    mat = np.load(npy_path)          # shape [seq_len, seq_len]
    row_means = np.array([
        float(np.mean(mat[i, :i+1])) for i in range(mat.shape[0])
    ])
    # Resample to n_bins
    x_orig = np.linspace(0, 1, len(row_means))
    x_new  = np.linspace(0, 1, n_bins)
    return np.interp(x_new, x_orig, row_means)


def load_position_scores(map_type, hmode, name):
    """Load position scores, preferring npy over PNG."""
    npy_path = os.path.join(NPY_DIR, f'onlytext_{hmode}_{name}_saliency.npy')
    png_path = os.path.join(
        OBS_DIR,
        f'onlytext_{hmode}_{name}_ALL_layer_avg_lower_{map_type}_normalized.png'
    )
    if map_type == 'saliency_map' and os.path.exists(npy_path):
        return npy_row_scores(npy_path), 'npy'
    elif os.path.exists(png_path):
        return png_row_scores(png_path), 'png'
    return None, None


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    use_npy = os.path.exists(NPY_DIR) and any(
        f.endswith('.npy') for f in os.listdir(NPY_DIR)
    )
    data_source = 'npy' if use_npy else 'PNG approximation'
    print(f'Data source: {data_source}')
    print(f'N_BINS (position segments): {N_BINS}')

    positions = np.linspace(0, 1, N_BINS)

    results = {}
    for map_type in ['saliency_map', 'weight_map']:
        hall_curves, real_curves = [], []
        for name in SAMPLES:
            h_scores, h_src = load_position_scores(map_type, 'hall', name)
            r_scores, r_src = load_position_scores(map_type, 'real', name)
            if h_scores is None or r_scores is None:
                print(f'  [skip] {name} {map_type}')
                continue
            hall_curves.append(h_scores)
            real_curves.append(r_scores)

        if not hall_curves:
            print(f'No data for {map_type}')
            continue

        hall_arr = np.array(hall_curves)   # [n_samples, N_BINS]
        real_arr = np.array(real_curves)

        hall_mean = hall_arr.mean(axis=0)
        hall_std  = hall_arr.std(axis=0)
        real_mean = real_arr.mean(axis=0)
        real_std  = real_arr.std(axis=0)

        results[map_type] = (hall_mean, hall_std, real_mean, real_std, hall_arr, real_arr)

        print(f'\n{map_type}:')
        print(f'  n_samples = {len(hall_curves)}')

        # Linear regression: does saliency decay with position?
        for label, arr_mean in [('Hall', hall_mean), ('Real', real_mean)]:
            slope, intercept, r, p, _ = stats.linregress(positions, arr_mean)
            print(f'  {label} slope: {slope:+.4f}  (r={r:.3f}, p={p:.4f})')

        # Gap at early vs late positions
        mid = N_BINS // 2
        early_gap = float((real_mean[:mid] - hall_mean[:mid]).mean())
        late_gap  = float((real_mean[mid:] - hall_mean[mid:]).mean())
        print(f'  Gap (real-hall) early positions: {early_gap:+.4f}')
        print(f'  Gap (real-hall) late  positions: {late_gap:+.4f}')

    if not results:
        print('No data loaded.')
        return

    plot(positions, results, data_source)
    save_summary(positions, results, data_source)


def plot(positions, results, data_source):
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)

    n_panels = len(results)
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5), sharey=False)
    if n_panels == 1:
        axes = [axes]

    x_labels = [f'P{i+1}' for i in range(len(positions))]
    titles = {
        'saliency_map': 'Saliency (|Attn × Grad|)',
        'weight_map':   'Attention Weight Only'
    }
    colors = {'hall': '#d62728', 'real': '#1f77b4'}

    for ax, (map_type, (hall_mean, hall_std, real_mean, real_std, hall_arr, real_arr)) \
            in zip(axes, results.items()):

        x = np.arange(len(positions))

        ax.plot(x, real_mean, 'o-', color=colors['real'],
                lw=2, ms=6, label='Correct (real)')
        ax.fill_between(x, real_mean - real_std, real_mean + real_std,
                        color=colors['real'], alpha=0.15)

        ax.plot(x, hall_mean, 's--', color=colors['hall'],
                lw=2, ms=6, label='Hallucinated')
        ax.fill_between(x, hall_mean - hall_std, hall_mean + hall_std,
                        color=colors['hall'], alpha=0.15)

        # Plot individual sample curves (light, transparent)
        for curve in real_arr:
            ax.plot(x, curve, color=colors['real'], lw=0.6, alpha=0.2)
        for curve in hall_arr:
            ax.plot(x, curve, color=colors['hall'], lw=0.6, alpha=0.2)

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_xlabel('Relative Token Position (P1=early, P10=late)')
        ax.set_ylabel('Mean Score')
        ax.set_title(f'{titles.get(map_type, map_type)}\n(n=10, {data_source})')
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle(
        'Position-Based Saliency Trend: Do Hallucinated Tokens Show Different Decay?',
        fontsize=12, fontweight='bold'
    )
    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'position_trend.png')
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f'\nSaved: {out}')
    plt.close()


def save_summary(positions, results, data_source):
    os.makedirs(RES_DIR, exist_ok=True)
    lines = ['=== Analysis 1: Position-Based Saliency Trend ===\n',
             f'Data source: {data_source}\n',
             f'N position bins: {N_BINS}\n\n']
    for map_type, (hall_mean, _, real_mean, _, _, _) in results.items():
        slope_h, _, r_h, p_h, _ = stats.linregress(
            np.linspace(0, 1, N_BINS), hall_mean)
        slope_r, _, r_r, p_r, _ = stats.linregress(
            np.linspace(0, 1, N_BINS), real_mean)
        mid = N_BINS // 2
        lines += [
            f'{map_type}:\n',
            f'  Hall slope: {slope_h:+.4f}  (r={r_h:.3f}, p={p_h:.4f})\n',
            f'  Real slope: {slope_r:+.4f}  (r={r_r:.3f}, p={p_r:.4f})\n',
            f'  Early gap (real-hall): {float((real_mean[:mid]-hall_mean[:mid]).mean()):+.4f}\n',
            f'  Late  gap (real-hall): {float((real_mean[mid:]-hall_mean[mid:]).mean()):+.4f}\n\n',
        ]
    path = os.path.join(RES_DIR, 'position_trend_summary.txt')
    with open(path, 'w') as f:
        f.writelines(lines)
    print(f'Saved: {path}')


if __name__ == '__main__':
    run()