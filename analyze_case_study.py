"""
Analysis 2: Qualitative Case Study

Side-by-side comparison of saliency heatmaps for three selected samples:
  Case A (541223) — largest positive gap (real >> hall, diff=+0.2816)
                    Most supportive of the paper's hypothesis.
  Case B (296775) — median gap (diff=+0.0508)
                    Typical/representative sample.
  Case C (428231) — anomalous (hall > real, diff=-0.2589)
                    Contradicts hypothesis — hallucinated token had
                    HIGHER saliency. Worth investigating.

Each case: 2×2 figure
  Row 1: Saliency map (hall) | Saliency map (real)
  Row 2: Weight map  (hall) | Weight map  (real)
"""

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

OBS_DIR = '/work/users/z/h/zhenqi/LVLMs-Saliency/observation'
FIG_DIR = '/work/users/z/h/zhenqi/LVLMs-Saliency/figures'
RES_DIR = '/work/users/z/h/zhenqi/LVLMs-Saliency/results'

# From analyze_ablation.py output
# sal_diff = real_mean - hall_mean
SAMPLE_DIFFS = {
    '011987': +0.0890,
    '124952': -0.0109,
    '137612': -0.1009,
    '279774': +0.0356,
    '296775': +0.0508,   # Case B: median positive gap
    '406451': +0.1608,
    '428231': -0.2589,   # Case C: anomalous (hall > real)
    '499775': +0.0865,
    '521509': +0.1492,
    '541223': +0.2816,   # Case A: largest positive gap
}

CASES = {
    'A': ('541223', '+0.28', 'Most Supportive: Real saliency >> Hallucinated'),
    'B': ('296775', '+0.05', 'Typical Sample: Moderate positive gap'),
    'C': ('428231', '-0.26', 'Anomalous: Hallucinated saliency > Correct (contradicts hypothesis)'),
}

HALL_SCORES = {
    '541223': 0.2474, '296775': 0.3393, '428231': 0.5412,
}
REAL_SCORES = {
    '541223': 0.5290, '296775': 0.3902, '428231': 0.2823,
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_png(map_type, hmode, name):
    fname = (f'onlytext_{hmode}_{name}_ALL_layer_avg_lower'
             f'_{map_type}_normalized.png')
    path = os.path.join(OBS_DIR, fname)
    if not os.path.exists(path):
        print(f'  [missing] {fname}')
        return None
    return np.array(Image.open(path).convert('RGB')) / 255.0


def mean_score(map_type, hmode, name):
    img = load_png(map_type, hmode, name)
    if img is None:
        return float('nan')
    h, w = img.shape[:2]
    crop = img[int(h * 0.08): h - int(h * 0.15),
               int(w * 0.12): w - int(w * 0.12)]
    lum  = (0.2126 * crop[:, :, 0] +
            0.7152 * crop[:, :, 1] +
            0.0722 * crop[:, :, 2])
    mn, mx = lum.min(), lum.max()
    if mx == mn:
        return 0.0
    norm   = (lum - mn) / (mx - mn)
    n      = norm.shape[0]
    values = norm[max(0, n - 5):, :][norm[max(0, n - 5):, :] > 0.05]
    return float(np.mean(values)) if len(values) > 0 else 0.0


# ── Per-case figure ───────────────────────────────────────────────────────────

def make_case_figure(case_label, name, diff_str, description):
    """2×2 panel: [saliency hall | saliency real] / [weight hall | weight real]"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(
        f'Case {case_label} — Image {name}  |  Score gap (real−hall) = {diff_str}\n{description}',
        fontsize=12, fontweight='bold'
    )

    panels = [
        (0, 0, 'saliency_map', 'hall', 'Saliency  |attn × grad|  —  Hallucinated'),
        (0, 1, 'saliency_map', 'real', 'Saliency  |attn × grad|  —  Correct'),
        (1, 0, 'weight_map',   'hall', 'Attention Weight Only  —  Hallucinated'),
        (1, 1, 'weight_map',   'real', 'Attention Weight Only  —  Correct'),
    ]

    for row, col, map_type, hmode, title in panels:
        ax = axes[row, col]
        img = load_png(map_type, hmode, name)
        score = mean_score(map_type, hmode, name)

        if img is not None:
            ax.imshow(img, aspect='auto')
            ax.set_title(f'{title}\n(mean score = {score:.4f})', fontsize=10)
        else:
            ax.text(0.5, 0.5, 'File not found', ha='center', va='center',
                    transform=ax.transAxes, fontsize=12, color='red')
            ax.set_title(title, fontsize=10)

        ax.axis('off')

    # Add annotation for anomalous case C
    if case_label == 'C':
        fig.text(
            0.5, 0.01,
            'Note: In this sample, the hallucinated token shows HIGHER saliency than the correct token.\n'
            'Hypothesis: the hallucinated object ("two") may be semantically proximate to a salient '
            'region (the kite / beach crowd), causing the model to attend strongly but incorrectly.',
            ha='center', fontsize=9, style='italic',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#fff3cd', alpha=0.8)
        )
        plt.subplots_adjust(bottom=0.12)

    plt.tight_layout(rect=[0, 0.05 if case_label == 'C' else 0, 1, 1])
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(RES_DIR, exist_ok=True)
    out = os.path.join(FIG_DIR, f'case_study_{case_label}.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    print(f'Saved: {out}')
    plt.close()
    return out


# ── Summary figure (3 cases × saliency maps only) ────────────────────────────

def make_summary_figure():
    """Compact 3×2 summary: one row per case, saliency hall + real only."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 16))
    fig.suptitle(
        'Case Study Summary: Saliency Maps (Hall vs. Correct)\n'
        '"Hallucination Begins Where Saliency Drops" — Reproduction',
        fontsize=13, fontweight='bold'
    )

    row_labels = []
    for row_idx, (case_label, (name, diff_str, description)) in enumerate(CASES.items()):
        hall_score = mean_score('saliency_map', 'hall', name)
        real_score = mean_score('saliency_map', 'real', name)

        for col_idx, hmode in enumerate(['hall', 'real']):
            ax = axes[row_idx, col_idx]
            img = load_png('saliency_map', hmode, name)
            score = hall_score if hmode == 'hall' else real_score
            hmode_label = 'Hallucinated' if hmode == 'hall' else 'Correct'

            if img is not None:
                ax.imshow(img, aspect='auto')
            else:
                ax.set_facecolor('#f0f0f0')
                ax.text(0.5, 0.5, 'Missing', ha='center', va='center',
                        transform=ax.transAxes, color='red')

            title = f'Case {case_label} ({name}) — {hmode_label}\nscore = {score:.4f}'
            ax.set_title(title, fontsize=9)
            ax.axis('off')

            # Case C border in orange to highlight anomaly
            if case_label == 'C':
                for spine in ax.spines.values():
                    spine.set_edgecolor('#ff7f0e')
                    spine.set_linewidth(3)
                    spine.set_visible(True)

        label = (f'Case {case_label}: gap={diff_str}\n'
                 + description.split(':')[0])
        row_labels.append(label)

    # Add row labels on the left
    for row_idx, label in enumerate(row_labels):
        axes[row_idx, 0].annotate(
            label, xy=(0, 0.5), xytext=(-10, 0),
            xycoords='axes fraction', textcoords='offset points',
            ha='right', va='center', fontsize=8, rotation=0,
            color='gray'
        )

    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'case_study_summary.png')
    plt.savefig(out, dpi=120, bbox_inches='tight')
    print(f'Saved: {out}')
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def run():
    print('=== Analysis 2: Qualitative Case Study ===\n')
    print('Cases selected based on sal_diff (real_mean - hall_mean):')
    for case_label, (name, diff_str, desc) in CASES.items():
        h = mean_score('saliency_map', 'hall', name)
        r = mean_score('saliency_map', 'real', name)
        print(f'  Case {case_label} ({name}): hall={h:.4f}  real={r:.4f}  '
              f'diff={diff_str}  — {desc.split(":")[0]}')

    print()
    for case_label, (name, diff_str, description) in CASES.items():
        print(f'Generating Case {case_label} figure ({name})...')
        make_case_figure(case_label, name, diff_str, description)

    print('Generating summary figure...')
    make_summary_figure()
    print('\nDone.')


if __name__ == '__main__':
    run()
