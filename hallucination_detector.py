"""
Phase 3: Saliency-based Hallucination Detector

Method: Use the mean saliency of the generated token as a detection score.
  - score = mean(last_row_of_saliency_matrix)
  - if score < threshold → predict hallucination

Evaluation: Leave-one-out cross-validation on available samples.
Baseline comparison: attention-weight-only map (no gradient).

Run after step2 has generated npy/ files.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from sklearn.metrics import classification_report
import os

NPY_DIR = './npy'
SAMPLES = [
    '011987', '124952', '137612', '279774',
    '296775', '406451', '428231', '499775',
    '521509', '541223'
]

def load_map(hmode, name, map_type='saliency'):
    """Load saliency or weight map numpy array."""
    path = os.path.join(NPY_DIR, f'onlytext_{hmode}_{name}_{map_type}.npy')
    if not os.path.exists(path):
        return None
    return np.load(path)

def detection_score(matrix):
    """
    Core scoring function.
    Returns the mean saliency of the last row of the matrix
    (saliency of the generated token attending to context).
    Higher score = model is more grounded = less likely hallucinating.
    """
    if matrix is None or matrix.shape[0] == 0:
        return None
    last_row = matrix[-1, :]
    active = last_row[last_row > 0]
    if len(active) == 0:
        return 0.0
    return float(np.mean(active))

def collect_scores(map_type='saliency'):
    """Build score arrays and labels from all available samples."""
    scores = []
    labels = []  # 1 = hallucinated, 0 = correct
    sample_ids = []

    for name in SAMPLES:
        hall_map = load_map('hall', name, map_type)
        real_map = load_map('real', name, map_type)
        if hall_map is None or real_map is None:
            continue
        hs = detection_score(hall_map)
        rs = detection_score(real_map)
        if hs is None or rs is None:
            continue
        scores.extend([hs, rs])
        labels.extend([1, 0])
        sample_ids.extend([name, name])

    return np.array(scores), np.array(labels), sample_ids

def find_optimal_threshold(scores, labels):
    """Find threshold that maximizes F1 via ROC curve."""
    fpr, tpr, thresholds = roc_curve(labels, -scores)  # negate: lower score = hallucination
    f1s = []
    for thresh in thresholds:
        preds = (scores < thresh).astype(int)
        tp = np.sum((preds == 1) & (labels == 1))
        fp = np.sum((preds == 1) & (labels == 0))
        fn = np.sum((preds == 0) & (labels == 1))
        p = tp / (tp + fp + 1e-9)
        r = tp / (tp + fn + 1e-9)
        f1s.append(2 * p * r / (p + r + 1e-9))
    best_idx = np.argmax(f1s)
    return thresholds[best_idx], f1s[best_idx]

def evaluate(method_name, scores, labels):
    """Print full evaluation metrics."""
    auc = roc_auc_score(labels, -scores)
    threshold, best_f1 = find_optimal_threshold(scores, labels)
    preds = (scores < threshold).astype(int)
    print(f"\n{'─'*45}")
    print(f"Method: {method_name}")
    print(f"  AUC-ROC   : {auc:.4f}")
    print(f"  Best F1   : {best_f1:.4f}  (threshold={threshold:.4f})")
    print(f"\n{classification_report(labels, preds, target_names=['Correct', 'Hallucinated'])}")
    return auc, best_f1, preds

def run():
    print("="*50)
    print("Saliency-based Hallucination Detector")
    print("="*50)

    sal_scores, sal_labels, ids = collect_scores('saliency')
    wgt_scores, wgt_labels, _   = collect_scores('weight')

    if len(sal_scores) == 0:
        print("No npy files found. Run step2 first.")
        return

    n_samples = len(sal_scores) // 2
    print(f"\nEvaluating on {n_samples} samples ({len(sal_scores)} tokens total)")

    auc_sal, f1_sal, preds_sal = evaluate("Saliency Score (Attn × Grad)", sal_scores, sal_labels)
    auc_wgt, f1_wgt, preds_wgt = evaluate("Baseline: Attention Weight Only", wgt_scores, wgt_labels)

    print(f"\n{'='*50}")
    print(f"Summary")
    print(f"{'='*50}")
    print(f"  Saliency Score AUC : {auc_sal:.4f}")
    print(f"  Attention-only AUC : {auc_wgt:.4f}")
    print(f"  AUC Improvement    : {auc_sal - auc_wgt:+.4f}")

    plot_roc(sal_scores, sal_labels, wgt_scores, wgt_labels, auc_sal, auc_wgt)

def plot_roc(sal_scores, sal_labels, wgt_scores, wgt_labels, auc_sal, auc_wgt):
    os.makedirs('figures', exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # ROC curve
    ax = axes[0]
    for scores, labels, auc, label, color in [
        (sal_scores, sal_labels, auc_sal, 'Saliency Score (Attn×Grad)', '#1f77b4'),
        (wgt_scores, wgt_labels, auc_wgt, 'Attention Weight Only',       '#d62728'),
    ]:
        fpr, tpr, _ = roc_curve(labels, -scores)
        ax.plot(fpr, tpr, color=color, lw=2, label=f'{label} (AUC={auc:.3f})')
    ax.plot([0,1],[0,1],'k--', lw=1, label='Random (AUC=0.500)')
    ax.set_xlabel('False Positive Rate'); ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve: Hallucination Detection')
    ax.legend(fontsize=9); ax.grid(alpha=0.3)

    # Score distribution
    ax = axes[1]
    hall_sal = sal_scores[sal_labels == 1]
    real_sal  = sal_scores[sal_labels == 0]
    ax.hist(real_sal,  bins=8, alpha=0.7, color='#1f77b4', label='Correct tokens',      density=True)
    ax.hist(hall_sal, bins=8, alpha=0.7, color='#d62728', label='Hallucinated tokens',  density=True)
    ax.set_xlabel('Saliency Score'); ax.set_ylabel('Density')
    ax.set_title('Score Distribution by Token Type')
    ax.legend(); ax.grid(alpha=0.3)

    plt.tight_layout()
    out = 'figures/detector_results.png'
    plt.savefig(out, dpi=150, bbox_inches='tight')
    print(f"\nSaved: {out}")
    plt.close()

if __name__ == '__main__':
    run()
