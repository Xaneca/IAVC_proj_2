"""Image classification PR/AP; shared without modification by C and D."""
import csv
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(labels, scores, output, category, split):
    """Save continuous-score PR, AP and deterministic descending ranking."""
    output = Path(output)
    ids = sorted(labels)
    y = np.asarray([labels[i] for i in ids])
    s = np.asarray([scores[i] for i in ids], dtype=float)
    if set(y) != {-1, 1} or not np.isfinite(s).all():
        raise ValueError('Evaluation requires both classes and finite scores')
    precision, recall, thresholds = precision_recall_curve(y, s, pos_label=1)
    ap = float(average_precision_score(y, s))
    write_csv(output/'precision_recall.csv', [dict(precision=float(p), recall=float(r),
        threshold=float(thresholds[i]) if i < len(thresholds) else '')
        for i, (p, r) in enumerate(zip(precision, recall))])
    ranked = sorted(ids, key=lambda i: (-float(scores[i]), i))
    write_csv(output/'ranking.csv', [dict(rank=n, image_id=i, label=labels[i],
        decision_score=float(scores[i]), split=split, category=category)
        for n, i in enumerate(ranked, 1)])
    metrics = dict(category=category, split=split, ap=ap,
        ap_definition='sklearn non-interpolated average precision on continuous SVM scores',
        images=len(ids), positives=int(np.sum(y == 1)), negatives=int(np.sum(y == -1)))
    (output/'ap.json').write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.step(recall, precision, where='post', label=f'AP = {ap:.4f}')
    ax.set(xlabel='Recall', ylabel='Precision', xlim=(0, 1), ylim=(0, 1.02),
           title=f'{category} / background — {split}')
    ax.legend()
    ax.grid(alpha=.25)
    fig.tight_layout()
    fig.savefig(output/'precision_recall.png', dpi=180)
    plt.close(fig)
    return metrics
