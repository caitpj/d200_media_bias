"""
visualise.py — Generate secondary analysis sentiment bar chart.

Reads stage2_on_target.tsv (on-target mentions only) and plots
mean sentiment relative to article-type mean for each party.

Usage:
    python scripts/visualise.py
"""

import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Matplotlib configuration — ICML 2025 style
#
# Key principle: design at the exact width LaTeX will render (3.25in for
# single-column), so \includegraphics[width=\columnwidth]{...} applies
# no scaling.  All font sizes therefore appear 1:1 in the final PDF.
#
# ICML guidelines require:
#   - figure text ≥ caption font size (9pt)
#   - lines ≥ 0.5pt thick
#   - vector format (PDF) for plots
#   - high contrast, legible at print resolution
# ---------------------------------------------------------------------------

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 8.5,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.6,
    'xtick.major.width': 0.6,
    'ytick.major.width': 0.6,
    'xtick.major.size': 3.5,
    'ytick.major.size': 3.5,
})

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data",
                         "secondary", "stage2_on_target.tsv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "report", "figures")

df = pd.read_csv(DATA_PATH, sep="\t")

parties_order = ['reform_uk', 'conservative', 'labour', 'plaid_cymru']
party_labels = ['Reform UK', 'Tory', 'Labour', 'Plaid Cymru']

# ---------------------------------------------------------------------------
# Compute article-type baselines and deviations
# ---------------------------------------------------------------------------

baselines = df.groupby('article_type')['sentiment_score'].mean()
print(f"Article-type baselines:")
print(f"  News mean:    {baselines['news']:.4f}")
print(f"  Opinion mean: {baselines['opinion']:.4f}")

news_dev, opin_dev = [], []
news_ci, opin_ci = [], []

for party in parties_order:
    for atype, dev_list, ci_list in [('news', news_dev, news_ci),
                                      ('opinion', opin_dev, opin_ci)]:
        scores = df[(df['party'] == party) &
                    (df['article_type'] == atype)]['sentiment_score']
        n = len(scores)
        mean = scores.mean()
        std = scores.std(ddof=1)
        deviation = mean - baselines[atype]
        ci = 1.96 * std / np.sqrt(n) if n > 0 else 0

        dev_list.append(deviation)
        ci_list.append(ci)
        print(f"  {party:15s} {atype:8s}  n={n:3d}  "
              f"dev={deviation:+.3f}  ci={ci:.3f}")

# ---------------------------------------------------------------------------
# Plot — sized to ICML single-column width (3.25in)
# ---------------------------------------------------------------------------

x = np.arange(len(parties_order))
width = 0.35

fig, ax = plt.subplots(figsize=(3.25, 2.5))

bars1 = ax.bar(x - width / 2, news_dev, width, label='News',
               color='#6a6a6a', edgecolor='black', linewidth=0.4,
               yerr=news_ci, capsize=2.5,
               error_kw={'elinewidth': 0.8, 'capthick': 0.8,
                         'color': 'black'})
bars2 = ax.bar(x + width / 2, opin_dev, width, label='Opinion',
               color='#b8b8b8', edgecolor='black', linewidth=0.4,
               hatch='///',
               yerr=opin_ci, capsize=2.5,
               error_kw={'elinewidth': 0.8, 'capthick': 0.8,
                         'color': 'black'})

ax.axhline(y=0, color='black', linewidth=0.8)
ax.set_ylabel('Relative sentiment')
ax.set_xticks(x)
ax.set_xticklabels(party_labels)
ax.set_ylim(-0.28, 0.68)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.yaxis.grid(True, linewidth=0.3, alpha=0.5, linestyle='-')
ax.set_axisbelow(True)

ax.legend(loc='upper left', frameon=True, edgecolor='#cccccc',
          fancybox=False, framealpha=1.0)

os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'sentiment_bars.pdf'),
            bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'sentiment_bars.png'),
            bbox_inches='tight')
print(f"\nSaved to {OUTPUT_DIR}/sentiment_bars.pdf")
print(f"Saved to {OUTPUT_DIR}/sentiment_bars.png")