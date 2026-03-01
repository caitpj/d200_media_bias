"""
make_figure.py — Generate secondary analysis sentiment bar chart.

Reads secondary_biased.csv (on-target mentions only) and plots
mean sentiment relative to article-type mean for each party.

Usage:
    python scripts/make_figure.py
"""

import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.labelsize': 10,
    'axes.titlesize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 0.5,
    'xtick.major.width': 0.5,
    'ytick.major.width': 0.5,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
})

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data",
                         "secondary", "secondary_biased.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "report", "figures")

df = pd.read_csv(DATA_PATH)

parties_order = ['reform_uk', 'conservative', 'labour', 'plaid_cymru']
party_labels = ['Reform UK', 'Conservative', 'Labour', 'Plaid Cymru']

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
        print(f"  {party:15s} {atype:8s}  n={n:3d}  dev={deviation:+.3f}  ci={ci:.3f}")

# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

x = np.arange(len(parties_order))
width = 0.32

fig, ax = plt.subplots(figsize=(5.5, 3.2))

bars1 = ax.bar(x - width/2, news_dev, width, label='News',
               color='#6a6a6a', edgecolor='black', linewidth=0.4,
               yerr=news_ci, capsize=2.5,
               error_kw={'elinewidth': 0.8, 'capthick': 0.8, 'color': 'black'})
bars2 = ax.bar(x + width/2, opin_dev, width, label='Opinion',
               color='#b8b8b8', edgecolor='black', linewidth=0.4,
               hatch='///',
               yerr=opin_ci, capsize=2.5,
               error_kw={'elinewidth': 0.8, 'capthick': 0.8, 'color': 'black'})

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
plt.savefig(os.path.join(OUTPUT_DIR, 'sentiment_bars.pdf'), bbox_inches='tight')
plt.savefig(os.path.join(OUTPUT_DIR, 'sentiment_bars.png'), bbox_inches='tight')
print("\nSaved to report/figures/sentiment_bars.pdf")
print("Saved to report/figures/sentiment_bars.png")