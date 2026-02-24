"""
visualise.py — Publication-ready figures and statistical tests for
media bias analysis of Nation.Cymru coverage of Welsh political parties.

Uses SciencePlots for academic styling. Produces figures sized for
NeurIPS single-column width (~5.5in).

Usage:
    python scripts/visualise.py

Requirements:
    pip install pandas numpy matplotlib scipy SciencePlots
"""

import os

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy import stats

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data",
                         "processed", "analysis_results.csv")
FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "figures")

# NeurIPS text width is ~5.5in
TEXT_WIDTH = 5.5

# Party order: left to right
PARTY_ORDER = ["plaid_cymru", "labour", "conservative", "reform_uk", "ukip"]

PARTY_LABELS = {
    "plaid_cymru": "Plaid Cymru",
    "labour": "Labour",
    "conservative": "Conservative",
    "reform_uk": "Reform UK",
    "ukip": "UKIP",
}

PARTY_COLOURS = {
    "plaid_cymru": "#005B54",
    "labour": "#E4003B",
    "conservative": "#0087DC",
    "reform_uk": "#12B6CF",
    "ukip": "#6D3177",
}

# Hatching patterns for grayscale readability
PARTY_HATCHES = {
    "plaid_cymru": "",
    "labour": "//",
    "conservative": "\\\\",
    "reform_uk": "xx",
    "ukip": "..",
}

PARTY_MARKERS = {
    "plaid_cymru": "o",
    "labour": "s",
    "conservative": "D",
    "reform_uk": "^",
    "ukip": "v",
}

# Apply SciencePlots style
try:
    plt.style.use(["science", "no-latex"])
except OSError:
    print("  SciencePlots style not found, using defaults")

# Override some rcParams for NeurIPS compatibility
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 6.5,
    "figure.figsize": (TEXT_WIDTH, TEXT_WIDTH * 0.6),
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fmt_pct(x, _):
    """Format y-axis as integer percentages."""
    return f"{x:.0f}%"


def wilson_ci(p, n, z=1.96):
    """Wilson score 95% confidence interval."""
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2*n)) / denom
    margin = z * np.sqrt((p*(1-p) + z**2/(4*n)) / n) / denom
    return margin


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_data():
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    df["publish_date"] = pd.to_datetime(df["publish_date"], errors="coerce")
    df["year"] = df["publish_date"].dt.year
    df["is_biased"] = (df["bias_label"] == "biased").astype(int)

    median_sent = df["sentiment_score"].median()
    df["is_neg_biased"] = (
        (df["is_biased"] == 1) & (df["sentiment_score"] < median_sent)
    ).astype(int)

    print(f"  {len(df)} mentions loaded")
    print(f"  Median sentiment: {median_sent:+.3f}")
    return df, median_sent


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def run_tests(df):
    print("\n" + "=" * 60)
    print("STATISTICAL TESTS")
    print("=" * 60)

    # Overall chi-squared
    contingency = pd.crosstab(df["party"], df["bias_label"])
    contingency = contingency.loc[PARTY_ORDER]
    chi2, p_chi2, dof, _ = stats.chi2_contingency(contingency)
    n = len(df)
    cramers_v = np.sqrt(chi2 / (n * (min(contingency.shape) - 1)))

    print(f"\n  Overall chi-squared (bias rate across parties):")
    print(f"    χ² = {chi2:.1f}, df = {dof}, p < {max(p_chi2, 1e-20):.1e}")
    print(f"    Cramér's V = {cramers_v:.3f}")

    # Pairwise z-tests
    pairs = [
        ("plaid_cymru", "reform_uk"),
        ("plaid_cymru", "conservative"),
        ("plaid_cymru", "labour"),
        ("labour", "conservative"),
        ("labour", "reform_uk"),
        ("conservative", "reform_uk"),
    ]

    print(f"\n  Pairwise z-tests (bias rate):")
    print(f"  {'Comparison':<40s} {'z':>7s} {'p':>10s} {'h':>8s}")
    print(f"  {'-'*67}")

    for p1, p2 in pairs:
        d1, d2 = df[df["party"] == p1], df[df["party"] == p2]
        n1, n2 = len(d1), len(d2)
        b1, b2 = d1["is_biased"].sum(), d2["is_biased"].sum()
        r1, r2 = b1/n1, b2/n2
        p_pooled = (b1 + b2) / (n1 + n2)
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
        z = (r2 - r1) / se if se > 0 else 0
        p_val = 2 * (1 - stats.norm.cdf(abs(z)))
        h = 2 * (np.arcsin(np.sqrt(r2)) - np.arcsin(np.sqrt(r1)))
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 \
            else "*" if p_val < 0.05 else ""
        label = f"{PARTY_LABELS[p1]} vs {PARTY_LABELS[p2]}"
        print(f"  {label:<40s} {z:+7.2f} {p_val:10.4f} {h:+8.3f} {sig}")

    # Pairwise t-tests for sentiment
    print(f"\n  Pairwise t-tests (sentiment score):")
    print(f"  {'Comparison':<40s} {'t':>7s} {'p':>10s} {'d':>8s}")
    print(f"  {'-'*67}")

    for p1, p2 in pairs:
        s1 = df[df["party"] == p1]["sentiment_score"]
        s2 = df[df["party"] == p2]["sentiment_score"]
        t_stat, p_val = stats.ttest_ind(s1, s2)
        pooled_std = np.sqrt(
            ((len(s1)-1)*s1.std()**2 + (len(s2)-1)*s2.std()**2)
            / (len(s1)+len(s2)-2))
        d = (s2.mean() - s1.mean()) / pooled_std if pooled_std > 0 else 0
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 \
            else "*" if p_val < 0.05 else ""
        label = f"{PARTY_LABELS[p1]} vs {PARTY_LABELS[p2]}"
        print(f"  {label:<40s} {t_stat:+7.2f} {p_val:10.4f} {d:+8.3f} {sig}")


# ---------------------------------------------------------------------------
# Figure 1: Bias rate by party (horizontal bar — cleaner for long labels)
# ---------------------------------------------------------------------------

def fig1_bias_rates(df):
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH * 0.5, 2.2))

    parties_rev = PARTY_ORDER[::-1]  # reverse for horizontal bars
    rates, cis, colours, hatches = [], [], [], []

    for party in parties_rev:
        subset = df[df["party"] == party]
        n = len(subset)
        p = subset["is_biased"].mean()
        rates.append(p * 100)
        cis.append(wilson_ci(p, n) * 100)
        colours.append(PARTY_COLOURS[party])
        hatches.append(PARTY_HATCHES[party])

    y = np.arange(len(parties_rev))
    bars = ax.barh(y, rates, xerr=cis, capsize=3, color=colours,
                   edgecolor="black", linewidth=0.4, height=0.6)

    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    for i, (rate, ci) in enumerate(zip(rates, cis)):
        ax.text(rate + ci + 0.8, i, f"{rate:.1f}%",
                va="center", ha="left", fontsize=7, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels([PARTY_LABELS[p] for p in parties_rev])
    ax.set_xlabel("Sentences with biased language (%)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(fmt_pct))
    ax.set_xlim(0, max(rates) + max(cis) + 6)
    ax.set_title("Bias rate by party")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 2: Bias over time
# ---------------------------------------------------------------------------

def fig2_bias_over_time(df):
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH * 0.5, 2.5))

    yearly = df.groupby(["year", "party"]).agg(
        bias_rate=("is_biased", "mean"),
        n=("is_biased", "count"),
    ).reset_index()
    yearly = yearly[yearly["n"] >= 20]

    for party in PARTY_ORDER:
        data = yearly[yearly["party"] == party].sort_values("year")
        if len(data) < 2:
            continue
        ax.plot(data["year"], data["bias_rate"] * 100,
                marker=PARTY_MARKERS[party], markersize=3.5,
                linewidth=1.2, color=PARTY_COLOURS[party],
                label=PARTY_LABELS[party])

    ax.set_xlabel("Year")
    ax.set_ylabel("Biased sentences (%)")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(fmt_pct))
    ax.set_title("Bias rate over time")
    ax.legend(loc="upper right", frameon=True, fancybox=False,
              edgecolor="black", framealpha=0.9)
    ax.set_xticks(range(2017, 2027))
    ax.set_xticklabels(range(2017, 2027), rotation=45, ha="right")
    ax.set_ylim(0, None)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 3: Negative bias rate
# ---------------------------------------------------------------------------

def fig3_neg_bias_rates(df):
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH * 0.5, 2.2))

    parties_rev = PARTY_ORDER[::-1]
    rates, colours, hatches = [], [], []

    for party in parties_rev:
        subset = df[df["party"] == party]
        rates.append(subset["is_neg_biased"].mean() * 100)
        colours.append(PARTY_COLOURS[party])
        hatches.append(PARTY_HATCHES[party])

    y = np.arange(len(parties_rev))
    bars = ax.barh(y, rates, color=colours, edgecolor="black",
                   linewidth=0.4, height=0.6)

    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)

    for i, rate in enumerate(rates):
        ax.text(rate + 0.5, i, f"{rate:.1f}%",
                va="center", ha="left", fontsize=7, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels([PARTY_LABELS[p] for p in parties_rev])
    ax.set_xlabel("Negatively biased mentions (%)")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(fmt_pct))
    ax.set_xlim(0, max(rates) + 5)
    ax.set_title("Negative bias rate by party")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 4: Sentiment distribution
# ---------------------------------------------------------------------------

def fig4_sentiment(df):
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH * 0.5, 2.2))

    parties_rev = PARTY_ORDER[::-1]
    y = np.arange(len(parties_rev))
    height = 0.6

    neg_r, neu_r, pos_r = [], [], []
    for party in parties_rev:
        subset = df[df["party"] == party]
        n = len(subset)
        neg_r.append((subset["sentiment"] == "negative").sum() / n * 100)
        neu_r.append((subset["sentiment"] == "neutral").sum() / n * 100)
        pos_r.append((subset["sentiment"] == "positive").sum() / n * 100)

    left_neu = np.array(neg_r)
    left_pos = left_neu + np.array(neu_r)

    ax.barh(y, neg_r, height, label="Negative", color="#c62828",
            edgecolor="black", linewidth=0.3)
    ax.barh(y, neu_r, height, left=left_neu, label="Neutral",
            color="#bdbdbd", edgecolor="black", linewidth=0.3)
    ax.barh(y, pos_r, height, left=left_pos, label="Positive",
            color="#2e7d32", edgecolor="black", linewidth=0.3)

    ax.set_yticks(y)
    ax.set_yticklabels([PARTY_LABELS[p] for p in parties_rev])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(fmt_pct))
    ax.set_xlabel("Share of mentions (%)")
    ax.set_title("Sentiment distribution by party")
    ax.legend(loc="lower right", frameon=True, fancybox=False,
              edgecolor="black", framealpha=0.9)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 5: Editorial vs quoted
# ---------------------------------------------------------------------------

def fig5_editorial_quoted(df):
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH * 0.5, 2.5))

    parties_rev = PARTY_ORDER[::-1]
    y = np.arange(len(parties_rev))
    height = 0.3

    ed_rates, qu_rates = [], []
    for party in parties_rev:
        ed = df[(df["party"] == party) & (df["is_quote"] == False)]
        qu = df[(df["party"] == party) & (df["is_quote"] == True)]
        ed_rates.append(ed["is_biased"].mean() * 100)
        qu_rates.append(qu["is_biased"].mean() * 100 if len(qu) > 0 else 0)

    ax.barh(y + height/2, ed_rates, height, label="Editorial framing",
            color="#424242", edgecolor="black", linewidth=0.3)
    ax.barh(y - height/2, qu_rates, height, label="Quoted speech",
            color="#bdbdbd", edgecolor="black", linewidth=0.3)

    ax.set_yticks(y)
    ax.set_yticklabels([PARTY_LABELS[p] for p in parties_rev])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(fmt_pct))
    ax.set_xlabel("Biased sentences (%)")
    ax.set_title("Bias: editorial framing vs quoted speech")
    ax.legend(loc="lower right", frameon=True, fancybox=False,
              edgecolor="black", framealpha=0.9)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 5b: Mean sentiment score by party
# ---------------------------------------------------------------------------

def fig5b_sentiment_score(df):
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH * 0.5, 2.2))

    parties_rev = PARTY_ORDER[::-1]
    means, stderrs, colours = [], [], []

    for party in parties_rev:
        subset = df[df["party"] == party]["sentiment_score"]
        means.append(subset.mean())
        stderrs.append(subset.std() / np.sqrt(len(subset)) * 1.96)  # 95% CI
        colours.append(PARTY_COLOURS[party])

    y = np.arange(len(parties_rev))
    bars = ax.barh(y, means, xerr=stderrs, capsize=2, color=colours,
                   edgecolor="black", linewidth=0.3, height=0.55)

    for i, (mean, se) in enumerate(zip(means, stderrs)):
        offset = -0.02 if mean < 0 else 0.02
        ax.text(mean + offset + (se if mean >= 0 else -se),
                i, f"{mean:+.3f}",
                va="center", ha="left" if mean >= 0 else "right",
                fontsize=6)

    ax.axvline(0, color="black", linewidth=0.5, linestyle="-")
    ax.set_yticks(y)
    ax.set_yticklabels([PARTY_LABELS[p] for p in parties_rev], fontsize=6.5)
    ax.set_xlabel("Mean sentiment score")
    ax.set_title("Sentiment toward each party")

    # Annotate direction
    ax.text(0.02, -0.12, "← More negative",
            transform=ax.transAxes, fontsize=5.5, color="gray", style="italic")
    ax.text(0.75, -0.12, "More positive →",
            transform=ax.transAxes, fontsize=5.5, color="gray", style="italic")

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 6: Combined 2-panel for report (bias rate + over time)
# ---------------------------------------------------------------------------

def fig6_combined(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 2.5))

    # (a) Bias rate
    parties_rev = PARTY_ORDER[::-1]
    rates, cis, colours, hatches = [], [], [], []
    for party in parties_rev:
        subset = df[df["party"] == party]
        n = len(subset)
        p = subset["is_biased"].mean()
        rates.append(p * 100)
        cis.append(wilson_ci(p, n) * 100)
        colours.append(PARTY_COLOURS[party])
        hatches.append(PARTY_HATCHES[party])

    y = np.arange(len(parties_rev))
    bars = ax1.barh(y, rates, xerr=cis, capsize=2, color=colours,
                    edgecolor="black", linewidth=0.3, height=0.6)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    for i, (rate, ci) in enumerate(zip(rates, cis)):
        ax1.text(rate + ci + 0.5, i, f"{rate:.1f}%",
                 va="center", ha="left", fontsize=6, fontweight="bold")
    ax1.set_yticks(y)
    ax1.set_yticklabels([PARTY_LABELS[p] for p in parties_rev], fontsize=7)
    ax1.set_xlabel("Biased sentences (%)")
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(fmt_pct))
    ax1.set_xlim(0, max(rates) + max(cis) + 6)
    ax1.set_title("(a) Bias rate by party")

    # (b) Over time
    yearly = df.groupby(["year", "party"]).agg(
        bias_rate=("is_biased", "mean"), n=("is_biased", "count")
    ).reset_index()
    yearly = yearly[yearly["n"] >= 20]
    for party in PARTY_ORDER:
        data = yearly[yearly["party"] == party].sort_values("year")
        if len(data) < 2:
            continue
        ax2.plot(data["year"], data["bias_rate"] * 100,
                 marker=PARTY_MARKERS[party], markersize=3,
                 linewidth=1.0, color=PARTY_COLOURS[party],
                 label=PARTY_LABELS[party])
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Biased sentences (%)")
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(fmt_pct))
    ax2.set_title("(b) Bias rate over time")
    ax2.legend(loc="upper right", frameon=True, fancybox=False,
               edgecolor="black", framealpha=0.9, fontsize=5.5)
    ax2.set_xticks(range(2017, 2027))
    ax2.set_xticklabels([str(y) for y in range(2017, 2027)],
                        rotation=45, ha="right", fontsize=6)
    ax2.set_ylim(0, None)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Figure 7: Combined 3-panel (bias + neg bias + time)
# ---------------------------------------------------------------------------

def fig7_three_panel(df):
    fig = plt.figure(figsize=(TEXT_WIDTH, 4.5))

    # Top row: two bar charts side by side
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    # Bottom row: time series spanning full width
    ax3 = fig.add_subplot(2, 1, 2)

    parties_rev = PARTY_ORDER[::-1]
    y = np.arange(len(parties_rev))

    # (a) Bias rate
    rates, cis, colours, hatches = [], [], [], []
    for party in parties_rev:
        subset = df[df["party"] == party]
        n = len(subset)
        p = subset["is_biased"].mean()
        rates.append(p * 100)
        cis.append(wilson_ci(p, n) * 100)
        colours.append(PARTY_COLOURS[party])
        hatches.append(PARTY_HATCHES[party])

    bars = ax1.barh(y, rates, xerr=cis, capsize=2, color=colours,
                    edgecolor="black", linewidth=0.3, height=0.55)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    for i, (rate, ci) in enumerate(zip(rates, cis)):
        ax1.text(rate + ci + 0.5, i, f"{rate:.1f}%",
                 va="center", ha="left", fontsize=6)
    ax1.set_yticks(y)
    ax1.set_yticklabels([PARTY_LABELS[p] for p in parties_rev], fontsize=6.5)
    ax1.set_xlabel("Biased sentences (%)")
    ax1.xaxis.set_major_formatter(plt.FuncFormatter(fmt_pct))
    ax1.set_xlim(0, max(rates) + max(cis) + 6)
    ax1.set_title("(a) Bias rate")

    # (b) Negative bias rate
    neg_rates = []
    for party in parties_rev:
        subset = df[df["party"] == party]
        neg_rates.append(subset["is_neg_biased"].mean() * 100)

    bars = ax2.barh(y, neg_rates, color=colours, edgecolor="black",
                    linewidth=0.3, height=0.55)
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    for i, rate in enumerate(neg_rates):
        ax2.text(rate + 0.4, i, f"{rate:.1f}%",
                 va="center", ha="left", fontsize=6)
    ax2.set_yticks(y)
    ax2.set_yticklabels([PARTY_LABELS[p] for p in parties_rev], fontsize=6.5)
    ax2.set_xlabel("Negatively biased mentions (%)")
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(fmt_pct))
    ax2.set_xlim(0, max(neg_rates) + 5)
    ax2.set_title("(b) Negative bias rate")

    # (c) Over time
    yearly = df.groupby(["year", "party"]).agg(
        bias_rate=("is_biased", "mean"), n=("is_biased", "count")
    ).reset_index()
    yearly = yearly[yearly["n"] >= 20]
    for party in PARTY_ORDER:
        data = yearly[yearly["party"] == party].sort_values("year")
        if len(data) < 2:
            continue
        ax3.plot(data["year"], data["bias_rate"] * 100,
                 marker=PARTY_MARKERS[party], markersize=3.5,
                 linewidth=1.2, color=PARTY_COLOURS[party],
                 label=PARTY_LABELS[party])
    ax3.set_xlabel("Year")
    ax3.set_ylabel("Biased sentences (%)")
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(fmt_pct))
    ax3.set_title("(c) Bias rate over time")
    ax3.legend(loc="upper right", frameon=True, fancybox=False,
               edgecolor="black", framealpha=0.9, fontsize=6,
               ncol=3)
    ax3.set_xticks(range(2017, 2027))
    ax3.set_xticklabels([str(yr) for yr in range(2017, 2027)],
                        rotation=0, fontsize=6.5)
    ax3.set_ylim(0, None)

    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    df, median_sent = load_data()

    run_tests(df)

    figures = {
        "fig1_bias_rates": fig1_bias_rates(df),
        "fig2_bias_over_time": fig2_bias_over_time(df),
        "fig3_neg_bias_rates": fig3_neg_bias_rates(df),
        "fig4_sentiment": fig4_sentiment(df),
        "fig5_editorial_quoted": fig5_editorial_quoted(df),
        "fig5b_sentiment_score": fig5b_sentiment_score(df),
        "fig6_combined_2panel": fig6_combined(df),
        "fig7_combined_3panel": fig7_three_panel(df),
    }

    for name, fig in figures.items():
        path = os.path.join(FIG_DIR, f"{name}.png")
        fig.savefig(path)
        # Also save PDF for LaTeX inclusion
        fig.savefig(os.path.join(FIG_DIR, f"{name}.pdf"))
        print(f"  Saved {name}.png + .pdf")
        plt.close(fig)

    print(f"\n  All figures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()