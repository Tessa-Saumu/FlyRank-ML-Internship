"""Make the capstone paper's figures from the committed receipts.

Reads the JSON receipts in work/outputs/ (produced by the executed w04-w07
notebooks), then:

  1. writes the paper's charts to work/figures/paper_fig*.png, and
  2. re-embeds them into work/paper/index.html (the paper's single,
     self-contained source file) as base64 data.

No warehouse access is needed: every number comes from a receipt, so the
figures always match the committed run. Run from the repo root:

  python work/scripts/make_paper_figures.py

Visual language follows the paper's page design (SaaS editorial &
documentation: Inter / Alegreya / IBM Plex Mono) and, above all, the hero's
"validation picture at a glance" chart:

  * white ground, thin #e5e5e5 grid, #cfcfcf axis edges, no chartjunk;
  * model bars/lines  #2878cf (logistic regression) / #1f5da8 (random forest);
  * hand rule         #383838 (charcoal);
  * base rate         #c9c9c9 (light gray) or a #7a7a7a dashed reference line;
  * status colors     #2f9e6b (honest / positive), #d64541 (leaked / negative),
                      blue tints #c9dcf5 / #a5c7ec / #6ba3e2 for context;
  * every bar carries its value in tabular mono figures, like the hero chart.
"""

import base64
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUTS = os.path.join(REPO, "work", "outputs")
FIG_DIR = os.path.join(REPO, "work", "figures")
PAGE = os.path.join(REPO, "work", "paper", "index.html")
FIG_NAMES = [
    "paper_fig1_results.png",
    "paper_fig2_walkforward.png",
    "paper_fig3_age_decay.png",
    "paper_fig4_playbook.png",
    "paper_fig5_leak_check.png",
]

# One colour per scorer, used in every chart so the paper reads consistently.
# Palette follows the page design tokens (--blue / --ink / --rule / status):
#   Model 1 (logistic regression) = --blue       #2878cf
#   Model 2 (random forest)       = --blue-dark  #1f5da8
#   Baseline / hand rule          = --ink        #383838
#   Base rate (random guessing)   = --rule-strong #c9c9c9 (bars)
#                                 = --ink-soft   #7a7a7a (dashed reference)
#   Positive / negative status    = --green      #2f9e6b / --red #d64541
C_RULE = "#383838"   # ink       - the hand-written rule
C_LR = "#2878cf"     # blue      - logistic regression (Model 1)
C_RF = "#1f5da8"     # blue-dark - random forest (Model 2)
C_BASE = "#c9c9c9"   # gray      - base-rate bars (picking at random)
C_REF = "#7a7a7a"    # ink-soft  - dashed reference lines (base rate / leak)
C_TEXT = "#383838"   # ink       - primary text
C_TICK = "#7a7a7a"   # ink-soft  - axis ticks / notes
C_GRID = "#e5e5e5"   # rule      - subtle gridlines
C_EDGE = "#cfcfcf"   # rule-strong - axis edges
C_GOOD = "#2f9e6b"   # green     - positive / honest
C_DANGER = "#d64541"  # red      - negative / leaked
C_TINT = "#c9dcf5"   # blue tint - context bars
C_TINT2 = "#a5c7ec"  # mid tint  - context bars
C_TINT3 = "#6ba3e2"  # deep tint - context bars

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 11,
    "axes.edgecolor": C_EDGE,
    "axes.labelcolor": C_TICK,
    "text.color": C_TEXT,
    "xtick.color": C_TICK,
    "ytick.color": C_TICK,
    "axes.labelsize": 10.5,
    "axes.grid": True,
    "grid.color": C_GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

MONO = {"family": "DejaVu Sans Mono"}


def load(name):
    with open(os.path.join(OUTS, name)) as f:
        return json.load(f)


def save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    path = os.path.join(FIG_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.relpath(path, REPO))


def label_bars(ax, bars, fmt="{:.2f}", dy=0.015):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy,
                fmt.format(b.get_height()), ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=C_TEXT, **MONO)


def strip(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


val = load("validation_audit_metrics.json")
play = load("action_playbook_summary.json")

# ---------------------------------------------------------------------------
# Figure 1 - the headline: the same three scorers under three tests.
# ---------------------------------------------------------------------------
rand = val["random_split"]["metrics"]
grouped = val["grouped_cv"]["summary"]
fwd = val["time_forward"]["metrics"]
rand_base = val["random_split"]["test_base_rate"]
fwd_base = val["time_forward"]["test_base_rate"]
fold_bases = [f["test_base_rate"] for f in val["grouped_cv"]["per_fold"]]
grouped_base = sum(fold_bases) / len(fold_bases)

fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.1), sharey=True)
panels = [
    ("A. Random split\n(the trap)", [rand["baseline"]["p50"], rand["logistic_regression"]["p50"], rand["random_forest"]["p50"]], rand_base,
     "34 of 40 clients on both sides"),
    ("B. Clients kept separate\n(5 folds, mean +/- spread)",
     [grouped["baseline"]["p50_mean"], grouped["logistic_regression"]["p50_mean"], grouped["random_forest"]["p50_mean"]], grouped_base,
     "0 clients shared per fold"),
    ("C. One month forward\n(fit March, score April)",
     [fwd["baseline"]["p50"], fwd["logistic_regression"]["p50"], fwd["random_forest"]["p50"]], fwd_base,
     "0.56 base rate in April"),
]
errs = [None,
        [grouped["baseline"]["p50_sd"], grouped["logistic_regression"]["p50_sd"], grouped["random_forest"]["p50_sd"]],
        None]
for ax, (title, vals_p, base, note), err in zip(axes, panels, errs):
    xs = range(3)
    bars = ax.bar(xs, vals_p, width=0.62,
                  color=[C_RULE, C_LR, C_RF], edgecolor="white", linewidth=1.2,
                  yerr=err, error_kw=dict(ecolor=C_TEXT, lw=1.4, capsize=5, capthick=1.4))
    label_bars(ax, bars)
    ax.axhline(base, color=C_REF, linestyle="--", linewidth=1.4)
    ax.text(2.42, base + 0.015, f"base rate\n{base:.2f}", color=C_REF,
            fontsize=8.5, ha="right", fontweight="bold", **MONO,
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.2))
    ax.set_xticks(list(xs))
    ax.set_xticklabels(["Rule", "Logistic\nregression", "Random\nforest"], fontsize=9.5)
    ax.set_title(title, fontsize=11.5, fontweight="bold", pad=10, color=C_TEXT)
    ax.set_ylim(0, 1.12)
    ax.text(0.5, -0.30, note, transform=ax.transAxes, ha="center", fontsize=9, color=C_TICK)
    strip(ax)
axes[0].set_ylabel("Precision@50", fontweight="bold")
fig.suptitle("The same three scorers, three increasingly honest tests",
             fontsize=13.5, fontweight="bold", y=1.04, color=C_TEXT)
fig.text(0.035, -0.055,
         "Precision@50 = share of the 50 highest-ranked pages that really declined next month. "
         "Bars above the dashed base-rate line beat picking pages at random.",
         fontsize=9.5, color=C_TICK)
save(fig, "paper_fig1_results.png")

# ---------------------------------------------------------------------------
# Figure 2 - walk-forward: does more training history fix the forward drop?
# ---------------------------------------------------------------------------
wf = val["walk_forward"]["origins"]
depth = [o["train_frames"] for o in wf]
lr = [o["metrics"]["logistic_regression"]["p50"] for o in wf]
rf = [o["metrics"]["random_forest"]["p50"] for o in wf]
rule = [o["metrics"]["baseline"]["p50"] for o in wf]
wf_base = wf[0]["test_base_rate"]

fig, ax = plt.subplots(figsize=(8.6, 4.3))
ax.plot(depth, lr, "-o", color=C_LR, linewidth=2.0, markersize=4.5, label="Logistic regression")
ax.plot(depth, rf, "-o", color=C_RF, linewidth=2.0, markersize=4.5, label="Random forest")
ax.plot(depth, rule, "-s", color=C_RULE, linewidth=1.4, markersize=4, label="Hand rule")
ax.axhline(wf_base, color=C_REF, linestyle="--", linewidth=1.4)
ax.text(4.02, wf_base + 0.012, f"base rate {wf_base:.2f}", color=C_REF,
        fontsize=9.5, fontweight="bold", ha="right", **MONO)
for d, y in zip(depth, lr):
    ax.annotate(f"{y:.2f}", (d, y), textcoords="offset points", xytext=(0, 9),
                ha="center", fontsize=9, fontweight="bold", color=C_LR, **MONO)
for d, y in zip(depth, rf):
    ax.annotate(f"{y:.2f}", (d, y), textcoords="offset points", xytext=(0, -16),
                ha="center", fontsize=9, fontweight="bold", color=C_RF, **MONO)
ax.set_xticks(depth)
ax.set_xticklabels([f"{d} month{'s' if d > 1 else ''} of training pages" for d in depth], fontsize=9.5)
ax.set_xlabel("Training history (frames ending Feb, Jan, Dec, Nov 2025 -> March 2026 labels)", fontweight="bold")
ax.set_ylabel("Precision@50 on the same April test", fontweight="bold")
ax.set_ylim(0.2, 0.85)
ax.set_title("More history helps briefly, then stops helping", fontsize=13, fontweight="bold", pad=10, color=C_TEXT)
ax.legend(loc="upper right", frameon=True, edgecolor=C_GRID, fontsize=9.5)
strip(ax)
fig.text(0.01, -0.04,
         "Fixed panel: the clients present in every frame (20 clients per training side); the test side is the same April rows throughout. "
         "Rule flat at 0.38 by construction of the fixed test.",
         fontsize=9, color=C_TICK)
save(fig, "paper_fig2_walkforward.png")

# ---------------------------------------------------------------------------
# Figure 3 - what the data shows: decline risk by content age (April frame).
# ---------------------------------------------------------------------------
buckets = play["age_bucket_decline_rate_april"]
names = [b["age_bucket"] for b in buckets]
rates = [b["decline_rate"] for b in buckets]
shares = [b["pages_pct"] for b in buckets]
apr_base = play["frames"]["score_april"]["base_rate"]

fig, ax = plt.subplots(figsize=(8.6, 4.3))
colors = [C_TINT if r != max(rates) else C_LR for r in rates]
bars = ax.bar(range(len(names)), rates, width=0.62, color=colors, edgecolor="white", linewidth=1.2)
label_bars(ax, bars, fmt="{:.2f}")
ax.axhline(apr_base, color=C_REF, linestyle="--", linewidth=1.4)
ax.text(4.45, apr_base + 0.012, f"April base rate {apr_base:.2f}", color=C_REF,
        fontsize=9.5, fontweight="bold", ha="right", **MONO)
ax.set_xticks(range(len(names)))
ax.set_xticklabels([f"{n}\n({s:.0f}% of pages)" for n, s in zip(names, shares)], fontsize=9.5)
ax.set_xlabel("Content age at the start of the month", fontweight="bold")
ax.set_ylabel("Share that declined next month", fontweight="bold")
ax.set_ylim(0, 0.78)
ax.set_title("Decline risk peaks at 3-6 months of age, then falls", fontsize=13, fontweight="bold", pad=10, color=C_TEXT)
strip(ax)
fig.text(0.01, -0.105,
         "April 2026 frame (99,279 pages). A straight-line age term cannot represent this shape - "
         "which is why the linear model's age coefficient came out with the wrong sign.",
         fontsize=9, color=C_TICK)
save(fig, "paper_fig3_age_decay.png")

# ---------------------------------------------------------------------------
# Figure 4 - what the playbook produces: funnel to the budget queue + actions.
# ---------------------------------------------------------------------------
gates = play["gates"]["hits_april"]
eligible = play["gates"]["eligible_rows"]
april_rows = play["frames"]["score_april"]["rows"]
queue_size = play["queue"]["size"]
queue_hits = round(play["queue"]["top50_decline_rate_after_gates"] * queue_size)

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
ax = axes[0]
stages = ["Pages in the\nApril frame", "Eligible after\ngates", "Budget queue\n(top 50)", "Really\ndeclined"]
nums = [april_rows, eligible, queue_size, queue_hits]
ypos = range(len(stages))[::-1]
bars = ax.barh(list(ypos), nums, height=0.62, color=[C_TINT, C_TINT2, C_TINT3, C_LR],
               edgecolor="white", linewidth=1.2)
ax.set_xscale("log")
ax.set_yticks(list(ypos))
ax.set_yticklabels(stages, fontsize=9.5)
for y, n in zip(ypos, nums):
    ax.text(n * 1.15, y, f"{n:,}", va="center", fontsize=10, fontweight="bold", color=C_TEXT, **MONO)
gate_note = (f"gates removed: new pages {gates['NEW_PAGE']:,} - risers {gates['RISER_1M']:,}\n"
             f"senior-look {gates['TOP2_NEAR_ZERO_CTR']:,} - verify {gates['EXTREME_JUMP_20X']:,}")
ax.set_title("From the month's pages to a 50-row queue", fontsize=12, fontweight="bold", pad=10, color=C_TEXT)
ax.set_xlabel("pages (log scale)", fontweight="bold")
ax.text(0.5, -0.30, gate_note, transform=ax.transAxes, ha="center", fontsize=8.5, color=C_TICK)
strip(ax)

ax = axes[1]
actions = play["action_distribution_april"]
anames = list(actions.keys())
avals = [actions[k] for k in anames]
action_colors = {
    "review_before_revert": C_RF,
    "monitor_only": C_GOOD,
    "verify_then_review": C_DANGER,
    "investigate_quiet_risk": C_LR,
}
ypos = range(len(anames))[::-1]
bars = ax.barh(list(ypos), avals, height=0.62,
               color=[action_colors.get(a, C_LR) for a in anames], edgecolor="white", linewidth=1.2)
ax.set_xscale("log")
ax.set_yticks(list(ypos))
ax.set_yticklabels([a.replace("_", "\n") for a in anames], fontsize=9)
for y, v in zip(ypos, avals):
    ax.text(v * 1.15, y, f"{v:,}", va="center", fontsize=10, fontweight="bold", color=C_TEXT, **MONO)
ax.set_title("The action attached to each April page", fontsize=12, fontweight="bold", pad=10, color=C_TEXT)
ax.set_xlabel("pages (log scale)", fontweight="bold")
ax.text(0.5, -0.30, "Actions are assigned before the budget cut;\nonly the top 50 eligible rows reach an editor.",
        transform=ax.transAxes, ha="center", fontsize=8.5, color=C_TICK)
strip(ax)
fig.suptitle("What the playbook produced for May 2026", fontsize=13.5, fontweight="bold", y=1.03, color=C_TEXT)
save(fig, "paper_fig4_playbook.png")

# ---------------------------------------------------------------------------
# Figure 5 - the leakage harness works: a planted leak is caught.
# ---------------------------------------------------------------------------
leak = val["leak_injection"]

fig, ax = plt.subplots(figsize=(6.8, 3.5))
bars = ax.bar([0, 1], [leak["honest_ap"], leak["injected_ap"]], width=0.5,
              color=[C_GOOD, C_DANGER], edgecolor="white", linewidth=1.2)
label_bars(ax, bars)
ax.set_xticks([0, 1])
ax.set_xticklabels(["Honest five features", "With the label's own\ncolumn planted in"], fontsize=10)
ax.set_ylabel("Average precision (one fold)", fontweight="bold")
ax.set_ylim(0, 1.12)
ax.set_title("The check can catch a leak", fontsize=12.5, fontweight="bold", pad=10, color=C_TEXT)
ax.annotate("0.99 = the score a leak\nbuys. Removed; every\nreported number uses\nthe honest five.",
            xy=(0.93, leak["injected_ap"] - 0.02), xytext=(0.30, 0.58),
            fontsize=8.5, color=C_DANGER,
            arrowprops=dict(arrowstyle="->", color=C_DANGER, lw=1.4))
strip(ax)
save(fig, "paper_fig5_leak_check.png")

# ---------------------------------------------------------------------------
# Embed the figures into the paper's source page (work/paper/index.html).
# ---------------------------------------------------------------------------
if os.path.exists(PAGE):
    with open(PAGE, encoding="utf-8") as f:
        html = f.read()
    for name in FIG_NAMES:
        with open(os.path.join(FIG_DIR, name), "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")

        def repl(m, b64=b64):
            return re.sub(r'src="[^"]*"', f'src="data:image/png;base64,{b64}"', m.group(0), count=1)

        html, n = re.subn(rf'<img[^>]*data-fig="{re.escape(name)}"[^>]*>', repl, html, count=1)
        if n != 1:
            raise SystemExit(f'could not find <img data-fig="{name}"> in work/paper/index.html')
    with open(PAGE, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"embedded {len(FIG_NAMES)} figures into work/paper/index.html "
          f"({os.path.getsize(PAGE):,} bytes) - the page needs no other files")
else:
    print("work/paper/index.html not found - figures left in work/figures/ only")

# ---------------------------------------------------------------------------
# Receipt check: print the exact numbers the paper quotes, straight from JSON.
# ---------------------------------------------------------------------------
print("\n--- receipt check: numbers quoted in the paper ---")
print(f"grouped  P@50  rule {grouped['baseline']['p50_mean']:.3f} ± {grouped['baseline']['p50_sd']:.3f}"
      f" | LR {grouped['logistic_regression']['p50_mean']:.3f} ± {grouped['logistic_regression']['p50_sd']:.3f}"
      f" | RF {grouped['random_forest']['p50_mean']:.3f} ± {grouped['random_forest']['p50_sd']:.3f}"
      f" | fold-mean base {grouped_base:.3f}")
print(f"random  P@50  rule {rand['baseline']['p50']:.2f} | LR {rand['logistic_regression']['p50']:.2f}"
      f" | RF {rand['random_forest']['p50']:.2f} | base {rand_base:.3f} | shared clients {val['random_split']['shared_clients_both_sides']}")
print(f"forward P@50  rule {fwd['baseline']['p50']:.2f} | LR {fwd['logistic_regression']['p50']:.2f}"
      f" | RF {fwd['random_forest']['p50']:.2f} | base {fwd_base:.3f}")
print(f"walk-forward  rule {rule[0]:.2f} flat | LR " + " ".join(f"{v:.2f}" for v in lr) +
      f" | RF " + " ".join(f"{v:.2f}" for v in rf) + f" | test base {wf_base:.3f}")
print(f"leak injection  honest AP {leak['honest_ap']:.3f} -> planted AP {leak['injected_ap']:.3f}")
print(f"age buckets  " + " ".join(f"{b['age_bucket']}:{b['decline_rate']:.2f}" for b in buckets))
print(f"queue  after-gates decline rate {play['queue']['top50_decline_rate_after_gates']:.2f}"
      f" | clients {play['queue']['top50_clients']} | largest {play['queue']['top50_largest_client_share']:.2f}")
print(f"frames  march {val['frames']['march']['rows']:,} rows/{val['frames']['march']['clients']} clients"
      f" | april {val['frames']['april']['rows']:,}/{val['frames']['april']['clients']}")
print("done")
