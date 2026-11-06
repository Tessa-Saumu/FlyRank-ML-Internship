"""Paper and notebook figures, drawn ONLY from committed receipts in work/outputs/*.json.

Theme matches the paper: blue academic palette, Times New Roman. Runs offline.
Writes work/figures/fig*.png
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT, FIG = HERE + "/../outputs", HERE + "/../figures"
BLUE, BLUE_DARK, BLUE_SOFT = "#2878cf", "#1f5da8", "#a9c8ee"
GREY, AMBER, RED, INK = "#9a9a9a", "#d9a21b", "#d64541", "#383838"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "font.size": 11, "axes.edgecolor": "#cfcfcf", "axes.labelcolor": INK, "xtick.color": INK,
                     "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})


def rj(name):
    return json.load(open(f"{OUT}/{name}"))


def fe_panel(ax, fe, keys, labels, colors, title):
    for k, lab, col in zip(keys, labels, colors):
        e = fe[k]["effects"]
        x = range(len(e))
        ax.plot(x, [r["ctr_multiplier"] for r in e], marker="o", color=col, label=f"{lab} ({fe[k]['clients']} clients)")
        ax.fill_between(x, [r["ci95"][0] for r in e], [r["ci95"][1] for r in e], color=col, alpha=0.12)
    ax.axhline(1, color=GREY, lw=1, ls="--")
    ax.set_xticks(range(len(e)))
    ax.set_xticklabels([r["month"][2:] for r in e], rotation=0, fontsize=9)
    ax.set_ylabel("CTR vs Oct 2025, same page, same position")
    ax.set_title(title, fontsize=12)
    ax.legend(frameon=False, fontsize=9)


def main():
    os.makedirs(FIG, exist_ok=True)
    fe = rj("panel_fe.json")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    fe_panel(ax[0], fe, ["pos_1_3", "pos_4_10", "pos_11_20"], ["Positions 1-3", "Positions 4-10", "Positions 11-20"],
             [BLUE_DARK, BLUE, BLUE_SOFT], "A. By position band")
    fe_panel(ax[1], fe, ["intent_transactional", "intent_commercial", "intent_informational"],
             ["Transactional", "Commercial", "Informational"], [BLUE_DARK, AMBER, GREY], "B. By search intent")
    fig.suptitle("Fig. 1  Click-through at a fixed position, page fixed effects, 95% CI clustered by client", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig1_fixed_position_ctr.png")
    plt.close(fig)

    dev, sealed = rj("erosion_model_metrics.json")["origins"], rj("erosion_model_metrics_sealed.json")["origins"]
    allo = {**dev, **sealed}
    origins = list(allo)
    series = [("Base rate", [allo[o]["base_rate"] for o in origins], "#d9d9d9"),
              ("Rule: top position + slipping", [allo[o]["metrics"]["rule_position_slipping"]["p@50"] for o in origins], AMBER),
              ("Best other baseline that month", [max(allo[o]["metrics"][b]["p@50"] for b in ["mean_reversion_ctr_excess", "client_shock", "persistence_prev_ratio"]) for o in origins], GREY),
              ("Logistic model (frozen)", [allo[o]["metrics"]["logreg"]["p@50"] for o in origins], BLUE)]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    w = 0.2
    for i, (lab, v, col) in enumerate(series):
        ax[0].bar([j + (i - 1.5) * w for j in range(len(origins))], v, w, color=col, label=lab)
    ax[0].set_xticks(range(len(origins)))
    ax[0].set_xticklabels([o + ("\n(sealed)" if o in sealed else "") for o in origins])
    ax[0].set_ylabel("Precision@50")
    ax[0].set_ylim(0, 1)
    ax[0].set_title("A. Precision of the 50-page queue", fontsize=12)
    ax[0].legend(frameon=False, fontsize=8.5, loc="upper left")
    for b, col, mk in [("logreg", BLUE, "o"), ("rule_position_slipping", AMBER, "s"), ("client_shock", GREY, "^"),
                       ("mean_reversion_ctr_excess", "#b0b0b0", "v"), ("persistence_prev_ratio", "#cfcfcf", "D")]:
        ax[1].plot(origins, [allo[o]["metrics"][b]["ap"] for o in origins], marker=mk, color=col, label=b.replace("_", " "))
    ax[1].plot(origins, [allo[o]["base_rate"] for o in origins], ls="--", color=INK, lw=1, label="base rate")
    ax[1].set_ylabel("PR-AUC (whole ranking)")
    ax[1].set_title("B. Quality of the whole ranking", fontsize=12)
    ax[1].legend(frameon=False, fontsize=8)
    fig.suptitle("Fig. 2  Forward validation: train on matured months only, score the next decision month", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig2_forward_validation.png")
    plt.close(fig)

    lk = rj("erosion_leakage_checks.json")
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    labs = ["Random rows\n(memorises clients)", "Clients held out", "Forward, April\n(honest)", "Forward + planted\nfuture feature"]
    vals = [lk["random_row_kfold"]["ap_mean"], lk["client_group_kfold"]["ap_mean"], lk["forward_2026-04_honest"]["ap"], lk["forward_2026-04_planted_future_ratio"]["ap"]]
    ax.bar(labs, vals, color=[RED, BLUE, BLUE_DARK, GREY])
    ax.axhline(lk["client_group_kfold"]["base_rate"], color=INK, ls="--", lw=1)
    ax.text(3.45, lk["client_group_kfold"]["base_rate"] + 0.01, "base rate", ha="right", fontsize=9)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=10)
    ax.set_ylabel("PR-AUC")
    ax.set_title("Fig. 3  Validation design changes the answer; the leak harness proves it can see a leak", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig3_validation_checks.png")
    plt.close(fig)

    e1 = rj("eda_01_landscape.json")
    tr = [r for r in e1["cohort_trend"]] if "cohort_trend" in e1 else []
    plat = {"ChatGPT": 0, "Gemini": 0, "Perplexity": 0, "Copilot": 0, "Claude": 0}
    for r in rj("eda_ai_monthly.json")["months"]:
        for k, c in [("ChatGPT", "gpt"), ("Gemini", "gem"), ("Perplexity", "ppx"), ("Copilot", "cop"), ("Claude", "cla")]:
            plat[k] += r[c] or 0
    m = rj("eda_ai_monthly.json")["months"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].bar([r["month"][2:] for r in m if r["ai"]], [r["ai"] for r in m if r["ai"]], color=BLUE)
    ax2 = ax[0].twinx()
    ax2.plot([r["month"][2:] for r in m if r["ai"]], [r["ga4_clients"] for r in m if r["ai"]], color=AMBER, marker="o")
    ax2.set_ylabel("clients with GA4", color=AMBER)
    ax[0].set_ylabel("AI-assistant sessions")
    ax[0].set_title("A. AI referrals grow with the panel, not per client", fontsize=11)
    tot = sum(plat.values())
    ax[1].barh(list(plat)[::-1], [plat[k] / tot for k in list(plat)[::-1]], color=BLUE)
    for i, k in enumerate(list(plat)[::-1]):
        ax[1].text(plat[k] / tot + 0.01, i, f"{plat[k] / tot:.0%}", va="center", fontsize=9)
    ax[1].set_xlim(0, 0.85)
    ax[1].set_title("B. Platform share of AI-assistant sessions", fontsize=11)
    fig.suptitle("Fig. 4  AI-assistant referrals in the warehouse (Oct 2025 to Jun 2026)", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig4_ai_referrals.png")
    plt.close(fig)
    print("figures written:", sorted(f for f in os.listdir(FIG) if f.startswith("fig")))


if __name__ == "__main__":
    main()
