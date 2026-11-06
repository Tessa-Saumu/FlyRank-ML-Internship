"""Paper and notebook figures, drawn ONLY from committed receipts in work/outputs/*.json (V2 receipts for figs 2, 3, 5).

Theme matches the paper: blue academic palette, Times New Roman. Runs offline.
Writes work/figures/fig*.png
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import v2_view as V  # noqa: E402
from v2_view import MONTH_NAME  # noqa: E402

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

    v = V.view()
    months = v["months"]
    labs = [f"{MONTH_NAME[m['month']]} 2026" + chr(10) + m["role"] for m in months]
    xs = list(range(len(months)))
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.5))
    base_style = {"rule_position_slipping": ("Editor's rule", AMBER, "s"), "mean_reversion_ctr_excess": ("CTR above peers", "#b0b0b0", "v"),
                  "client_shock": ("Client shock", GREY, "^"), "persistence_prev_ratio": ("Persistence", "#cfcfcf", "D")}
    for b, (lab, col, mk) in base_style.items():
        ax[0].plot(xs, [m["base"][b]["ap"] for m in months], marker=mk, color=col, lw=1, ms=5, label=lab)
    ax[0].errorbar(xs, [m["ap"] for m in months], yerr=[[m["ap"] - m["ap_ci"][0] for m in months], [m["ap_ci"][1] - m["ap"] for m in months]],
                   marker="o", color=BLUE, lw=2, ms=7, capsize=3, label="V2 logistic (95% CI)")
    ax[0].plot(xs, [m["base_rate"] for m in months], ls="--", color=INK, lw=1, label="Base rate")
    ax[0].set_xticks(xs)
    ax[0].set_xticklabels(labs, fontsize=9)
    ax[0].set_ylabel("Average precision (whole ranking)")
    ax[0].set_ylim(0.1, 0.65)
    ax[0].set_title("A. Whole-ranking quality (AP, the selection criterion)", fontsize=11)
    ax[0].legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    w = 0.2
    series = [("Base rate", [m["base_rate"] for m in months], "#d9d9d9"), ("Editor's rule", [m["rule"]["p@50"] for m in months], AMBER),
              ("Best other baseline", [max(m["base"][b]["p@50"] for b in V.BASELINES[1:]) for m in months], GREY),
              ("V2 logistic (95% CI)", [m["p50"] for m in months], BLUE)]
    for i, (lab, vals, col) in enumerate(series):
        pos = [j + (i - 1.5) * w for j in xs]
        if lab.startswith("V2"):
            ax[1].bar(pos, vals, w, color=col, label=lab, yerr=[[m["p50"] - m["p50_ci"][0] for m in months], [m["p50_ci"][1] - m["p50"] for m in months]],
                      capsize=2, error_kw={"lw": 1, "ecolor": INK})
            ax[1].scatter(pos, [m["p50_unm"] for m in months], marker="D", s=34, facecolor="white", edgecolor=RED, lw=1.6, zorder=5,
                          label="V2 queue, unmeasurable pages counted as misses")
        else:
            ax[1].bar(pos, vals, w, color=col, label=lab)
            if lab == "Editor's rule":
                ax[1].scatter(pos, [m["rule"]["p@50_unmeasurable_as_miss"] for m in months], marker="s", s=30, facecolor="white", edgecolor=RED, lw=1.6, zorder=5,
                              label="Editor's rule, unmeasurable pages counted as misses")
    ax[1].set_xticks(xs)
    ax[1].set_xticklabels(labs, fontsize=9)
    ax[1].set_ylabel("Precision@50")
    ax[1].set_ylim(0, 1.08)
    ax[1].set_title("B. The 50-page queue (P@50), with the unmeasurable-page penalty", fontsize=11)
    ax[1].legend(frameon=False, fontsize=7.5, loc="upper left", ncol=2)
    for a in ax:
        a.axvspan(len(months) - 1.5, len(months) - 0.5, color=BLUE_SOFT, alpha=0.18, lw=0)
    fig.suptitle("Fig. 2  V2 forward validation: train on matured months only, score the next decision month", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig2_forward_validation.png")
    plt.close(fig)

    lk = v["leak"]
    fig, ax = plt.subplots(figsize=(8, 3.9))
    rvg, pl = lk["random_vs_client_grouped_feb_mar"], lk["planted_future_feature_fold_2"]
    NL = chr(10)
    labs3 = ["Random rows" + NL + "(Feb + Mar)", "Clients held out" + NL + "(Feb + Mar)", "Honest logistic" + NL + "(Mar fold)",
             "Mar fold + planted" + NL + "future feature"]
    vals3 = [rvg["random_row_kfold"]["ap_mean"], rvg["client_group_kfold"]["ap_mean"], pl["LOGISTIC_honest"]["ap"], pl["LOGISTIC_planted_future_ratio"]["ap"]]
    ax.bar(labs3, vals3, color=[RED, BLUE, BLUE_DARK, GREY])
    ax.hlines(rvg["base_rate"], -0.4, 1.4, color=INK, ls="--", lw=1)
    ax.hlines(pl["LOGISTIC_honest"]["base_rate"], 1.6, 3.4, color=INK, ls="--", lw=1)
    ax.text(1.4, rvg["base_rate"] + 0.012, "base rate", ha="right", fontsize=9)
    ax.text(3.4, pl["LOGISTIC_honest"]["base_rate"] + 0.012, "base rate", ha="right", fontsize=9)
    for i, val in enumerate(vals3):
        ax.text(i, val + 0.012, f"{val:.3f}", ha="center", fontsize=10)
    ax.set_ylim(0, 0.95)
    ax.set_ylabel("Average precision")
    ax.set_title("Fig. 3  The split changes the answer a little; a planted leak changes it a lot", fontsize=11)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig3_validation_checks.png")
    plt.close(fig)

    sel = v["sel"]
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2), gridspec_kw={"width_ratios": [1.1, 1]})
    pen_style = {"l2": (BLUE_DARK, "o", "L2"), "l1": (BLUE, "s", "L1 (lasso)"), "elasticnet": (BLUE_SOFT, "^", "Elastic net (0.5)")}
    for pn, (col, mk, lab) in pen_style.items():
        rows = [r for r in sel["configs"]["LOGISTIC"] if r["cfg"]["penalty"] == pn]
        ax[0].plot([r["cfg"]["C"] for r in rows], [r["mean_ap"] for r in rows], marker=mk, color=col, label=lab, lw=1.6)
    ax[0].set_xscale("log")
    ax[0].set_xlabel("Inverse regularisation strength C")
    ax[0].set_ylabel("Mean AP over validation folds 1 and 2")
    ax[0].set_title("A. The logistic penalty options", fontsize=11)
    ax[0].legend(frameon=False, fontsize=9, loc="lower right")
    chn = list(sel["challenger_vs_champion"])
    for i, n in enumerate(chn):
        r = sel["challenger_vs_champion"][n]
        ax[1].errorbar(r["mean_gain"], i, xerr=[[r["mean_gain"] - r["boot_95_ci"][0]], [r["boot_95_ci"][1] - r["mean_gain"]]], marker="o", color=BLUE,
                       capsize=4, lw=2, ms=7)
        ax[1].text(r["boot_95_ci"][1] + 0.002, i, f"lower bound {r['lower_bound']:+.3f}", va="center", fontsize=8.5)
    ax[1].axvline(0, color=INK, ls="--", lw=1)
    ax[1].set_yticks(range(len(chn)))
    ax[1].set_yticklabels(["Boosted classifier", "Poisson-rate boosted", "Rank-average ensemble"][:len(chn)], fontsize=9.5)
    ax[1].invert_yaxis()
    ax[1].set_xlim(-0.05, 0.075)
    ax[1].set_xlabel("AP gain over the logistic champion (95% client bootstrap)")
    ax[1].set_title("B. Champion versus challengers: none clears zero", fontsize=11)
    fig.suptitle("Fig. 5  Model selection on the two validation folds only", fontsize=12)
    fig.tight_layout()
    fig.savefig(f"{FIG}/fig5_selection.png")
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
