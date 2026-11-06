"""Build work/paper/index.html: one static page, blue academic theme (work/paper/theme.css,
Times New Roman), figures embedded as base64, EVERY number read from work/outputs/*.json (V2 receipts through v2_view.py).
Runs offline. Publish with work/scripts/deploy_paper.py (gh-pages)."""
import base64
import json
import os

import v2_view as VV

HERE = os.path.dirname(os.path.abspath(__file__))
W = HERE + "/.."
REPO = "https://github.com/Tessa-Saumu/FlyRank-ML-Internship"


def rj(n):
    return json.load(open(f"{W}/outputs/{n}", encoding="utf-8"))


def img(n):
    return "data:image/png;base64," + base64.b64encode(open(f"{W}/figures/{n}", "rb").read()).decode()


f2, f3, ci2, ci3 = VV.f2, VV.f3, VV.ci2, VV.ci3


def build():
    N = VV.view(W)
    fe, e4, mon = rj("panel_fe.json"), rj("eda_04_client_level.json"), rj("eda_ai_monthly.json")
    summ, e1, e3 = rj("erosion_frame_summary.json"), rj("eda_01_landscape.json"), rj("eda_03_inference.json")
    v2s = rj("v2_summary.json")
    pb, sel, fz, feats = N["playbook"], N["sel"], N["freeze"], N["feats"]
    months, by = N["months"], N["by"]
    feb, mar, apr, may = by["2026-02"], by["2026-03"], by["2026-04"], by["2026-05"]
    top3 = next(r for r in fe["pos_1_3"]["effects"] if r["month"] == "2026-04")
    tra = next(r for r in fe["intent_transactional"]["effects"] if r["month"] == "2026-04")
    inf = next(r for r in fe["intent_informational"]["effects"] if r["month"] == "2026-04")
    c10 = next(r for r in e4["A_client_level"] if r["segment"] == "1-10")
    ai_total = sum(r["ai"] or 0 for r in mon["months"])
    conc = e1["ai_concentration"][0]
    onset = {r["feature"]: r["within_client_auc"] for r in e3["B_onset_signal_audit"]}
    nulls = [o["null_rate"] for o in summ["origins"] if o["null_rate"] is not None and o["null_rate"] == o["null_rate"]]
    pen, ch, mean2, bst = N["penalty_best"], N["ch"], N["mean2"], N["boost"]
    sh = lambda h: h[:10]

    def names(pred):
        ms = [VV.MONTH_NAME[m["month"]] for m in months if pred(m)]
        return ", ".join(ms[:-1]) + " and " + ms[-1] if len(ms) > 1 else (ms[0] if ms else "none")

    SHORT = {"rule_position_slipping": "Editor's rule", "mean_reversion_ctr_excess": "CTR above peers", "client_shock": "Client shock",
             "persistence_prev_ratio": "Persistence"}
    nulls_all = nulls + [m["noise"] for m in months]
    assert all(m["rule"]["p@50_unmeasurable_as_miss"] > m["p50_unm"] for m in months)
    RULE_SENT = ("Counted this way, the editor's rule scores higher than the champion at 50 pages in every month; "
                 "the champion's queue favours pages that later lose visibility.")
    rule_best_months = names(lambda m: m["best_p50_name"] == "rule_position_slipping")
    ap_c1 = next(r for r in sel["configs"]["LOGISTIC"] if r["cfg"] == {"C": 1.0, "penalty": "l2"})
    clause_c = (f"The selected C = {pen['l2']['C']:g} sits at the edge of the searched grid, but on a plateau (mean AP {ap_c1['mean_ap']:.3f} at C = 1 "
                f"against {pen['l2']['mean_ap']:.3f} at C = {pen['l2']['C']:g}), so the edge position is harmless.")
    AP_WHY = ("Model choice used AP because it scores the whole ranking and is far less noisy than P@50 at about 5 effective clients "
              "(P@50 intervals are roughly plus or minus 0.2); P@50 remains the reported operational metric.")

    # ---- tables
    data_rows = ""
    for o in summ["origins"]:
        m = o["origin"]
        if m in by:
            ev = by[m]
            role = {"validation": "validation", "confirmation": "confirmation", "test": "test"}[ev["role"]]
            data_rows += (f"<tr><td>{m}</td><td>{role}</td><td class='num'>{ev['n_eligible']:,}</td><td class='num'>{ev['n_measurable']:,}</td>"
                          f"<td class='num'>{ev['lost_visibility'] * 100:.1f}%</td><td class='num'>{ev['n_clients']}</td><td class='num'>{ev['eff_clients']:.1f}</td>"
                          f"<td class='num'>{f3(ev['base_rate'])}</td><td class='num'>{f3(ev['noise'])}</td></tr>")
        elif o.get("live_no_outcome"):
            data_rows += (f"<tr><td>{m}</td><td>live, no outcome yet</td><td class='num'>{o['eligible']:,}</td><td class='num'></td><td class='num'></td>"
                          f"<td class='num'>{o['clients']}</td><td class='num'></td><td class='num'></td><td class='num'></td></tr>")
        else:
            data_rows += (f"<tr><td>{m}</td><td>training only</td><td class='num'>{o['eligible']:,}</td><td class='num'>{o['measurable']:,.0f}</td>"
                          f"<td class='num'>{o['lost_visibility'] * 100:.1f}%</td><td class='num'>{o['clients']}</td><td class='num'></td>"
                          f"<td class='num'>{f3(o['base_rate'])}</td><td class='num'>{f3(o['null_rate'])}</td></tr>")
    split_rows = "".join(
        f"<tr><td>{a}</td><td>{b}</td><td>{c}</td><td>{d}</td><td>{e}</td></tr>" for a, b, c, d, e in [
            ("Validation fold 1", "Feb 2026", "Feb to Mar", "Dec", "yes: tuning and the champion test"),
            ("Validation fold 2", "Mar 2026", "Mar to Apr", "Dec, Jan", "yes: tuning and the champion test"),
            ("Confirmation fold", "Apr 2026", "Apr to May", "Dec to Feb", "<strong>no</strong>, reported only"),
            ("Test (sealed)", "May 2026", "May to Jun", "Dec to Mar (refit)", "<strong>no</strong>, scored once after the freeze"),
            ("Live", "Jul 2026", "none yet", "Dec to May", "queue only")])
    pen_rows = "".join(
        f"<tr><td>{lab}</td><td class='num'>{pen[k]['C']:g}</td><td class='num'>{pen[k]['mean_ap']:.4f}</td><td class='num'>{pen[k]['fold_ap'][0]:.3f}</td><td class='num'>{pen[k]['fold_ap'][1]:.3f}</td></tr>"
        for k, lab in [("l2", "L2"), ("l1", "L1 (lasso)"), ("elasticnet", "Elastic net (0.5)")])
    chal_rows = "".join(
        f"<tr><td>{lab}</td><td class='num'>{ch[k]['mean_gain']:+.4f}</td><td class='num'>{ch[k]['boot_95_ci'][0]:+.4f} to {ch[k]['boot_95_ci'][1]:+.4f}</td>"
        f"<td class='num'>{ch[k]['lower_bound']:+.4f}</td><td class='num'>{ch[k]['bonferroni_3_lower_bound_info_only']:+.4f}</td>"
        f"<td class='num'>{ch[k]['loco_clients_with_gain_le_0']} of {ch[k]['loco_n_clients']}</td><td>no</td></tr>"
        for k, lab in [("HGB_CLF", "Boosted classifier"), ("POISSON_RATE", "Poisson-rate boosted"), ("ENSEMBLE", "Rank-average ensemble")])
    rows = ""
    for m in months:
        rows += (f"<tr><td>{m['month']}</td><td>{m['role']}</td><td class='num'>{f2(m['base_rate'])}</td><td class='num'>{f2(m['rule']['p@50'])}</td>"
                 f"<td class='num'>{f2(m['best_p50'])} <span class='cap'>({SHORT[m['best_p50_name']]})</span></td>"
                 f"<td class='num'><strong>{f2(m['p50'])}</strong> <span class='cap'>[{f2(m['p50_ci'][0])}, {f2(m['p50_ci'][1])}]</span></td>"
                 f"<td class='num'><strong>{f2(m['p50_unm'])}</strong></td><td class='num'>{f2(m['rule']['p@50_unmeasurable_as_miss'])}</td>"
                 f"<td class='num'>{f3(m['best_ap'])}</td><td class='num'><strong>{f3(m['ap'])}</strong> <span class='cap'>[{f3(m['ap_ci'][0])}, {f3(m['ap_ci'][1])}]</span></td>"
                 f"<td class='num'>{m['d_ap']:+.3f} <span class='cap'>[{m['d_ap_ci'][0]:+.3f}, {m['d_ap_ci'][1]:+.3f}]</span></td></tr>")
    unm_rows = "".join(
        f"<tr><td>{m['month']}</td><td>{m['role']}</td><td class='num'>{f2(m['rule']['p@50'])}</td><td class='num'>{f2(m['p50'])}</td>"
        f"<td class='num'><strong>{f2(m['rule']['p@50_unmeasurable_as_miss'])}</strong></td><td class='num'><strong>{f2(m['p50_unm'])}</strong></td>"
        f"<td class='num'>{m['rule']['clicks_lost_captured'] * 100:.1f}%</td><td class='num'>{m['clk'] * 100:.1f}%</td></tr>" for m in months)
    v1v2_rows = "".join(
        f"<tr><td>{m['month']}</td><td>{m['role']}</td><td class='num'>{f3(m['v1_ap'])}</td><td class='num'>{f3(m['ap'])}</td><td class='num'>{m['ap'] - m['v1_ap']:+.3f}</td>"
        f"<td class='num'>{f2(m['v1_p50'])}</td><td class='num'>{f2(m['p50'])}</td><td class='num'>{m['p50'] - m['v1_p50']:+.2f}</td></tr>" for m in months)
    fe_rows = ""
    for k, lab in [("pos_1_3", "Positions 1-3"), ("pos_4_10", "Positions 4-10"), ("pos_11_20", "Positions 11-20"),
                   ("intent_transactional", "Transactional"), ("intent_commercial", "Commercial"), ("intent_informational", "Informational")]:
        r = next(e for e in fe[k]["effects"] if e["month"] == "2026-04")
        fe_rows += f"<tr><td>{lab}</td><td class='num'>{fe[k]['clients']}</td><td class='num'>{fe[k]['pages']:,}</td><td class='num'>{r['ctr_multiplier']:.3f}</td><td class='num'>{r['ci95'][0]} to {r['ci95'][1]}</td></tr>"
    rh_rows = "".join(
        f"<tr><td><code>{r['feature']}</code></td><td>{r['expected_direction']}</td><td class='num'>{r['within_client_auc']:.3f}</td><td class='num'>{ci3(r['ci95'])}</td><td><strong>{r['verdict']}</strong></td></tr>"
        for r in v2s["rhythm_signal_audit"]["features"])
    coef = {r["feature"]: r["std_coef"] for r in v2s["champion_coefficients"]["top_positive"] + v2s["champion_coefficients"]["top_negative"]}
    coef_new = {r["feature"]: r["std_coef"] for r in v2s["champion_coefficients"]["new_features"]}
    css = open(f"{W}/paper/theme.css", encoding="utf-8").read()
    sec = lambda i, sid, title: f'<section id="{sid}"><h2 class="sec"><span class="secnum">{i}</span>{title}<a class="backtop" href="#top">&uarr; Top</a></h2>'
    DISC = VV.DISCLOSURE
    ap_sentence = (f"The V2 champion's AP is the highest of all five rankers in {N['ap_lead_months']} of 4 evaluated months, and the 95% interval of its AP gain over the best baseline "
                   f"excludes 0 in {N['d_ap_excl0_months']} of them ({names(lambda m: m['d_ap_excl0'])}); in {names(lambda m: not m['d_ap_excl0'])} the interval includes 0")
    p50_sentence = (f"Its P@50 is above all four baselines in {N['p50_lead_months']} of 4 months (not in {names(lambda m: not m['p50_above_all'])}, where the editor's rule scores "
                    f"{f2(mar['rule']['p@50'])} against {f2(mar['p50'])}), and the interval of its P@50 gain over the best-AP baseline excludes 0 in only {N['d_p50_excl0_months']} month "
                    f"({names(lambda m: m['d_p50_excl0'])}), so P@50 differences are not established")
    unm_sentence = (f"counting pages that later lost visibility as misses, the test month's P@50 falls from {f2(may['p50'])} to {f2(may['p50_unm'])} and the confirmation month's from "
                    f"{f2(apr['p50'])} to {f2(apr['p50_unm'])}")

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Same Rank, Fewer Clicks</title>
<meta name="description" content="FlyRank ML Internship capstone (freestyle): an early-warning ranking of pages losing Google clicks at a stable rank, the pattern AI answers would cause, built with a pre-registered V2 pipeline. Built on the FlyRank ML Internship dataset.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&display=swap" rel="stylesheet">
<!-- Generated by work/scripts/make_paper.py from the receipts in work/outputs/*.json. Do not edit by hand. -->
<style>
{css}
body{{background:#ffffff}}
.hero .chart img{{display:block;width:100%;height:auto;border-radius:8px}}
td .cap{{display:inline}}
</style>
</head>
<body>
<div id="top"></div>
<header class="hero"><div class="hero-inner">
  <div class="hero-text">
    <p class="kicker">Research paper · FlyRank ML Internship · Capstone (freestyle track)</p>
    <h1>Same Rank, Fewer Clicks: An Early Warning for Pages Losing Google Clicks to AI Answers</h1>
    <p class="byline">Theresia Saumu <span class="dot"></span> November 2026</p>
    <p class="sub">Click-through at a fixed position fell by about a third at the top of Google in six months. This paper measures it on FlyRank's warehouse, then asks whether a model can tell an editor which 50 pages to protect next month, with the process written down before it ran and the caveats in plain view.</p>
    <div class="chips"><a class="chip" href="{REPO}">GitHub repo</a><a class="chip" href="{REPO}/blob/main/work/notebooks/capstone.ipynb">Capstone notebook</a><a class="chip" href="#repro">How to rerun</a><a class="chip" href="#credit">Data credit</a></div>
  </div>
  <div class="chart"><h2>Same page, same position: CTR vs Oct 2025</h2><img src="{img('fig1_fixed_position_ctr.png')}" alt="Click-through at a fixed position by month, by position band and intent">
  <p class="chart-note">Page fixed effects, impression weights, 95% CI clustered by client. Top-3 positions, April 2026: {top3['ctr_multiplier']:.3f} [{top3['ci95'][0]}, {top3['ci95'][1]}].</p></div>
</div></header>
<div class="kpis-wrap"><div class="kpis">
  <div class="kpi"><div class="kpi-num">{top3['ctr_multiplier']:.2f}&times;</div><div class="kpi-label">Top-3 CTR, Apr vs Oct</div><div class="kpi-sub">same page, same position, {fe['pos_1_3']['clients']} clients</div></div>
  <div class="kpi"><div class="kpi-num">{c10['clients_down']} / {c10['clients']}</div><div class="kpi-label">Clients with lower CTR</div><div class="kpi-sub">at a stable rank, Wilcoxon p = {c10['wilcoxon_p']}</div></div>
  <div class="kpi"><div class="kpi-num">{N['ap_lead_months']} / 4</div><div class="kpi-label">Months the champion led on AP</div><div class="kpi-sub">whole-ranking AP, not P@50; the {names(lambda m: not m['d_ap_excl0']).replace('Apr', 'April')} gain interval includes 0</div></div>
  <div class="kpi"><div class="kpi-num">{f2(may['p50'])} &rarr; {f2(may['p50_unm'])}</div><div class="kpi-label">Test P@50, with and without the unmeasurable pages</div><div class="kpi-sub">base rate {f2(may['base_rate'])}; pages that lost visibility counted as misses. The editor's rule scores {f2(may['rule']['p@50'])} and {f2(may['rule']['p@50_unmeasurable_as_miss'])}</div></div>
</div></div>
<nav class="toc"><div class="toc-inner"><a href="#abstract">Abstract</a><a href="#problem">1 Introduction</a><a href="#data">2 Data</a><a href="#method">3 Methodology</a><a href="#results">4 Results</a><a href="#limits">5 Limitations</a><a href="#recs">6 Recommendations</a><a href="#repro">7 Reproducibility</a><a href="#credit">8 Credit</a></div></nav>
<main><div class="layout"><div class="article">

{sec('A', 'abstract', 'Abstract')}
<p>AI answers in search are the new variable in FlyRank's business, so I asked which top-10 pages are about to lose clicks at the rank they already hold, so that an editor can protect 50 of them first each month. I used FlyRank's pseudonymized internship warehouse ({mon['daily_rows']/1e6:.1f} million daily rows, January 2025 to June 2026), where page fixed effects with client-clustered errors put top-3 click-through in April 2026 at {top3['ctr_multiplier']:.3f} times its October 2025 level for the same page in the same position band (95% CI {top3['ci95'][0]} to {top3['ci95'][1]}). The monthly ranking comes from a pipeline that was specified and hashed before it ran: a label that visible momentum cannot give away, {N['n_features']} features, four frozen baselines, four learned candidates compared under a selection rule fixed in advance, and a test month scored once. The logistic champion had the highest average precision (AP) in {N['ap_lead_months']} of 4 evaluated months, although the {names(lambda m: not m['d_ap_excl0']).replace('Apr', 'April')} gain interval includes 0; on the test month its AP was {f3(may['ap'])} (95% CI {ci3(may['ap_ci'])}) against {f3(may['best_ap'])} for the best baseline, and its precision at 50 pages (P@50) was {f2(may['p50'])} against a base rate of {f2(may['base_rate'])}, which falls to {f2(may['p50_unm'])} when pages that later lost visibility count as misses, below the editor's rule at {f2(may['rule']['p@50_unmeasurable_as_miss'])}. The output is a reviewed monthly queue for editors, a predictive and not causal decision aid whose intervals are wide at {N['eff_range'][0]:.1f} to {N['eff_range'][1]:.1f} effective clients and whose test month was seen during V1, so the result may be optimistic.</p>
</section>

<section class="tldr" id="tldr"><h2>In one minute</h2><dl>
<dt>Question</dt><dd>Which ranking pages will lose clicks at the rank they hold, so an editor can protect 50 of them first each month?</dd>
<dt>Finding</dt><dd>Same page, same position: top-3 CTR at {top3['ctr_multiplier']:.2f}&times; in six months; CTR fell in {c10['clients_down']} of {c10['clients']} clients.</dd>
<dt>Method</dt><dd>Leak-checked label, {N['n_features']} features, four frozen rules, four candidates under a rule fixed in advance, two validation folds, a confirmation fold, one test month scored once.</dd>
<dt>Result</dt><dd>The logistic champion had the top AP in {N['ap_lead_months']} of 4 months; P@50 {f2(may['p50'])} on the test month against a base rate of {f2(may['base_rate'])}, and {f2(may['p50_unm'])} if pages that lost visibility count as misses. {RULE_SENT}</dd>
<dt>Caveat</dt><dd>About five effective clients; the test month was seen in V1; no challenger model was distinguishable from the champion.</dd>
<dt>Use</dt><dd>A reviewed monthly queue with reason codes. Predictive, not causal.</dd>
</dl></section>

<section id="claims"><h2 class="sec"><span class="secnum">&#9670;</span>What the evidence supports<a class="backtop" href="#top">&uarr; Top</a></h2>
<div class="claims-grid">
<div class="claim"><div class="lbl">Claim 1 · observed</div><p>At a fixed page and position, click-through fell after October 2025: {c10['clients_down']} of {c10['clients']} clients (median client ratio {c10['median_client_ctr_ratio']}); top-3 at {top3['ctr_multiplier']:.3f} by April 2026. <a href="#results">Section 4</a>.</p></div>
<div class="claim"><div class="lbl">Claim 2 · measured</div><p>Transactional pages fell to {tra['ctr_multiplier']:.3f} [{tra['ci95'][0]}, {tra['ci95'][1]}]; informational pages did not fall significantly ({inf['ctr_multiplier']:.3f} [{inf['ci95'][0]}, {inf['ci95'][1]}]).</p></div>
<div class="claim"><div class="lbl">Claim 3 · decision support</div><p>The V2 champion's AP was above all four frozen baselines in {N['ap_lead_months']} of 4 months. Test month: AP {f3(may['ap'])} against {f3(may['best_ap'])}, P@50 {f2(may['p50'])} against a base rate of {f2(may['base_rate'])} ({f2(may['p50_unm'])} counting lost-visibility pages as misses, against {f2(may['rule']['p@50_unmeasurable_as_miss'])} for the editor's rule).</p></div>
<div class="claim"><div class="lbl">Not claimed</div><p>That AI answers caused the fall; that V2 improved on V1 (test AP {f3(may['ap'])} against {f3(may['v1_ap'])}, P@50 {f2(may['p50'])} against {f2(may['v1_p50'])}, both inside the noise); that the P@50 lead over the best rule is established; that a more complex model is better; or that acting on the queue restores clicks.</p></div>
</div></section>

{sec('1', 'problem', 'Introduction')}
<p>FlyRank produces and maintains content for client sites, and AI answers in search (Google AI Overviews and AI Mode, ChatGPT, Perplexity, Gemini) are changing what a ranking is worth. FlyRank's <em>State of AI-Driven SEO 2026</em> report measures AI referral sessions, but not what AI answers do to Google clicks at a given rank. Industry studies report click-through falling where AI Overviews appear, but they use cross-sectional keyword samples, not a panel of client pages followed over time.</p>
<p>My earlier work on Lane 2 (refresh), archived in the repository, taught the lesson that shaped this study: when the label is largely answered by momentum that is already visible, a two-line rule beats any model. This study therefore starts with a target that momentum does not give away, and with the decision an editor faces: a budget of 50 reviews a month, and the question of which pages to protect before their clicks disappear.</p>
<p>The first version of the pipeline (V1) compared two models and opened one sealed month once. V2 is the same question, unit and label with a more rigorous process: a specification written and hashed before any V2 run, four learned candidates, selection on validation folds under a rule fixed in advance, and a three-way split. The V2 changes are changes of process; they are not an attempt to raise the score, and V1's numbers stay in the repository, labelled as V1.</p>
</section>

{sec('2', 'data', 'Data')}
<p>The FlyRank internship warehouse release (build v20260703, Hugging Face <code>FlyRank/internship-warehouse</code>, pinned revision <code>{summ['warehouse_revision'][:8]}</code>), cached locally once and queried with DuckDB: the daily performance fact table ({mon['daily_rows']:,} rows of Search Console and GA4 metrics, {mon['first_date']} to {mon['last_date']}), <code>dim_content</code> and <code>dim_clients</code>. Identities are pseudonymized and none appear here. The warehouse has one revision and ends on 2026-06-30, so there is no more data from which to draw a fresh holdout.</p>
<div class="tw"><table><thead><tr><th>Decision month</th><th>Role</th><th>Eligible pages</th><th>Measurable</th><th>Unmeasurable share</th><th>Clients</th><th>Effective clients</th><th>Base rate</th><th>Noise-only rate</th></tr></thead><tbody>
{data_rows}
</tbody></table></div>
<p class="cap">Effective clients = 1 divided by the sum of squared client row shares. Unmeasurable pages are eligible pages whose expected outcome clicks are under 10 (mostly pages that lost visibility); the label cannot be measured for them. The live month has no outcome yet.</p>
<p class="cap">Client counts use two definitions: for the four evaluated months, clients with at least one measurable page in the scored rows; for training-only and live months, clients in the frame receipt (all eligible pages). They can differ by a few clients for the same month.</p>
<h3>What I excluded and why</h3>
<ul>
<li>Months before October 2025 for any comparison: Google's September 2025 removal of deep-result impressions shifted every position average (the share of impressions in the top 10 rose from 69% to 95%).</li>
<li>Pages outside the top 10, or with fewer than 200 impressions or 20 clicks in the feature window (their CTR is too noisy to judge).</li>
<li>Unmeasurable pages (expected outcome clicks under 10); this uses outcome impressions, so it is a disclosed population choice, and Section 4 shows what it hides.</li>
<li>The 90-day query table: its window (April to June 2026) overlaps every outcome window.</li>
<li>The column <code>provider_used</code>, which contains brand-like strings; it is never selected.</li>
<li>Three V1 features: <code>page_prev_ratio</code> and <code>client_prev_erode_rate</code> (empty in the earliest training rows, and the label anti-persists) and <code>word_count</code> (from a July 2026 snapshot; {min(d['share_content_updated_on_or_after_origin'] for d in feats['snapshot_diagnostics_dim_content'].values()) * 100:.0f}% to {max(d['share_content_updated_on_or_after_origin'] for d in feats['snapshot_diagnostics_dim_content'].values()) * 100:.1f}% of pages were updated after each origin).</li>
<li>AI-assistant referral sessions as a target: {ai_total:,.0f} sessions in total, only from October 2025, with two clients holding 68% of them.</li>
</ul>
</section>

{sec('3', 'method', 'Methodology')}
<h3>Label</h3>
<p>At each monthly decision point, features use the two closed months before it and the label uses the two months after. A page erodes if its outcome clicks fall below 75% of what its current click-through predicts at its new position. "Current" is the latest month's CTR, shrunk toward the client's position-adjusted CTR, with a position curve fixed on October and November 2025. Two earlier label versions leaked momentum and made every model look better; both were fixed during V1 development. A Poisson simulation puts the noise-only rate at {min(nulls_all):.3f} to {max(nulls_all):.3f} across the training, validation, confirmation and test months, against observed base rates several times higher.</p>
<h3>Features and baselines</h3>
<ul>
<li>{N['n_features']} features, exported from code: exposure and trajectory, rank, CTR against expectation, content, GA4 channels and client context. They are the {N['n_v1_features']} V1 features minus three, plus {N['n_new']} date and rhythm features (<code>weekday_share</code>, <code>days_since_last_click</code>, <code>longest_zero_click_run</code>, <code>days_since_peak</code>, <code>keyword_to_content_gap_days</code>). Those five come from a mentor's advice that a date column yields many derived features; they are not claimed to be motivated by V1's errors. Every feature is computed from months that closed before the decision.</li>
<li>Four frozen baselines, identical to V1: the editor's rule (top position plus slipping CTR), CTR above peers, client-wide shock and persistence.</li>
</ul>
<h3>Four candidates, and why four</h3>
<ul>
<li><strong>Logistic regression (the champion)</strong> with three penalty options (L2, L1, elastic net) and six values of C: 18 configurations. L1 and elastic net are options of this one candidate, not extra candidates.</li>
<li><strong>Boosted classifier</strong> and <strong>Poisson-rate boosted regressor</strong> (histogram gradient boosting, up to 20 random configurations each, trees chosen by early stopping, five seeds).</li>
<li><strong>Rank-average ensemble</strong> of the tuned logistic and boosted classifier, declared in advance.</li>
</ul>
<p>With about five effective clients, the AP of one configuration on two validation folds is noisy, and choosing the best of many candidates on the rows that score them gives optimistic estimates whose bias grows with the number of candidates (Cawley and Talbot, 2010). So the set is small, diverse and fixed in advance. No model family wins on every tabular dataset (McElfresh et al., 2023), which is why a linear model, tree ensembles with two different losses and one blend were compared, and tree ensembles remain strong on medium-sized tabular data (Grinsztajn et al., 2022), which is why deep networks were not.</p>
<h3>Selection rule, fixed before any run</h3>
<p>Each configuration is scored by its mean AP over the two validation folds, and the best configuration per family is kept. AP is the only criterion. {AP_WHY} The logistic model is the champion. A challenger replaces it only if the client-bootstrap 95% lower bound of its paired AP gain is above 0 and the mean gain stays above 0 when any single client is left out. There are {N['n_comparisons']} challenger comparisons, and the rule does not correct for multiplicity; Bonferroni-adjusted bounds are reported for information. A nested check repeats the procedure on fold 1 only and scores fold 2.</p>
<h3>Validation: no overlap between training, validation and test outcome windows</h3>
<div class="tw"><table><thead><tr><th>Role</th><th>Scored month</th><th>Outcome window</th><th>Trained on origins</th><th>Decides anything?</th></tr></thead><tbody>{split_rows}</tbody></table></div>
<p>A model scoring month <em>o</em> trains only on origins whose outcome windows closed before <em>o</em> begins. The two validation windows end before the test window opens. The confirmation fold's window overlaps the test window, which is why it is computed after the selection is written to disk and cannot change it. Tuning and the champion test use the same two folds, so the validation numbers of the tuned models are optimistic; the confirmation fold and the test month are the less-biased reads. The test month was scored once, after <code>v2_freeze.json</code> ({fz['frozen_at']}) was written; no decision followed it, and the code refuses a second run.</p>
<div class="alert warn"><strong>Disclosure.</strong> {DISC}</div>
<p>Checks: a planted future feature lifts the champion's AP from {N['planted']['LOGISTIC_honest']['ap']:.3f} to {N['planted']['LOGISTIC_planted_future_ratio']['ap']:.3f} (+{N['planted']['LOGISTIC_ap_lift']:.2f}) on validation fold 2, so the harness can see a leak. A random row split reports AP {N['rvg']['random_row_kfold']['ap_mean']:.3f} against {N['rvg']['client_group_kfold']['ap_mean']:.3f} with clients held out. Timeline assertions hold for all five roles. A secondary, client-relative label is evaluated for the champion only and never used to choose anything.</p>
<figure class="figure"><div class="frame"><img src="{img('fig3_validation_checks.png')}" alt="Validation checks"></div><figcaption>Fig. 3. Random rows flatter the model a little; a planted leak changes the answer a lot, so the harness detects leaks.</figcaption></figure>
</section>

{sec('4', 'results', 'Results')}
<h3>4.1 The click loss at a fixed position</h3>
<div class="tw"><table><thead><tr><th>Segment</th><th>Clients</th><th>Pages</th><th>CTR Apr 2026 vs Oct 2025</th><th>95% CI</th></tr></thead><tbody>{fe_rows}</tbody></table></div>
<p>Page fixed effects remove page mix, position bands remove rank changes, and errors are clustered by client. December dips in every segment (the holiday season). Positions 11 to 20 partly recover by June.</p>
<h3>4.2 Model selection: the logistic champion stays</h3>
<div class="tw"><table><thead><tr><th>Logistic penalty</th><th>Best C</th><th>Mean AP, folds 1 and 2</th><th>Fold 1 AP</th><th>Fold 2 AP</th></tr></thead><tbody>{pen_rows}</tbody></table></div>
<p>The three penalties differ by {max(p['mean_ap'] for p in pen.values()) - min(p['mean_ap'] for p in pen.values()):.4f} AP at their best, so sparsity did not help; L2 with C = {pen['l2']['C']:g} is selected. {clause_c} Mean validation AP of the four candidates is {mean2['LOGISTIC']['ap']:.4f} (logistic), {mean2['HGB_CLF']['ap']:.4f} (boosted classifier), {mean2['POISSON_RATE']['ap']:.4f} (Poisson-rate boosted) and {mean2['ENSEMBLE']['ap']:.4f} (ensemble).</p>
<div class="tw"><table><thead><tr><th>Challenger</th><th>Mean AP gain over the champion</th><th>95% interval</th><th>Lower bound</th><th>Bonferroni-3 bound (information)</th><th>Clients whose removal makes the gain &le; 0</th><th>Passes</th></tr></thead><tbody>{chal_rows}</tbody></table></div>
<p><strong>No challenger passed.</strong> The ensemble came closest (lower bound {N['closest_lb']:+.4f}), and that is not enough under a rule written in advance. The four candidates are statistically indistinguishable on these two folds, which says what two folds and about five effective clients can show; it does not say the logistic model is better. The selected boosted classifier stops after {bst['HGB_CLF']['validation_median_iter_selected'][0]} and {bst['HGB_CLF']['validation_median_iter_selected'][1]} trees on the validation folds, a sign that there is little structure beyond what a linear model captures. Repeating the whole procedure on fold 1 only selects the logistic model again and scores AP {sel['nested_time_ordered_check']['fold_2_evaluation']['models']['NESTED_SELECTED']['ap']:.4f} on fold 2.</p>
<figure class="figure"><div class="frame"><img src="{img('fig5_selection.png')}" alt="Model selection"></div><figcaption>Fig. 5. Left: the logistic penalty options. Right: the paired AP gain of each challenger over the champion with its client-bootstrap interval; none clears zero.</figcaption></figure>
<h3>4.3 The ranking, month by month</h3>
<div class="alert warn"><strong>Unmeasurable pages.</strong> The headline P@50 is computed on measurable pages. Pages that lose most of their visibility cannot be labelled, so they drop out. If the queue ranks them, the headline does not see them. {unm_sentence[0].upper() + unm_sentence[1:]}; the conservative bound counts them as misses, and what happened to those pages cannot be measured with this label. The editor's rule, under the same count, scores {f2(may['rule']['p@50_unmeasurable_as_miss'])} on the test month and {f2(apr['rule']['p@50_unmeasurable_as_miss'])} on the confirmation month. <strong>{RULE_SENT}</strong> The queue also captures only {min(m['clk'] for m in months) * 100:.1f}% to {max(m['clk'] for m in months) * 100:.1f}% of a month's total click shortfall.</div>
<div class="tw"><table><thead><tr><th>Month</th><th>Role</th><th>Editor's rule P@50</th><th>V2 P@50</th><th>Editor's rule P@50, unmeasurable = miss</th><th>V2 P@50, unmeasurable = miss</th><th>Editor's rule: share of click shortfall in its top 50</th><th>V2: share of click shortfall in its top 50</th></tr></thead><tbody>{unm_rows}</tbody></table></div>
<p class="cap">The first two columns count only measurable pages; the next two count pages that later lost visibility as misses. The click-shortfall shares are on measurable pages (unmeasurable pages have no measured clicks), and the rule leads on that share in {names(lambda m: m['rule']['clicks_lost_captured'] > m['clk'])}.</p>
<div class="tw"><table><thead><tr><th>Month</th><th>Role</th><th>Base rate</th><th>Editor's rule P@50</th><th>Best baseline P@50</th><th>V2 P@50 [95% CI]</th><th>V2 P@50, unmeasurable = miss</th><th>Editor's rule P@50, unmeasurable = miss</th><th>Best baseline AP</th><th>V2 AP [95% CI]</th><th>AP gain over best baseline [95% CI]</th></tr></thead><tbody>{rows}</tbody></table></div>
<p class="cap">In {rule_best_months} the best baseline at 50 pages is the editor's rule itself, so the "Editor's rule P@50" and "Best baseline P@50" columns repeat there.</p>
<figure class="figure"><div class="frame"><img src="{img('fig2_forward_validation.png')}" alt="Forward validation"></div><figcaption>Fig. 2. Left: whole-ranking quality (AP). Right: the 50-page queue; the red-outlined markers show the editor's rule (squares) and the V2 queue (diamonds) with unmeasurable pages counted as misses. Validation months were tuned on; the shaded month is the test.</figcaption></figure>
<p>{ap_sentence}. {p50_sentence}. {RULE_SENT} The champion's AUC is {f3(may['auc'])} on the test month and its lift is {f2(may['lift'])}; the effective number of clients is {N['eff_range'][0]:.1f} to {N['eff_range'][1]:.1f}.</p>
<div class="alert info"><strong>Reading the test month.</strong> The champion reached P@50 of {f2(may['p50'])} (CI {ci2(may['p50_ci'])}) against a base rate of {f2(may['base_rate'])} and AP {f3(may['ap'])} (CI {ci3(may['ap_ci'])}) against {f3(may['best_ap'])} for the best baseline, a gain of {may['d_ap']:+.3f} (CI {ci3(may['d_ap_ci'])}) that stays between {may['loco']['min']:+.3f} and {may['loco']['max']:+.3f} when any single client is left out. At exactly 50 pages the lead is not established: the interval of the P@50 gain over the best-AP baseline is {ci2(may['d_p50_ci'])}. The best other baseline by P@50 drew its 50 picks from only {may['models']['client_shock']['clients_in_top50']} clients, and the editor's rule had an AUC of {f3(may['rule']['auc'])}, worse than random order. {DISC}</div>
<h3>4.4 V1 against V2</h3>
<div class="tw"><table><thead><tr><th>Month</th><th>V2 role</th><th>V1 AP</th><th>V2 AP</th><th>V2 minus V1</th><th>V1 P@50</th><th>V2 P@50</th><th>V2 minus V1</th></tr></thead><tbody>{v1v2_rows}</tbody></table></div>
<p>V1's February to April were its development months and its May its sealed month, opened once in V1; V2's February and March were selection folds. On the test month V2's AP is slightly lower than V1's ({f3(may['ap'])} against {f3(may['v1_ap'])}) and its P@50 higher ({f2(may['p50'])} against {f2(may['v1_p50'])}); V1's values lie inside V2's 95% intervals, so the receipts do not show a difference in either direction. V2 is a process upgrade, not a better result.</p>
<h3>4.5 What the model learned, and the new features</h3>
<p>The largest standardised weights are <code>l_imp_f</code> ({coef['l_imp_f']:+.2f}) with a matching negative <code>l_clk_f</code> ({coef['l_clk_f']:+.2f}) (many impressions per click), <code>ctr_vs_ref</code> ({coef['ctr_vs_ref']:+.2f}, the page earns more clicks than its position normally does) and <code>ctr_growth</code> ({coef['ctr_growth']:+.2f}). So the pages losing clicks next are the ones currently earning more than their rank normally earns, or riding a recent CTR bump, the opposite of the editor's intuition ("slipping keeps slipping"). The champion is only weakly regularised (C = {v2s['champion_coefficients']['config']['C']:g}), so correlated features share weight and the coefficients show where it leans rather than importances. A single-signal rule built on the same idea ranks at about the base rate, so the edge comes from combining signals.</p>
<div class="tw"><table><thead><tr><th>New V2 feature</th><th>Direction written down first</th><th>Within-client AUC</th><th>95% CI</th><th>Verdict</th></tr></thead><tbody>{rh_rows}</tbody></table></div>
<p>Tested on the December to March origins (outcome windows end before the test window opens), after the V2 test was scored, so this changes no decision. The longest click-free spell is the clearest signal, yet the champion gives the five new features small weights (largest <code>longest_zero_click_run</code>, {coef_new['longest_zero_click_run']:+.2f}), so V2's gain is in process, not in new signal.</p>
<h3>4.6 AI referrals and discovery</h3>
<p>AI-assistant sessions are thin ({ai_total:,.0f}) and concentrated: {conc['pages_with_any_ai']:,} of {conc['pages_with_ga4_sessions']:,} GA4-tracked pages ever received one. Within a client, the first AI referral follows exposure (prior impressions, AUC {onset['l_imp']:.3f}) and not Google rank (AUC {onset['pos']:.3f}).</p>
<figure class="figure"><div class="frame"><img src="{img('fig4_ai_referrals.png')}" alt="AI referrals"></div><figcaption>Fig. 4. AI referrals grow with the number of tracked clients rather than per client; ChatGPT sends most of them.</figcaption></figure>
</section>

{sec('5', 'limits', 'Limitations and honest framing')}
<div class="alert warn"><strong>Disclosure.</strong> {DISC}</div>
<ul>
<li><strong>Predictive, not causal.</strong> Search Console has no AI-answer flag. Seasonality, other search-feature changes and impression inflation are competing explanations for the click loss.</li>
<li><strong>About five effective clients</strong> ({N['eff_range'][0]:.1f} to {N['eff_range'][1]:.1f} per evaluated month, from 20 to 30 clients). Every interval resamples whole clients and is wide (about &plusmn;0.2 at 50 pages), so differences in P@50 between rankers are not established.</li>
<li><strong>Unmeasurable pages.</strong> The headline metrics drop pages that lost most of their visibility; counting them as misses lowers P@50 to {f2(may['p50_unm'])} on the test month and {f2(apr['p50_unm'])} on the confirmation month, while the editor's rule scores {f2(may['rule']['p@50_unmeasurable_as_miss'])} and {f2(apr['rule']['p@50_unmeasurable_as_miss'])}. {RULE_SENT}</li>
<li>Tuning and the champion test share two validation folds, so the validation numbers are optimistic; three challenger comparisons were made and not corrected for multiplicity; the confirmation fold and the test month are the less-biased reads.</li>
<li>Four evaluated months only. The V1 comparison is not a like-for-like test of improvement.</li>
<li>Eligibility keeps pages still in the top 10 with enough clicks, which favours survivors; measurability uses outcome impressions.</li>
<li>Snapshot attributes (<code>dim_content</code>, keyword fields) are as of July 2026, not as of each origin.</li>
<li>The per-client cap and the 90-day rule in the playbook are guardrails. The cap was introduced in V1 after the test month was opened and plays no part in any V2 selection or headline metric.</li>
<li>The label was designed on the V1 development origins.</li>
</ul>
<div class="alert info">Predictive, not causal. Decision support for human reviewers; no claim that acting on the queue changes traffic.</div>
</section>

{sec('6', 'recs', 'Ranked recommendations')}
<table class="spec"><tbody>
<tr><th>1</th><td><strong>Protect before repairing.</strong> Each month, review the 50-page queue (at most {pb['guardrails']['client_cap']} per client, pages at least 90 days old). Start with pages earning more than their rank normally earns ({pb['reason_codes']['CTR_ABOVE_PEERS']} of the {pb['queue_size']} July pages): make them the source an AI answer cites (answer-first summary, FAQ, structured data) before any rewrite.</td></tr>
<tr><th>2</th><td><strong>Verify visibility and check the live results page first.</strong> Confirm the page still ranks and still earns impressions, record whether an AI answer appears, and confirm demand if impressions jumped threefold or the page has one month of data.</td></tr>
<tr><th>3</th><td><strong>Prioritise transactional and commercial pages</strong> in AI-search planning for this panel; informational pages did not show a significant fall.</td></tr>
<tr><th>4</th><td><strong>Do not use Google rank as a proxy for AI visibility.</strong> AI discovery followed exposure, not rank.</td></tr>
<tr><th>5</th><td><strong>Measure the action.</strong> Leave 10 of each month's 50 pages untouched and compare, so the value of the work can be estimated.</td></tr>
<tr><th>6</th><td><strong>Retrain monthly, and score the frozen July queue.</strong> Pause the queue if a matured month's precision falls below 1.3&times; its base rate. The July queue is frozen by its SHA-256; its own outcome is the one untouched test left, once a later data release exists.</td></tr>
</tbody></table>
<p class="cap">July 2026 queue: {pb['eligible_pages']:,} eligible pages; 50 queued across {pb['queue_clients']} clients; about {round(pb['clicks_at_risk_2_months_if_25pct_lost'], -3):,.0f} clicks at stake over two months if each lost a quarter. {pb['overlap_with_v1_queue_pages']} of the 50 pages are also in V1's July queue.</p>
</section>

{sec('7', 'repro', 'Reproducibility')}
<p>Code: <code>work/scripts/</code>, one module per stage; <code>v2_*.py</code> is the V2 pipeline. Notebooks: <code>work/notebooks/</code>, one per assignment card, executed with outputs. Receipts: <code>work/outputs/*.json</code>; every number on this page is read from them by <code>make_paper.py</code>. Seed 42; warehouse revision <code>{summ['warehouse_revision'][:8]}</code>; single-threaded fits, so results are bit-reproducible.</p>
<p><strong>Pre-registration and one logged deviation.</strong> The V2 specification (<code>work/V2_SPEC.md</code> and <code>v2_spec.json</code>) was committed and hashed before any V2 code ran. One implementation deviation is logged in its section 13 (a column that carries no information in a given boosted fit is left out of that fit only), which means the specification's current hash differs from the pre-registered one. <code>v2_freeze.json</code> records both: the markdown spec was <code>{sh(N['spec_hash_pre']['work/V2_SPEC.md'])}</code> at pre-registration and <code>{sh(N['spec_hash_now']['work/V2_SPEC.md'])}</code> at the freeze. The edit was made before any V2 result was used.</p>
<p>Order of operations, each with a receipt: spec, selection, confirmation, secondary label, checks, freeze ({fz['frozen_at']}), test scored once ({N['scored_at']}), live queue. The unseal step refuses a second run. Offline check: <code>python work/scripts/verify_evidence.py</code> re-executes the capstone notebook and matches the headline numbers to the receipts. To rerun from data: set <code>HF_TOKEN</code> as an environment variable or Colab secret, run the notebooks in card order, and set <code>RERUN_PIPELINE = True</code> to recompute the V2 steps in a scratch copy.</p>
</section>

{sec('8', 'credit', 'Acknowledgments and data credit')}
<p>Built on the <a href="https://flyrank.ai">FlyRank ML Internship dataset</a>: the pseudonymized warehouse release, the starter repository and the curriculum. Thanks to the FlyRank mentors. All errors and interpretations are mine.</p>
<p class="cap">References: Cawley, G. C. and Talbot, N. L. C. (2010), On over-fitting in model selection and subsequent selection bias in performance evaluation, <em>Journal of Machine Learning Research</em> 11. McElfresh, D. et al. (2023), When do neural nets outperform boosted trees on tabular data?, <em>NeurIPS Datasets and Benchmarks</em>. Grinsztajn, L., Oyallon, E. and Varoquaux, G. (2022), Why do tree-based models still outperform deep learning on typical tabular data?, <em>NeurIPS Datasets and Benchmarks</em>.</p>
</section>

</div></div></main>
<footer><div class="foot-inner">
<p class="credit">Built on the <a href="https://flyrank.ai">FlyRank ML Internship dataset</a>.</p>
<p>Predictive, not causal · decision support for human reviewers · refit monthly.</p>
<p><a href="{REPO}">Repository</a> · <a href="{REPO}/blob/main/work/notebooks/capstone.ipynb">Capstone notebook</a> · <a href="https://huggingface.co/datasets/FlyRank/internship-warehouse">Dataset release</a></p>
</div></footer>
</body></html>
"""
    os.makedirs(f"{W}/paper", exist_ok=True)
    open(f"{W}/paper/index.html", "w", encoding="utf-8").write(html)
    print("paper written:", len(html) // 1024, "KB")


if __name__ == "__main__":
    build()
