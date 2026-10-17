"""Final submission stages; reuse the tested warehouse query and ranking definitions.

Private page-level frames/queues stay under the ignored local directory. Only
aggregate receipts are written under outputs. No fresh holdout is claimed.
"""
from pathlib import Path
import hashlib
import json
import os
from dataclasses import asdict
import duckdb
import numpy as np
import pandas as pd
import warehouse_feature_experiments as study

ROOT = study.ROOT
LOCAL = Path(os.environ.get('FLYRANK_FINAL_CACHE', ROOT / 'work/local_final_pipeline'))
OUT = Path(os.environ.get('FLYRANK_FINAL_OUTPUT', ROOT / 'work/outputs/final_pipeline'))
CONFIG = study.ExperimentConfig(reporting_lag_sensitivity_days=())
EXPERIMENT = ROOT / 'work/outputs/warehouse_feature_experiments/experiment_results.json'
REVISION = json.loads(EXPERIMENT.read_text(encoding='utf-8'))['source']['huggingface_revision']
RULE = 'coverage_adjusted_acceleration_rule'


def save(name, value):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / (name + '.json')).write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + '\n', encoding='utf-8')
    return value


def provenance():
    return {'revision': REVISION, 'config': asdict(CONFIG),
            'query_sha256': hashlib.sha256(Path(study.__file__).read_bytes()).hexdigest(),
            'pipeline_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'package_versions': study._package_versions()}


def frames():
    manifest = json.loads((LOCAL / 'manifest.json').read_text(encoding='utf-8'))
    if manifest['provenance'] != json.loads(json.dumps(provenance())):
        raise ValueError('Local cache configuration/source changed. Rebuild W03 with rebuild=True.')
    return {m: pd.read_parquet(LOCAL / (m + '.parquet')) for m in CONFIG.feature_months}


def contract(rebuild=False):
    LOCAL.mkdir(parents=True, exist_ok=True)
    if rebuild or not (LOCAL / 'manifest.json').exists():
        from huggingface_hub import hf_hub_download
        months = study.months_between(
            study._feature_window(study._month_end(CONFIG.feature_months[0]) - study.timedelta(days=7))['prior_start'],
            study._month_end(CONFIG.feature_months[-1]) + study.timedelta(days=30))
        assert max(months) <= '2026-05' and '2026-06' not in months
        token = study.resolve_token()
        def download(name):
            return hf_hub_download('FlyRank/internship-warehouse', name, repo_type='dataset', revision=REVISION, token=token)
        paths = {m: download(f'fact_content_daily_performance/month={m}/data_0.parquet') for m in months}
        dim = download('dim_content.parquet')
        con = duckdb.connect()
        try:
            con.execute("SET memory_limit='2GB'")
            con.execute('SET threads=1')
            con.execute('SET preserve_insertion_order=false')
            built = study.load_origin_frames(con, CONFIG.feature_months, CONFIG, (7,), paths, dim)[7]
        finally:
            con.close()
        for m, f in built.items():
            f.to_parquet(LOCAL / (m + '.parquet'), index=False)
        (LOCAL / 'manifest.json').write_text(json.dumps({'provenance': provenance()}, default=str), encoding='utf-8')
    audit = []
    for month, f in frames().items():
        assert not f.duplicated(['client_hash_id', 'content_hash_id']).any()
        assert f.recent30_days.ge(14).all() and f.recent30_impressions.ge(100).all()
        assert (pd.to_datetime(f.feature_cutoff_date) < pd.to_datetime(f.outcome_start)).all()
        assert f.loc[f.future30_days.lt(14), study.TARGET].isna().all()
        expected = f.future30_impressions < .8 * f.recent30_impressions
        known = f[study.TARGET].notna()
        assert (f.loc[known, study.TARGET].astype(bool) == expected[known]).all()
        audit.append({'origin': month, 'candidates': len(f), 'clients': int(f.client_hash_id.nunique()),
                      'measurable': int(known.sum()), 'unknown': int((~known).sum()),
                      'prevalence_measurable': float(f[study.TARGET].mean()),
                      'source_partition_bounds': f.attrs.get('source_partition_bounds', {}),
                      'feature_cutoff': str(f.feature_cutoff_date.iloc[0]),
                      'outcome_end': str(f.outcome_end.iloc[0])})
    return save('contract', {'provenance': provenance(), 'audit': audit})


def leakage():
    fs = frames()
    rows = []
    for origin in CONFIG.prediction_origins:
        choices = study._mature_training_frames(fs, origin, 7)
        months, train = choices['latest_only']
        test = fs[origin]
        x, z, _, _ = study._design_matrices(train, test, 'momentum')
        assert not set(x) & {'future30_impressions', study.TARGET, 'client_hash_id', 'content_hash_id'}
        for month in months:
            assert study._month_end(month) + study.timedelta(days=37) <= study._month_end(origin)
        rows.append({'origin': origin, 'training_months': list(months), 'features': list(x),
                     'training_rows': len(train), 'test_rows': len(test)})
    return save('leakage', {'checks': rows, 'scope': 'Forward time validation; repeated clients are allowed. No unseen-client claim.'})


def signals():
    rows = []
    for month, f in frames().items():
        score = study._rule_scores(f)[RULE]
        # Average ranks make a stable split even when ties prevent qcut bins.
        bucket = pd.cut(pd.Series(score).rank(pct=True), [0,.25,.5,.75,1], include_lowest=True)
        grouped = f.assign(bucket=bucket.to_numpy()).groupby('bucket', observed=True)
        for name, g in grouped:
            n = int(g[study.TARGET].notna().sum())
            rows.append({'origin': month, 'score_quartile': str(name), 'n': len(g), 'measurable': n,
                         'decline_rate': float(g[study.TARGET].mean()) if n else None,
                         'verdict_status': 'REVIEW_TABLE' if n >= 50 else 'INSUFFICIENT_DATA'})
    return save('signals', {'tables': rows, 'note': 'Larger scores mean faster recent deterioration. Review direction across months before writing a verdict.'})


def evaluate(models=False):
    fs = frames()
    rows = []
    for origin in CONFIG.prediction_origins:
        f = fs[origin]
        if models:
            months, train = study._mature_training_frames(fs, origin, 7)['latest_only']
            candidates = [(feature, model, study._fit_score(train, f, feature, model))
                          for feature in ('control', 'momentum') for model in study.MODELS]
        else:
            months = ()
            candidates = [(name, 'rule', values) for name, values in study._rule_scores(f).items()
                          if name in ('static_rule', 'acceleration_rule', RULE)]
            candidates.append(('random_reference', 'random', np.random.default_rng(CONFIG.seed).random(len(f))))
        for candidate, model, scores in candidates:
            row, _ = study._metric_row(f, scores, origin=origin, candidate=candidate,
                                      model=model, history_policy='latest_only' if models else 'not_applicable',
                                      training_months=months, reporting_lag_days=7)
            row['prevalence_confirmed_lower'] = float(f[study.TARGET].eq(1).mean())
            row['prevalence_confirmed_upper'] = float((f[study.TARGET].eq(1) | f[study.TARGET].isna()).mean())
            rows.append(row)
    return save('models' if models else 'baselines', {'provenance': provenance(), 'results': rows})


def validation():
    sources = [json.loads((OUT / (name + '.json')).read_text()) for name in ('baselines', 'models')]
    for source in sources:
        assert source['provenance'] == json.loads(json.dumps(provenance())), 'Stale receipts: rerun W04 and W05.'
    table = pd.DataFrame([r for s in sources for r in s['results']])
    for _, g in table.groupby('origin'):
        assert g.candidate_count.nunique() == 1 and g.labeled_count.nunique() == 1
    summary = table.groupby(['candidate', 'model']).agg(
        mean_p50=('p_at_50_confirmed_lower', 'mean'), worst_p50=('p_at_50_confirmed_lower','min'),
        mean_p20=('p_at_20_confirmed_lower', 'mean'), mean_p100=('p_at_100_confirmed_lower', 'mean'),
        origins=('origin','nunique')).reset_index()
    baseline = table.loc[table.candidate.eq(RULE)].set_index('origin').p_at_50_confirmed_lower
    decisions = []
    for (candidate, model), g in table[table.model.isin(study.MODELS)].groupby(['candidate','model']):
        delta = g.set_index('origin').p_at_50_confirmed_lower - baseline
        decisions.append({'candidate': candidate, 'model': model, 'mean_delta': float(delta.mean()),
                          'positive_origins': int(delta.gt(0).sum()),
                          'development_hurdle_met': bool(delta.mean() >= .10 and delta.gt(0).sum() >= 4),
                          'adopt': False})
    return save('validation', {'summary': summary.to_dict('records'), 'paired_comparisons': decisions,
                               'scope': 'Development rerun after selection on these months. No sealed holdout. Hurdle: +5 cases per 50 on average and positive in at least 4 origins. A new prospective evaluation is required for adoption.'})


def playbook():
    fs = frames()
    month = CONFIG.prediction_origins[-1]
    f = fs[month]
    scores = study._rule_scores(f)[RULE]
    queue = f.iloc[study._rank_indices(f, scores, 50)].copy()
    queue['reason'] = np.where(queue.recent7_days.lt(3), 'insufficient_recent7_tracking_neutral_score',
                              np.where(scores[study._rank_indices(f, scores, 50)] > -1,
                                       'recent_daily_impressions_falling', 'no_recent_deterioration'))
    queue['review_action'] = 'Check tracking, search demand, indexing and content relevance before recommending an edit.'
    queue['confidence_note'] = 'Development ranking; not a calibrated probability or an estimate of edit impact.'
    queue['manual_review_note'] = 'INPUT AFTER PRIVATE REVIEW: evidence, likely false alarm, proposed action and reason.'
    # Exclude all outcomes from the editor-facing queue.
    cols = ['client_hash_id','content_hash_id','recent30_impressions','recent30_days','recent7_days',
            'reason','review_action','confidence_note','manual_review_note']
    queue[cols].to_csv(LOCAL / 'private_review_queue.csv', index=False)
    return save('playbook', {'origin': month, 'selected_rule': RULE, 'queue_size': len(queue),
                             'diagnostics': study._queue_diagnostics(f, scores),
                             'manual_review_status': 'PENDING: review at least top 20 privately; document weak picks and 3 wrong cases without identifiers.',
                             'action': 'Human review only; no automatic refresh or causal traffic claim.'})
