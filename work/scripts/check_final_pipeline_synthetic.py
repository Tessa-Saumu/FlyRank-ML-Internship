from pathlib import Path
import sys, tempfile, json, os
import duckdb, nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager
sys.path.insert(0,str(Path('work/scripts').resolve()))
import final_submission_pipeline as p
with tempfile.TemporaryDirectory(prefix='flyrank_final_check_') as tmp:
 base=Path(tmp); facts=base/'facts'; facts.mkdir(); cache=base/'cache';cache.mkdir()
 con=duckdb.connect()
 con.execute("""CREATE TABLE fact AS SELECT CAST(d AS DATE) report_date,
 'client_'||CAST(i%12 AS VARCHAR) client_hash_id,'page_'||CAST(i AS VARCHAR) content_hash_id,
 CASE WHEN i%10=0 AND month(d)=2 THEN day(d)<=10 ELSE TRUE END gsc_data_available,
 CAST(CASE WHEN i>=120 THEN i-87 ELSE 10000*exp((CASE WHEN i%2=0 THEN -0.01 ELSE 0.002 END)*date_diff('day',DATE '2025-08-01',CAST(d AS DATE))) + i END AS INTEGER) gsc_impressions,
 4 gsc_clicks,8.0 gsc_avg_position,TRUE ga4_data_available,3 ga4_sessions,
 4 ga4_pageviews,2 ga4_engaged_sessions,1 scroll_events
 FROM generate_series(DATE '2025-08-01',DATE '2026-05-31',INTERVAL 1 DAY) ds(d) CROSS JOIN range(122) r(i)""")
 files={}
 for month in p.study.months_between(p.study.date(2025,8,1),p.study.date(2026,5,31)):
  path=facts/(month+'.parquet'); con.execute(f"COPY (SELECT * FROM fact WHERE strftime(report_date,'%Y-%m')='{month}') TO '{path.as_posix()}' (FORMAT PARQUET)"); files[month]=path.as_posix()
 dim=base/'dim.parquet';con.execute(f"COPY (SELECT DISTINCT client_hash_id,content_hash_id,DATE '2020-01-01' content_created_date FROM fact) TO '{dim.as_posix()}' (FORMAT PARQUET)")
 fs=p.study.load_origin_frames(con,p.CONFIG.feature_months,p.CONFIG,(7,),files,dim.as_posix())[7];con.close()
 assert p.CONFIG.minimum_impressions==1000
 for f in fs.values():
  assert f.recent30_impressions.ge(1000).all()
  assert 'page_120' not in set(f.content_hash_id), '990-impression page must be excluded'
  assert 'page_121' in set(f.content_hash_id), '1020-impression page must remain eligible'
 assert any(f[p.study.TARGET].isna().any() for f in fs.values()), 'Fixture must exercise unknown outcomes'
 for month,f in fs.items():f.to_parquet(cache/(month+'.parquet'),index=False)
 (cache/'manifest.json').write_text(json.dumps({'provenance':p.provenance()},default=str))
 spec=base/'kernels'/'flyrank-test';spec.mkdir(parents=True)
 (spec/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'test','language':'python'}))
 env=dict(os.environ,FLYRANK_FINAL_CACHE=str(cache),FLYRANK_FINAL_OUTPUT=str(base/'receipts'))
 order=['w03_data_contract','w03_feature_leakage_check','w04_signal_audit','w04_baseline_score','w05_model','w06_validation_audit','w07_action_playbook','capstone_revised']
 for name in order:
  km=KernelManager(kernel_name='flyrank-test',kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(base/'kernels')]))
  n=nbformat.read(Path('work/notebooks')/(name+'.ipynb'),as_version=4)
  try:
   NotebookClient(n,km=km,timeout=600,resources={'metadata':{'path':str(Path.cwd())}}).execute(env=env)
  finally:
   if km.has_kernel: km.shutdown_kernel(now=True)
  print('PASS',name,flush=True)
 validation=json.loads((base/'receipts/validation.json').read_text())
 assert validation['primary_baseline']=='static_rule'
 baseline_rows=json.loads((base/'receipts/baselines.json').read_text())['results']
 assert {'static_rule','acceleration_rule','coverage_adjusted_acceleration_rule'}.issubset({r['candidate'] for r in baseline_rows})
 static={r['origin']:r['p_at_50_confirmed_lower'] for r in baseline_rows if r['candidate']=='static_rule'}
 model_rows=json.loads((base/'receipts/models.json').read_text())['results']
 for decision in validation['paired_comparisons']:
  matches=[r for r in model_rows if r['candidate']==decision['candidate'] and r['model']==decision['model']]
  expected=sum(r['p_at_50_confirmed_lower']-static[r['origin']] for r in matches)/len(matches)
  assert abs(expected-decision['mean_delta'])<1e-12
 assert json.loads((base/'receipts/playbook.json').read_text())['selected_rule']=='static_rule'
 print('Synthetic warehouse query, eight notebooks and frozen-baseline comparisons passed. No synthetic receipts retained.',flush=True)
