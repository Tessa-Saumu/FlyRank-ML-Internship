"""Execute receipt checks and figure regeneration without modifying evidence files.

This is an offline artifact check, not a warehouse rebuild or model retraining.
Run from any directory: python work/scripts/verify_evidence.py
"""
from pathlib import Path
import shutil
import tempfile

import nbformat
from nbclient import NotebookClient


def main():
    repo = Path(__file__).resolve().parents[2]
    # Check the onset receipt up front: the capstone reads it, and a missing file would
    # otherwise surface as a download error from inside the notebook.
    if not (repo / 'work/outputs/onset_screening.json').exists():
        raise SystemExit(
            'work/outputs/onset_screening.json is missing. Run Week 5 section 3c, or '
            'python work/scripts/onset_screening.py, then run this check again.'
        )
    with tempfile.TemporaryDirectory(prefix='flyrank-evidence-') as scratch:
        root = Path(scratch)
        for folder in ('outputs', 'paper', 'scripts'):
            shutil.copytree(repo / 'work' / folder, root / 'work' / folder)
        notebook = nbformat.read(repo / 'work/notebooks/capstone.ipynb', as_version=4)
        NotebookClient(
            notebook, timeout=120, kernel_name='python3',
            resources={'metadata': {'path': str(root)}},
        ).execute()
        final_text = ''.join(
            output.get('text', '') for cell in notebook.cells
            for output in cell.get('outputs', [])
        )
        if 'ALL 31 CHECKS PASS' not in final_text:
            raise RuntimeError('Expected headline receipt-check confirmation absent.')
        if 'ONSET CHECKS PASS' not in final_text:
            raise RuntimeError(
                'Onset receipt checks did not pass. Run Week 5 section 3c (or '
                'work/scripts/onset_screening.py) and commit work/outputs/onset_screening.json.'
            )
        print('All capstone cells executed; 31 headline checks and the onset checks passed;')
        print('figures regenerated from receipts.')
        print('No warehouse access or retraining; source artifacts unchanged.')


if __name__ == '__main__':
    main()
