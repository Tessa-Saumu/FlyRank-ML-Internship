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
        print('All capstone cells executed; 31 receipt checks passed; figures regenerated.')
        print('No warehouse access or retraining; source artifacts unchanged.')


if __name__ == '__main__':
    main()
