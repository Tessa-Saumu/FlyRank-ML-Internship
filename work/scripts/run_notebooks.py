"""Execute the submission notebooks top to bottom in place (cwd = work/notebooks)."""
import os
import sys
import time

import nbformat
from nbclient import NotebookClient

NB = os.path.dirname(os.path.abspath(__file__)) + "/../notebooks"
ORDER = ["w01_research_question.ipynb", "w02_ml_task_framing.ipynb", "w03_data_contract.ipynb", "w03_feature_leakage_check.ipynb",
         "w04_signal_audit.ipynb", "w04_baseline_score.ipynb", "w05_model.ipynb", "w06_validation_audit.ipynb",
         "w07_action_playbook.ipynb", "capstone.ipynb"]
for name in (sys.argv[1:] or ORDER):
    t = time.time()
    nb = nbformat.read(f"{NB}/{name}", as_version=4)
    NotebookClient(nb, timeout=3600, kernel_name="python3", resources={"metadata": {"path": NB}}).execute()
    nbformat.write(nb, f"{NB}/{name}")
    print(f"executed {name} in {time.time() - t:.0f}s", flush=True)
