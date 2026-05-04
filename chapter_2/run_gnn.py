import subprocess
import sys

scripts = [
    "chapter_2/gnn_epo_train.py",
    "chapter_2/gnn_epo_eval.py"
]

for script in scripts:
    print(f"Running {script}...")
    result = subprocess.run([sys.executable, script])