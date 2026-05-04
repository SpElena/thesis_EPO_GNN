import subprocess
import sys

scripts = [
    "chapter_1/epo_model.py",
    "chapter_1/epo_exp.py", 
    "chapter_1/epo_plot.py"
]

for script in scripts:
    print(f"Running {script}...")
    result = subprocess.run([sys.executable, script])