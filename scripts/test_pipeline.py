"""
test_pipeline.py
-----------------
This script runs all steps of the pipeline to validate end-to-end functionality.

Usage:
    python test_pipeline.py
"""
import subprocess

subprocess.run(["python", "scripts/preprocessing.py"])
subprocess.run(["python", "scripts/training_model.py"])
subprocess.run(["python", "scripts/evaluate_model.py"])
