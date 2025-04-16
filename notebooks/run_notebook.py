import os
import sys

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def run_notebook(notebook_path):
    # Check if the notebook exists
    if not os.path.exists(notebook_path):
        raise FileNotFoundError(f"Notebook '{notebook_path}' not found.")
    # Get the directory of the notebook
    notebook_dir = os.path.dirname(os.path.abspath(notebook_path))
    # Set the working directory to the notebook directory
    os.chdir(notebook_dir)
    # Read the notebook
    with open(notebook_path) as f:
        notebook_content = nbformat.read(f, as_version=4)
    # Create the notebook processor and exporter
    ep = ExecutePreprocessor(timeout=600, kernel_name="python3")
    try:
        # Execute the notebook
        ep.preprocess(notebook_content, {"metadata": {"path": "./"}})
        # Optionally: You can extract results and verify them here
        print(f"Notebook {notebook_path} executed successfully.")
    except Exception as e:
        print(f"Error executing notebook {notebook_path}: {e!s}")
        sys.exit(1)


if __name__ == "__main__":
    notebook_path = sys.argv[1]  # First argument is the notebook path
    run_notebook(notebook_path)
