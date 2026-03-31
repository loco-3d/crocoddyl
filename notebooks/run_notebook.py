import os
import re
import sys

import nbformat
from nbconvert.preprocessors import ExecutePreprocessor


def prepare_notebook_for_test(notebook_content):
    for cell in notebook_content.cells:
        if cell.cell_type != "code":
            continue
        source = cell.source
        source = re.sub(r"\bn_iter\s*=\s*1000000\b", "n_iter = 10000", source)
        source = re.sub(
            r"HTML\(anim\.to_html5_video\(\)\)",
            'print("Skipping animation rendering in notebook test mode.")',
            source,
        )
        source = re.sub(
            r"HTML\(anim\.to_jshtml\(\)\)",
            'print("Skipping animation rendering in notebook test mode.")',
            source,
        )
        cell.source = source


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
    if os.environ.get("CROCODDYL_NOTEBOOK_TEST") == "1":
        os.environ.setdefault("MPLBACKEND", "Agg")
        prepare_notebook_for_test(notebook_content)
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
