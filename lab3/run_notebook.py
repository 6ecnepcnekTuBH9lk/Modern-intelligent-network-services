from pathlib import Path
import argparse
import sys

from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpec
import nbformat
from nbclient import NotebookClient


class LocalKernel(KernelManager):
    @property
    def kernel_spec(self):
        return KernelSpec(argv=[sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
                          display_name="Python 3", language="python")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--review-only", action="store_true")
    args = parser.parse_args()
    path = Path(__file__).with_name("rag_experiments.ipynb").resolve()
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    manager = LocalKernel()
    to_run = notebook
    if args.review_only:
        last = notebook.cells[-1]
        if last.cell_type != "code" or "finalize_review" not in last.source:
            raise ValueError("Последняя ячейка разметки не найдена")
        to_run = nbformat.v4.new_notebook(cells=[last])
    client = NotebookClient(to_run, km=manager, timeout=3600, record_timing=False,
                            resources={"metadata": {"path": str(path.parent)}})
    client.on_cell_executed = lambda cell, cell_index, execute_reply: print(
        f"Ячейка {cell_index + 1}/{len(notebook.cells)} выполнена.", flush=True)
    client.execute()
    if args.review_only:
        last.execution_count = 1 + max(c.execution_count for c in notebook.cells[:-1] if c.cell_type == "code")
    for cell in notebook.cells:
        cell.metadata.pop("execution", None)
    nbformat.validate(notebook)
    nbformat.write(notebook, args.output or path)
    print("Ноутбук сохранён с результатами.")
