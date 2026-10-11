"""Small, dependency-free notebook authoring helpers for this course."""
import json
import textwrap
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def md(source):
    return {"cell_type": "markdown", "id": uuid.uuid4().hex[:8], "metadata": {},
            "source": textwrap.dedent(source).strip() + "\n"}


def code(source):
    return {"cell_type": "code", "id": uuid.uuid4().hex[:8], "metadata": {},
            "execution_count": None, "outputs": [],
            "source": textwrap.dedent(source).strip() + "\n"}


def bootstrap():
    return code('''
        from pathlib import Path
        import sys
        ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
                     if (p / "course_utils.py").exists()), None)
        if ROOT is None:
            raise RuntimeError("Open Jupyter from the ObjectDetectionPractical folder.")
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from course_utils import seed_everything, get_device, SMOKE_TEST, ARTIFACTS
        seed_everything(42)
        device = get_device()
        print("Device:", device, "| smoke test:", SMOKE_TEST)
    ''')


def write_notebook(filename, cells):
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    notebook = {"cells": cells, "metadata": {
        "kernelspec": {"display_name": "Image Classification (GPU)",
                       "language": "python", "name": "image-classification"},
        "language_info": {"name": "python", "version": "3.13"}},
        "nbformat": 4, "nbformat_minor": 5}
    path = NOTEBOOKS / filename
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(path.name)
