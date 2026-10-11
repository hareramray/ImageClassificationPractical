"""Validate notebooks and optionally execute each in a fresh Jupyter kernel.

python tools/validate_notebooks.py --execute --smoke
python tools/validate_notebooks.py --execute --pattern 07_*.ipynb
"""
import argparse
import ast
import fnmatch
import json
import os
from pathlib import Path
import re
import sys
import time
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--pattern", default="*.ipynb")
    parser.add_argument("--exclude", action="append", default=[])
    parser.add_argument("--timeout", type=int, default=1200)
    args = parser.parse_args()
    os.environ["COURSE_SMOKE_TEST"] = "1" if args.smoke else "0"
    if args.device:
        os.environ["COURSE_DEVICE"] = args.device
    os.environ["PYTHONUTF8"] = "1"
    os.environ["MPLBACKEND"] = "module://matplotlib_inline.backend_inline"
    mode = "smoke" if args.smoke else "full"
    output = ROOT / "artifacts" / "verification" / f"{mode}_{args.device or 'auto'}"
    output.mkdir(parents=True, exist_ok=True)
    paths = [p for p in sorted((ROOT / "notebooks").glob(args.pattern))
             if not any(fnmatch.fnmatch(p.name, pattern) for pattern in args.exclude)]
    if not paths:
        parser.error("No matching notebooks")
    results = []
    for path in paths:
        started = time.perf_counter()
        print(f"Checking {path.name}", flush=True)
        notebook = nbformat.read(path, as_version=4)
        status, error = "passed", None
        try:
            nbformat.validate(notebook)
            for i, cell in enumerate(notebook.cells):
                if cell.cell_type == "code":
                    ast.parse(cell.source, filename=f"{path.name}:cell {i}")
                else:
                    for target in re.findall(r"\]\(([^)]+\.ipynb)\)", cell.source):
                        if not (path.parent / target).exists():
                            raise ValueError(f"Broken notebook link: {target}")
            if args.execute:
                client = NotebookClient(notebook, timeout=args.timeout,
                                        kernel_name="image-classification",
                                        resources={"metadata": {"path": str(path.parent)}})
                client.execute()
        except Exception as exc:
            status, error = "failed", str(exc)
            print(error, flush=True)
        if args.execute:
            nbformat.write(notebook, output / path.name)
        row = {"notebook": path.name, "status": status,
               "seconds": round(time.perf_counter() - started, 2),
               "code_cells": sum(c.cell_type == "code" for c in notebook.cells),
               "error": error}
        results.append(row)
        (output / (path.stem + ".result.json")).write_text(
            json.dumps(row, indent=2), encoding="utf-8")
        print(f"  {status} ({row['seconds']} s)", flush=True)
        recorded = [json.loads(p.read_text(encoding="utf-8"))
                    for p in sorted(output.glob("*.result.json"))]
        (output / "results.json").write_text(json.dumps({
            "mode": mode, "device": args.device or "auto",
            "executed": args.execute, "python": sys.executable,
            "results": recorded}, indent=2), encoding="utf-8")
    failures = sum(r["status"] != "passed" for r in results)
    print(f"{len(results) - failures}/{len(results)} notebooks passed.")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
