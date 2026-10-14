from __future__ import annotations

import ast
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NOTEBOOK = next(REPO.glob("*.ipynb"))


def test_python_module_parses():
    ast.parse((REPO / "glass_quality.py").read_text(encoding="utf-8"))


def test_notebook_is_valid_json_and_has_no_stale_outputs():
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    assert notebook["nbformat"] >= 4
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell.get("source", [])), filename=str(NOTEBOOK))
            assert cell.get("outputs", []) == []
            assert cell.get("execution_count") is None


def test_notebook_has_no_local_machine_paths_or_removed_submission_dependency():
    text = NOTEBOOK.read_text(encoding="utf-8")
    assert "/home/" not in text
    assert "C:\\" not in text
    assert "Sample_Submission" not in text
    assert "distplot" not in text
    assert "extratrees_addv1.xls\"" not in text


def test_required_engineering_guards_are_documented_in_code():
    source = (REPO / "glass_quality.py").read_text(encoding="utf-8")
    for term in (
        "FileNotFoundError",
        "StratifiedKFold",
        "random_state=RANDOM_STATE",
        "labels=fitted.classes_",
    ):
        assert term in source
