"""Allowlisted Kaggle source bundle: exclusions, credentials, SHA-256 manifest."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

from knee.kaggle_bundle import (
    BUNDLE_NAME,
    build_bundle,
    iter_allowlisted_files,
    would_include,
)
from knee.config import PROJECT_ROOT


def test_would_include_allowlist_and_forbidden_paths():
    assert would_include("knee/kaggle_audit.py")
    assert would_include("scripts/build_kaggle_bundle.py")
    assert would_include("configs/baseline.yaml")
    assert would_include("notebooks/01_kaggle_real_data_audit.ipynb")
    assert would_include("notebooks/03_kaggle_report_labeler_v2.ipynb")
    assert would_include("configs/report_labeler_v1.yaml")
    assert would_include("notebooks/04_kaggle_second_reader_gold_pilot.ipynb")
    assert would_include("configs/second_reader_pilot.yaml")
    assert would_include("literature/ontology.yaml")
    assert would_include("configs/report_labeler.yaml")
    assert would_include("knee/report_labels.py")
    assert not would_include("knee/weak_labels.csv")
    assert not would_include("knee/gold_predictions.csv")
    assert not would_include("knee/language_assignments.csv")
    assert not would_include("knee/gold_second_reader_predictions.csv")
    assert not would_include("knee/gold_consensus_predictions.csv")
    assert would_include("literature/ontology.yaml")
    assert would_include("pyproject.toml")
    assert not would_include(".git/config")
    assert not would_include(".venv/lib/python3.12/site.py")
    assert not would_include("data/train.csv")
    assert not would_include("knee/secret.dcm")
    assert not would_include("knee/model.pt")
    assert not would_include("knee/kaggle.json")
    assert not would_include("knee/.env")
    assert not would_include("outputs/preds.csv")
    assert not would_include("predictions/submission.csv")
    assert not would_include("checkpoints/model.ckpt")
    assert not would_include("reports/data_audit.md")
    assert not would_include("knee/weights.pkl")
    assert not would_include("scripts/id_rsa")


def test_bundle_from_temp_repo_excludes_credentials_and_dicom(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "knee").mkdir()
    (repo / "knee" / "config.py").write_text("TARGET_KEYS = ()\n")
    (repo / "knee" / "kaggle_audit.py").write_text("def run_notebook_audit():\n    return {}\n")
    (repo / "knee" / "kaggle.json").write_text('{"username": "x", "key": "SECRETTOKEN"}\n')
    (repo / "knee" / ".env").write_text("KAGGLE_KEY=abc\n")
    (repo / "knee" / "weights.pt").write_bytes(b"fake-weights")
    (repo / "knee" / "leak.dcm").write_bytes(b"DICM")
    (repo / "scripts").mkdir()
    (repo / "scripts" / "build_kaggle_bundle.py").write_text("print(1)\n")
    (repo / "configs").mkdir()
    (repo / "configs" / "baseline.yaml").write_text("seed: 1\n")
    (repo / "notebooks").mkdir()
    (repo / "notebooks" / "01_kaggle_real_data_audit.ipynb").write_text("{}\n")
    (repo / "pyproject.toml").write_text("[project]\nname='x'\n")
    (repo / "data").mkdir()
    (repo / "data" / "train.csv").write_text("StudyInstanceUID\n1\n")
    (repo / "outputs").mkdir()
    (repo / "outputs" / "preds.csv").write_text("p\n")
    (repo / ".git").mkdir()
    (repo / ".git" / "config").write_text("ok\n")
    (repo / "reports").mkdir()
    (repo / "reports" / "with_clinical_text.md").write_text("UNIQUE_REPORT hello\n")

    dest = tmp_path / "bundle_out"
    manifest = build_bundle(repo_root=repo, output_dir=dest)
    names = [row["path"] for row in manifest["files"]]
    assert "knee/config.py" in names
    assert "scripts/build_kaggle_bundle.py" in names
    assert "knee/kaggle.json" not in names
    assert "knee/.env" not in names
    assert "knee/weights.pt" not in names
    assert "knee/leak.dcm" not in names
    assert "data/train.csv" not in names
    assert "outputs/preds.csv" not in names
    assert ".git/config" not in names
    assert "reports/with_clinical_text.md" not in names
    assert all("sha256" in row and len(row["sha256"]) == 64 for row in manifest["files"])

    zip_path = dest / BUNDLE_NAME
    with zipfile.ZipFile(zip_path) as zf:
        zip_names = zf.namelist()
    assert "knee/kaggle.json" not in zip_names
    blob = zip_path.read_bytes()
    assert b"SECRETTOKEN" not in blob
    assert b"KAGGLE_KEY" not in blob
    assert manifest["zip_sha256"] == __import__("hashlib").sha256(blob).hexdigest()


def test_real_repo_bundle_skips_venv_and_git(tmp_path):
    dest = tmp_path / "out"
    manifest = build_bundle(repo_root=PROJECT_ROOT, output_dir=dest)
    names = [row["path"] for row in manifest["files"]]
    assert any(name.startswith("knee/") and name.endswith(".py") for name in names)
    assert "scripts/build_kaggle_bundle.py" in names
    assert "notebooks/01_kaggle_real_data_audit.ipynb" in names
    assert not any(name.startswith(".git/") for name in names)
    assert not any(".venv" in name for name in names)
    assert not any(name.startswith("data/") for name in names)
    assert not any(name.endswith(".dcm") for name in names)
    assert not any(name.endswith(".pt") for name in names)
    assert "kaggle.json" not in names
    allowlisted = {str(p.relative_to(PROJECT_ROOT).as_posix()) for p in iter_allowlisted_files(PROJECT_ROOT)}
    assert set(names) == allowlisted
    manifest_path = Path(manifest["manifest_path"])
    loaded = json.loads(manifest_path.read_text())
    assert loaded["n_files"] == len(names)


def test_build_kaggle_bundle_cli(tmp_path):
    import subprocess
    import sys

    dest = tmp_path / "cli_out"
    proc = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "build_kaggle_bundle.py"),
            "--output-dir",
            str(dest),
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (dest / BUNDLE_NAME).is_file()
    payload = json.loads(proc.stdout)
    assert payload["n_files"] > 0
    assert Path(payload["zip_path"]).is_file()
