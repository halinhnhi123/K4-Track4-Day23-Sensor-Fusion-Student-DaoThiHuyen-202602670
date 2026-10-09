"""Tests for tools/check_submission.py on a throwaway git repo."""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

from fusion_lab.evaluation import aggregate_records

REPO_ROOT = Path(__file__).resolve().parents[2]

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not installed")


def _load_checker():
    spec = importlib.util.spec_from_file_location("check_submission", REPO_ROOT / "tools" / "check_submission.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


checker = _load_checker()


def _record(mode: str, frame: int) -> dict:
    return {"mode": mode, "frame": frame, "det_tp": 2, "det_fp": 1, "det_fn": 0, "valid_gt": 2,
            "confirmed": 2, "matches": 2, "sum_sq_err": 0.08, "ghosts": 0, "misses": 0}


def _write_artifacts(artifacts: Path) -> None:
    artifacts.mkdir(parents=True)
    frames, seed, segment = [0, 2], 0, "segment.tfrecord"
    records = [_record(mode, f) for mode in ("lidar", "fused") for f in range(frames[0], frames[1] + 1)]
    metrics = aggregate_records(records, "compare", frames, seed, segment)
    (artifacts / "metrics.json").write_text(json.dumps(metrics))
    (artifacts / "grade_run.log").write_text("".join(json.dumps(r) + "\n" for r in records))
    for mode in ("lidar", "fused"):
        mode_records = [r for r in records if r["mode"] == mode]
        (artifacts / f"metrics_{mode}.json").write_text(
            json.dumps(aggregate_records(mode_records, mode, frames, seed, segment)))
        (artifacts / f"grade_run_{mode}.log").write_text("".join(json.dumps(r) + "\n" for r in mode_records))


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def _commit_all(root: Path) -> None:
    _git(root, "add", "-A")
    _git(root, "-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false", "commit", "-qm", "work")


@pytest.fixture
def submission(tmp_path: Path) -> Path:
    root = tmp_path / "fork"
    for path in checker.EXERCISE_FILES.values():
        (root / path).parent.mkdir(parents=True, exist_ok=True)
        (root / path).write_text("def solved():\n    return 1\n")
    _write_artifacts(root / "student" / "artifacts")
    (root / checker.SUBMISSION).write_text(
        "- Họ tên: Nguyen Van A\n- MSSV: 2A20260000\n"
        "- Link repo (fork): https://github.com/a/K4-L2L3-DAY23-NguyenVanA-2A20260000-SensorFusion\n"
        "- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): Không dùng AI\n",
        encoding="utf-8",
    )
    _git(root, "init", "-q")
    _commit_all(root)
    return root


def _failed(root: Path) -> list[str]:
    return [name for name, ok, _ in checker.run_checks(root) if not ok]


def test_complete_submission_passes(submission: Path) -> None:
    assert _failed(submission) == []


def test_remaining_todo_fails_its_part(submission: Path) -> None:
    path = submission / checker.EXERCISE_FILES["F"]
    path.write_text('def f():\n    raise NotImplementedError("TODO: implement")\n')
    _commit_all(submission)
    assert _failed(submission) == ["Part F — student/workspace/association.py"]


def test_metrics_inconsistent_with_log_fails(submission: Path) -> None:
    metrics_path = submission / "student/artifacts/metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["tracking"]["fused"]["matches"] += 1
    metrics_path.write_text(json.dumps(metrics))
    _commit_all(submission)
    assert _failed(submission) == ["metrics.json khớp grade_run.log (compare, đủ hai mode)"]


def test_blank_submission_field_fails(submission: Path) -> None:
    (submission / checker.SUBMISSION).write_text(
        "- Họ tên:\n- MSSV: 1\n", encoding="utf-8"
    )
    _commit_all(submission)
    assert _failed(submission) == ["SUBMISSION.md: thông tin + khai báo AI"]


def test_forbidden_file_and_secret_fail(submission: Path) -> None:
    (submission / "student/config").mkdir(parents=True)
    (submission / "student/config/paths.yaml").write_text("waymo_dir: x\n")
    # Built at runtime so this test file does not itself look like a leaked key.
    (submission / "notes.py").write_text("api_" + 'key = "' + "x" * 26 + '"\n')
    _commit_all(submission)
    assert _failed(submission) == [
        "Không commit dữ liệu/weights/paths.yaml/file nén",
        "Không lộ API key / token",
    ]


def test_uncommitted_change_fails(submission: Path) -> None:
    (submission / checker.SUBMISSION).write_text(
        "- Họ tên: changed\n", encoding="utf-8"
    )
    assert "Mọi thay đổi trong student/ đã commit" in _failed(submission)
