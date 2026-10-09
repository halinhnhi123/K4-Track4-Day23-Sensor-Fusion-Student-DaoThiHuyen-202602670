#!/usr/bin/env python3
"""Self-check a Day 23 submission before pushing and submitting on LMS.

Run from anywhere inside the repo:

    python tools/check_submission.py

It checks structure, artifacts, SUBMISSION.md, forbidden files and leaked keys.
It does not grade the work. Exit code 0 means ready to submit, 1 otherwise.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_FILE_MB = 20

EXERCISE_FILES = {
    "E": "student/workspace/kalman.py",
    "F": "student/workspace/association.py",
    "G": "student/workspace/camera_fusion.py",
    "H": "student/workspace/track_management.py",
}
ARTIFACT_FILES = (
    "student/artifacts/metrics.json",
    "student/artifacts/grade_run.log",
    "student/artifacts/metrics_lidar.json",
    "student/artifacts/metrics_fused.json",
    "student/artifacts/grade_run_lidar.log",
    "student/artifacts/grade_run_fused.log",
)
SUBMISSION = "student/SUBMISSION.md"
# SUBMISSION.md fields that must not be left blank.
REQUIRED_FIELDS = (
    "Họ tên",
    "MSSV",
    "Link repo (fork)",
    "Công cụ đã dùng (ChatGPT, Copilot, Claude, …)",
)
FORBIDDEN_SUFFIXES = (".tfrecord", ".pth", ".pt", ".ckpt", ".pickle", ".zip", ".tar", ".gz", ".7z", ".rar")
FORBIDDEN_NAMES = ("paths.yaml", ".env")
SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_-]{20,}"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"hf_[A-Za-z0-9]{30,}"),
    re.compile(r"gh[pousr]_[A-Za-z0-9]{36,}"),
    re.compile(r"AIza[0-9A-Za-z_-]{35}"),
    re.compile(r"(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*['\"][^'\"\s]{12,}['\"]"),
)
TODO_MARKER = re.compile(r"raise\s+NotImplementedError\(\s*[\"']TODO")


def git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return result.stdout


def tracked_files(root: Path) -> list[str]:
    return [line for line in git(root, "ls-files", "-z").split("\0") if line]


def dirty_files(root: Path, pathspec: str) -> list[str]:
    output = git(root, "status", "--porcelain", "--untracked-files=no", "--", pathspec)
    return [line[3:] for line in output.splitlines() if line.strip()]


def submission_fields(text: str) -> dict[str, str]:
    """Map '- Label: value' lines to their stripped values."""
    fields = {}
    for line in text.splitlines():
        match = re.match(r"\s*-\s+([^:]+?)\s*:\s*(.*)$", line)
        if match:
            fields.setdefault(match.group(1).replace("`", "").strip(), match.group(2).strip())
    return fields


def validate_artifacts(root: Path) -> str | None:
    """Return None when metrics.json matches grade_run.log under the grading rules."""
    platform_dir = root / "platform"
    if str(platform_dir) not in sys.path:
        sys.path.insert(0, str(platform_dir))
    from fusion_lab.evaluation import read_records, validate_metrics_records

    artifacts = root / "student" / "artifacts"
    try:
        metrics = json.loads((artifacts / "metrics.json").read_text())
        if metrics.get("fusion_mode") != "compare":
            return f"fusion_mode = {metrics.get('fusion_mode')!r}, cần 'compare'"
        if any(metrics.get("tracking", {}).get(mode) is None for mode in ("lidar", "fused")):
            return "tracking.lidar hoặc tracking.fused là null"
        validate_metrics_records(metrics, read_records(artifacts / "grade_run.log"))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def run_checks(root: Path) -> list[tuple[str, bool, str]]:
    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, ok, detail))

    try:
        tracked = tracked_files(root)
    except (OSError, subprocess.CalledProcessError):
        check("Repo git", False, "không đọc được git; chạy trong thư mục clone của fork")
        return results
    tracked_set = set(tracked)

    for part, path in EXERCISE_FILES.items():
        file = root / path
        if not file.is_file():
            check(f"Part {part} — {path}", False, "thiếu file")
            continue
        remaining = len(TODO_MARKER.findall(file.read_text(encoding="utf-8")))
        check(f"Part {part} — {path}", remaining == 0,
              f"còn {remaining} hàm `NotImplementedError(\"TODO ...\")`" if remaining else "")

    missing = [path for path in ARTIFACT_FILES if path not in tracked_set]
    check("Artifacts đã commit", not missing, "chưa commit: " + ", ".join(missing) if missing else "")
    if not missing:
        error = validate_artifacts(root)
        check("metrics.json khớp grade_run.log (compare, đủ hai mode)", error is None, error or "")
        if error is None:
            metrics = json.loads((root / "student/artifacts/metrics.json").read_text())
            check("Lần chạy chấm điểm dùng seed 0", metrics["seed"] == 0, f"seed = {metrics['seed']}")
            check("Khoảng frame bắt đầu từ 0", metrics["frames"][0] == 0, f"frames = {metrics['frames']}")

    if SUBMISSION not in tracked_set:
        check("SUBMISSION.md đã commit", False, f"thiếu {SUBMISSION}")
    else:
        fields = submission_fields((root / SUBMISSION).read_text(encoding="utf-8"))
        blank = [label for label in REQUIRED_FIELDS if not fields.get(label)]
        check("SUBMISSION.md: thông tin + khai báo AI", not blank,
              "còn trống: " + ", ".join(blank) if blank else "")

    forbidden = [path for path in tracked
                 if path.lower().endswith(FORBIDDEN_SUFFIXES) or Path(path).name in FORBIDDEN_NAMES]
    check("Không commit dữ liệu/weights/paths.yaml/file nén", not forbidden, ", ".join(forbidden[:10]))

    large = [path for path in tracked
             if (root / path).is_file() and (root / path).stat().st_size > MAX_FILE_MB * 1024 * 1024]
    check(f"Không có file > {MAX_FILE_MB} MB", not large, ", ".join(large))

    leaked = []
    for path in tracked:
        file = root / path
        if not file.is_file() or file.stat().st_size > 2 * 1024 * 1024:
            continue
        try:
            text = file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(pattern.search(text) for pattern in SECRET_PATTERNS):
            leaked.append(path)
    check("Không lộ API key / token", not leaked, ", ".join(leaked))

    dirty = dirty_files(root, "student")
    check("Mọi thay đổi trong student/ đã commit", not dirty, ", ".join(dirty[:10]))
    return results


def main() -> int:
    results = run_checks(ROOT)
    for name, ok, detail in results:
        line = f"[{'PASS' if ok else 'FAIL'}] {name}"
        print(f"{line} — {detail}" if detail and not ok else line)
    failed = sum(not ok for _, ok, _ in results)
    print()
    if failed:
        print(f"KẾT QUẢ: CHƯA SẴN SÀNG — {failed} mục FAIL. Sửa rồi chạy lại.")
        return 1
    print("KẾT QUẢ: SẴN SÀNG NỘP")
    print("Tiếp theo: git push, rồi nộp link repo + `git rev-parse HEAD` trên LMS (xem SUBMISSION.md ở gốc repo).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
