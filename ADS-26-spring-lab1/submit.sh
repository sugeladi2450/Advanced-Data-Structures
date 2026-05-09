#!/usr/bin/env sh
set -eu

if [ $# -lt 1 ]; then
    echo "Usage: ./submit.sh <student_id>"
    echo "       sh submit.sh <student_id>"
    echo "Example: ./submit.sh 523030910000"
    exit 1
fi

STUDENT_ID="$1"

SCRIPT_DIR=$(CDPATH= cd "$(dirname "$0")" && pwd)
cd "$SCRIPT_DIR"

PYTHON_MODE=

if command -v python3 >/dev/null 2>&1 && python3 -c "import sys; raise SystemExit(sys.version_info[0] < 3)" >/dev/null 2>&1; then
    PYTHON_MODE=python3
elif command -v python >/dev/null 2>&1 && python -c "import sys; raise SystemExit(sys.version_info[0] < 3)" >/dev/null 2>&1; then
    PYTHON_MODE=python
elif command -v py >/dev/null 2>&1 && py -3 -c "import sys" >/dev/null 2>&1; then
    PYTHON_MODE=py
elif command -v uv >/dev/null 2>&1 && uv run python -c "import sys; raise SystemExit(sys.version_info[0] < 3)" >/dev/null 2>&1; then
    PYTHON_MODE=uv
else
    echo "Error: Python 3 was not found."
    echo "Please install Python 3, or run this script in the lab environment after setting up Python."
    exit 1
fi

run_python() {
    case "$PYTHON_MODE" in
        python3)
            python3 "$@"
            ;;
        python)
            python "$@"
            ;;
        py)
            py -3 "$@"
            ;;
        uv)
            uv run python "$@"
            ;;
    esac
}

run_python - "$STUDENT_ID" <<'PY'
import sys
import zipfile
from pathlib import Path

student_id = sys.argv[1]
zip_name = f"lab1_{student_id}.zip"

required_files = [
    "graph/graph_db.py",
    "data/load_graph_json.py",
    "skills/graph_query_skills.py",
]
optional_files = [
    "report.md",
]

root = Path.cwd()
missing = [path for path in required_files if not (root / path).is_file()]

if missing:
    for path in missing:
        print(f"Error: required file '{path}' not found.")
    sys.exit(1)

files_to_zip = list(required_files)
for path in optional_files:
    if (root / path).is_file():
        files_to_zip.append(path)
    else:
        print(f"Warning: {path} not found. Make sure to include it before final submission.")

zip_path = root / zip_name

try:
    if zip_path.exists():
        print(f"Overwriting existing {zip_name}")
        zip_path.unlink()

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files_to_zip:
            archive.write(root / path, path)

    with zipfile.ZipFile(zip_path) as archive:
        infos = archive.infolist()

    print(f"Successfully created {zip_name}")
    print("Contents:")
    print(f"Archive:  {zip_name}")
    print("  Length      Date    Time    Name")
    print("---------  ---------- -----   ----")

    total_size = 0
    for info in infos:
        year, month, day, hour, minute, _ = info.date_time
        total_size += info.file_size
        print(f"{info.file_size:9d}  {month:02d}-{day:02d}-{year:04d} {hour:02d}:{minute:02d}   {info.filename}")

    print("---------                     -------")
    print(f"{total_size:9d}                     {len(infos)} files")
except OSError as exc:
    print(f"Error: failed to create {zip_name}: {exc}")
    sys.exit(1)
except zipfile.BadZipFile as exc:
    print(f"Error: failed to verify {zip_name}: {exc}")
    sys.exit(1)
PY
