#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: ./submit.sh <student_id>"
    echo "Example: ./submit.sh 523030910000"
    exit 1
fi

STUDENT_ID="$1"
ZIP_NAME="lab0_${STUDENT_ID}.zip"

# Files students must submit
REQUIRED_FILES=(
    agent/loop.py
    skills/graph_skill/SKILL.md
    skills/range_query_skill/SKILL.md
    report.md
)

# Check required files exist
for f in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$f" ]; then
        echo "Error: required file '$f' not found."
        exit 1
    fi
done

# Check report exists
if [ ! -f "report.md" ]; then
    echo "Warning: report.md not found. Make sure to include it before final submission."
fi

# Package the project, excluding non-essential files
zip -r "$ZIP_NAME" \
    agent/ \
    skills/ \
    examples/ \
    tests/ \
    docs/ \
    main.py \
    pyproject.toml \
    pytest.ini \
    README.md \
    report.md \
    -x "*/__pycache__/*" \
    2>/dev/null || true

# Verify the zip was created
if [ -f "$ZIP_NAME" ]; then
    echo "Successfully created $ZIP_NAME"
    echo "Contents:"
    unzip -l "$ZIP_NAME"
else
    echo "Error: failed to create zip file."
    exit 1
fi
