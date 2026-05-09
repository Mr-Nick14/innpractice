#!/bin/sh
set -eu

mkdir -p artifacts/test-reports

python3 -m pytest tests -v --junitxml=artifacts/test-reports/test-results.xml
