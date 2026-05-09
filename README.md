# Monte Carlo Cashflow CI Project

This repository contains a Python cashflow calculation project with CI configured through Bamboo.

## Local test run

```bash
python3 -m pip install -r requirements_test.txt
sh scripts/run_tests.sh
```

The test command writes a JUnit report to `artifacts/test-reports/test-results.xml`.

## Docker test run

```bash
docker build -t monte-carlo-cashflow-ci .
docker run --rm -v "$PWD/artifacts:/app/artifacts" monte-carlo-cashflow-ci
```

## Bamboo CI

The repository stores Bamboo YAML Specs in `bamboo-specs/bamboo.yaml`.

The plan:

- runs on the `python:3.12-slim` Docker image;
- installs test dependencies from `requirements_test.txt`;
- runs `pytest` through `scripts/run_tests.sh`;
- publishes `artifacts/test-reports/*.xml` as JUnit results;
- creates branch plans for pull requests via `branches: create: for-pull-request`.

Before importing the Specs, create or reuse the Bamboo project with key `CASH`, or change
`plan.project-key` in `bamboo-specs/bamboo.yaml` to the key used in your Bamboo instance.
