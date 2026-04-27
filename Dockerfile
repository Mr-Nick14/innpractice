FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

RUN mkdir -p /app/monte_carlo /app/tests /artifacts

COPY requirements_test.txt /tmp/requirements_test.txt

RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r /tmp/requirements_test.txt

COPY monte_carlo/__init__.py /app/monte_carlo/__init__.py
COPY monte_carlo/mc_cashflow_engine.py /app/monte_carlo/mc_cashflow_engine.py
COPY monte_carlo/mc_stochastic_drivers.py /app/monte_carlo/mc_stochastic_drivers.py
COPY tests /app/tests

CMD ["python", "-m", "pytest", "tests", "-v", "--junitxml=/artifacts/test-results.xml"]
