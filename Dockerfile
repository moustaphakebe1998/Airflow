FROM apache/airflow:3.1.5

USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

RUN pip install --no-cache-dir --upgrade pip

RUN pip install --no-cache-dir \
    "fastapi>=0.115.0" \
    "pydantic>=2.7.0" \
    pandas \
    scikit-learn \
    elasticsearch \
    apache-airflow-providers-amazon \
    apache-airflow-providers-elasticsearch  \
    apache-airflow-providers-trino

WORKDIR /opt/airflow