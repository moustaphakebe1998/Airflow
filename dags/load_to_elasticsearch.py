from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from elasticsearch import Elasticsearch, helpers
import pandas as pd
import os

# --------------------------
# Variables
# --------------------------
CSV_FILE_PATH = "/opt/airflow/data/language_detection.csv"  # chemin vers ton fichier CSV
ES_URL = "https://elastic:Z25ft0VLU2fpiqOXRSEc@host.docker.internal:9200"
ES_INDEX = "language"

# --------------------------
# Fonction de test ES
# --------------------------
def test_es_connection():
    es = Elasticsearch(
        ES_URL,
        verify_certs=False,
        ssl_show_warn=False
    )
    if es.ping():
        print("Connexion réussie à Elasticsearch !")
    else:
        raise Exception("Impossible de se connecter à Elasticsearch.")

# --------------------------
# Fonction d'import CSV -> ES
# --------------------------
def import_csv_to_es():
    es = Elasticsearch(
        ES_URL,
        verify_certs=False,
        ssl_show_warn=False
    )

    if not es.ping():
        raise Exception("Impossible de se connecter à Elasticsearch.")

    if not os.path.exists(CSV_FILE_PATH):
        raise FileNotFoundError(f"Fichier CSV non trouvé : {CSV_FILE_PATH}")

    # Lire le CSV
    df = pd.read_csv(CSV_FILE_PATH)

    # Préparer les documents pour Elasticsearch
    actions = [
        {
            "_index": ES_INDEX,
            "_source": {
                "text": row["Text"],
                "language": row["Language"]
            }
        }
        for _, row in df.iterrows()
    ]

    if actions:
        helpers.bulk(es, actions)
        print(f"{len(actions)} documents insérés dans l'index '{ES_INDEX}' !")
    else:
        print("Aucun document à insérer.")

# --------------------------
# DAG Airflow
# --------------------------
with DAG(
    dag_id="csv_to_elasticsearch_language",
    start_date=datetime(2025, 11, 16),
    schedule_interval=None,
    catchup=False,
    tags=["elasticsearch", "csv", "language_detection"]
) as dag:

    test_connection_task = PythonOperator(
        task_id="test_es_connection",
        python_callable=test_es_connection
    )

    import_csv_task = PythonOperator(
        task_id="import_csv_to_es",
        python_callable=import_csv_to_es
    )

    # test_connection_task >> import_csv_task
