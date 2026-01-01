import os
import ssl
import urllib3
import pandas as pd
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)



ca_cert_path = '/Users/moustaphakebe1998gmail.com/airflow-project/key/http_ca.crt'

ssl_context = ssl.create_default_context(cafile=ca_cert_path)
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE



es = Elasticsearch(
    ['https://localhost:9200'],
    basic_auth=('elastic', 'Z25ft0VLU2fpiqOXRSEc'),
    ssl_context=ssl_context
)

# Vérification de la connexion Elasticsearch
if es.ping():
    print("Connexion réussie à Elasticsearch.")
else:
    print("Impossible de se connecter à Elasticsearch. Vérifiez l'URL et les paramètres SSL.")



from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator
from elasticsearch import Elasticsearch

# --------------------------
# Fonction de test ES
# --------------------------
def test_es_connection():
    ES_URL = "https://elastic:Z25ft0VLU2fpiqOXRSEc@host.docker.internal:9200"
    
    es = Elasticsearch(
        ES_URL,
        verify_certs=False,   # ignore le certificat SSL
        ssl_show_warn=False   # supprime les warnings SSL
    )

    if es.ping():
        print("Connexion réussie à Elasticsearch !")
    else:
        raise Exception("Impossible de se connecter à Elasticsearch.")

# --------------------------
# Définition du DAG
# --------------------------
with DAG(
    dag_id="test_elasticsearch_connection",
    start_date=datetime(2025, 11, 16),
    schedule_interval=None,  # DAG manuel
    catchup=False,
    tags=["test", "elasticsearch"]
) as dag:

    test_connection_task = PythonOperator(
        task_id="test_es_connection",
        python_callable=test_es_connection
    )

    test_connection_task

