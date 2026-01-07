from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

DBT_PROJECT_DIR = "/opt/airflow/dags/r57_..."

default_args = {
    'owner': 'data_team',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='test_arrondissement_service_seuil',
    default_args=default_args,
    description='Test du modèle arrondissement_service_seuil',
    schedule=None,
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['dbt', 'test', 'r57_...'],
    params={
        "date_maj": "2025-10-22",
        "filter_query": "s3a://datalake-...../service_seuil_20251022_070104.csv"
    }
) as dag:
    # Vérification de la configuration
    dbt_debug = BashOperator(
        task_id='dbt_debug',
        bash_command=f"""
        cd {DBT_PROJECT_DIR} && \
        dbt debug --profiles-dir . --target dev
        """,
    )

    # Lister les modèles disponibles
    dbt_list = BashOperator(
        task_id='dbt_list_models',
        bash_command=f"""
        cd {DBT_PROJECT_DIR} && \
        dbt list --profiles-dir . --target dev
        """,
    )

    # Exécuter le modèle
    dbt_run_model = BashOperator(
        task_id='dbt_run_arrondissement_service_seuil',
        bash_command=f"""
        cd {DBT_PROJECT_DIR} && \
        dbt run \
        --select arrondissement_service_seuil \
        --profiles-dir . \
        --target dev \
        --vars '{{"date_maj": "{{{{ params.date_maj }}}}", "filter_query": "{{{{ params.filter_query }}}}"}}' \
        --full-refresh
        """,
    )

    # Tester le modèle
    dbt_test_model = BashOperator(
        task_id='dbt_test_arrondissement_service_seuil',
        bash_command=f"""
        cd {DBT_PROJECT_DIR} && \
        dbt test \
        --select arrondissement_service_seuil \
        --profiles-dir . \
        --target dev
        """,
    )

    dbt_debug >> dbt_list >> dbt_run_model >> dbt_test_model