#!/bin/bash

echo "=========================================="
echo "  AIRFLOW 3.1.5 - Configuration Setup"
echo "=========================================="
echo ""

# Couleurs pour les messages
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# 1. Créer les dossiers nécessaires
echo -e "${YELLOW}[1/5]${NC} Création des dossiers..."
mkdir -p ./dags ./logs ./config ./plugins ./data ./output ./models
echo -e "${GREEN}✓${NC} Dossiers créés"
echo ""

# 2. Générer la clé Fernet
echo -e "${YELLOW}[2/5]${NC} Génération de la clé Fernet..."

# Méthode 1 : Essayer avec Docker (recommandé)
if command -v docker &> /dev/null; then
    echo "Utilisation de Docker pour générer la clé Fernet..."
    FERNET_KEY=$(docker run --rm python:3.11-slim bash -c "pip install -q cryptography && python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'")
    if [ ! -z "$FERNET_KEY" ]; then
        echo -e "${GREEN}✓${NC} Clé Fernet générée: ${FERNET_KEY}"
    else
        echo -e "${RED}✗${NC} Erreur lors de la génération avec Docker"
        # Fallback : générer manuellement avec OpenSSL
        echo "Utilisation d'OpenSSL comme alternative..."
        FERNET_KEY=$(dd if=/dev/urandom bs=32 count=1 2>/dev/null | base64)
        echo -e "${GREEN}✓${NC} Clé Fernet générée (OpenSSL): ${FERNET_KEY}"
    fi
elif command -v python3 &> /dev/null; then
    # Méthode 2 : Essayer avec Python local
    FERNET_KEY=$(python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())" 2>/dev/null)
    if [ -z "$FERNET_KEY" ]; then
        echo -e "${YELLOW}⚠${NC} Cryptography non installé, génération alternative..."
        # Fallback : générer manuellement avec OpenSSL
        FERNET_KEY=$(dd if=/dev/urandom bs=32 count=1 2>/dev/null | base64)
        echo -e "${GREEN}✓${NC} Clé Fernet générée (OpenSSL): ${FERNET_KEY}"
    else
        echo -e "${GREEN}✓${NC} Clé Fernet générée: ${FERNET_KEY}"
    fi
else
    echo -e "${RED}✗${NC} Ni Docker ni Python3 disponible. Installation impossible."
    exit 1
fi
echo ""

# 3. Obtenir l'UID de l'utilisateur
echo -e "${YELLOW}[3/5]${NC} Détection de l'UID utilisateur..."
if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "darwin"* ]]; then
    USER_UID=$(id -u)
    echo -e "${GREEN}✓${NC} UID détecté: ${USER_UID}"
else
    USER_UID=50000
    echo -e "${YELLOW}⚠${NC} OS non-Linux/Mac, utilisation de l'UID par défaut: ${USER_UID}"
fi
echo ""

# 4. Demander les identifiants personnalisés
echo -e "${YELLOW}[4/5]${NC} Configuration des identifiants..."
echo ""

read -p "Username Airflow (défaut: admin): " AIRFLOW_USER
AIRFLOW_USER=${AIRFLOW_USER:-admin}

read -sp "Password Airflow (défaut: admin123): " AIRFLOW_PASS
echo ""
AIRFLOW_PASS=${AIRFLOW_PASS:-admin123}

read -p "PostgreSQL Database (défaut: airflow_db): " POSTGRES_DB
POSTGRES_DB=${POSTGRES_DB:-airflow_db}

read -p "PostgreSQL User (défaut: airflow_user): " POSTGRES_USER
POSTGRES_USER=${POSTGRES_USER:-airflow_user}

read -sp "PostgreSQL Password (défaut: postgres123): " POSTGRES_PASS
echo ""
POSTGRES_PASS=${POSTGRES_PASS:-postgres123}

echo ""
echo -e "${GREEN}✓${NC} Identifiants configurés"
echo ""

# 5. Créer le fichier .env
echo -e "${YELLOW}[5/5]${NC} Création du fichier .env..."

cat > .env << EOF
# ==========================================
# AIRFLOW CONFIGURATION
# ==========================================

AIRFLOW_UID=${USER_UID}
AIRFLOW_IMAGE_NAME=apache/airflow:3.1.5
AIRFLOW_PROJ_DIR=.

# ==========================================
# IDENTIFIANTS AIRFLOW WEB UI
# ==========================================

_AIRFLOW_WWW_USER_USERNAME=${AIRFLOW_USER}
_AIRFLOW_WWW_USER_PASSWORD=${AIRFLOW_PASS}

# ==========================================
# POSTGRESQL CONFIGURATION
# ==========================================

POSTGRES_DB=${POSTGRES_DB}
POSTGRES_USER=${POSTGRES_USER}
POSTGRES_PASSWORD=${POSTGRES_PASS}

# ==========================================
# AIRFLOW CORE SETTINGS
# ==========================================

AIRFLOW__CORE__LOAD_EXAMPLES=false
AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION=true
AIRFLOW__CORE__FERNET_KEY=${FERNET_KEY}
AIRFLOW__CORE__DEFAULT_TIMEZONE=Europe/Paris

# ==========================================
# PACKAGES PYTHON ADDITIONNELS
# ==========================================

_PIP_ADDITIONAL_REQUIREMENTS=
EOF

echo -e "${GREEN}✓${NC} Fichier .env créé"
echo ""

# Résumé
echo "=========================================="
echo "  CONFIGURATION TERMINÉE"
echo "=========================================="
echo ""
echo -e "${GREEN}Identifiants Airflow:${NC}"
echo "  URL: http://localhost:8080"
echo "  Username: ${AIRFLOW_USER}"
echo "  Password: ${AIRFLOW_PASS}"
echo ""
echo -e "${GREEN}Identifiants PostgreSQL:${NC}"
echo "  Host: localhost"
echo "  Port: 5432"
echo "  Database: ${POSTGRES_DB}"
echo "  User: ${POSTGRES_USER}"
echo "  Password: ${POSTGRES_PASS}"
echo ""
echo -e "${YELLOW}Prochaines étapes:${NC}"
echo "  1. docker-compose build"
echo "  2. docker-compose up -d"
echo "  3. Attendez 2-3 minutes pour l'initialisation"
echo "  4. Accédez à http://localhost:8080"
echo ""
echo "=========================================="