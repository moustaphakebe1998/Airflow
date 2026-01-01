from fastapi import FastAPI, HTTPException, Header
from typing import List, Optional
import pickle

app = FastAPI(title="Language Detection API")

# Token d'administration
ADMIN_TOKEN = "adminkebe"

# Variables globales pour les modèles
pipeline = None
encoder = None


@app.on_event("startup")
def load_models():
    """Charger les modèles UNE SEULE FOIS au démarrage"""
    global pipeline, encoder
    
    print("🚀 Chargement des modèles...")
    
    try:
        pipeline = pickle.load(open('/app/models/trained_pipeline.pkl', 'rb'))
        encoder = pickle.load(open('/app/models/label_encoder.pkl', 'rb'))
        print(f"✅ Modèles chargés avec succès")
        print(f"📊 Langues supportées: {list(encoder.classes_)}")
    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")


@app.get("/")
def home():
    """Page d'accueil"""
    return {
        "service": "Language Detection API",
        "model_loaded": pipeline is not None,
        "languages": list(encoder.classes_) if encoder else []
    }


@app.post("/predict")
def predict(texts: List[str]):
    """Prédire la langue des textes"""
    if not pipeline or not encoder:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    
    predictions = pipeline.predict(texts)
    results = []
    
    for text, pred in zip(texts, predictions):
        results.append({
            "text": text,
            "language": encoder.classes_[pred]
        })
    
    return results


@app.post("/reload")
def reload(authorization: Optional[str] = Header(None)):
    """
    Recharger les modèles (après un nouvel entraînement)
    
    Requiert un token d'administration dans le header:
    Authorization: adminkebe
    """
    # Vérifier le token
    if authorization != ADMIN_TOKEN:
        raise HTTPException(
            status_code=401,
            detail="Token d'administration invalide ou manquant"
        )
    
    global pipeline, encoder
    
    try:
        pipeline = pickle.load(open('/app/models/trained_pipeline.pkl', 'rb'))
        encoder = pickle.load(open('/app/models/label_encoder.pkl', 'rb'))
        print("🔄 Modèles rechargés par l'administrateur")
        return {
            "status": "success",
            "message": "Modèles rechargés avec succès",
            "languages": list(encoder.classes_)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))