# app.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Crée l'app FastAPI
app = FastAPI(title="API Analyse Risque Projet", version="1.0")

# Modèle Pydantic pour valider l'entrée
class Project(BaseModel):
    Budget: float
    Montant_collecte: float

# Charger le modèle ML
model = joblib.load("model.pkl")

# Endpoint de test simple
@app.get("/")
def home():
    return {"message": "API OK"}

# Endpoint prédiction
@app.post("/predict")
def predict(data: Project):
    try:
        # Préparer les données pour le modèle avec les colonnes exactes
        X = np.array([[data.Budget, data.Montant_collecte]])

        # Faire la prédiction
        prediction = model.predict(X)

        # Optionnel : calculer les probabilités
        proba = model.predict_proba(X)

        # Retourner un JSON avec la classe de risque et les probabilités
        return {
            "risk_class": int(prediction[0]),          # 0 = faible, 1 = élevé
            "risk_probability": proba[0].tolist()      # probabilités [faible, élevé]
        }

    except Exception as e:
        # Retourne l'erreur pour debug si quelque chose plante
        return {"error": str(e)}