from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from joblib import load
import os

# Charger le modèle
MODEL_PATH = "model.joblib"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Le modèle {MODEL_PATH} est introuvable. Veuillez l'entraîner et le sauvegarder d'abord.")

model = load(MODEL_PATH)

# Initialiser l'application FastAPI
app = FastAPI()

# Schéma de la requête entrante
class SymptomInput(BaseModel):
    symptoms: str

# Route POST pour la prédiction
@app.post("/predict")
def predict(input_data: SymptomInput):
    try:
        prediction = model.predict([input_data.symptoms])
        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur de prédiction : {str(e)}")

