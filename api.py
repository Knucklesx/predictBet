from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
from extract_features import extract_features
from load_dataset import load_dataset
from sklearn.preprocessing import LabelEncoder

app = FastAPI()

# CORS para permitir chamadas do frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ou especifique: ["http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Carrega modelo, recursos esperados, imputer e encoder
model, expected_features = joblib.load("xgb_model.pkl")
imputer = joblib.load("imputer.pkl")
le_result = joblib.load("label_encoder_result.pkl")
df = load_dataset("BRA.csv")

le_team = LabelEncoder()
le_team.fit(df["Home"].tolist() + df["Away"].tolist())

class MatchRequest(BaseModel):
    home_team: str
    away_team: str

@app.post("/api/predict")
def predict_result(request: MatchRequest):
    try:
        match_date = pd.Timestamp.now()
        print(f"🔍 Prevendo: {request.home_team} x {request.away_team}")

        features = extract_features(df, request.home_team, request.away_team, match_date, le_team)
        print("✅ Features extraídas com sucesso")

        features = features[expected_features]
        features_imputed = imputer.transform(features)

        pred = model.predict(features_imputed)[0]
        result = le_result.inverse_transform([pred])[0]
        return {"result": result}

    except Exception as e:
        print("❌ Erro:", e)
        raise HTTPException(status_code=500, detail=f"Erro na previsão: {str(e)}")
