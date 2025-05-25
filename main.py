import pandas as pd
from fastapi import FastAPI
import argparse
from model_pipeline import prepare_data, train_model, evaluate_model, optimize_model, save_model, load_model, predict_symptoms
from sklearn.model_selection import train_test_split

def main(args):
    if args.step == "train":
        X, y, vectorizer, label_encoder, df = prepare_data(args.file)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = train_model(X_train, y_train)
        evaluate_model(model, X_test, y_test)
        save_model(model, vectorizer, label_encoder, args.model)
    elif args.step == "optimize":
        X, y, vectorizer, label_encoder, df = prepare_data(args.file)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = optimize_model(X_train, y_train)
        evaluate_model(model, X_test, y_test)
        save_model(model, vectorizer, label_encoder, args.model)
    elif args.step == "predict":
        model, vectorizer, label_encoder = load_model(args.model)
        df = pd.read_csv(args.file)
        user_input = input("Entrez vos symptômes séparés par des virgules : ")
        predict_symptoms(model, vectorizer, label_encoder, df, user_input)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--step', type=str, required=True, help="train | optimize | predict")
    parser.add_argument('--file', type=str, default='final_dataset.csv', help="Fichier de données")
    parser.add_argument('--model', type=str, default='model.joblib', help="Fichier du modèle")
    args = parser.parse_args()
    main(args)

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "API du modèle ML opérationnelle"}

@app.post("/predict")
def predict(data: SomeInputType):
    # Charger le modèle et renvoyer une prédiction
    ...
