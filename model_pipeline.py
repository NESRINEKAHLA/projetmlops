import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

def prepare_data(filepath, sample_size=600):
    df = pd.read_csv(filepath)
    df = df.sample(n=sample_size, random_state=42)
    df.fillna("Unknown", inplace=True)
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df['Symptoms'])
    y = df['Disease']
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    return X, y_encoded, vectorizer, label_encoder, df

def train_model(X_train, y_train):
    model = MultinomialNB()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    print("\nRapport de classification :")
    print(classification_report(y_test, y_pred))
    print("\nMatrice de confusion :")
    conf_matrix = confusion_matrix(y_test, y_pred)
    print(conf_matrix)
    print("\nPrécision globale :", accuracy_score(y_test, y_pred))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title('Matrice de confusion')
    plt.xlabel('Prédictions')
    plt.ylabel('Réalité')
    plt.show()
    return y_pred

def optimize_model(X_train, y_train):
    param_grid = {'alpha': [0.1, 0.5, 1.0, 1.5]}
    grid = GridSearchCV(MultinomialNB(), param_grid, cv=5)
    grid.fit(X_train, y_train)
    print("Meilleurs paramètres :", grid.best_params_)
    return grid.best_estimator_

def save_model(model, vectorizer, label_encoder, path='model.joblib'):
    joblib.dump((model, vectorizer, label_encoder), path)

def load_model(path='model.joblib'):
    return joblib.load(path)

def predict_symptoms(model, vectorizer, label_encoder, df, user_input):
    user_vect = vectorizer.transform([user_input])
    pred = model.predict(user_vect)
    disease_name = label_encoder.inverse_transform(pred)[0]
    print(f"\n🧠 Maladie prédite : {disease_name}")
    precautions = df[df['Disease'] == disease_name][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].iloc[0]
    print("🩺 Précautions recommandées :")
    for p in precautions:
        print(f"- {p}")
