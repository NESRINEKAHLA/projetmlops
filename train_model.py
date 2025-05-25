from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from joblib import dump

# Données d'exemple
symptoms = [
    "fever cough fatigue",
    "headache nausea dizziness",
    "vomiting diarrhea",
    "fever sore_throat"
]
labels = ["flu", "migraine", "infection", "flu"]

# Création du pipeline
model = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

# Entraînement
model.fit(symptoms, labels)

# Sauvegarde dans model.joblib
dump(model, "model.joblib")
print("✅ Modèle entraîné et sauvegardé dans model.joblib")
