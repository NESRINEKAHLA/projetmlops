from elasticsearch import Elasticsearch
from datetime import datetime

# Connexion à Elasticsearch
es = Elasticsearch("http://localhost:9200")

# Vérifier la connexion
if es.ping():
    print("✅ Connecté à Elasticsearch !")
else:
    print("❌ Échec de connexion.")
    exit()

# Préparer les données
log_doc = {
    "metric_name": "accuracy",
    "value": 0.93,
    "timestamp": datetime.utcnow().isoformat()
}

# Envoyer à l'index mlflow-metrics
response = es.index(index="mlflow-metrics", document=log_doc)
print("📤 Document envoyé avec ID :", response["_id"])

