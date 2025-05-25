
# 📦 Installer toutes les dépendances à partir du fichier requirements.txt
install:
	pip install -r requirements.txt

# 🔁 Entraîner le modèle et sauvegarder model.joblib
train:
	python train_model.py

# 🚀 Démarrer l'application FastAPI avec rechargement automatique
run:
	uvicorn app:app --reload --host 0.0.0.0 --port 8000

# 🧪 Lancer l'API + Ouvrir Swagger manuellement
test:
	@echo "✅ API disponible sur : http://127.0.0.1:8000/docs"
	@echo "Utilisez make run pour démarrer le serveur."

# 🧩 Pipeline complet : Installer + Entraîner + Démarrer
full:
	make install
	make train
	make run

docker-build:
       docker build -t kahlanesrine_ds2_mlop .

docker-run:
       docker run -d -p 8000:8000  kahlanesrine_ds2_mlop

docker-push:
       docker tag  kahlanesrine_ds2_mlop   kahlanesrine/kahlanesrine_ds2_mlop
       docker push  kahlanesrine/kahlanesrine_ds2_mlop
