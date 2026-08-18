# Car Spotter AI

Application de reconnaissance de voitures américaines classiques basée sur
YOLOv8 et Streamlit. Un détecteur localise chaque voiture, puis un classifieur
spécialisé identifie son modèle, sa période et, lorsque pertinent, sa carrosserie.

## Architecture

```text
.
├── app.py                              # Interface Streamlit
├── dataset_review_app.py               # Validation humaine des images
├── dataset_config.py                   # Chargement typé taxonomie/profils
├── model.py                            # API publique d'inférence
├── photo_model.py                      # Détection + classification fine
├── review_store.py                     # Persistance atomique des décisions
├── train_classifier.py                 # Entraînement YOLOv8-cls
├── config/
│   ├── taxonomy.json                   # Classes et sources Wikimedia
│   └── profiles/mustang_mvp.json       # Sous-ensemble du premier modèle
├── scripts/
│   ├── crop_dataset.py                 # Recadrage automatique par YOLO
│   ├── download_wikimedia.py           # Collecte avec licences
│   ├── prepare_classification_dataset.py
│   └── validate_classification_dataset.py
├── requirements.txt
├── Dockerfile
└── weights/
    └── classifier-best.pt              # Classifieur entraîné (à fournir)
```

La taxonomie couvre 1960 à 1974. Pour la Mustang, les carrosseries hardtop,
fastback/SportsRoof et convertible sont séparées lorsqu'il existe assez de
données. Les autres modèles sont regroupés par génération visuelle.

## MVP Mustang

Le premier poids est volontairement limité à cinq classes : Mustang hardtop et
fastback pour 1964–1966 et 1967–1968, plus `other_car`. Ce périmètre permet de
valider tout le cycle de données et la distinction de carrosserie avant
d'ajouter les autres générations, puis Charger, Challenger, Camaro, Corvette et
Impala. Le profil est défini dans `config/profiles/mustang_mvp.json` sans réduire
la taxonomie cible complète.

## Construction du dataset

Téléchargez d'abord un petit échantillon pour auditer la qualité des catégories :

```bash
python -m scripts.download_wikimedia \
  --profile config/profiles/mustang_mvp.json \
  --limit-per-category 30
```

Les images arrivent dans `datasets/raw/images` et chaque attribution est
enregistrée dans `datasets/raw/manifest.jsonl`. La commande reprend la collecte
sans retélécharger les fichiers déjà présents. Pour une collecte ciblée,
`--class-slug` reste disponible à la place de `--profile`.

Recadrez ensuite la voiture principale avec le détecteur COCO générique :

```bash
python -m scripts.crop_dataset \
  --profile config/profiles/mustang_mvp.json
```

Une photo sans voiture détectée est marquée `no_detection`. Une photo avec deux
voitures de taille comparable est marquée `ambiguous`. Ces cas ne sont pas
envoyés à la revue. Les autres recadrages et leurs coordonnées sont consignés
dans `datasets/cropped/manifest.jsonl`.

Lancez l'interface de contrôle :

```bash
streamlit run dataset_review_app.py
```

Chaque recadrage doit être accepté, corrigé vers une autre classe du profil, ou
rejeté. Les décisions sont sauvegardées au fil de l'eau dans
`datasets/review/decisions.json`. Le dataset final exige par défaut une décision
humaine positive.

La classe `other_car` est alimentée avec des véhicules proches mais hors cible :
Pontiac GTO, Firebird, Chevelle, Barracuda, ainsi que des versions modernes de
nos six modèles. Ces négatifs difficiles réduisent les faux positifs. On pourra
ensuite compléter cette classe avec le dataset Roboflow audité ; toute image
ajoutée devra conserver son origine et sa licence dans le manifeste.

Préparez et validez les splits :

```bash
python -m scripts.prepare_classification_dataset \
  --manifest datasets/cropped/manifest.jsonl
python -m scripts.validate_classification_dataset --minimum-per-split 5
```

Le découpage est déterministe et groupé par auteur/source afin de réduire les
fuites entre entraînement et évaluation. Pour un véritable modèle, visez bien
plus que le minimum technique de cinq images par classe et par split.
`--allow-unreviewed` existe uniquement pour les essais techniques et ne doit pas
servir à produire le poids publié.

## Entraînement sur Apple Silicon

Sur le Mac M1 Pro, entraînez d'abord `yolov8s-cls.pt` avec MPS :

```bash
python train_classifier.py \
  --model yolov8s-cls.pt \
  --device mps \
  --epochs 100 \
  --image-size 320 \
  --batch-size 8
```

Le meilleur checkpoint est créé dans :

```text
runs/classify/classic-car-classifier-v1/weights/best.pt
```

Copiez-le ensuite vers `weights/classifier-best.pt`. Ce fichier n'est pas
versionné dans Git.

## Exécution locale

Python 3.10 ou supérieur est requis.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Placez le classifieur entraîné dans `weights/classifier-best.pt`, ou configurez
un autre emplacement :

```bash
export CAR_SPOTTER_CLASSIFIER_PATH=/chemin/vers/best.pt
export CAR_SPOTTER_DEVICE=mps
streamlit run app.py
```

Le détecteur générique utilise `yolov8n.pt` par défaut. Il peut être remplacé via
`CAR_SPOTTER_DETECTOR_PATH`. Les autres réglages disponibles sont
`CAR_SPOTTER_DETECTION_CONFIDENCE`, `CAR_SPOTTER_CLASSIFICATION_CONFIDENCE`,
`CAR_SPOTTER_IOU`, `CAR_SPOTTER_DEVICE` et `CAR_SPOTTER_TAXONOMY_PATH`.

## Docker

Le poids n'est pas versionné dans Git. Ajoutez-le dans
`weights/classifier-best.pt` avant le build, puis lancez :

```bash
docker build -t car-spotter-ai .
docker run --rm -p 8501:8501 car-spotter-ai
```

L'application est ensuite disponible sur <http://localhost:8501>.
