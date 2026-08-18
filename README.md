# Car Spotter AI

Application de reconnaissance de voitures américaines classiques basée sur
YOLOv8 et Streamlit. Un détecteur localise chaque voiture, puis un classifieur
spécialisé identifie son modèle, sa période et, lorsque pertinent, sa carrosserie.

## Architecture

```text
.
├── app.py                              # Interface Streamlit
├── model.py                            # API publique d'inférence
├── photo_model.py                      # Détection + classification fine
├── train_classifier.py                 # Entraînement YOLOv8-cls
├── config/
│   └── taxonomy.json                   # Classes et sources Wikimedia
├── scripts/
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

## Construction du dataset

Téléchargez d'abord un petit échantillon pour auditer la qualité des catégories :

```bash
python scripts/download_wikimedia.py \
  --class-slug ford_mustang_fastback_1967_1968 \
  --limit-per-category 30
```

Lorsque les catégories ont été vérifiées, retirez `--class-slug` pour collecter
toutes les classes. La limite par défaut est de 80 images par catégorie afin de
maîtriser le volume. Les images arrivent dans `datasets/raw/images` et chaque
attribution est enregistrée dans `datasets/raw/manifest.jsonl`.

La collecte automatique doit toujours être revue manuellement : supprimez les
photos d'intérieurs, de moteurs, les images contenant plusieurs modèles ambigus
et les erreurs de catégorie avant de préparer les splits.

La classe `other_car` est alimentée avec des véhicules proches mais hors cible :
Pontiac GTO, Firebird, Chevelle, Barracuda, ainsi que des versions modernes de
nos six modèles. Ces négatifs difficiles réduisent les faux positifs. On pourra
ensuite compléter cette classe avec le dataset Roboflow audité ; toute image
ajoutée devra conserver son origine et sa licence dans le manifeste.

Préparez et validez les splits :

```bash
python scripts/prepare_classification_dataset.py
python scripts/validate_classification_dataset.py --minimum-per-split 5
```

Le découpage est déterministe et groupé par auteur/source afin de réduire les
fuites entre entraînement et évaluation. Pour un véritable modèle, visez bien
plus que le minimum technique de cinq images par classe et par split.

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
