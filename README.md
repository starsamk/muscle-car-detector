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
│   ├── collect_open_images_negatives.py
│   ├── download_wikimedia.py           # Collecte avec licences
│   ├── merge_review_dataset.py          # Fusion traçable des files de revue
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

### Transition vers les carrosseries Mustang

La v2 prépare un modèle plus robuste : il ne prédit plus une année, mais une
carrosserie Mustang classique (`Fastback`, `Hardtop` ou `Convertible`). La v1 et
son poids actuel restent inchangés pendant la transition. La nouvelle taxonomie
est dans `config/taxonomy_mustang_body_style_v2.json` et son profil dans
`config/profiles/mustang_body_style_v2.json`.

Les images et décisions déjà revues peuvent être migrées sans les recopier ni
les réviser. La commande écrit uniquement de nouveaux manifestes dérivés :

```bash
python -m scripts.migrate_taxonomy
```

Les sorties par défaut sont `datasets/mustang_body_style_v2/`. Pour ouvrir la
revue de cette future version, configurez les chemins suivants avant Streamlit :

```bash
export CAR_SPOTTER_REVIEW_MANIFEST=datasets/mustang_body_style_v2/cropped/manifest.jsonl
export CAR_SPOTTER_REVIEW_DECISIONS=datasets/mustang_body_style_v2/review/decisions.json
export CAR_SPOTTER_REVIEW_TAXONOMY_PATH=config/taxonomy_mustang_body_style_v2.json
export CAR_SPOTTER_REVIEW_PROFILE_PATH=config/profiles/mustang_body_style_v2.json
streamlit run dataset_review_app.py
```

### Taxonomie véhicule V3

La V3 définit le premier périmètre multi-modèles : Mustang Fastback, Mustang
Hardtop, Mustang Convertible, Camaro, Corvette, Charger et `other_car`. Les
années servent à documenter les sources, mais ne sont pas affichées comme une
prédiction indépendante. Les fichiers sont `config/taxonomy_vehicle_v3.json`
et `config/profiles/vehicle_taxonomy_v3.json`.

Les images déjà validées V2 ont été migrées vers cette taxonomie sans recopier
les sources. Elles alimentent actuellement 222 Fastback, 287 Hardtop et 135
`other_car`. Les classes Convertible, Camaro, Corvette et Charger restent
volontairement vides jusqu'à leur collecte et validation dédiées. `other_car`
conserve ses négatifs validés (Firebird, Chevelle, GTO, Barracuda, Thunderbird,
Monte Carlo et Road Runner) ; ils ne sont pas artificiellement transformés en
une des nouvelles classes.

Pour régénérer les métadonnées dérivées localement :

```bash
python -m scripts.migrate_taxonomy \
  --manifest datasets/mustang_body_style_v2/cropped/manifest.jsonl \
  --decisions datasets/mustang_body_style_v2/review/decisions.json \
  --target-taxonomy config/taxonomy_vehicle_v3.json \
  --mapping config/mappings/body_style_v2_to_vehicle_v3.json \
  --output-manifest datasets/vehicle_taxonomy_v3/cropped/manifest.jsonl \
  --output-decisions datasets/vehicle_taxonomy_v3/review/decisions.json
```

Le profil `config/profiles/vehicle_taxonomy_v3_bootstrap.json` est réservé à
la vérification technique des trois classes actuellement disponibles. Il ne
doit pas servir à produire le poids final multi-modèles : ce poids attendra la
collecte des quatre classes manquantes.

```bash
python -m scripts.prepare_classification_dataset \
  --manifest datasets/vehicle_taxonomy_v3/cropped/manifest.jsonl \
  --decisions datasets/vehicle_taxonomy_v3/review/decisions.json \
  --taxonomy config/taxonomy_vehicle_v3.json \
  --profile config/profiles/vehicle_taxonomy_v3_bootstrap.json \
  --output datasets/classification_vehicle_v3_bootstrap

python -m scripts.validate_classification_dataset \
  --data datasets/classification_vehicle_v3_bootstrap \
  --taxonomy config/taxonomy_vehicle_v3.json \
  --profile config/profiles/vehicle_taxonomy_v3_bootstrap.json \
  --minimum-per-split 5
```

### Collecte V3

La collecte des quatre classes manquantes utilise un profil dédié afin de ne
pas retélécharger les Fastback et Hardtop déjà validés :

```bash
python -m scripts.download_wikimedia \
  --taxonomy config/taxonomy_vehicle_v3.json \
  --profile config/profiles/vehicle_taxonomy_v3_missing_positive_collection.json \
  --output datasets/vehicle_taxonomy_v3/raw \
  --limit-per-category 120 \
  --target-per-class 150
```

Les négatifs difficiles Wikimedia sont collectés séparément dans la même
taxonomie avec `config/profiles/vehicle_taxonomy_v3_hard_negative_collection.json`.
`--target-per-class` permet de reprendre une collecte sans dépasser un objectif
de volume déjà présent dans le manifeste.
Ils restent des candidats jusqu'à leur validation humaine.

Pour ajouter des négatifs Open Images, téléchargez seulement les métadonnées de
validation nécessaires :

```bash
mkdir -p datasets/metadata/open_images_v5
curl -L -o datasets/metadata/open_images_v5/validation-annotations-bbox.csv \
  https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv
curl -L -o datasets/metadata/open_images_v5/validation-images-with-rotation.csv \
  https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv
curl -L -o datasets/metadata/open_images_v5/class-descriptions-boxable.csv \
  https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv

python -m scripts.collect_open_images_negatives \
  --annotations datasets/metadata/open_images_v5/validation-annotations-bbox.csv \
  --metadata datasets/metadata/open_images_v5/validation-images-with-rotation.csv \
  --class-descriptions datasets/metadata/open_images_v5/class-descriptions-boxable.csv \
  --output datasets/vehicle_taxonomy_v3/open_images_negatives \
  --limit-per-class 200 \
  --max-images 1000
```

Le filtre exclut les boîtes intérieures et les groupes d'objets. Les candidats
Open Images gardent leur URL, leur auteur et leur licence ; ils doivent tout de
même passer par la revue Car Spotter avant d'intégrer `other_car`.

Après recadrage, les trois sources peuvent être réunies dans une file de revue
unique. Le premier manifeste reste prioritaire et les doublons SHA-256 sont
ignorés :

```bash
python -m scripts.merge_review_dataset \
  --manifest datasets/vehicle_taxonomy_v3/cropped/manifest.jsonl \
  --manifest datasets/vehicle_taxonomy_v3/new_collection/cropped/manifest.jsonl \
  --manifest datasets/vehicle_taxonomy_v3/open_images_negatives/cropped/manifest.jsonl \
  --decisions datasets/vehicle_taxonomy_v3/review/decisions.json \
  --output-manifest datasets/vehicle_taxonomy_v3/review_queue/cropped/manifest.jsonl \
  --output-decisions datasets/vehicle_taxonomy_v3/review_queue/decisions.json
```

Pour ouvrir cette file dans Streamlit :

```bash
export CAR_SPOTTER_REVIEW_MANIFEST=datasets/vehicle_taxonomy_v3/review_queue/cropped/manifest.jsonl
export CAR_SPOTTER_REVIEW_DECISIONS=datasets/vehicle_taxonomy_v3/review_queue/decisions.json
export CAR_SPOTTER_REVIEW_TAXONOMY_PATH=config/taxonomy_vehicle_v3.json
export CAR_SPOTTER_REVIEW_PROFILE_PATH=config/profiles/vehicle_taxonomy_v3.json
streamlit run dataset_review_app.py --server.port 8503
```

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

Le découpage est déterministe, groupé par auteur/source et stratifié par classe
afin de réduire les fuites entre entraînement et évaluation tout en gardant les
classes représentées dans les trois splits. Pour un véritable modèle, visez
bien plus que le minimum technique de cinq images par classe et par split.
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
