# Évaluation du candidat V5

## Configuration

- Dataset : `classification_vehicle_v5` (1 332 images, 7 classes).
- Initialisation : `classic-car-classifier-v4/weights/best.pt`.
- Fine-tuning : AdamW, `lr0=0.0003`, couches `0` à `8` gelées, patience 10.
- Checkpoint : `runs/classify/classic-car-classifier-v5-final/weights/best.pt`.

## Résultats

Ultralytics mesure 79,9 % de top-1 sur les 202 images du split de test V5.
L'inférence directe du classifieur sur ce split donne les métriques par classe
suivantes :

| Classe | Résultat |
| --- | --- |
| Camaro | 19/19 (100,0 %) |
| Corvette | 21/21 (100,0 %) |
| Charger | 14/14 (100,0 %) |
| Mustang Convertible | 15/19 (78,9 %) |
| Mustang Fastback | 26/39 (66,7 %) |
| Mustang Hardtop | 35/46 (76,1 %) |
| Other car | 39/44 (88,6 %) |

Sur les 80 photos du lot terrain indépendant :

| Seuil de classification | Précision cible | Faux positifs `other_car` |
| --- | --- | --- |
| 0,40 | 60,4 % | 18,8 % |
| 0,50 | 60,4 % | 15,6 % |
| 0,70 | 52,1 % | 6,2 % |

## Décision

Le candidat V5 **n'est pas promu**. Il échoue aux seuils définis avant
l'entraînement : Fastback, Hardtop et `other_car` sont sous leurs objectifs
internes, et le taux de faux positifs au seuil applicatif 0,50 reste supérieur
à 12,5 %. Le poids actif V3 demeure inchangé.
