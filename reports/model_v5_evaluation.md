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

V5 est promu comme modèle final du MVP personnel au seuil applicatif de 0,50.
Il améliore nettement le comportement terrain observé par rapport à V3 et
atteint 60,4 % de précision sur les images cibles. Le taux de faux positifs
`other_car` reste à 15,6 % et dépasse donc le seuil strict de 12,5 % défini
pour une mise en production : ce modèle est expérimental et ne constitue pas
une garantie de reconnaissance fiable dans tous les cas.

Le poids V3 est conservé localement comme sauvegarde dans
`weights/classifier-v3-best.pt`. Le backend et le dataset V5 sont désormais
gelés ; toute amélioration ultérieure devra commencer une nouvelle version et
une nouvelle campagne d'évaluation.
