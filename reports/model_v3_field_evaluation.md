# Évaluation terrain du classifieur V3

## Protocole

- Poids : `weights/classifier-best.pt` (YOLOv8s-cls V3).
- Pipeline évalué : détection YOLOv8n puis classification du plus grand véhicule.
- Seuil de classification de l'application : 0,40.
- 80 photos Wikimedia sous licence compatible.
- 48 photos positives : 8 par classe cible.
- 32 négatifs : Pontiac GTO, Fiat Multipla, Volkswagen Golf VII et Toyota
  Corolla E100.
- Aucun chevauchement de titre, page source ou SHA-256 avec le manifeste de
  revue utilisé pour l'entraînement.

Ce pilote utilise les catégories Wikimedia comme vérité terrain. Il mesure le
comportement hors échantillon, mais ne remplace pas un audit humain plus large
des images et des annotations.

## Résultats au seuil 0,40

| Mesure | Résultat |
|---|---:|
| Exactitude end-to-end | 48,75 % |
| Exactitude sur les classes cibles | 56,25 % |
| Faux positifs sur `other_car` | 56,25 % |
| Absences de détection | 8 / 80 |

| Classe attendue | Correct |
|---|---:|
| Chevrolet Camaro | 5 / 8 |
| Chevrolet Corvette | 8 / 8 |
| Dodge Charger | 4 / 8 |
| Mustang Convertible | 3 / 8 |
| Mustang Fastback | 3 / 8 |
| Mustang Hardtop | 4 / 8 |
| Other car | 12 / 32 |

## Faux positifs négatifs

| Catégorie | Faux positifs |
|---|---:|
| Pontiac GTO 1967 | 2 / 8 (25,0 %) |
| Fiat Multipla I | 4 / 8 (50,0 %) |
| Toyota Corolla E100 | 5 / 8 (62,5 %) |
| Volkswagen Golf VII | 7 / 8 (87,5 %) |

Les erreurs dominantes sont `other_car` vers Mustang Fastback, Corvette et
Camaro. Plusieurs erreurs ont une confiance supérieure à 0,80 ; augmenter le
seuil ne suffit donc pas à corriger le manque de négatifs variés.

## Balayage du seuil

| Seuil | Exactitude globale | Exactitude cibles | Faux positifs |
|---:|---:|---:|---:|
| 0,40 | 48,75 % | 56,25 % | 56,25 % |
| 0,50 | 52,50 % | 56,25 % | 46,88 % |
| 0,60 | 57,50 % | 54,17 % | 31,25 % |
| 0,70 | 58,75 % | 54,17 % | 28,12 % |
| 0,80 | 60,00 % | 54,17 % | 25,00 % |
| 0,90 | 63,75 % | 50,00 % | 9,38 % |
| 0,95 | 58,75 % | 41,67 % | 9,38 % |

## Décision

Le poids V3 constitue un baseline utile, mais ne doit pas être présenté comme
robuste en monde ouvert. La prochaine itération doit enrichir `other_car` avec
des voitures ordinaires et modernes, puis réentraîner et rejouer exactement ce
jeu d'évaluation sans l'ajouter au train.
