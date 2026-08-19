# Sources de donnees et politique de collecte

Ce projet ne versionne ni les images ni les poids entraines. Chaque image
retenue garde toutefois dans son manifeste son origine, son auteur, sa licence
et, quand elle existe, sa boite source.

## Sources retenues

| Source | Usage | Statut |
| --- | --- | --- |
| Wikimedia Commons | Modeles cibles et negatifs difficiles nommes | Retenue : licence de chaque image verifiee par le collecteur |
| Open Images V5/V7 | Negatifs generiques `other_car` recadres avec boites | Retenue : la licence, l'auteur et l'URL sont conserves par image |
| Roboflow Universe | Petit apport optionnel et manuel | A auditer projet par projet avant toute integration |
| Stanford Cars / Cars196 | Reference de recherche, validation hors-distribution eventuelle | Non retenue pour le dataset redistribuable : licence ImageNet distincte |

## Campagne `other_car` V4

La classe `other_car` doit apprendre a rejeter de vraies voitures et non des
vehicules generiques, des intérieurs, ou des images sans rapport. La campagne
vise environ 600 **candidats** a revoir, repartis ainsi :

- 300 a 350 voitures precises issues de Wikimedia : muscle cars voisins,
  versions modernes des classes cibles, Fiat Multipla I, Volkswagen Golf VII et
  Toyota Corolla E100 ;
- 250 a 300 voitures de route variees issues d'Open Images, uniquement avec le
  label `Car`, une grande boite, sans groupe, interieur, dessin ni recadrage
  tronque ;
- aucun fichier de `datasets/field_evaluation_v3/` : ce jeu reste reserve a la
  mesure des faux positifs apres entrainement.

Roboflow Universe est utile pour explorer des candidats, notamment
[`Car Make Model Year`](https://universe.roboflow.com/senior-design-mzwsh/car-make-model-year)
(10 111 images, CC BY 4.0 affiche) et
[`Vehicle detection`](https://universe.roboflow.com/cars-hgtd4/vehicle-detection-mvhuc-qicc4)
(725 images, CC BY 4.0 affiche). Ils ne sont pas importes automatiquement :
leurs fiches ne donnent pas assez de provenance image par image pour garantir
notre tracabilite. Avant import, controler la licence de la version exacte, son
origine, les doublons avec le jeu local et effectuer une revue humaine.

## Regle de revue

Accepter `other_car` seulement si le vehicule est visible et si son cadrage
correspond a celui attendu en inference. Rejeter les intérieurs, les voitures
trop petites, les groupes ambigus, les illustrations et les photos hors sujet.
Un cadrage imparfait mais une photo source utile doit etre conserve comme
candidate puis recadre avec le detecteur ; il ne faut pas l'ajouter telle quelle
au dataset de classification.
