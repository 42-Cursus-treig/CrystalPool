# CrystalPool

Estimer, pendant la piscine de 42, la probabilité qu'un piscineux soit
retenu — et comprendre ce qui détermine réellement cette issue.

Entraîné sur **4 782 piscineux** répartis sur **44 sessions** et 6 campus
(Perpignan, Nice, Angoulême, Le Havre, Mulhouse, Lyon), de juillet 2023 à
septembre 2025.

---

## Ce que le modèle prédit

Une probabilité d'admission, à cinq horizons : 3, 7, 14, 21 et 26 jours
de piscine.

| horizon | AUC | ce dont le modèle dispose |
|---|---|---|
| J3 | 0.805 | présence, corrections, premiers exercices — **aucun exam** |
| J7 | 0.883 | + Exam 00 |
| J14 | 0.915 | + Exam 01 |
| J21 | 0.930 | + Exam 02 |
| J26 | 0.942 | + Exam final |

L'AUC mesure la capacité à classer : 0.5 c'est le hasard, 1.0 c'est
parfait. À 0.915, si on tire au hasard un admis et un non-admis, le
modèle place le bon devant dans 91 % des cas.

**Une probabilité n'est pas un verdict.** À J14 le modèle se trompe sur
17 % des cas. Un score de 30 % signifie que sur dix personnes ayant ce
profil, trois passent — pas que celle-ci échouera.

---

## Trois résultats

### 1. Tout se joue plus tôt qu'on ne croit

Dès le troisième jour, avant le moindre exam, le modèle atteint 0.805
d'AUC uniquement à partir du comportement : régularité de présence,
corrections données et reçues, premiers exercices validés.

Ce n'est pas une prédiction sûre, mais c'est très au-dessus du hasard —
deux semaines avant que quiconque puisse se prononcer.

### 2. Les exams ne révèlent presque rien de nouveau

À J14, retirer **toutes** les features d'exam fait passer l'AUC de 0.915
à 0.901. Un coût de 0.014.

Autrement dit : le travail quotidien contient déjà l'information que
l'exam révélera. L'exam confirme, il ne décide pas — sauf pour la
minorité dont le résultat était incertain.

### 3. Le soin des corrections compte, indépendamment du temps passé

La longueur moyenne des commentaires laissés en correction prédit
l'admission de façon stable et indépendante (VIF 1.4, confirmée par
SHAP). Écrire un retour construit à un camarade pèse davantage que rester
douze heures au cluster.

Autres effets robustes, tirés des croisements bruts :

- **Retenter est bon signe** : 5 % d'admission avec zéro rattrapage,
  57 % avec cinq ou plus. Ceux qui ne retentent jamais sont ceux qui ont
  décroché.
- **Avoir des chantiers ouverts est bon signe** : 4 % d'admission avec
  aucun projet en cours, 65 % avec cinq ou plus.
- **La régularité bat l'intensité** : le nombre de jours de présence
  prédit mieux que le nombre d'heures par jour.
- **Un signalement de triche fait chuter l'admission** de 44 % à 26 %,
  mais ne concerne que 31 personnes sur 4 782.

---

## Ce que le modèle observe

Quatre familles de signaux, extraits de l'API de l'intra et bornés à la
fenêtre temporelle de l'horizon considéré.

**Présence** — nombre de jours venus, heures totales, heures par jour,
régularité (écart-type des journées), part nocturne, journée la plus
longue. Le nombre de *jours* prédit mieux que le nombre d'*heures* : la
régularité l'emporte sur l'intensité.

**Progression** — exercices rendus, validés, taux de réussite, note
moyenne et maximale, rattrapages, chantiers ouverts non rendus. C'est la
famille la plus prédictive : `proj_mark_mean` domine tous les classements
d'importance.

**Évaluation entre pairs** — corrections données et reçues, notes
attribuées et obtenues, longueur moyenne des commentaires, absences aux
corrections, signalements (triche, travail vide). La longueur des
commentaires est le signal le plus indépendant du lot.

**Exams** — note et passage pour Exam 00, 01, 02 et l'exam final, plus
des agrégats. Une note absente vaut `-1` : « pas encore passé » et
« a eu zéro » sont deux informations différentes, même si les données
montrent qu'elles ont des taux d'admission très proches (15,0 % contre
13,2 %).

Chaque feature existe en valeur brute et en **rang percentile intra-session** ;
seule cette seconde version alimente le modèle. Les features trop creuses
— un signalement de triche ne concerne que 0,6 % des piscineux — restent
en valeur brute : leur rang encoderait la composition de la promo plutôt
que l'individu.

Liste exhaustive des 35 colonnes : [PIPELINE.md](PIPELINE.md).

---

## Ce que le modèle ne peut pas faire

**Prédire le taux d'admission d'une promo.** Les features sont des rangs
au sein de chaque session, donc uniformément répartis par construction.
Le modèle classe les gens *les uns par rapport aux autres*, il ne situe
pas le niveau global d'une session. Les six piscines de juillet 2026 sont
toutes prédites entre 46 % et 48 %, alors que l'historique s'étale de
29 % à 61 %.

**Scorer quelqu'un isolément.** Pour dire « cette personne est dans le
top 20 % », il faut les 80 % restants. Un piscineux sans sa promo n'est
pas évaluable — il n'existe pas d'échelle absolue.

**Expliquer une décision individuelle.** Beaucoup de features mesurent la
même chose (VIF jusqu'à 40) : la régression logistique répartit alors le
poids arbitrairement entre elles, avec des signes qui peuvent s'inverser.
Les contributions affichées par `predict_user.py` se lisent avec
prudence ; seules celles à VIF bas sont fiables.

**Prédire à J26 quelque chose d'utile.** À la fin de la piscine,
`exam_final >= 50` sépare les classes presque sans erreur : le modèle
paraphrase le règlement. Sa valeur est aux horizons précoces.

---

## Comment il fonctionne

**Rangs intra-session.** L'admission dépend d'un classement dans la
promo, pas d'un score absolu — les taux observés vont de 29 % à 61 %
selon les sessions. Chaque feature est donc convertie en rang percentile
au sein de sa session. Un logtime de 200 h n'a pas le même sens à Lyon en
2023 et à Nice en 2025 ; un rang de 0.8, si.

**Filtrage temporel strict.** Les données brutes contiennent toute
l'activité des comptes, y compris après la piscine. Or seuls les admis
continuent à se connecter : sans filtrage, un modèle atteint 0.99 d'AUC
en apprenant « s'est logué en novembre, donc a été pris ». Chaque
événement est borné à la fenêtre `[début de session, début + horizon]`.

**Validation leave-one-pool-out.** Chaque session sert de test à tour de
rôle, entraînée sur les 43 autres. Un split aléatoire mettrait des
camarades de promo des deux côtés, et le modèle verrait le contexte de la
session qu'il est censé prédire.

**Régression logistique.** LightGBM ne fait pas mieux (écart de 0.002 à
0.015, sous l'écart-type entre sessions). La logistique est calibrée
nativement et ses coefficients se lisent. LightGBM est conservé pour les
effets de seuil qu'un modèle linéaire ne peut pas représenter — le
signalement de triche, par exemple.

---

## Validation prospective

Les probabilités des **603 piscineux de juillet 2026** ont été calculées
et figées avant que les résultats soient connus de quiconque
(`data/processed/predictions_j26.csv`, horodaté par commit).

`scripts/evaluate_prospective.py` les confrontera aux résultats réels
début 2027. C'est le seul test qui ne puisse pas être truqué a posteriori.

La métrique sera l'**AUC intra-session** : comparer taux prédit et taux
observé échouerait par construction, pour la raison expliquée plus haut.

---

## Données et éthique

Données personnelles de personnes identifiables, sur un sujet sensible.

- Les logins sont hachés dès la collecte, avec un sel local jamais
  versionné. Aucun nom n'apparaît dans `data/`.
- Le texte des commentaires d'évaluation est jeté à la collecte : il peut
  nommer des tiers. Seule sa longueur est conservée.
- Le dataset n'est pas publié.
- Les prédictions individuelles portant sur des piscineux dont le
  résultat n'est pas encore public se discutent avec la personne
  concernée. Elles ne se diffusent pas, et ne servent pas à produire un
  classement nominatif.

---

## Utilisation

```bash
python scripts/predict_user.py --login <login>              # un piscineux
python scripts/predict_user.py --pool <session> --top 30    # une session
python scripts/train.py                                     # performances
python scripts/interpret.py --horizon 14 --no-exams --shap  # ce qui pèse
```

Installation, collecte et reproduction : [PIPELINE.md](PIPELINE.md).
