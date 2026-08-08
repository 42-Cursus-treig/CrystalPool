# Pipeline — installation, collecte, reproduction

Détail opérationnel du projet. Pour comprendre ce que fait le modèle et
ce qu'il a trouvé, voir [README.md](README.md).

Collecte et analyse des traces d'activité d'une piscine 42 pour estimer,
à différents horizons temporels (J3, J7, J14…), la probabilité qu'un
piscineux soit retenu.

## Démarrage

```bash
pip install -r requirements.txt
cp .env.example .env          # puis remplis FT_CLIENT_ID / FT_CLIENT_SECRET
python -c "import secrets; print(secrets.token_hex(32))"   # -> HASH_SALT dans .env

python scripts/discover.py     # IDs campus + cursus
python scripts/find_pools.py   # LE bon cursus piscine + sessions disponibles
# reporte POOL_CURSUS_ID et la liste POOLS dans config.py

python scripts/scan_labels.py --campus 60,41,31,62,48,9   # dimensionne
python scripts/collect.py --dry-run   # vérifie les effectifs
python scripts/collect.py --limit 5   # valide le schéma sur un échantillon
python scripts/inspect_sample.py      # les champs attendus existent-ils ?

python scripts/collect.py             # collecte complète (~33 h, reprenable)
python scripts/explore.py             # contrôle qualité
python scripts/features.py            # features à J3/J7/J14/J21/J26
python scripts/check_leakage.py       # détection de fuite
```

## Ce que fait le collecteur

Pour chaque piscineux de la cohorte :

| Endpoint | Ce qu'on en tire |
|---|---|
| `projects_users` | exercices/projets, notes, `marked_at` → timelines |
| `locations` | logtime, régularité, horaires |
| `scale_teams/as_corrector` | corrections données |
| `scale_teams/as_corrected` | corrections reçues, feedbacks |
| `cursus_users` | **le label** : inscription ultérieure au cursus principal |

Le staff et les comptes anonymisés sont écartés de la cohorte.

## Garanties

- **Idempotent** — chaque réponse HTTP est cachée sur disque (`cache/`),
  chaque piscineux traité est écrit dans `data/raw/`. Relancer ne
  recollecte rien.
- **Rate limit respecté** — token bucket seconde + heure, backoff sur 429,
  refresh automatique du token OAuth.
- **Pseudonymisé** — aucun login n'est écrit dans `data/`. Le hash dépend
  d'un sel local. `cache/` contient en revanche les réponses brutes de
  l'API : il est gitignoré, ne le partage pas.

## Périmètre et coût

`--campus` restreint la collecte sans toucher à `config.py`. Le cache est
conservé : ajouter un campus plus tard ne recollecte que le nouveau.

| périmètre cumulé | lignes | admis | sessions | durée |
|---|---|---|---|---|
| Perpignan | 725 | 314 | 6 | ~3 h |
| + Angoulême | 1509 | 719 | 15 | ~7 h |
| + Nice | 2087 | 1018 | 23 | ~10 h |
| + Le Havre, Mulhouse, Lyon | 4782 | 2111 | 44 | ~23 h |

En dessous de ~8 sessions, la validation leave-one-pool-out devient
instable : chaque fold ne compte qu'une centaine de personnes.

### Plusieurs applications

Le quota de 1200 req/h est **par application**. Déclarer une seconde app
dans `.env` (`FT_CLIENT_ID_2` / `FT_CLIENT_SECRET_2`) double le débit :
le client choisit à chaque requête celle dont le quota se libère le plus
tôt, et bascule automatiquement si l'une reçoit un 429.

| applications | débit | collecte complète |
|---|---|---|
| 1 | 1140 req/h | ~26 h |
| 2 | 2280 req/h | ~13 h |
| 3 | 3420 req/h | ~9 h |

À noter : ce plafond est fixé volontairement par 42. Le contourner en
multipliant les applications reste possible mais peut valoir une
révocation d'accès si c'est repéré.

### Disque

Le cache HTTP double les données sur disque (~200 Ko par piscineux, dont
l'essentiel en `locations`). Une fois une session entièrement collectée,
`data/raw/` suffit à `features.py` — `cache/http/` peut être purgé, au
prix d'une recollecte si tu y reviens.

## Train vs target

Une piscine n'est exploitable pour l'entraînement que si son label est
stabilisé — compte 6 mois après la fin. Avant ça, les admis ne se sont pas
tous inscrits au cursus principal, et un label à 0 serait un faux négatif.

Les sessions récentes sont déclarées `role="target"` dans `config.POOLS` :
leur label est écrit à `None`, jamais à `0`. Elles servent uniquement à
prédire, et à faire une **validation prospective** — geler les prédictions
maintenant, les confronter à la réalité dans 6 mois.

## Fuite de données : le piège central

Les données brutes contiennent **toute** l'activité des comptes, pas
seulement la piscine. `filter[cursus]` ne filtre pas côté API : on
retrouve des projets 42cursus et des logs post-piscine dans les payloads.

Or seuls les admis continuent à se connecter après la piscine. Sans
filtrage, un modèle atteint 0.99 d'AUC en apprenant « s'est logué en
novembre => a été pris ». `features.py` borne donc chaque événement à
`[début de session, début + horizon]`, et `check_leakage.py` vérifie
après coup qu'aucune feature ne sépare les classes toute seule.

## Entraînement

```bash
python scripts/train.py                    # les 5 horizons
python scripts/train.py --horizon 7        # un seul, en détail
python scripts/train.py --horizon 26 --predict   # + cibles 2026
```

Validation **leave-one-pool-out** : chaque session sert de test à tour de
rôle. Un split aléatoire mettrait des camarades de promo des deux côtés
et gonflerait les scores. Le modèle ne consomme que les colonnes `_rank`,
seules comparables d'une session à l'autre.

Sortie : AUC, PR-AUC, Brier, écart-type entre sessions, table de
calibration, et un test de généralisation par campus (entraîner sans
Lyon, tester sur Lyon).


## Carte des scripts

Quinze scripts, mais tu n'en utilises que trois au quotidien. Les autres
correspondent à des phases traversées une fois.

### Usage courant

| script | rôle |
|---|---|
| `predict_user.py` | **l'outil final** : probabilité d'un piscineux (`--login`) ou classement d'une session (`--pool`) |
| `train.py` | performance par horizon, calibration, généralisation par campus |
| `interpret.py` | coefficients, effets marginaux, SHAP, croisements bruts |

### Une fois, à l'installation (déjà fait)

| script | rôle |
|---|---|
| `discover.py` | IDs de campus et de cursus → `config.py` |
| `find_pools.py` | *superseded par `scan_labels.py`, supprimable* |
| `inspect_sample.py` | vérifie que les champs de l'API sont ceux attendus |

### Collecte — à ressortir pour ajouter un campus ou une session

| script | rôle |
|---|---|
| `scan_labels.py` | dimensionne un périmètre (effectifs, taux) sans collecter |
| `collect.py` | la collecte elle-même, reprenable |
| `check_ratelimit.py` | quotas réels par application |
| `status.py` | avancement session par session |
| `explore.py` | contrôle qualité des données brutes |

### Contrôles — à relancer après toute modification de `features.py`

| script | rôle |
|---|---|
| `features.py` | reconstruit les features aux 5 horizons |
| `check_leakage.py` | trajectoire des AUC : détecte une fuite |
| `diagnose_pool.py` | pourquoi une session est parfaitement séparée |

### Dans six mois

| script | rôle |
|---|---|
| `evaluate_prospective.py` | compare les prédictions figées aux résultats réels |

## Nettoyage

```bash
./cleanup.sh --dry-run    # montre ce qui partirait
./cleanup.sh
```

Supprime le cache HTTP (le plus gros poste), les `__pycache__` et les
`features_*.csv`. Préserve `data/raw/`, `.env` et surtout
`predictions_j26.csv` — la seule chose non régénérable du projet, dont la
valeur tient à son antériorité sur les résultats.

## Validation prospective

Les prédictions sur les piscines de juillet 2026 sont écrites avant que
les résultats soient connus de quiconque — c'est le seul test qui ne
puisse pas être truqué.

```bash
python scripts/train.py --horizon 26 --predict
git commit -am "Prédictions figées"          # l'horodatage fait foi

# ~6 mois plus tard, une fois les inscriptions au 42cursus stabilisées :
python scripts/evaluate_prospective.py
```

**La métrique est l'AUC intra-session.** Les features étant des rangs
percentiles, elles sont uniformément réparties dans chaque promo par
construction : le modèle classe les gens *à l'intérieur* d'une session et
ne peut pas prédire le niveau d'admission de cette session. Les six
cibles sont d'ailleurs toutes prédites entre 46 % et 48 %, alors que
l'historique s'étale de 29 % à 61 %. Comparer taux prédit et taux observé
échouera toujours, et ne voudra rien dire.

## Interprétation

```bash
python scripts/diagnose_pool.py --horizon 26   # pourquoi une session est parfaite
python scripts/interpret.py --horizon 7        # coefficients + effets marginaux
python scripts/interpret.py --horizon 14 --shap
```

À J26, `exam_final_mark >= 50` sépare les classes presque sans erreur :
c'est la règle d'admission, pas une découverte. Le modèle y paraphrase le
règlement.

L'intérêt est aux horizons précoces. À J14, retirer *toutes* les features
d'exam ne coûte que 0.018 d'AUC : le travail quotidien contient déjà
l'information que l'exam final révélera deux semaines plus tard. C'est ce
qui rend une alerte précoce possible.

## Note

Données personnelles de personnes identifiables sur un sujet sensible.
Ne publie pas le dataset, ne redistribue pas le sel, et préviens le staff
de ton campus si tu comptes montrer les résultats publiquement.

## Liste exhaustive des features

37 features, générées par `scripts/features.py` à chacun
des 5 horizons. Chacune existe aussi en suffixe `_rank` (rang percentile
intra-session) ; c'est cette version que consomment les modèles.

### Présence

Extraites de `/v2/users/:id/locations`. Le nombre de *jours* prédit mieux que le nombre d'*heures*.

| feature | description |
|---|---|
| `logtime_h` | heures totales passées au campus |
| `logtime_days` | nombre de jours distincts de présence |
| `logtime_sessions` | nombre de sessions de log (plusieurs par jour possible) |
| `logtime_h_per_day` | heures moyennes par jour de présence |
| `logtime_regularity` | écart-type des heures quotidiennes — bas = rythme constant |
| `logtime_night_ratio` | part des heures commencées entre 22 h et 6 h |
| `logtime_max_day` | durée de la journée la plus longue |
| `logtime_data_missing` | activité présente mais aucune location : donnée manquante, pas absence |
| `inactive_in_window` | rien du tout dans la fenêtre : véritable abandon |

### Progression

Extraites de `/v2/users/:id/projects_users`, filtrées localement sur `cursus_ids` : le `filter[cursus]` de l'API ne filtre pas.

| feature | description |
|---|---|
| `proj_marked` | exercices notés à cette date |
| `proj_validated` | exercices validés |
| `proj_success_rate` | part de validés parmi les notés |
| `proj_mark_mean` | note moyenne — la feature la plus prédictive du modèle |
| `proj_mark_max` | meilleure note obtenue |
| `proj_retries` | nombre de rattrapages ; en retenter beaucoup est bon signe |
| `proj_started_unfinished` | chantiers ouverts non rendus à cette date ; en avoir est bon signe |

### Évaluation entre pairs

Extraites des `scale_teams`. Le champ de date est `filled_at` — `corrected_at` n'existe pas dans le payload. Le texte des commentaires n'est jamais conservé, seulement sa longueur.

| feature | description |
|---|---|
| `corr_given` | corrections effectuées pour des camarades |
| `corr_received` | corrections subies |
| `corr_ratio` | rapport données / reçues — implication dans la promo |
| `corr_mark_given_mean` | note moyenne attribuée aux autres |
| `corr_mark_received_mean` | note moyenne obtenue en correction |
| `corr_comment_len_mean` | longueur moyenne des commentaires laissés ; prédit indépendamment du temps passé |
| `corr_truant_given` | corrections où l'évalué ne s'est pas présenté |
| `corr_truant_received` | corrections où le correcteur ne s'est pas présenté |
| `corr_flag_cheat` | signalements de triche reçus ; 44 % -> 26 % d'admission, mais 31 cas seulement |
| `corr_flag_empty` | signalements de rendu vide |

### Exams

Isolés par nom (`C Piscine Exam 00`…) pour exclure les `Exam Rank XX`, qui appartiennent au 42cursus et non à la piscine.

| feature | description |
|---|---|
| `exam00_mark` | note à l'Exam 00 ; -1 = pas encore passé |
| `exam01_mark` | note à l'Exam 01 |
| `exam02_mark` | note à l'Exam 02 |
| `exam_final_mark` | note à l'Exam final ; >= 50 vaut admission quasi systématique |
| `exam00_taken` | l'Exam 00 a-t-il été passé à cette date |
| `exam01_taken` | l'Exam 01 a-t-il été passé |
| `exam02_taken` | l'Exam 02 a-t-il été passé |
| `exam_final_taken` | l'Exam final a-t-il été passé |
| `exams_taken` | nombre d'exams passés |
| `exams_passed` | nombre d'exams à 30 ou plus |
| `exam_mark_best` | meilleure note d'exam obtenue |
