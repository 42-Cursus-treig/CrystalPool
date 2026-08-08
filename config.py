"""Configuration. Lance discover.py puis find_pools.py avant de remplir."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# --- Identifiants API (dans .env, jamais en dur ici) ----------------------
# Une ou plusieurs applications. Chacune dispose de son propre quota
# (1200 req/h) : le client bascule sur celle dont le quota se libère le
# plus tôt, ce qui divise d'autant la durée de collecte.
#
#   FT_CLIENT_ID / FT_CLIENT_SECRET          -> app1
#   FT_CLIENT_ID_2 / FT_CLIENT_SECRET_2      -> app2
#   ... et ainsi de suite


def _load_credentials() -> list[tuple[str, str]]:
    creds = []
    first = (os.getenv("FT_CLIENT_ID", ""), os.getenv("FT_CLIENT_SECRET", ""))
    if all(first):
        creds.append(first)
    i = 2
    while True:
        pair = (os.getenv(f"FT_CLIENT_ID_{i}", ""), os.getenv(f"FT_CLIENT_SECRET_{i}", ""))
        if not all(pair):
            break
        creds.append(pair)
        i += 1
    return creds


CREDENTIALS = _load_credentials()

# Conservés pour compatibilité avec les scripts existants.
CLIENT_ID = CREDENTIALS[0][0] if CREDENTIALS else ""
CLIENT_SECRET = CREDENTIALS[0][1] if CREDENTIALS else ""

# Sel de pseudonymisation :
#   python -c "import secrets; print(secrets.token_hex(32))"
HASH_SALT = os.getenv("HASH_SALT", "")

# --- Rate limit -----------------------------------------------------------
# Par application. Vérifie avec scripts/check_ratelimit.py.
# Rester sous le plafond réel (2/s, 1200/h) évite les 429 en série.
REQ_PER_SECOND = 1.8
REQ_PER_HOUR = 1140

# --- Chemins --------------------------------------------------------------
ROOT = Path(__file__).parent
CACHE_DIR = ROOT / "cache" / "http"
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"

# --- Cursus ---------------------------------------------------------------
MAIN_CURSUS_ID = 21   # 42cursus — c'est lui qui définit le label
POOL_CURSUS_ID = 9    # C Piscine — À CONFIRMER avec scripts/find_pools.py

# Délai après la fin d'une piscine avant que le label soit fiable.
# En dessous, des admis ne sont pas encore inscrits au cursus principal.
LABEL_GRACE_DAYS = 180


@dataclass(frozen=True)
class Pool:
    """Une session de piscine sur un campus donné."""
    campus_id: int
    campus_name: str
    start: str          # borne basse sur begin_at (large : -3 jours)
    end: str            # borne haute sur begin_at (large : +3 jours)
    role: str           # "train" | "target"

    @property
    def tag(self) -> str:
        # date complète, pas seulement le mois : certains campus ont deux
        # sessions dans le même mois (Angoulême, août 2024)
        return f"{self.campus_name.lower()}_{self.start}"

    @property
    def is_target(self) -> bool:
        return self.role == "target"


# --- Sessions à collecter -------------------------------------------------
# "train"  : label stabilisé
# "target" : piscine récente, label INEXISTANT, uniquement pour prédire
#
# COUPURE À 2023-07. Avant, les campus venaient d'ouvrir et admettaient
# 76 à 100 % des piscineux : il n'y avait pas de sélection à apprendre.
# Le Havre 2023-07 (83 %) est écarté pour la même raison.
#
# Bornes élargies de 3 jours autour des begin_at observés.
POOLS = [
    # ---- Perpignan (6 sessions, n=725, 43% d'admis) ----
    Pool(60, "Perpignan", "2023-07-07", "2023-07-13", "train"),   # n=112
    Pool(60, "Perpignan", "2023-08-18", "2023-08-24", "train"),   # n=128
    Pool(60, "Perpignan", "2024-07-12", "2024-07-19", "train"),   # n=142
    Pool(60, "Perpignan", "2024-08-30", "2024-09-08", "train"),   # n=134
    Pool(60, "Perpignan", "2025-07-04", "2025-07-12", "train"),   # n=102
    Pool(60, "Perpignan", "2025-09-12", "2025-09-18", "train"),   # n=107

    # ---- Nice (8 sessions, n=578, 52%) ----
    Pool(41, "Nice", "2024-02-09", "2024-02-15", "train"),        # n=87
    Pool(41, "Nice", "2024-03-08", "2024-03-14", "train"),        # n=30
    Pool(41, "Nice", "2024-07-12", "2024-07-18", "train"),        # n=68
    Pool(41, "Nice", "2024-08-09", "2024-08-15", "train"),        # n=34
    Pool(41, "Nice", "2024-09-06", "2024-09-12", "train"),        # n=73
    Pool(41, "Nice", "2025-01-24", "2025-01-30", "train"),        # n=95
    Pool(41, "Nice", "2025-07-04", "2025-07-10", "train"),        # n=93
    Pool(41, "Nice", "2025-08-29", "2025-09-04", "train"),        # n=98

    # ---- Angoulême (9 sessions, n=784, 52%) ----
    Pool(31, "Angouleme", "2023-07-07", "2023-07-13", "train"),   # n=82
    Pool(31, "Angouleme", "2023-08-11", "2023-08-17", "train"),   # n=91
    Pool(31, "Angouleme", "2023-09-15", "2023-09-21", "train"),   # n=95
    Pool(31, "Angouleme", "2024-07-05", "2024-07-11", "train"),   # n=97
    Pool(31, "Angouleme", "2024-08-02", "2024-08-08", "train"),   # n=82
    Pool(31, "Angouleme", "2024-08-30", "2024-09-06", "train"),   # n=89
    Pool(31, "Angouleme", "2025-07-04", "2025-07-10", "train"),   # n=84
    Pool(31, "Angouleme", "2025-08-01", "2025-08-07", "train"),   # n=77
    Pool(31, "Angouleme", "2025-08-29", "2025-09-04", "train"),   # n=87

    # ---- Le Havre (6 sessions, n=688, 46%) ----
    Pool(62, "LeHavre", "2023-08-04", "2023-08-10", "train"),     # n=123
    Pool(62, "LeHavre", "2024-06-28", "2024-07-04", "train"),     # n=121
    Pool(62, "LeHavre", "2024-08-02", "2024-08-11", "train"),     # n=122
    Pool(62, "LeHavre", "2024-09-06", "2024-09-12", "train"),     # n=85
    Pool(62, "LeHavre", "2025-07-04", "2025-07-10", "train"),     # n=142
    Pool(62, "LeHavre", "2025-08-22", "2025-08-28", "train"),     # n=95

    # ---- Mulhouse (6 sessions, n=607, 43%) ----
    Pool(48, "Mulhouse", "2023-07-07", "2023-07-13", "train"),    # n=45
    Pool(48, "Mulhouse", "2023-08-18", "2023-08-25", "train"),    # n=93
    Pool(48, "Mulhouse", "2024-07-12", "2024-07-18", "train"),    # n=104
    Pool(48, "Mulhouse", "2024-08-30", "2024-09-06", "train"),    # n=150
    Pool(48, "Mulhouse", "2025-07-04", "2025-07-18", "train"),    # n=100
    Pool(48, "Mulhouse", "2025-08-29", "2025-09-11", "train"),    # n=115

    # ---- Lyon (9 sessions, n=1400, 36%) ----
    # Attention : Lyon pèse 29% du dataset et a le taux le plus bas.
    # Vérifier que le modèle ne se contente pas d'apprendre Lyon.
    Pool(9, "Lyon", "2023-06-30", "2023-07-06", "train"),         # n=106
    Pool(9, "Lyon", "2023-08-04", "2023-08-17", "train"),         # n=165
    Pool(9, "Lyon", "2023-09-08", "2023-09-14", "train"),         # n=173
    Pool(9, "Lyon", "2024-06-28", "2024-07-18", "train"),         # n=185
    Pool(9, "Lyon", "2024-08-02", "2024-08-08", "train"),         # n=158
    Pool(9, "Lyon", "2024-09-06", "2024-09-12", "train"),         # n=162
    Pool(9, "Lyon", "2025-07-04", "2025-07-12", "train"),         # n=171
    Pool(9, "Lyon", "2025-08-08", "2025-08-14", "train"),         # n=141
    Pool(9, "Lyon", "2025-09-12", "2025-09-18", "train"),         # n=139

    # ---- Cibles : terminées en juillet 2026, aucun label ----
    Pool(60, "Perpignan", "2026-07-03", "2026-07-09", "target"),  # n=83
    Pool(41, "Nice", "2026-07-03", "2026-07-09", "target"),       # n=93
    Pool(31, "Angouleme", "2026-07-03", "2026-07-09", "target"),  # n=78
    Pool(62, "LeHavre", "2026-07-03", "2026-07-09", "target"),    # n=100
    Pool(48, "Mulhouse", "2026-07-10", "2026-07-16", "target"),   # n=77
    Pool(9, "Lyon", "2026-07-09", "2026-07-16", "target"),        # n=172
    # Session qui DÉMARRE le 3 août 2026 : aucune activité à collecter.
    # À réactiver dans quelques semaines pour une prédiction à J3.
    # Pool(31, "Angouleme", "2026-07-31", "2026-08-06", "target"),  # n=40
]

TRAIN_POOLS = [p for p in POOLS if p.role == "train"]
TARGET_POOLS = [p for p in POOLS if p.is_target]
