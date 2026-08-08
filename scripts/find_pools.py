"""
Étape 0bis : identifier LE bon cursus piscine, et lister les sessions
disponibles sur ton campus.

    python scripts/find_pools.py

Interroge chaque cursus piscine candidat et compte les inscrits. Le bon
cursus est celui qui a des effectifs crédibles (100-400 par session).
Affiche ensuite les sessions détectées avec leur utilisabilité.
"""

import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from api42.client import make_client

# Tous les cursus dont le slug évoque une piscine C généraliste.
CANDIDATES = {
    4: "Piscine C",
    6: "Piscine C décloisonnée",
    7: "Piscine C à distance",
    9: "C Piscine",
    66: "C-Piscine-Reloaded",
}

CAMPUS_ID = 60  # Perpignan
SCAN_FROM = "2023-01-01"
SCAN_TO = "2026-12-31"


def main() -> None:
    client = make_client()

    print(f"Campus {CAMPUS_ID}, période {SCAN_FROM} -> {SCAN_TO}\n")

    all_rows: dict[int, list] = {}
    for cursus_id, name in CANDIDATES.items():
        rows = client.get_all(
            f"/v2/cursus/{cursus_id}/cursus_users",
            params={
                "filter[campus_id]": CAMPUS_ID,
                "range[begin_at]": f"{SCAN_FROM},{SCAN_TO}",
            },
        )
        all_rows[cursus_id] = rows
        flag = "  <-- candidat sérieux" if len(rows) > 50 else ""
        print(f"  cursus {cursus_id:>2} ({name:<26}) : {len(rows):>5} inscrits{flag}")

    best = max(all_rows, key=lambda k: len(all_rows[k]))
    if not all_rows[best]:
        print("\nAucun inscrit trouvé. Vérifie CAMPUS_ID et la période.")
        return

    print(f"\n=> POOL_CURSUS_ID = {best}  ({CANDIDATES[best]})\n")

    # ---- sessions détectées -------------------------------------------
    now = datetime.now(timezone.utc)
    by_month: dict[str, list] = {}
    for row in all_rows[best]:
        begin = row.get("begin_at")
        if begin:
            by_month.setdefault(begin[:7], []).append(row)

    print(f"{'session':<10} {'inscrits':>9}  {'fin réelle':<12} statut")
    print("-" * 62)
    for month in sorted(by_month):
        rows = by_month[month]
        n = len(rows)
        if n < 20:
            continue  # bruit : inscriptions isolées, pas une vraie session
        # vraie date de fin, pas une estimation
        ends = [r.get("end_at") for r in rows if r.get("end_at")]
        if ends:
            approx_end = datetime.fromisoformat(max(ends).replace("Z", "+00:00"))
        else:
            approx_end = datetime.fromisoformat(f"{month}-01").replace(
                tzinfo=timezone.utc
            ) + timedelta(days=45)
        age = (now - approx_end).days
        if age < 0:
            status = "EN COURS — inutilisable"
        elif age < config.LABEL_GRACE_DAYS:
            status = f"label non stabilisé ({age}j) — CIBLE uniquement"
        else:
            status = f"label fiable ({age}j) — entraînement OK"
        print(f"{month:<10} {n:>9}  {approx_end:%Y-%m-%d}   {status}")

    print(
        f"\nRappel : une session < {config.LABEL_GRACE_DAYS} jours ne peut PAS servir "
        "à entraîner\nni à évaluer — tous les labels seraient à 0 par construction."
    )


if __name__ == "__main__":
    main()
