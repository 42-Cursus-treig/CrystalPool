"""
Étape 0ter : dimensionner le dataset AVANT de lancer la grosse collecte.

    python scripts/scan_labels.py --campus 60,41,31,62,48

Récupère par campus l'ensemble des inscrits au cursus principal, puis
l'intersecte avec les cohortes piscine. Coût : ~40 requêtes par campus
au lieu d'une par piscineux. Quelques minutes pour tout scanner.

Sortie : effectifs et taux d'admission par session, plus un bloc POOLS
prêt à coller dans config.py.
"""

import argparse
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from api42.client import Client42, make_client

SCAN_FROM = "2022-01-01"
SCAN_TO = "2027-12-31"
MIN_COHORT = 20  # en dessous : inscriptions isolées, pas une vraie session
PAD = timedelta(days=3)  # marge sur les bornes de begin_at


def parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def is_real_user(user: dict) -> bool:
    login = user.get("login")
    return bool(login) and not user.get("staff?") and not str(login).startswith("3b3")


def main_cursus_dates(client: Client42, campus_id: int) -> dict[int, datetime]:
    """user_id -> date d'entrée au cursus principal, pour tout le campus."""
    rows = client.get_all(
        f"/v2/cursus/{config.MAIN_CURSUS_ID}/cursus_users",
        params={"filter[campus_id]": campus_id},
    )
    out: dict[int, datetime] = {}
    for row in rows:
        uid = (row.get("user") or {}).get("id")
        begin = parse_dt(row.get("begin_at"))
        if uid is not None and begin:
            # on garde la plus ancienne inscription
            if uid not in out or begin < out[uid]:
                out[uid] = begin
    return out


def scan_campus(client: Client42, campus_id: int, name: str) -> list[dict]:
    main_dates = main_cursus_dates(client, campus_id)
    pool_rows = client.get_all(
        f"/v2/cursus/{config.POOL_CURSUS_ID}/cursus_users",
        params={
            "filter[campus_id]": campus_id,
            "range[begin_at]": f"{SCAN_FROM},{SCAN_TO}",
        },
    )

    sessions: dict[str, list[dict]] = defaultdict(list)
    for row in pool_rows:
        user = row.get("user") or {}
        begin = parse_dt(row.get("begin_at"))
        if not begin or not is_real_user(user):
            continue
        sessions[begin.strftime("%Y-%m")].append(row)

    now = datetime.now(timezone.utc)
    results = []

    for month, rows in sorted(sessions.items()):
        if len(rows) < MIN_COHORT:
            continue

        begins = [parse_dt(r["begin_at"]) for r in rows]
        ends = sorted(e for e in (parse_dt(r.get("end_at")) for r in rows) if e)
        # médiane et non max : un seul cursus_user jamais clôturé décalerait
        # la date de fin de plusieurs années
        real_end = ends[len(ends) // 2] if ends else max(begins)
        age_days = (now - real_end).days

        n_pos = 0
        for row in rows:
            uid = row["user"]["id"]
            pool_begin = parse_dt(row["begin_at"])
            main_begin = main_dates.get(uid)
            # admis = entré au cursus principal APRÈS le début de cette piscine
            # (sinon on compte quelqu'un déjà inscrit via une piscine antérieure)
            if main_begin and pool_begin and main_begin > pool_begin:
                n_pos += 1

        role = "train" if age_days >= config.LABEL_GRACE_DAYS else "target"
        results.append(
            {
                "campus_id": campus_id,
                "campus_name": name,
                "month": month,
                "n": len(rows),
                "n_pos": n_pos,
                "rate": n_pos / len(rows),
                "end": real_end,
                "age_days": age_days,
                "role": role,
                # marge de 3 jours : sans elle, une session dont tous les
                # begin_at tombent le même jour donne un intervalle nul
                "start_bound": (min(begins) - PAD).strftime("%Y-%m-%d"),
                "end_bound": (max(begins) + PAD).strftime("%Y-%m-%d"),
            }
        )

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--campus",
        default="60",
        help="IDs séparés par des virgules, ex: 60,41,31,62,48",
    )
    parser.add_argument(
        "--since",
        default=None,
        help="ignore les sessions antérieures (ex: 2023-07). Les campus qui "
             "viennent d'ouvrir admettent ~100%% : autre régime, à exclure.",
    )
    args = parser.parse_args()
    campus_ids = [int(c.strip()) for c in args.campus.split(",")]

    client = make_client()

    # noms de campus, pour des tags lisibles
    campuses = {c["id"]: c["name"] for c in client.get_all("/v2/campus")}

    all_results = []
    for cid in campus_ids:
        name = campuses.get(cid, f"campus{cid}")
        print(f"\n--- {name} (id {cid}) ---")
        results = scan_campus(client, cid, name)
        if args.since:
            results = [r for r in results if r["month"] >= args.since]
        if not results:
            print("  aucune session détectée")
            continue

        print(f"  {'session':<9} {'n':>5} {'admis':>6} {'taux':>7}  {'fin':<11} rôle")
        for r in results:
            print(
                f"  {r['month']:<9} {r['n']:>5} {r['n_pos']:>6} {r['rate']:>6.1%}  "
                f"{r['end']:%Y-%m-%d}  {r['role']}"
                + ("  <- cible" if r["role"] == "target" else "")
            )
        all_results.extend(results)

    # ---- bilan --------------------------------------------------------
    train = [r for r in all_results if r["role"] == "train"]
    n_tot = sum(r["n"] for r in train)
    n_pos = sum(r["n_pos"] for r in train)

    print(f"\n{'=' * 62}\nBILAN ENTRAÎNEMENT")
    print(f"  sessions exploitables : {len(train)}")
    print(f"  piscineux             : {n_tot}")
    print(f"  admis                 : {n_pos} ({n_pos / n_tot:.1%})" if n_tot else "")

    if n_tot:
        rates = [r["rate"] for r in train]
        print(f"  taux min/max          : {min(rates):.1%} / {max(rates):.1%}")
        if max(rates) - min(rates) > 0.15:
            print("  [!] forte variation entre sessions — la normalisation")
            print("      par piscine n'est pas optionnelle")

    saturated = [r for r in train if r["rate"] > 0.75]
    if saturated:
        last = max(r["month"] for r in saturated)
        print(f"\n  [!] {len(saturated)} sessions à plus de 75% d'admission "
              f"(jusqu'à {last}).")
        print("      Campus en ouverture : tout le monde passe, le signal est")
        print(f"      absent. Relance avec --since {last[:4]}-{int(last[5:7]) + 1:02d}")

    if n_pos < 250:
        print("\n  [!] moins de 250 admis : ajoute des campus avant de modéliser")
    elif len(train) < 8:
        print("\n  [!] moins de 8 sessions : la validation leave-one-pool-out")
        print("      sera instable, ajoute des campus")
    else:
        print("\n  Volume correct pour démarrer.")

    # ---- config prête à coller ----------------------------------------
    print(f"\n{'=' * 62}\nÀ COLLER DANS config.py :\n")
    print("POOLS = [")
    for r in sorted(all_results, key=lambda x: (x["role"] != "train", x["month"])):
        print(
            f'    Pool({r["campus_id"]}, "{r["campus_name"]}", '
            f'"{r["start_bound"]}", "{r["end_bound"]}", "{r["role"]}"),'
            f'  # n={r["n"]}'
        )
    print("]")


if __name__ == "__main__":
    main()
