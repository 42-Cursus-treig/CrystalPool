"""
Étape 7 : boucler la validation prospective, une fois les résultats
connus (compter ~6 mois après la fin des piscines cibles).

    python scripts/evaluate_prospective.py

Relit les prédictions figées, va chercher les labels réels via l'API, et
compare. C'est le seul test qui ne peut pas être truqué : les
prédictions ont été écrites avant que l'issue soit connue de quiconque.

ATTENTION À LA MÉTRIQUE. Les features sont des rangs intra-session, donc
uniformément répartis dans chaque promo par construction. Le modèle
classe les gens À L'INTÉRIEUR d'une session ; il ne peut pas prédire le
NIVEAU d'admission de cette session. Comparer taux prédit et taux observé
échouera toujours, et ne signifiera rien. La métrique juste est l'AUC
calculée session par session.
"""

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from api42.client import make_client


def parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def fetch_labels(client, pools: list[config.Pool]) -> dict[str, int]:
    """
    uid_hash -> admis, pour les sessions cibles.

    On refait le même hachage que collect.py, avec le même sel : sans
    lui, impossible de relier les prédictions aux personnes.
    """
    import hashlib

    if not config.HASH_SALT:
        raise SystemExit("HASH_SALT absent : les hash ne correspondront pas.")

    labels: dict[str, int] = {}
    for pool in pools:
        main = client.get_all(
            f"/v2/cursus/{config.MAIN_CURSUS_ID}/cursus_users",
            params={"filter[campus_id]": pool.campus_id},
        )
        admitted = {}
        for row in main:
            uid = (row.get("user") or {}).get("id")
            begin = parse_dt(row.get("begin_at"))
            if uid is not None and begin:
                if uid not in admitted or begin < admitted[uid]:
                    admitted[uid] = begin

        cohort = client.get_all(
            f"/v2/cursus/{config.POOL_CURSUS_ID}/cursus_users",
            params={
                "filter[campus_id]": pool.campus_id,
                "range[begin_at]": f"{pool.start},{pool.end}",
            },
        )
        n_pos = 0
        for row in cohort:
            user = row.get("user") or {}
            login = user.get("login")
            if not login or user.get("staff?") or str(login).startswith("3b3"):
                continue
            uid_hash = hashlib.sha256(
                f"{config.HASH_SALT}:{login}".encode()
            ).hexdigest()[:16]
            pool_begin = parse_dt(row.get("begin_at"))
            main_begin = admitted.get(user["id"])
            retenu = int(bool(main_begin and pool_begin and main_begin > pool_begin))
            labels[uid_hash] = retenu
            n_pos += retenu
        print(f"  {pool.tag:<26} {len(cohort):>4} pers., {n_pos:>4} admis")

    return labels


def main() -> None:
    preds = sorted(config.OUT_DIR.glob("predictions_j*.csv"))
    if not preds:
        raise SystemExit(
            "Aucune prédiction figée. Lance :\n"
            "  python scripts/train.py --horizon 26 --predict"
        )

    client = make_client()
    print("Récupération des labels réels :")
    labels = fetch_labels(client, config.TARGET_POOLS)

    for path in preds:
        df = pd.read_csv(path)
        df["retenu"] = df["uid_hash"].map(labels)
        known = df[df["retenu"].notna()].copy()
        known["retenu"] = known["retenu"].astype(int)

        print(f"\n{'=' * 62}\n  {path.name}  —  {len(known)}/{len(df)} labels trouvés\n"
              f"{'=' * 62}")
        if known["retenu"].nunique() < 2:
            print("  Une seule classe : évaluation impossible.")
            continue

        print(f"  {'session':<26} {'n':>5} {'observé':>9} {'AUC':>7}")
        aucs = []
        for tag, grp in known.groupby("pool_tag"):
            if grp["retenu"].nunique() < 2:
                continue
            auc = roc_auc_score(grp["retenu"], grp["proba"])
            aucs.append(auc)
            print(f"  {tag:<26} {len(grp):>5} {grp['retenu'].mean():>8.1%} {auc:>7.3f}")

        if aucs:
            print(f"\n  AUC moyenne intra-session : {sum(aucs) / len(aucs):.3f}")
            print(f"  min / max                 : {min(aucs):.3f} / {max(aucs):.3f}")
        print(f"  Brier (calibration)       : "
              f"{brier_score_loss(known['retenu'], known['proba']):.3f}")

        print(f"\n  Taux prédit {known['proba'].mean():.1%} vs observé "
              f"{known['retenu'].mean():.1%}")
        print("  (écart attendu et non significatif : les rangs intra-session")
        print("   effacent par construction le niveau propre à chaque promo)")


if __name__ == "__main__":
    main()
