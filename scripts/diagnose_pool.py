"""
Étape 4bis : comprendre pourquoi une session est (presque) parfaitement
séparée.

    python scripts/diagnose_pool.py --horizon 26
    python scripts/diagnose_pool.py --pool nice_2024-08-09 --horizon 26

Une AUC de 0.99 sur une session a deux explications possibles, et il faut
les distinguer avant de conclure quoi que ce soit :

  - RÈGLE DE SÉLECTION : une feature qui EST le critère officiel sépare
    proprement autour d'un seuil net. « exam final >= 50 » qui donne 40
    admis sur 40 au-dessus et 0 sur 45 en dessous, ce n'est pas une
    anomalie : c'est le règlement de l'école, appris correctement.

  - FUITE : la séparation vient d'une feature qui n'a aucune raison
    d'être décisive (logtime, nombre de corrections), ou d'un seuil qui
    ne correspond à rien.

Le script affiche, pour chaque session concernée, la feature la plus
séparatrice et la table de contingence autour de son meilleur seuil.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

# Features qui constituent le critère officiel d'admission. Qu'elles
# séparent parfaitement est attendu, pas suspect.
OFFICIAL = ("exam", "proj_mark", "proj_validated", "proj_success")


def is_official(name: str) -> bool:
    return any(name.startswith(p) for p in OFFICIAL)


def best_threshold(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Seuil maximisant (sensibilité + spécificité), et l'AUC associée."""
    fpr, tpr, thr = roc_curve(y, x)
    best = np.argmax(tpr - fpr)
    return float(thr[best]), roc_auc_score(y, x)


def contingency(x: np.ndarray, y: np.ndarray, thr: float) -> str:
    above, below = x >= thr, x < thr
    lines = [
        f"      {'':>14} {'admis':>7} {'non admis':>11}",
        f"      {'>= seuil':>14} {int(y[above].sum()):>7} "
        f"{int((1 - y[above]).sum()):>11}",
        f"      {'<  seuil':>14} {int(y[below].sum()):>7} "
        f"{int((1 - y[below]).sum()):>11}",
    ]
    errors = int((1 - y[above]).sum() + y[below].sum())
    lines.append(f"      -> {errors} erreur(s) sur {len(y)} avec cette seule règle")
    return "\n".join(lines)


def diagnose(df: pd.DataFrame, pool: str) -> None:
    grp = df[df["pool_tag"] == pool]
    y = grp["retenu"].values.astype(int)
    if len(np.unique(y)) < 2:
        return

    raw = [
        c for c in grp.columns
        if not c.endswith("_rank")
        and c not in {"retenu", "campus_id", "uid_hash"}
        and pd.api.types.is_numeric_dtype(grp[c])
        and grp[c].nunique() > 1
    ]

    scores = {}
    for col in raw:
        try:
            auc = roc_auc_score(y, grp[col])
        except ValueError:
            continue
        scores[col] = max(auc, 1 - auc)

    if not scores:
        return
    ranked = pd.Series(scores).sort_values(ascending=False)
    top = ranked.index[0]

    print(f"\n  {pool}  (n={len(grp)}, {y.mean():.0%} d'admis)")
    print(f"    features les plus séparatrices :")
    for name, auc in ranked.head(4).items():
        tag = "critère officiel" if is_official(name) else "COMPORTEMENTALE"
        print(f"      {name:<30} {auc:.3f}   {tag}")

    x = grp[top].values.astype(float)
    if roc_auc_score(y, x) < 0.5:
        x = -x
    thr, _ = best_threshold(x, y)
    print(f"    meilleur seuil sur {top} : {abs(thr):.1f}")
    print(contingency(x, y, thr))

    if not is_official(top):
        print("    [!] une feature comportementale ne devrait PAS suffire à")
        print("        séparer une promo. Fuite probable sur cette session.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=26)
    parser.add_argument("--pool", default=None)
    parser.add_argument("--threshold", type=float, default=0.99)
    args = parser.parse_args()

    path = config.OUT_DIR / f"features_j{args.horizon:02d}.csv"
    if not path.exists():
        raise SystemExit(f"{path} absent. Lance scripts/features.py.")

    df = pd.read_csv(path)
    df = df[(df["pool_role"] == "train") & df["retenu"].notna()].copy()
    df["retenu"] = df["retenu"].astype(int)

    if args.pool:
        pools = [args.pool]
    else:
        # sessions séparables par une feature UNIQUE au-dessus du seuil
        pools = []
        for tag, grp in df.groupby("pool_tag"):
            y = grp["retenu"].values.astype(int)
            if len(np.unique(y)) < 2:
                continue
            best = 0.0
            for col in grp.columns:
                if col.endswith("_rank") or col in {"retenu", "campus_id", "uid_hash"}:
                    continue
                if not pd.api.types.is_numeric_dtype(grp[col]) or grp[col].nunique() < 2:
                    continue
                try:
                    auc = roc_auc_score(y, grp[col])
                except ValueError:
                    continue
                best = max(best, auc, 1 - auc)
            if best >= args.threshold:
                pools.append(tag)

    print(f"J{args.horizon} — {len(pools)} session(s) séparable(s) à "
          f"{args.threshold:.2f} par une feature seule")

    for pool in pools:
        diagnose(df, pool)

    if not pools:
        print("\n  Aucune. La performance vient de la combinaison des")
        print("  features, pas d'une règle unique — c'est le cas sain.")


if __name__ == "__main__":
    main()
