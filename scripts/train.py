"""
Étape 5 : entraîner et évaluer, horizon par horizon.

    python scripts/train.py
    python scripts/train.py --horizon 7      # un seul horizon, en détail
    python scripts/train.py --predict        # + prédictions sur les cibles

Trois choix de méthode, tous dictés par la structure des données :

1. VALIDATION LEAVE-ONE-POOL-OUT. Un split aléatoire mettrait des
   camarades de la même session des deux côtés : le modèle verrait le
   contexte de la promo et les scores seraient gonflés. Ici chaque
   session sert de test à tour de rôle, entraînée sur les 43 autres.

2. FEATURES EN RANGS. Les taux d'admission observés vont de 29% à 60%
   selon les sessions. Un logtime de 200 h n'a donc pas le même sens
   partout ; seul le rang intra-promo est comparable.

3. RÉGRESSION LOGISTIQUE EN PREMIER. Sur ~4800 lignes et 40 features,
   elle est calibrée nativement et ne surapprend pas. LightGBM sert de
   contrôle : s'il ne fait pas nettement mieux, il n'apporte rien.
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    roc_auc_score,
)
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

warnings.filterwarnings("ignore", category=ConvergenceWarning)


EXCLUDE = {"retenu_rank", "campus_id_rank", "uid_hash_rank"}


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Uniquement les rangs intra-session : seuls comparables entre promos."""
    return [c for c in df.columns if c.endswith("_rank") and c not in EXCLUDE]


def load(horizon: int) -> pd.DataFrame:
    path = config.OUT_DIR / f"features_j{horizon:02d}.csv"
    if not path.exists():
        raise SystemExit(f"{path} absent. Lance scripts/features.py.")
    return pd.read_csv(path)


def make_model(kind: str):
    if kind == "logreg":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced"),
        )
    from lightgbm import LGBMClassifier

    return LGBMClassifier(
        n_estimators=300,
        learning_rate=0.03,
        num_leaves=15,
        min_child_samples=40,
        subsample=0.8,
        colsample_bytree=0.7,
        reg_lambda=5.0,
        verbose=-1,
    )


def cross_validate(df: pd.DataFrame, kind: str) -> dict:
    """Leave-one-pool-out : chaque session testée, entraînée sur les autres."""
    cols = feature_columns(df)
    X, y = df[cols].values, df["retenu"].values
    groups = df["pool_tag"].values

    oof = np.zeros(len(df))
    per_pool = []

    for train_idx, test_idx in LeaveOneGroupOut().split(X, y, groups):
        if len(np.unique(y[train_idx])) < 2:
            continue
        model = make_model(kind)
        model.fit(X[train_idx], y[train_idx])
        proba = model.predict_proba(X[test_idx])[:, 1]
        oof[test_idx] = proba

        if len(np.unique(y[test_idx])) == 2:
            per_pool.append({
                "pool": groups[test_idx][0],
                "n": len(test_idx),
                "rate": y[test_idx].mean(),
                "auc": roc_auc_score(y[test_idx], proba),
            })

    pools = pd.DataFrame(per_pool)
    return {
        "oof": oof,
        "auc": roc_auc_score(y, oof),
        "pr_auc": average_precision_score(y, oof),
        "brier": brier_score_loss(y, oof),
        # écart-type entre sessions : mesure la stabilité réelle
        "auc_std": pools["auc"].std() if len(pools) else float("nan"),
        "per_pool": pools,
    }


def calibration_table(y: np.ndarray, proba: np.ndarray, bins: int = 5) -> str:
    """Une probabilité de 30% doit correspondre à 30% d'admis observés."""
    edges = np.linspace(0, 1, bins + 1)
    lines = [f"    {'prédit':>12}  {'observé':>9}  {'n':>5}"]
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (proba >= lo) & (proba < hi if hi < 1 else proba <= 1)
        if mask.sum() < 10:
            continue
        lines.append(
            f"    {lo:.0%}-{hi:.0%}".ljust(16)
            + f"  {y[mask].mean():>8.0%}  {mask.sum():>5}"
        )
    return "\n".join(lines)


def campus_holdout(df: pd.DataFrame, campus_id: int, kind: str) -> float | None:
    """
    Entraîne sans un campus, teste dessus. Répond à : le modèle a-t-il
    appris la sélection, ou les habitudes d'un campus en particulier ?
    """
    cols = feature_columns(df)
    tr, te = df[df.campus_id != campus_id], df[df.campus_id == campus_id]
    if len(te) < 50 or te["retenu"].nunique() < 2:
        return None
    model = make_model(kind)
    model.fit(tr[cols].values, tr["retenu"].values)
    return roc_auc_score(te["retenu"], model.predict_proba(te[cols].values)[:, 1])


def run_horizon(horizon: int, detailed: bool) -> dict:
    df = load(horizon)
    train = df[(df["pool_role"] == "train") & df["retenu"].notna()].copy()
    train["retenu"] = train["retenu"].astype(int)

    results = {}
    for kind in ("logreg", "lgbm"):
        try:
            results[kind] = cross_validate(train, kind)
        except ImportError:
            print("  (lightgbm non installé, ignoré)")

    base = train["retenu"].mean()
    print(f"\n{'=' * 64}")
    print(f"  J{horizon}  —  {len(train)} lignes, "
          f"{train['pool_tag'].nunique()} sessions, {base:.1%} d'admis")
    print("=" * 64)
    print(f"  {'modèle':<10} {'AUC':>7} {'±':>6} {'PR-AUC':>8} {'Brier':>8}")
    for kind, r in results.items():
        print(f"  {kind:<10} {r['auc']:>7.3f} {r['auc_std']:>6.3f} "
              f"{r['pr_auc']:>8.3f} {r['brier']:>8.3f}")
    print(f"  {'hasard':<10} {0.5:>7.3f} {'':>6} {base:>8.3f} "
          f"{base * (1 - base):>8.3f}")

    # Une session parfaitement séparée n'est pas un exploit : c'est le
    # symptôme d'une information qui n'aurait pas dû être disponible.
    for kind, r in results.items():
        perfect = r["per_pool"][r["per_pool"]["auc"] >= 0.99]
        if len(perfect):
            print(f"\n  [!!] {kind} : {len(perfect)} session(s) à AUC >= 0.99")
            for _, row in perfect.iterrows():
                print(f"       {row['pool']:<26} {row['auc']:.3f} (n={row['n']:.0f})")
            print("       Deux causes possibles, à trancher avec :")
            print(f"         python scripts/diagnose_pool.py --horizon {horizon}")
            print("       - une feature qui EST le critère officiel (exam final")
            print("         >= 50) : normal, le modèle apprend le règlement ;")
            print("       - une feature comportementale : là, c'est une fuite.")
            break

    if detailed and results:
        best = min(results.items(), key=lambda kv: kv[1]["brier"])
        name, r = best
        print(f"\n  Calibration ({name}) :")
        print(calibration_table(train["retenu"].values, r["oof"]))

        pools = r["per_pool"].sort_values("auc")
        print(f"\n  Sessions les moins bien prédites ({name}) :")
        for _, row in pools.head(3).iterrows():
            print(f"    {row['pool']:<26} AUC {row['auc']:.3f}  (n={row['n']:.0f})")
        print(f"  Les mieux prédites :")
        for _, row in pools.tail(3).iterrows():
            print(f"    {row['pool']:<26} AUC {row['auc']:.3f}  (n={row['n']:.0f})")

        print(f"\n  Généralisation par campus ({name}) :")
        for cid, cname in [(9, "Lyon"), (60, "Perpignan"), (31, "Angouleme")]:
            auc = campus_holdout(train, cid, name)
            if auc is not None:
                delta = auc - r["auc"]
                flag = "  [!] chute nette" if delta < -0.05 else ""
                print(f"    entraîné sans {cname:<12} testé dessus : "
                      f"{auc:.3f} ({delta:+.3f}){flag}")

    return results


def predict_targets(horizon: int) -> None:
    """Prédictions sur les piscines de juillet 2026, sans label."""
    df = load(horizon)
    train = df[(df["pool_role"] == "train") & df["retenu"].notna()].copy()
    train["retenu"] = train["retenu"].astype(int)
    targets = df[df["pool_role"] == "target"].copy()
    if targets.empty:
        print("Aucune piscine cible.")
        return

    cols = feature_columns(df)
    model = make_model("logreg")
    model.fit(train[cols].values, train["retenu"].values)
    targets["proba"] = model.predict_proba(targets[cols].values)[:, 1]

    out = config.OUT_DIR / f"predictions_j{horizon:02d}.csv"
    targets[["uid_hash", "pool_tag", "proba"]].sort_values(
        ["pool_tag", "proba"], ascending=[True, False]
    ).to_csv(out, index=False)

    print(f"\n{'=' * 64}\n  PRÉDICTIONS J{horizon} — {len(targets)} piscineux\n{'=' * 64}")
    print(f"  {'session':<26} {'n':>5} {'taux prédit':>13}")
    for tag, grp in targets.groupby("pool_tag"):
        print(f"  {tag:<26} {len(grp):>5} {grp['proba'].mean():>12.1%}")
    print(f"\n  Écrit : {out}")
    print("\n  Ces prédictions portent sur des personnes identifiables dont")
    print("  les résultats ne sont pas encore publics. À garder pour soi ;")
    print("  l'agrégé par session, lui, ne pose pas de problème.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--predict", action="store_true")
    args = parser.parse_args()

    horizons = [args.horizon] if args.horizon else [3, 7, 14, 21, 26]
    summary = {}

    for h in horizons:
        res = run_horizon(h, detailed=bool(args.horizon))
        if res:
            summary[h] = {k: v["auc"] for k, v in res.items()}

    if len(summary) > 1:
        print(f"\n{'=' * 64}\n  PROGRESSION DE LA PRÉDICTIBILITÉ\n{'=' * 64}")
        print(f"  {'horizon':<10} {'logreg':>9} {'lgbm':>9}")
        for h, aucs in summary.items():
            print(f"  J{h:<9} {aucs.get('logreg', float('nan')):>9.3f} "
                  f"{aucs.get('lgbm', float('nan')):>9.3f}")
        first, last = list(summary)[0], list(summary)[-1]
        gain = summary[last]["logreg"] - summary[first]["logreg"]
        print(f"\n  Gain J{first} -> J{last} : {gain:+.3f}")
        if gain < 0.03:
            print("  [!] progression quasi nulle : soit tout se joue très tôt,")
            print("      soit le filtrage temporel ne fait pas son travail.")

    if args.predict:
        predict_targets(args.horizon or 26)


if __name__ == "__main__":
    main()
