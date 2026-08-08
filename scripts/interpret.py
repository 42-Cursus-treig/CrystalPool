"""
Étape 6 : comprendre ce que le modèle a appris.

    python scripts/interpret.py --horizon 7
    python scripts/interpret.py --horizon 14 --shap

Trois sorties :

1. COEFFICIENTS de la régression logistique. Les features étant des rangs
   percentiles standardisés, ils sont directement comparables entre eux :
   un coefficient deux fois plus grand pèse deux fois plus.

2. EFFET MARGINAL : de combien la probabilité d'admission bouge quand on
   passe du 1er au 3e quartile sur une feature, tout le reste au médian.
   C'est la forme sous laquelle une réponse est utilisable : « corriger
   beaucoup fait gagner X points de probabilité ».

3. MODÈLE SANS EXAMS. À J26, prédire avec les notes d'exam revient à
   relire le règlement (exam final >= 50 => admis). Le modèle sans exams
   mesure ce que le comportement seul permet d'anticiper — c'est lui qui
   a une valeur d'alerte précoce.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

EXAM_MARKERS = ("exam",)


def is_exam(col: str) -> bool:
    return any(m in col for m in EXAM_MARKERS)


def feature_columns(df: pd.DataFrame, with_exams: bool = True) -> list[str]:
    cols = [
        c for c in df.columns
        if c.endswith("_rank")
        and c not in {"retenu_rank", "campus_id_rank", "uid_hash_rank"}
    ]
    return cols if with_exams else [c for c in cols if not is_exam(c)]


def load(horizon: int) -> pd.DataFrame:
    path = config.OUT_DIR / f"features_j{horizon:02d}.csv"
    if not path.exists():
        raise SystemExit(f"{path} absent. Lance scripts/features.py.")
    df = pd.read_csv(path)
    df = df[(df["pool_role"] == "train") & df["retenu"].notna()].copy()
    df["retenu"] = df["retenu"].astype(int)
    return df


def fit_logreg(df: pd.DataFrame, cols: list[str]):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced"),
    )
    model.fit(df[cols].values, df["retenu"].values)
    return model


def cv_auc(df: pd.DataFrame, cols: list[str]) -> float:
    X, y, g = df[cols].values, df["retenu"].values, df["pool_tag"].values
    oof = np.zeros(len(df))
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        if len(np.unique(y[tr])) < 2:
            continue
        m = fit_logreg(df.iloc[tr], cols)
        oof[te] = m.predict_proba(X[te])[:, 1]
    return roc_auc_score(y, oof)


def vif(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """
    Facteur d'inflation de la variance. VIF_i = 1 / (1 - R²_i), où R²_i
    est obtenu en régressant la feature i sur toutes les autres.

    Au-delà de 5, la feature est largement redéductible des autres : son
    coefficient devient instable et son SIGNE peut s'inverser sans que
    cela veuille dire quoi que ce soit. C'est ce qui produit des
    aberrations du type « valider un projet défavorise l'admission ».
    """
    from sklearn.linear_model import LinearRegression

    X = df[cols].values
    out = {}
    for i, col in enumerate(cols):
        others = np.delete(X, i, axis=1)
        r2 = LinearRegression().fit(others, X[:, i]).score(others, X[:, i])
        out[col] = 1.0 / max(1e-9, 1.0 - r2)
    return pd.Series(out)


def show_coefficients(df: pd.DataFrame, cols: list[str], top: int = 14) -> None:
    model = fit_logreg(df, cols)
    coefs = pd.Series(model[-1].coef_[0], index=cols).sort_values(key=abs, ascending=False)
    vifs = vif(df, cols)

    print(f"\n  {'feature':<32} {'coef':>7} {'VIF':>7}   effet")
    for name, c in coefs.head(top).items():
        v = vifs[name]
        bar = "+" * min(16, int(abs(c) * 12))
        sense = "favorise" if c > 0 else "défavorise"
        tags = " [exam]" if is_exam(name) else ""
        if v > 5:
            tags += " [colinéaire]"
        v_txt = ">999" if v > 999 else f"{v:.1f}"
        print(f"  {name.replace('_rank',''):<32} {c:>+7.2f} {v_txt:>7}   "
              f"{sense:<11}{bar}{tags}")

    bad = vifs[vifs > 5]
    if len(bad):
        print(f"\n  [!] {len(bad)} features à VIF > 5 : leur coefficient — et")
        print("      surtout son signe — n'est pas interprétable isolément.")
        print("      Se fier à SHAP pour l'importance, et aux groupes ci-dessous")
        print("      plutôt qu'aux features prises une à une.")


def correlated_groups(df: pd.DataFrame, cols: list[str], thr: float = 0.8) -> None:
    """Regroupe les features quasi redondantes : elles se lisent ensemble."""
    corr = df[cols].corr().abs()
    seen: set[str] = set()
    groups: list[list[str]] = []
    for c in cols:
        if c in seen:
            continue
        block = [o for o in cols if o != c and corr.loc[c, o] >= thr]
        if block:
            group = [c] + [b for b in block if b not in seen]
            seen.update(group)
            groups.append(group)

    if not groups:
        print(f"\n  Aucun groupe de features corrélées à plus de {thr:.0%}.")
        return
    print(f"\n  Groupes de features corrélées à plus de {thr:.0%} "
          "(à lire ensemble) :")
    for g in groups:
        names = ", ".join(n.replace("_rank", "") for n in g)
        print(f"    - {names}")


def marginal_effects(df: pd.DataFrame, cols: list[str], top: int = 10) -> None:
    """Probabilité prédite en passant du 1er au 3e quartile, reste au médian."""
    model = fit_logreg(df, cols)
    med = df[cols].median().values.reshape(1, -1)
    base = model.predict_proba(med)[0, 1]

    effects = []
    for i, col in enumerate(cols):
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        lo, hi = med.copy(), med.copy()
        lo[0, i], hi[0, i] = q1, q3
        p_lo = model.predict_proba(lo)[0, 1]
        p_hi = model.predict_proba(hi)[0, 1]
        effects.append((col, p_lo, p_hi, p_hi - p_lo))

    effects.sort(key=lambda e: abs(e[3]), reverse=True)
    print(f"\n  Profil médian : {base:.0%} de probabilité d'admission.")
    print(f"  Effet du passage Q1 -> Q3, toutes choses égales :\n")
    print(f"  {'feature':<34} {'Q1':>7} {'Q3':>7} {'écart':>8}")
    for col, p_lo, p_hi, d in effects[:top]:
        tag = " [exam]" if is_exam(col) else ""
        print(f"  {col.replace('_rank',''):<34} {p_lo:>6.0%} {p_hi:>6.0%} {d:>+8.0%}{tag}")


def compare_with_without_exams(horizon: int) -> None:
    df = load(horizon)
    with_e = feature_columns(df, True)
    without_e = feature_columns(df, False)
    auc_w = cv_auc(df, with_e)
    auc_wo = cv_auc(df, without_e)
    print(f"\n  {'modèle':<26} {'features':>9} {'AUC':>8}")
    print(f"  {'avec exams':<26} {len(with_e):>9} {auc_w:>8.3f}")
    print(f"  {'comportement seul':<26} {len(without_e):>9} {auc_wo:>8.3f}")
    print(f"  {'coût du retrait':<26} {'':>9} {auc_wo - auc_w:>+8.3f}")
    if auc_w - auc_wo < 0.03:
        print("\n  Les exams n'apportent presque rien : le comportement")
        print("  quotidien contient déjà l'information. C'est ce qui rend")
        print("  une alerte précoce possible.")


def rare_feature_effects(df: pd.DataFrame, max_levels: int = 6) -> None:
    """
    Taux d'admission croisé par valeur, pour les features de comptage.

    Un modèle linéaire ne voit qu'une pente : il suppose que passer de 0 à
    1 signalement de triche a le même effet que de 3 à 4. Or ces
    événements agissent plutôt comme des couperets. Le croisement brut
    montre la vraie forme de l'effet — et si elle est plate, la feature
    ne veut rien dire malgré son importance SHAP.
    """
    candidates = [
        c for c in df.columns
        if not c.endswith("_rank")
        and c not in {"retenu", "campus_id", "uid_hash"}
        and pd.api.types.is_numeric_dtype(df[c])
        and (df[c] % 1 == 0).all()
        and 1 < df[c].nunique() <= 40
    ]
    if not candidates:
        print("\n  Aucune feature de comptage exploitable.")
        return

    base = df["retenu"].mean()
    print(f"\n  Taux d'admission de référence : {base:.1%}\n")

    for col in candidates:
        vals = df[col]
        # au-delà de max_levels valeurs, on regroupe la queue
        levels = sorted(vals.unique())
        if len(levels) > max_levels:
            cut = levels[max_levels - 1]
            key = vals.clip(upper=cut)
            labels = {v: (f"{v:.0f}" if v < cut else f"{cut:.0f}+") for v in sorted(key.unique())}
        else:
            key = vals
            labels = {v: f"{v:.0f}" for v in levels}

        grouped = df.groupby(key)["retenu"].agg(["mean", "size"])
        grouped = grouped[grouped["size"] >= 30]
        if len(grouped) < 2:
            continue

        spread = grouped["mean"].max() - grouped["mean"].min()
        if spread < 0.10:
            continue  # effet trop faible pour être signalé

        print(f"  {col}")
        for val, row in grouped.iterrows():
            delta = row["mean"] - base
            bar = ("+" if delta > 0 else "-") * min(18, int(abs(delta) * 60))
            print(f"    {labels.get(val, str(val)):>5} : {row['mean']:>6.1%} "
                  f"({row['size']:>5.0f} pers.)  {delta:>+6.1%}  {bar}")
        print()


def shap_summary(df: pd.DataFrame, cols: list[str], top: int = 14) -> None:
    try:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import shap
        from lightgbm import LGBMClassifier
    except ImportError:
        print("\n  (shap ou lightgbm non installé : pip install shap lightgbm)")
        return

    model = LGBMClassifier(
        n_estimators=300, learning_rate=0.03, num_leaves=15,
        min_child_samples=40, subsample=0.8, colsample_bytree=0.7,
        reg_lambda=5.0, verbose=-1,
    )
    model.fit(df[cols].values, df["retenu"].values)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        values = shap.TreeExplainer(model).shap_values(df[cols].values)
    # selon la version, shap renvoie une liste (une matrice par classe)
    if isinstance(values, list):
        values = values[1]
    elif values.ndim == 3:
        values = values[:, :, 1]

    imp = pd.Series(np.abs(values).mean(axis=0), index=cols).sort_values(ascending=False)
    print(f"\n  Importance SHAP (moyenne des |contributions|) :\n")
    top_imp = imp.head(top)
    scale = top_imp.iloc[0]
    for name, v in top_imp.items():
        bar = "#" * max(1, int(20 * v / scale))
        tag = " [exam]" if is_exam(name) else ""
        print(f"  {name.replace('_rank',''):<34} {v:.3f}  {bar}{tag}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--horizon", type=int, default=7)
    parser.add_argument("--shap", action="store_true")
    parser.add_argument("--no-exams", action="store_true",
                        help="interpréter le modèle comportemental seul")
    args = parser.parse_args()

    df = load(args.horizon)
    cols = feature_columns(df, with_exams=not args.no_exams)

    label = "comportement seul" if args.no_exams else "toutes features"
    print(f"{'=' * 66}")
    print(f"  J{args.horizon} — {len(df)} lignes, {df['pool_tag'].nunique()} sessions, "
          f"{len(cols)} features ({label})")
    print("=" * 66)

    print("\n--- COEFFICIENTS (régression logistique) ---")
    show_coefficients(df, cols)

    print("\n--- REDONDANCE ENTRE FEATURES ---")
    correlated_groups(df, cols)

    print("\n--- EFFETS MARGINAUX ---")
    marginal_effects(df, cols)

    print("\n--- AVEC ET SANS EXAMS ---")
    compare_with_without_exams(args.horizon)

    print("\n--- FEATURES DE COMPTAGE (effet réel, non linéaire) ---")
    rare_feature_effects(df)

    if args.shap:
        print("\n--- SHAP ---")
        shap_summary(df, cols)


if __name__ == "__main__":
    main()
