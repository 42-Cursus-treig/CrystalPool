"""
L'outil final : la probabilité d'admission d'un piscineux donné.

    python scripts/predict_user.py --login trbonnet
    python scripts/predict_user.py --login trbonnet --horizon 7
    python scripts/predict_user.py --pool perpignan_2026-07-03 --top 20

Le modèle raisonne en RANGS INTRA-SESSION : il compare une personne à sa
promo, pas à une échelle absolue. Il faut donc que sa session ait été
collectée — ce que fait déjà collect.py pour les piscines déclarées dans
config.POOLS.

La probabilité affichée est un classement traduit en pourcentage, pas une
prédiction individuelle certaine. Le taux d'erreur réel à l'horizon
demandé est calculé et affiché en fin de sortie.
"""

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

EXCLUDE = {"retenu_rank", "campus_id_rank", "uid_hash_rank"}


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.endswith("_rank") and c not in EXCLUDE]


def pseudonymize(login: str) -> str:
    if not config.HASH_SALT:
        raise SystemExit("HASH_SALT absent de .env : impossible de retrouver le login.")
    return hashlib.sha256(f"{config.HASH_SALT}:{login}".encode()).hexdigest()[:16]


def load(horizon: int) -> pd.DataFrame:
    path = config.OUT_DIR / f"features_j{horizon:02d}.csv"
    if not path.exists():
        raise SystemExit(f"{path} absent. Lance scripts/features.py.")
    return pd.read_csv(path)


def fit(df: pd.DataFrame, cols: list[str]):
    train = df[(df["pool_role"] == "train") & df["retenu"].notna()]
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced"),
    )
    model.fit(train[cols].values, train["retenu"].astype(int).values)
    return model


def error_rate(df: pd.DataFrame, cols: list[str]) -> float:
    """
    Taux d'erreur réel à l'horizon demandé, mesuré hors échantillon.

    Le découpage se fait par session : mélanger des camarades de promo
    entre entraînement et test donnerait un taux trop flatteur. Cinq plis
    suffisent ici — on veut un ordre de grandeur honnête à afficher, pas
    l'évaluation complète que produit train.py.
    """
    train = df[(df["pool_role"] == "train") & df["retenu"].notna()].copy()
    train["retenu"] = train["retenu"].astype(int)
    X, y, g = train[cols].values, train["retenu"].values, train["pool_tag"].values

    oof = np.zeros(len(train))
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        m = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, C=0.5, class_weight="balanced"),
        )
        m.fit(X[tr], y[tr])
        oof[te] = m.predict_proba(X[te])[:, 1]
    return float(((oof >= 0.5).astype(int) != y).mean())


def explain(model, cols: list[str], x: np.ndarray, top: int = 6) -> None:
    """Contribution de chaque feature : coefficient x valeur standardisée."""
    scaler, clf = model[0], model[-1]
    z = scaler.transform(x.reshape(1, -1))[0]
    contrib = pd.Series(z * clf.coef_[0], index=cols).sort_values(ascending=False)

    # séparer par signe : sinon, avec peu de features, head() et tail()
    # se recouvrent et la même contribution s'affiche deux fois
    plus = contrib[contrib > 0.01].head(top)
    moins = contrib[contrib < -0.01].tail(top).sort_values()

    if len(plus):
        print("\n  En sa faveur :")
        for name, v in plus.items():
            print(f"    {name.replace('_rank', ''):<32} {v:>+6.2f}")
    if len(moins):
        print("\n  En sa défaveur :")
        for name, v in moins.items():
            print(f"    {name.replace('_rank', ''):<32} {v:>+6.2f}")
    if not len(plus) and not len(moins):
        print("\n  Profil très proche de la médiane de sa promo.")


def show_one(df: pd.DataFrame, model, cols: list[str], row: pd.Series,
             horizon: int, detail: bool) -> None:
    x = row[cols].values.astype(float)
    proba = model.predict_proba(x.reshape(1, -1))[0, 1]

    pool = df[df["pool_tag"] == row["pool_tag"]]
    print(f"\n  session   : {row['pool_tag']} ({len(pool)} piscineux)")
    print(f"  horizon   : J{horizon}")
    print(f"  PROBABILITÉ D'ADMISSION : {proba:.0%}")

    # position dans la promo : c'est ce que le modèle mesure vraiment
    all_p = model.predict_proba(pool[cols].values)[:, 1]
    rank = (all_p < proba).mean()
    print(f"  position dans la promo  : {rank:.0%} "
          f"(mieux classé que {int(rank * len(pool))} personnes sur {len(pool)})")

    if row.get("retenu") == row.get("retenu") and pd.notna(row.get("retenu")):
        reel = "admis" if row["retenu"] == 1 else "non admis"
        print(f"  résultat réel           : {reel}")

    if detail:
        explain(model, cols, x)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--login", default=None, help="login intra du piscineux")
    parser.add_argument("--pool", default=None, help="classer toute une session")
    parser.add_argument("--horizon", type=int, default=26)
    parser.add_argument("--top", type=int, default=15, help="avec --pool")
    parser.add_argument("--no-detail", action="store_true")
    args = parser.parse_args()

    if not args.login and not args.pool:
        raise SystemExit("Donne --login <login> ou --pool <session>.")

    df = load(args.horizon)
    cols = feature_columns(df)
    model = fit(df, cols)

    if args.login:
        uid_hash = pseudonymize(args.login)
        match = df[df["uid_hash"] == uid_hash]
        if match.empty:
            print(f"'{args.login}' introuvable dans les données collectées.")
            print("\nSessions disponibles :")
            for tag in sorted(df["pool_tag"].unique()):
                print(f"  {tag}")
            print("\nSi sa piscine n'est pas dans la liste, ajoute-la à")
            print("config.POOLS puis relance collect.py et features.py.")
            return
        print("=" * 62)
        print(f"  {args.login}")
        print("=" * 62)
        show_one(df, model, cols, match.iloc[0], args.horizon, not args.no_detail)

    if args.pool:
        pool = df[df["pool_tag"] == args.pool].copy()
        if pool.empty:
            raise SystemExit(f"Session '{args.pool}' inconnue.")
        pool["proba"] = model.predict_proba(pool[cols].values)[:, 1]
        pool = pool.sort_values("proba", ascending=False)

        print("=" * 62)
        print(f"  {args.pool} — J{args.horizon} — {len(pool)} piscineux")
        print("=" * 62)
        print(f"  probabilité moyenne : {pool['proba'].mean():.0%}")
        print(f"\n  {'rang':>5}  {'hash':<18} {'proba':>7}")
        for i, (_, r) in enumerate(pool.head(args.top).iterrows(), 1):
            reel = ""
            if pd.notna(r.get("retenu")):
                reel = "  (admis)" if r["retenu"] == 1 else "  (non admis)"
            print(f"  {i:>5}  {r['uid_hash']:<18} {r['proba']:>6.0%}{reel}")
        if len(pool) > args.top:
            print(f"  ... {len(pool) - args.top} autres")

    err = error_rate(df, cols)
    print(f"\n  Un score n'est pas un verdict : à J{args.horizon}, le modèle se")
    print(f"  trompe encore sur {err:.0%} des cas ({err * 100:.0f} personnes sur 100).")
    print("  Sur une piscine récente, ces chiffres se discutent avec la")
    print("  personne concernée, ils ne se diffusent pas.")


if __name__ == "__main__":
    main()
