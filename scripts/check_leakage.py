"""
Étape 4 : distinguer une fuite d'un signal légitimement fort.

    python scripts/check_leakage.py

Le niveau d'AUC ne suffit pas à trancher : une feature qui EST le critère
de sélection (une note d'exam) atteint légitimement 0.90. Ce qui
distingue une fuite, c'est sa TRAJECTOIRE dans le temps.

  - fuite      : AUC déjà élevée au tout premier horizon, puis plate.
                 L'information ne vient pas de la piscine, elle est là
                 depuis le début — typiquement de l'activité postérieure
                 qui a échappé au filtrage.
  - légitime   : AUC qui monte avec l'horizon. Le signal s'accumule au
                 fil de la piscine, ce qui est exactement attendu.

Contrôle supplémentaire : la performance en retirant les exams. Si tout
l'effet vient d'eux, le modèle ne fait que relire le barème de l'école.
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

EARLY_SUSPECT = 0.85   # AUC dès le premier horizon : anormal
RISE_LEGIT = 0.05      # progression minimale pour parler d'accumulation
EXAM_PREFIXES = ("exam",)


def is_exam(col: str) -> bool:
    return col.startswith(EXAM_PREFIXES) or "_exam" in col


def univariate_auc(df: pd.DataFrame) -> pd.Series:
    y = df["retenu"]
    skip = {"retenu", "campus_id", "uid_hash"}
    scores = {}
    for col in df.columns:
        if col in skip or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        if df[col].nunique() < 2:
            continue
        try:
            auc = roc_auc_score(y, df[col])
        except ValueError:
            continue
        scores[col] = max(auc, 1 - auc)
    return pd.Series(scores)


def main() -> None:
    files = sorted(config.OUT_DIR.glob("features_j*.csv"))
    if not files:
        raise SystemExit(f"Rien dans {config.OUT_DIR}. Lance scripts/features.py.")

    matrix = {}
    frames = {}
    for path in files:
        df = pd.read_csv(path)
        train = df[(df["pool_role"] == "train") & df["retenu"].notna()]
        if train["retenu"].nunique() < 2:
            continue
        label = path.stem.replace("features_j", "J").lstrip("0") or "J0"
        matrix[label] = univariate_auc(train)
        frames[label] = train

    if len(matrix) < 2:
        raise SystemExit("Il faut au moins deux horizons pour juger.")

    auc = pd.DataFrame(matrix)
    horizons = list(auc.columns)
    first, last = horizons[0], horizons[-1]

    # Une feature peut être INDÉFINIE à un horizon précoce : à J3 aucun
    # exam n'est passé, la colonne est constante et l'AUC incalculable.
    # C'est le meilleur signe possible d'un filtrage temporel correct, et
    # surtout pas une anomalie — d'où le traitement séparé.
    def first_defined(row: pd.Series) -> str | None:
        for h in horizons:
            if pd.notna(row[h]):
                return h
        return None

    auc["apparait"] = auc.apply(first_defined, axis=1)
    auc["Δ"] = auc.apply(
        lambda r: r[last] - r[r["apparait"]] if r["apparait"] else float("nan"),
        axis=1,
    )

    n = len(next(iter(frames.values())))
    rate = next(iter(frames.values()))["retenu"].mean()
    print(f"{n} lignes, {rate:.0%} d'admis, horizons {first} -> {last}\n")

    # ---- verdict -------------------------------------------------------
    # Suspect = présente ET déjà séparatrice dès le premier horizon,
    # sans progresser ensuite.
    suspects = auc[
        auc[first].notna()
        & (auc[first] >= EARLY_SUSPECT)
        & (auc["Δ"].fillna(0) < RISE_LEGIT)
    ]

    print("=" * 66)
    if len(suspects):
        print("  [!!] FUITE PROBABLE")
        print(f"  Ces features séparent déjà les classes à {first} et ne")
        print("  progressent pas : l'information ne vient pas de la piscine.")
        for name, row in suspects.iterrows():
            print(f"    {name:<32} {first}={row[first]:.3f}  {last}={row[last]:.3f}")
    else:
        print("  AUCUNE FUITE DÉTECTÉE")
        print(f"  Aucune feature n'atteint {EARLY_SUSPECT} dès {first} sans")
        print("  progresser ensuite. Le signal s'accumule bien dans le temps.")
    print("=" * 66)

    # ---- trajectoires --------------------------------------------------
    raw = auc[~auc.index.str.endswith("_rank")]
    top = raw.sort_values(last, ascending=False).head(12)
    print(f"\nTrajectoires (features brutes, hors rangs) :")
    print("  " + "feature".ljust(26) + "".join(f"{c:>8}" for c in horizons)
          + f"{'Δ':>8}   verdict")
    for name, row in top.iterrows():
        traj = "".join(
            f"{row[h]:>8.3f}" if pd.notna(row[h]) else f"{'—':>8}"
            for h in horizons
        )
        delta = row["Δ"]
        if row["apparait"] != first:
            verdict = f"apparaît à {row['apparait']}"
        elif pd.notna(delta) and delta >= RISE_LEGIT:
            verdict = "croissante"
        else:
            verdict = "plate"
        d = f"{delta:>+8.3f}" if pd.notna(delta) else f"{'—':>8}"
        print(f"  {name:<26}{traj}{d}   {verdict}")

    # ---- que reste-t-il sans les exams ? -------------------------------
    print(f"\n{'=' * 66}\n  SIGNAL HORS EXAMS\n{'=' * 66}")
    print("  Les exams sont le critère officiel : les retirer montre ce que")
    print("  le comportement seul permet de prédire.\n")
    print(f"  {'horizon':<8} {'tout':>7}  {'':<26} {'hors exam':>7}")
    for label in horizons:
        col = auc[label].dropna()
        best_all = col.idxmax()
        non_exam = col[[i for i in col.index if not is_exam(i)]]
        best_ne = non_exam.idxmax()
        gap = col[best_all] - non_exam[best_ne]
        print(f"  {label:<8} {col[best_all]:>7.3f}  {best_all:<26}"
              f" {non_exam[best_ne]:>7.3f}  {best_ne:<26} écart {gap:+.3f}")

    print("\n  Si l'écart entre les deux colonnes est faible aux horizons")
    print("  précoces, le comportement porte une information propre, non")
    print("  redondante avec les exams — c'est là que le projet a un intérêt.")


if __name__ == "__main__":
    main()
