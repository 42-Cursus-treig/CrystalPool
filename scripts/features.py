"""
Étape 3 : construire les features, à plusieurs horizons temporels.

    python scripts/features.py

Deux principes, non négociables :

1. FILTRAGE TEMPOREL STRICT. Les données brutes contiennent toute
   l'activité des comptes, y compris post-piscine. Or seuls les admis
   continuent à se loguer : sans filtre, le modèle apprend "s'est
   connecté en novembre => a été pris". Chaque événement est donc borné
   à [début de piscine, début + horizon].

2. NORMALISATION PAR PISCINE. L'admission dépend d'un rang dans la
   promo, pas d'un score absolu. Les taux observés vont de 29% à 60%
   selon les sessions : un logtime de 200h n'a pas le même sens partout.
   Chaque feature brute est doublée d'un rang percentile intra-session.

Sortie : data/processed/features_j{H}.csv, un fichier par horizon.
"""

import json
import sys
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path
from statistics import median, pstdev

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

# Horizons de prédiction, en jours depuis le début de la piscine.
# 26 ≈ fin de piscine (utile comme borne haute de référence).
HORIZONS = [3, 7, 14, 21, 26]

# Exams de la piscine C. Les "Exam Rank XX" appartiennent au 42cursus :
# ils ne doivent jamais entrer dans les features.
POOL_EXAMS = {
    "C Piscine Exam 00": "exam00",
    "C Piscine Exam 01": "exam01",
    "C Piscine Exam 02": "exam02",
    "C Piscine Final Exam": "exam_final",
}

NIGHT_START, NIGHT_END = 22, 6  # heures considérées comme nocturnes


# ------------------------------------------------------------------ utils

def parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def in_window(item: dict, field: str, start: datetime, cutoff: datetime) -> bool:
    dt = parse_dt(item.get(field))
    return dt is not None and start <= dt <= cutoff


def _comment_len(item: dict) -> int:
    """Longueur du commentaire d'évaluation, quel que soit le format stocké."""
    if "comment_len" in item:
        return int(item.get("comment_len") or 0)
    return len(item.get("comment") or "")


def belongs_to_pool(pu: dict) -> bool:
    """filter[cursus] ne filtre pas côté API : on le refait localement."""
    return config.POOL_CURSUS_ID in (pu.get("cursus_ids") or [])


# --------------------------------------------------------------- features

def logtime_features(locations: list[dict], start: datetime, cutoff: datetime) -> dict:
    """Présence physique. La feature la plus robuste du lot."""
    hours_by_day: dict[str, float] = {}
    night_hours = total = 0.0
    n_sessions = 0

    for loc in locations:
        begin = parse_dt(loc.get("begin_at"))
        if begin is None or not (start <= begin <= cutoff):
            continue
        end = parse_dt(loc.get("end_at"))
        if end is None or end <= begin:
            continue
        # une session à cheval sur le cutoff est tronquée, pas ignorée
        end = min(end, cutoff)
        dur = (end - begin).total_seconds() / 3600
        if dur <= 0:
            continue

        total += dur
        n_sessions += 1
        hours_by_day[begin.date().isoformat()] = (
            hours_by_day.get(begin.date().isoformat(), 0) + dur
        )
        if begin.hour >= NIGHT_START or begin.hour < NIGHT_END:
            night_hours += dur

    n_days = len(hours_by_day)
    daily = list(hours_by_day.values())

    return {
        "logtime_h": round(total, 2),
        "logtime_days": n_days,
        "logtime_sessions": n_sessions,
        "logtime_h_per_day": round(total / n_days, 2) if n_days else 0.0,
        # PAS de "presence_rate = n_days / span_days" : span_days est le
        # même pour tout le monde à un horizon donné, donc cette feature
        # est proportionnelle à logtime_days — et strictement identique
        # une fois passée en rangs. Doublon parfait (VIF infini).
        # régularité : un écart-type faible = rythme constant
        "logtime_regularity": round(pstdev(daily), 2) if len(daily) > 1 else 0.0,
        "logtime_night_ratio": round(night_hours / total, 3) if total else 0.0,
        "logtime_max_day": round(max(daily), 2) if daily else 0.0,
    }


def project_features(projects: list[dict], start: datetime, cutoff: datetime) -> dict:
    """Progression sur les exercices, et notes d'exam."""
    pool_projects = [p for p in projects if belongs_to_pool(p)]
    done = [p for p in pool_projects if in_window(p, "marked_at", start, cutoff)]

    marks = [p["final_mark"] for p in done if p.get("final_mark") is not None]
    validated = [p for p in done if p.get("validated?")]
    retries = sum(max(0, (p.get("occurrence") or 0)) for p in done)

    feats = {
        "proj_marked": len(done),
        "proj_validated": len(validated),
        "proj_success_rate": round(len(validated) / len(done), 3) if done else 0.0,
        "proj_mark_mean": round(sum(marks) / len(marks), 2) if marks else 0.0,
        "proj_mark_max": max(marks) if marks else 0,
        "proj_retries": retries,
        # Projets commencés avant la coupure et pas encore rendus À CETTE
        # DATE. Le champ 'status' ne peut PAS servir ici : il reflète
        # l'état au moment de la collecte, pas à l'horizon considéré.
        # L'utiliser ferait entrer l'issue de la piscine dans les
        # features de J3.
        "proj_started_unfinished": sum(
            1 for p in pool_projects
            if in_window(p, "created_at", start, cutoff)
            and not in_window(p, "marked_at", start, cutoff)
        ),
    }

    # Notes d'exam, une colonne par exam. -1 = pas encore passé à cet horizon,
    # ce qui est une information en soi (à ne pas confondre avec un 0).
    by_name = {}
    for p in done:
        name = (p.get("project") or {}).get("name")
        if name in POOL_EXAMS:
            by_name[POOL_EXAMS[name]] = p.get("final_mark")

    for key in POOL_EXAMS.values():
        mark = by_name.get(key)
        feats[f"{key}_mark"] = mark if mark is not None else -1
        feats[f"{key}_taken"] = int(key in by_name)

    passed = [m for m in by_name.values() if m is not None and m >= 30]
    feats["exams_taken"] = len(by_name)
    feats["exams_passed"] = len(passed)
    feats["exam_mark_best"] = max(
        (m for m in by_name.values() if m is not None), default=-1
    )
    return feats


def correction_features(
    given: list[dict], received: list[dict], start: datetime, cutoff: datetime
) -> dict:
    """
    Comportement d'évaluation. 'filled_at' et non 'corrected_at' :
    ce dernier n'existe pas dans le payload (vérifié sur 25k évaluations).
    """
    def window(items: list[dict]) -> list[dict]:
        return [
            it for it in items
            if in_window(it, "begin_at", start, cutoff)
            or in_window(it, "filled_at", start, cutoff)
        ]

    g, r = window(given), window(received)

    marks_r = [it["final_mark"] for it in r if it.get("final_mark") is not None]
    marks_g = [it["final_mark"] for it in g if it.get("final_mark") is not None]

    flags_r = Counter(
        (it.get("flag") or {}).get("name") for it in r if it.get("flag")
    )
    n_cheat = sum(v for k, v in flags_r.items() if k and "cheat" in k.lower())
    n_empty = sum(v for k, v in flags_r.items() if k and "empty" in k.lower())

    return {
        "corr_given": len(g),
        "corr_received": len(r),
        # ratio : donner beaucoup relativement à ce qu'on reçoit traduit
        # de l'implication dans la promo
        "corr_ratio": round(len(g) / len(r), 3) if r else 0.0,
        "corr_mark_received_mean": round(sum(marks_r) / len(marks_r), 2) if marks_r else 0.0,
        "corr_mark_given_mean": round(sum(marks_g) / len(marks_g), 2) if marks_g else 0.0,
        # truant : ne s'est pas présenté à une correction
        "corr_truant_given": sum(1 for it in g if it.get("truant")),
        "corr_truant_received": sum(1 for it in r if it.get("truant")),
        "corr_flag_cheat": n_cheat,
        "corr_flag_empty": n_empty,
        # PAS de "corr_feedback_given" (nombre de corrections commentées) :
        # presque toutes le sont, la feature dupliquait corr_given à 99%
        # (VIF > 400) et les deux coefficients se compensaient en signe.
        #
        # collect.py ne conserve que la LONGUEUR des commentaires : leur
        # texte peut nommer des tiers. La longueur moyenne suffit comme
        # proxy du soin apporté aux corrections, et elle, est indépendante.
        "corr_comment_len_mean": round(
            sum(_comment_len(it) for it in g) / len(g), 1
        ) if g else 0.0,
    }


# ------------------------------------------------------------------ build

def load_records() -> list[dict]:
    records = []
    for pool_dir in sorted(p for p in config.RAW_DIR.glob("*") if p.is_dir()):
        for path in pool_dir.glob("*.json"):
            with path.open(encoding="utf-8") as fh:
                records.append(json.load(fh))
    return records


# Durée plausible d'une piscine. Hors de ces bornes, la date de fin
# relevée est aberrante (cursus_user jamais clôturé) et on retombe sur
# une durée par défaut.
MIN_POOL_DAYS, MAX_POOL_DAYS = 14, 40
DEFAULT_POOL_DAYS = 26


def pool_reference_dates(records: list[dict]) -> dict[str, tuple[datetime, datetime]]:
    """
    Bornes (début, fin) communes à toute une session, prises comme les
    médianes de begin_at et end_at.

    Le début commun évite que le "jour 3" tombe à des dates différentes
    d'une personne à l'autre, ce qui fausserait les rangs intra-session.

    La FIN est indispensable : toutes les piscines ne durent pas 26 jours.
    Sans elle, la fenêtre d'une session courte déborde après la fin, et
    n'y capte plus que l'activité des admis — les seuls à revenir. Les
    scale_teams sont particulièrement exposés : ils ne sont filtrés que
    par date, donc une correction 42cursus postérieure y entrerait.
    """
    begins: dict[str, list[datetime]] = {}
    ends: dict[str, list[datetime]] = {}
    for r in records:
        tag = r["pool_tag"]
        if (b := parse_dt(r.get("pool_begin_at"))):
            begins.setdefault(tag, []).append(b)
        if (e := parse_dt(r.get("pool_end_at"))):
            ends.setdefault(tag, []).append(e)

    out: dict[str, tuple[datetime, datetime]] = {}
    for tag, dates in begins.items():
        if not dates:
            continue
        start = sorted(dates)[len(dates) // 2]
        fallback = start + timedelta(days=DEFAULT_POOL_DAYS)

        candidates = sorted(ends.get(tag, []))
        end = candidates[len(candidates) // 2] if candidates else fallback
        duration = (end - start).days
        if not (MIN_POOL_DAYS <= duration <= MAX_POOL_DAYS):
            end = fallback
        out[tag] = (start, end)
    return out


def build_row(record: dict, start: datetime, cutoff: datetime) -> dict:
    row = {
        "uid_hash": record["uid_hash"],
        "pool_tag": record["pool_tag"],
        "campus_id": record["campus_id"],
        "pool_role": record.get("pool_role", "train"),
        "retenu": record.get("retenu"),
    }
    row.update(logtime_features(record.get("locations", []), start, cutoff))
    row.update(project_features(record.get("projects_users", []), start, cutoff))
    row.update(
        correction_features(
            record.get("scale_teams_as_corrector", []),
            record.get("scale_teams_as_corrected", []),
            start,
            cutoff,
        )
    )

    # Zéro jour de présence recouvre DEUX populations opposées :
    #
    #   - le vrai absent : rien du tout dans la fenêtre, ni projet ni
    #     correction. Il ne viendra pas, il n'est pas admis.
    #   - le trou de données : des projets rendus et des corrections
    #     faites, mais aucune location. La personne était évidemment
    #     présente, sa trace de log manque.
    #
    # Les confondre revient à classer les seconds au dernier rang de
    # toutes les features de présence alors qu'ils sont dans la moyenne.
    other_activity = (
        row["proj_marked"] > 0
        or row["corr_given"] > 0
        or row["corr_received"] > 0
    )
    row["logtime_data_missing"] = int(row["logtime_sessions"] == 0 and other_activity)
    row["inactive_in_window"] = int(row["logtime_sessions"] == 0 and not other_activity)
    return row


# Au-delà de cette proportion de valeurs identiques, le rang cesse d'avoir
# un sens : tous les ex æquo partagent un rang qui dépend du NOMBRE
# d'ex æquo dans leur session. Pour un signalement de triche présent chez
# 0,6% des gens, le "rang" d'un non-tricheur encode surtout combien sa
# promo comptait de tricheurs — une information de session, pas d'individu.
SPARSE_THRESHOLD = 0.90


def add_pool_ranks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rang percentile intra-session pour chaque feature numérique.

    C'est cette version que le modèle consomme : elle est comparable d'un
    campus et d'une session à l'autre, contrairement aux valeurs brutes.

    Exception faite des features trop creuses, laissées en valeur brute.
    """
    skip = {"retenu", "campus_id"}
    numeric = [
        c for c in df.columns
        if c not in skip and pd.api.types.is_numeric_dtype(df[c])
    ]

    modal_share = {
        c: df[c].value_counts(normalize=True).iloc[0] for c in numeric
    }
    dense = [c for c in numeric if modal_share[c] < SPARSE_THRESHOLD]
    sparse = [c for c in numeric if modal_share[c] >= SPARSE_THRESHOLD]

    parts = [df]
    if dense:
        parts.append(
            df.groupby("pool_tag")[dense]
            .rank(pct=True, method="average")
            .add_suffix("_rank")
        )
    if sparse:
        # suffixe _rank conservé pour que le modèle les sélectionne,
        # mais la valeur reste brute
        parts.append(df[sparse].add_suffix("_rank"))

    out = pd.concat(parts, axis=1)
    out.attrs["sparse_features"] = sparse
    return out


def main() -> None:
    records = load_records()
    if not records:
        raise SystemExit(f"Rien dans {config.RAW_DIR}. Lance scripts/collect.py.")

    refs = pool_reference_dates(records)
    print(f"{len(records)} piscineux sur {len(refs)} sessions")

    durations = sorted((end - start).days for start, end in refs.values())
    median_dur = durations[len(durations) // 2]
    print(f"  durée des sessions : {durations[0]}-{durations[-1]} j "
          f"(médiane {median_dur} j)")

    # On ne signale que les sessions ANORMALEMENT courtes. Comparer à
    # max(HORIZONS) ferait sonner l'alerte pour toutes les sessions dès
    # que la durée standard est inférieure au dernier horizon.
    short = [
        (tag, (end - start).days)
        for tag, (start, end) in refs.items()
        if (end - start).days < median_dur - 3
    ]
    if short:
        print(f"  [!] {len(short)} session(s) nettement plus courtes que la "
              "médiane : fenêtre tronquée à leur fin")
        for tag, d in sorted(short, key=lambda x: x[1])[:5]:
            print(f"    {tag:<26} {d} j")

    config.OUT_DIR.mkdir(parents=True, exist_ok=True)

    for horizon in HORIZONS:
        rows = []
        for record in records:
            ref = refs.get(record["pool_tag"])
            if ref is None:
                continue
            start, pool_end = ref
            # la fenêtre ne dépasse JAMAIS la fin réelle de la session
            cutoff = min(start + timedelta(days=horizon), pool_end)
            rows.append(build_row(record, start, cutoff))

        df = pd.DataFrame(rows)
        df = add_pool_ranks(df)
        sparse = df.attrs.get("sparse_features", [])

        out = config.OUT_DIR / f"features_j{horizon:02d}.csv"
        df.to_csv(out, index=False)

        train = df[df["pool_role"] == "train"]
        n_feat = sum(1 for c in df.columns if c.endswith("_rank"))
        active = (train["logtime_h"] > 0).mean() if len(train) else 0
        print(
            f"  J{horizon:<2} -> {out.name:<22} "
            f"{len(df):>5} lignes, {n_feat} features "
            f"({len(sparse)} creuses, laissées brutes), "
            f"{active:.0%} avec activité"
        )

    print(f"\nÉcrit dans {config.OUT_DIR}")
    print("Prochaine étape : contrôle de fuite (scripts/check_leakage.py)")


if __name__ == "__main__":
    main()
