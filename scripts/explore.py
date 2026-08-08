import json
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config


def load_pool(pool_dir: Path) -> list[dict]:
    records = []
    for path in sorted(pool_dir.glob("*.json")):
        with path.open(encoding="utf-8") as fh:
            records.append(json.load(fh))
    return records


def parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def logtime_hours(locations: list[dict]) -> float:
    total = 0.0
    for loc in locations:
        begin, end = parse_dt(loc.get("begin_at")), parse_dt(loc.get("end_at"))
        if begin and end and end > begin:
            total += (end - begin).total_seconds() / 3600
    return total


def main() -> None:
    pools = sorted(p for p in config.RAW_DIR.glob("*") if p.is_dir())
    if not pools:
        raise SystemExit(f"Aucune donnée dans {config.RAW_DIR}. Lance scripts/collect.py.")

    for pool_dir in pools:
        records = load_pool(pool_dir)
        n = len(records)
        print(f"\n{'=' * 62}\n  {pool_dir.name}  —  {n} piscineux\n{'=' * 62}")
        if not n:
            continue

        # ---- label -------------------------------------------------------
        n_pos = sum(r["retenu"] for r in records)
        print(f"\nLABEL")
        print(f"  retenus         : {n_pos:>4} ({n_pos / n:.1%})")
        print(f"  non retenus     : {n - n_pos:>4} ({1 - n_pos / n:.1%})")
        if n_pos < 30 or n - n_pos < 30:
            print("  [!] classe minoritaire très faible : une seule piscine ne suffira pas")

        # ---- abandons ----------------------------------------------------
        no_activity = [r for r in records if not r["projects_users"]]
        no_locations = [r for r in records if not r["locations"]]
        print(f"\nABANDONS / COMPTES VIDES")
        print(f"  zéro projet          : {len(no_activity):>4} ({len(no_activity) / n:.1%})")
        print(f"  zéro log de présence : {len(no_locations):>4} ({len(no_locations) / n:.1%})")
        print("  -> à exclure ou à traiter comme classe à part, pas comme des 0")

        # ---- qualité des locations ---------------------------------------
        n_loc = sum(len(r["locations"]) for r in records)
        n_open = sum(
            1 for r in records for loc in r["locations"] if not loc.get("end_at")
        )
        print(f"\nLOCATIONS")
        print(f"  sessions totales     : {n_loc}")
        print(f"  end_at manquant      : {n_open} ({n_open / n_loc:.1%})" if n_loc else "")
        print("  -> sessions ouvertes : à imputer ou à ignorer, sinon logtime faussé")

        hours = [logtime_hours(r["locations"]) for r in records if r["locations"]]
        if hours:
            hours.sort()
            print(f"  logtime médian       : {median(hours):.0f} h")
            print(f"  p10 / p90            : {hours[len(hours) // 10]:.0f} h / "
                  f"{hours[len(hours) * 9 // 10]:.0f} h")

        # ---- projets -----------------------------------------------------
        statuses = Counter(
            pu.get("status")
            for r in records
            for pu in r["projects_users"]
        )
        print(f"\nPROJETS — répartition des statuts")
        for status, count in statuses.most_common():
            print(f"  {str(status):<24} {count}")

        marked = [
            pu for r in records for pu in r["projects_users"] if pu.get("marked_at")
        ]
        print(f"  avec marked_at        : {len(marked)} "
              f"({len(marked) / max(1, sum(statuses.values())):.1%})")
        print("  -> marked_at est ta seule source pour reconstruire les timelines")

        # ---- corrections -------------------------------------------------
        given = [len(r["scale_teams_as_corrector"]) for r in records]
        received = [len(r["scale_teams_as_corrected"]) for r in records]
        print(f"\nCORRECTIONS")
        print(f"  données  — médiane : {median(given):.0f}")
        print(f"  reçues   — médiane : {median(received):.0f}")

        # signal brut : différence de comportement entre retenus et non retenus
        for label, name in ((1, "retenus"), (0, "non retenus")):
            grp = [r for r in records if r["retenu"] == label]
            if not grp:
                continue
            g = median(len(r["scale_teams_as_corrector"]) for r in grp)
            h = [logtime_hours(r["locations"]) for r in grp if r["locations"]]
            print(f"  {name:<12} : {g:.0f} corrections données, "
                  f"{median(h) if h else 0:.0f} h de log (médianes)")

        # ---- fenêtre temporelle ------------------------------------------
        begins = [parse_dt(r.get("pool_begin_at")) for r in records]
        begins = [b for b in begins if b]
        if begins:
            span = (max(begins) - min(begins)).days
            print(f"\nFENÊTRE")
            print(f"  début le plus tôt / tard : écart de {span} jours")
            if span > 7:
                print("  [!] écart important : tu as peut-être mélangé deux sessions")


if __name__ == "__main__":
    main()
