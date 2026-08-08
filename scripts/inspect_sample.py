import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config

# Champs dont features.py aura besoin, par bloc.
EXPECTED = {
    "projects_users": ["status", "final_mark", "marked_at", "validated?", "project"],
    "locations": ["begin_at", "end_at", "campus_id", "host"],
    "scale_teams_as_corrector": ["begin_at", "corrected_at", "final_mark", "correcteds"],
    "scale_teams_as_corrected": ["begin_at", "corrected_at", "final_mark", "corrector"],
}


def load_sample(limit_per_pool: int = 5) -> list[dict]:
    records = []
    for pool_dir in sorted(p for p in config.RAW_DIR.glob("*") if p.is_dir()):
        for path in sorted(pool_dir.glob("*.json"))[:limit_per_pool]:
            with path.open(encoding="utf-8") as fh:
                records.append(json.load(fh))
    return records


def report_block(records: list[dict], block: str) -> None:
    items = [it for r in records for it in r.get(block, [])]
    print(f"\n--- {block} ---")
    if not items:
        print("  [!] AUCUN élément — endpoint vide ou filtre trop restrictif")
        return

    print(f"  {len(items)} éléments sur {len(records)} piscineux "
          f"(médiane ~{len(items) // max(1, len(records))}/personne)")

    keys = Counter(k for it in items for k in it)
    for field in EXPECTED[block]:
        n = keys.get(field, 0)
        if n == 0:
            print(f"  [!] '{field}' ABSENT — mon hypothèse est fausse")
            continue
        filled = sum(1 for it in items if it.get(field) is not None)
        flag = "  [!] souvent vide" if filled < len(items) * 0.5 else ""
        print(f"  '{field}' : présent {n}/{len(items)}, renseigné {filled}{flag}")

    extra = [k for k in keys if k not in EXPECTED[block]]
    if extra:
        print(f"  autres champs disponibles : {', '.join(sorted(extra)[:14])}")


def main() -> None:
    records = load_sample()
    if not records:
        raise SystemExit(
            f"Rien dans {config.RAW_DIR}. Lance d'abord :\n"
            "  python scripts/collect.py --limit 5"
        )

    print(f"Échantillon : {len(records)} piscineux sur "
          f"{len(set(r['pool_tag'] for r in records))} sessions")

    # ---- filtre staff : est-ce qu'il fait quelque chose ? --------------
    print("\n=== FILTRE staff? ===")
    print("  Le filtre s'applique au bloc 'user' de la cohorte, absent des")
    print("  fichiers sauvés. Vérification directe ci-dessous via l'API.")

    # ---- blocs de données ---------------------------------------------
    for block in EXPECTED:
        report_block(records, block)

    # ---- statuts de projets -------------------------------------------
    statuses = Counter(
        pu.get("status") for r in records for pu in r.get("projects_users", [])
    )
    print(f"\n--- statuts observés ---")
    for status, n in statuses.most_common():
        print(f"  {str(status):<26} {n}")

    # ---- noms de projets : repérer les exams --------------------------
    names = Counter()
    for r in records:
        for pu in r.get("projects_users", []):
            proj = pu.get("project") or {}
            if proj.get("name"):
                names[proj["name"]] += 1
    print(f"\n--- projets les plus fréquents ({len(names)} distincts) ---")
    for name, n in names.most_common(12):
        print(f"  {name:<30} {n}")
    exams = [n for n in names if "exam" in n.lower()]
    print(f"  exams détectés : {exams if exams else '[!] AUCUN — vérifier le filtre cursus'}")

    # ---- cohérence temporelle -----------------------------------------
    print("\n--- cohérence temporelle ---")
    n_before, n_total = 0, 0
    for r in records:
        pool_begin = r.get("pool_begin_at")
        if not pool_begin:
            continue
        for pu in r.get("projects_users", []):
            if pu.get("marked_at"):
                n_total += 1
                if pu["marked_at"] < pool_begin:
                    n_before += 1
    if n_total:
        print(f"  projets marqués AVANT le début de la piscine : {n_before}/{n_total}")
        if n_before:
            print("  [!] à exclure : ce sont d'anciennes piscines du même compte")

    # ---- label ---------------------------------------------------------
    labels = Counter(r.get("label_status") for r in records)
    print(f"\n--- label ---")
    for status, n in labels.items():
        print(f"  {status}: {n}")


if __name__ == "__main__":
    main()
