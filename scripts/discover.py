import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from api42.client import make_client


def main() -> None:
    client = make_client()

    print("\n=== CAMPUS ===")
    campuses = client.get_all("/v2/campus")
    for c in sorted(campuses, key=lambda x: x.get("country") or ""):
        print(f"  {c['id']:>4}  {c['name']:<24} {c.get('country', '')}")

    print("\n=== CURSUS ===")
    cursus = client.get_all("/v2/cursus")
    for c in sorted(cursus, key=lambda x: x["id"]):
        print(f"  {c['id']:>4}  {c['name']:<32} (slug: {c.get('slug')})")

    print(
        "\nCherche le cursus piscine (souvent un slug contenant 'piscine' ou 'c-piscine')"
        "\net le cursus principal (souvent nommé '42cursus').\n"
        "Reporte les 3 IDs dans config.py : CAMPUS_ID, POOL_CURSUS_ID, MAIN_CURSUS_ID.\n"
    )


if __name__ == "__main__":
    main()
