"""
Affiche les limites réelles de l'API, pour calibrer config.REQ_PER_HOUR.

    python scripts/check_ratelimit.py

La valeur par défaut (1150/h) est prudente. Si ton application dispose
d'un plafond supérieur, l'augmenter réduit proportionnellement la durée
de collecte.
"""

import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from api42.client import API_ROOT, make_client


def main() -> None:
    client = make_client(verbose=False)
    print(f"{len(client.creds)} application(s) configurée(s)\n")

    hourly_total = 0

    for cred in client.creds:
        resp = requests.get(
            f"{API_ROOT}/v2/campus",
            headers={"Authorization": f"Bearer {cred.token(client.session, False)}"},
            params={"page[size]": 1},
            timeout=30,
        )
        print(f"--- {cred.name} (statut {resp.status_code}) ---")

        headers = {
            k.lower(): v for k, v in resp.headers.items()
            if "ratelimit" in k.lower() or "retry" in k.lower()
        }
        if not headers:
            print("  aucun header de rate limit renvoyé")
            continue
        for key, value in headers.items():
            print(f"  {key}: {value}")

        hourly = headers.get("x-hourly-ratelimit-limit", "")
        if hourly.isdigit():
            hourly_total += int(hourly)

    print(f"\nConfiguration actuelle : {config.REQ_PER_SECOND}/s, "
          f"{config.REQ_PER_HOUR}/h PAR application")

    if hourly_total:
        per_app = hourly_total // len(client.creds)
        safe = int(per_app * 0.95)
        print(f"\nSuggestion pour config.py :")
        print(f"  REQ_PER_HOUR = {safe}      # par application")
        print(f"\nDébit cumulé : {safe * len(client.creds)} req/h")
        remaining = 5385 * 5.5
        print(f"Collecte complète (~{remaining:.0f} req) : "
              f"~{remaining / (safe * len(client.creds)):.0f} h")


if __name__ == "__main__":
    main()
