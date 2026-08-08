"""
Étape 1 : collecter les piscines déclarées dans config.POOLS.

    python scripts/collect.py              # tout
    python scripts/collect.py --train      # sessions d'entraînement seulement
    python scripts/collect.py --target     # piscine cible seulement
    python scripts/collect.py --dry-run    # taille des cohortes, sans collecter

Idempotent et reprenable : Ctrl-C sans risque, relance sans rien reperdre.
Sortie : data/raw/<tag>/<uid_hash>.json, un fichier par piscineux.
Aucun login n'est écrit en clair.
"""

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from api42.client import Client42, make_client


# Champs conservés par bloc. Tout le reste est jeté à l'écriture.
# Deux raisons : le volume (questions_with_answers et feedbacks pèsent
# lourd sur 550k évaluations), et la confidentialité — les payloads
# contiennent les logins EN CLAIR des camarades correcteurs/corrigés,
# qui ne sont couverts par aucune pseudonymisation.
KEEP = {
    "projects_users": (
        "id", "status", "final_mark", "marked_at", "validated?",
        "occurrence", "cursus_ids", "created_at", "updated_at",
    ),
    "locations": ("begin_at", "end_at", "campus_id", "host"),
    "scale_teams_as_corrector": (
        "id", "begin_at", "filled_at", "final_mark", "truant", "scale_id",
    ),
    "scale_teams_as_corrected": (
        "id", "begin_at", "filled_at", "final_mark", "truant", "scale_id",
    ),
}


def slim(block: str, items: list[dict]) -> list[dict]:
    """Réduit un bloc à l'essentiel, sans données personnelles de tiers."""
    keep = KEEP[block]
    out = []
    for item in items:
        row = {k: item.get(k) for k in keep if k in item}

        # nom du projet : nécessaire pour isoler les exams
        proj = item.get("project")
        if isinstance(proj, dict):
            row["project"] = {"id": proj.get("id"), "name": proj.get("name")}

        # flag d'évaluation (Ok / Cheat / Empty work / Crash...)
        flag = item.get("flag")
        if isinstance(flag, dict):
            row["flag"] = {"id": flag.get("id"), "name": flag.get("name")}

        # le TEXTE des commentaires est jeté : il peut nommer des tiers.
        # On garde seulement sa longueur, qui suffit comme proxy d'effort.
        if "comment" in item:
            row["comment_len"] = len(item.get("comment") or "")
        if "feedback" in item:
            row["feedback_len"] = len(item.get("feedback") or "")

        # nombre de personnes corrigées, sans leurs identités
        if "correcteds" in item:
            row["n_correcteds"] = len(item.get("correcteds") or [])

        out.append(row)
    return out


def pseudonymize(login: str) -> str:
    if not config.HASH_SALT:
        raise SystemExit("HASH_SALT vide dans .env — génère-le avant de collecter.")
    return hashlib.sha256(f"{config.HASH_SALT}:{login}".encode()).hexdigest()[:16]


def fetch_cohort(client: Client42, pool: config.Pool) -> list[dict]:
    rows = client.get_all(
        f"/v2/cursus/{config.POOL_CURSUS_ID}/cursus_users",
        params={
            "filter[campus_id]": pool.campus_id,
            "range[begin_at]": f"{pool.start},{pool.end}",
        },
    )
    keep = []
    reasons: dict[str, int] = {"staff": 0, "anonymisé": 0, "sans login": 0}
    has_staff_key = any("staff?" in (r.get("user") or {}) for r in rows)

    for row in rows:
        user = row.get("user") or {}
        login = user.get("login")
        if not login:
            reasons["sans login"] += 1
        elif user.get("staff?"):
            reasons["staff"] += 1
        elif str(login).startswith("3b3"):
            reasons["anonymisé"] += 1
        else:
            keep.append(row)
            continue

    detail = ", ".join(f"{n} {k}" for k, n in reasons.items() if n)
    print(f"  cohorte : {len(keep)} piscineux" + (f" ({detail} écartés)" if detail else ""))

    if rows and not has_staff_key:
        # sans cette alerte, .get("staff?") renverrait None pour tout le monde
        # et le filtre ne ferait rien du tout, sans erreur
        print("  [!] clé 'staff?' absente du payload — le filtre staff est inopérant")

    return keep


def fetch_user(client: Client42, user_id: int, campus_id: int) -> dict:
    # use_cache=False : le fichier par utilisateur dans data/raw assure déjà
    # la reprise. Cacher en plus les réponses HTTP doublerait l'espace disque
    # pour ne gagner que la reprise en cours d'utilisateur.
    def fetch(path: str, params: dict | None = None) -> list[dict]:
        return client.get_all(path, params=params, use_cache=False)

    return {
        "projects_users": slim(
            "projects_users",
            fetch(f"/v2/users/{user_id}/projects_users"),
        ),
        "locations": slim(
            "locations",
            fetch(f"/v2/users/{user_id}/locations",
                  {"filter[campus_id]": campus_id}),
        ),
        "scale_teams_as_corrector": slim(
            "scale_teams_as_corrector",
            fetch(f"/v2/users/{user_id}/scale_teams/as_corrector"),
        ),
        "scale_teams_as_corrected": slim(
            "scale_teams_as_corrected",
            fetch(f"/v2/users/{user_id}/scale_teams/as_corrected"),
        ),
        "cursus_users": fetch(f"/v2/users/{user_id}/cursus_users"),
    }


def compute_label(cursus_users: list[dict], pool: config.Pool) -> dict:
    """
    Sur une piscine cible, le label n'existe PAS encore : on écrit None,
    surtout pas 0. Un 0 ici serait un faux négatif qui contaminerait
    silencieusement l'entraînement.
    """
    if pool.is_target:
        return {"retenu": None, "main_cursus_begin_at": None, "label_status": "pending"}

    for cu in cursus_users:
        if (cu.get("cursus") or {}).get("id") == config.MAIN_CURSUS_ID:
            return {
                "retenu": 1,
                "main_cursus_begin_at": cu.get("begin_at"),
                "label_status": "settled",
            }
    return {"retenu": 0, "main_cursus_begin_at": None, "label_status": "settled"}


def collect_pool(client: Client42, pool: config.Pool, dry_run: bool,
                 limit: int | None = None) -> None:
    print(f"\n{'=' * 62}\n  {pool.tag}  [{pool.role}]  {pool.start} -> {pool.end}\n{'=' * 62}")

    cohort = fetch_cohort(client, pool)
    if dry_run or not cohort:
        return
    if limit:
        cohort = cohort[:limit]
        print(f"  --limit {limit} : échantillon seulement")

    out_dir = config.RAW_DIR / pool.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    total = len(cohort)
    started = time.time()
    n_done = n_skipped = 0

    for i, row in enumerate(cohort, 1):
        user = row["user"]
        uid_hash = pseudonymize(user["login"])
        target = out_dir / f"{uid_hash}.json"

        if target.exists():
            n_skipped += 1
            continue

        payload = fetch_user(client, user["id"], pool.campus_id)
        record = {
            "uid_hash": uid_hash,
            "pool_tag": pool.tag,
            "pool_role": pool.role,
            "campus_id": pool.campus_id,
            "pool_begin_at": row.get("begin_at"),
            "pool_end_at": row.get("end_at"),
            "pool_level": row.get("level"),
            "collected_at": datetime.now(timezone.utc).isoformat(),
            **compute_label(payload.pop("cursus_users"), pool),
            **payload,
        }

        tmp = target.with_suffix(".tmp")
        tmp.write_text(json.dumps(record), encoding="utf-8")
        tmp.replace(target)
        n_done += 1

        if n_done % 10 == 0 or i == total:
            elapsed = time.time() - started
            rate = n_done / elapsed if elapsed else 0
            eta = (total - i) / rate / 60 if rate else 0
            print(
                f"  [{i}/{total}] {n_done} collectés, {n_skipped} en cache | "
                f"{client.n_requests} req | ~{eta:.0f} min"
            )

    print(f"  -> {n_done} nouveaux, {n_skipped} déjà présents")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="sessions d'entraînement")
    parser.add_argument("--target", action="store_true", help="piscine cible")
    parser.add_argument("--dry-run", action="store_true", help="compte sans collecter")
    parser.add_argument("--limit", type=int, default=None,
                        help="n premiers piscineux par session (validation rapide)")
    parser.add_argument("--campus", default=None,
                        help="restreint à certains campus, ex: 60 ou 60,31. "
                             "Le cache est conservé : étendre plus tard ne "
                             "recollecte pas ce qui est déjà là.")
    args = parser.parse_args()

    if args.train:
        pools = config.TRAIN_POOLS
    elif args.target:
        pools = config.TARGET_POOLS
    else:
        pools = config.POOLS

    if args.campus:
        wanted = {int(c.strip()) for c in args.campus.split(",")}
        pools = [p for p in pools if p.campus_id in wanted]

    if not pools:
        raise SystemExit("Aucune piscine sélectionnée. Vérifie config.POOLS.")

    n_train = sum(1 for p in pools if p.role == "train")
    if n_train < 8:
        print(f"[info] {n_train} sessions d'entraînement seulement. La validation")
        print("       leave-one-pool-out sera peu stable ; privilégie une")
        print("       régression logistique à un modèle à base d'arbres.\n")

    client = make_client()

    for pool in pools:
        collect_pool(client, pool, args.dry_run, args.limit)

    n_train = sum(1 for p in pools if p.role == "train")
    print(f"\nTerminé. {len(pools)} sessions ({n_train} exploitables pour l'entraînement).")
    if not args.dry_run:
        print("Prochaine étape : python scripts/explore.py")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrompu. Relance pour reprendre où tu en étais.")
