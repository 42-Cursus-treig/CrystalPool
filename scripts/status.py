import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config


def main() -> None:
    print(f"{'session':<26} {'rôle':<8} {'collectés':>10}")
    print("-" * 48)

    total = 0
    done_pools = 0

    for pool in config.POOLS:
        d = config.RAW_DIR / pool.tag
        n = len(list(d.glob("*.json"))) if d.exists() else 0
        total += n
        if n:
            done_pools += 1
        bar = "#" * min(20, n // 10) if n else ""
        print(f"{pool.tag:<26} {pool.role:<8} {n:>10}  {bar}")

    n_target = sum(
        len(list((config.RAW_DIR / p.tag).glob("*.json")))
        for p in config.TARGET_POOLS
        if (config.RAW_DIR / p.tag).exists()
    )

    print("-" * 48)
    print(f"{'TOTAL':<26} {'':<8} {total:>10}")
    print(f"\n  sessions entamées : {done_pools}/{len(config.POOLS)}")
    print(f"  cibles collectées : {n_target} piscineux "
          f"sur {len(config.TARGET_POOLS)} sessions")

    remaining = 5385 - total
    if remaining > 0:
        print(f"  restant estimé    : ~{remaining} piscineux "
              f"(~{remaining * 5.5 / 1140:.0f} h)")
    else:
        print("  collecte a priori terminée")


if __name__ == "__main__":
    main()