"""
Client pour l'API v2 de l'intra 42.

Garanties :
  - le rate limit est respecté PAR APPLICATION (compteur seconde + heure)
  - plusieurs jeux d'identifiants peuvent être fournis : le client bascule
    automatiquement sur celui dont le quota se libère le plus tôt
  - toute réponse est cachée sur disque : relancer ne recollecte rien
  - chaque token est rafraîchi indépendamment avant expiration
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import deque
from pathlib import Path
from typing import Any

import requests

API_ROOT = "https://api.intra.42.fr"
TOKEN_URL = f"{API_ROOT}/oauth/token"


class RateLimitError(RuntimeError):
    pass


class RateLimiter:
    """Double limite : X requêtes/seconde et Y requêtes/heure."""

    def __init__(self, per_second: float = 2.0, per_hour: int = 1140):
        self.min_interval = 1.0 / per_second
        self.per_hour = per_hour
        self._calls: deque[float] = deque()
        self._last = 0.0
        self._lock = threading.Lock()

    def _purge(self, now: float) -> None:
        while self._calls and now - self._calls[0] > 3600:
            self._calls.popleft()

    def wait_time(self) -> float:
        """Secondes à attendre avant de pouvoir émettre, sans bloquer."""
        with self._lock:
            now = time.monotonic()
            self._purge(now)
            wait = max(0.0, self.min_interval - (now - self._last))
            if len(self._calls) >= self.per_hour:
                wait = max(wait, 3600 - (now - self._calls[0]) + 1)
            return wait

    def record(self) -> None:
        with self._lock:
            now = time.monotonic()
            self._calls.append(now)
            self._last = now

    def used(self) -> int:
        with self._lock:
            self._purge(time.monotonic())
            return len(self._calls)


class Credential:
    """Un jeu d'identifiants, avec son quota et son token propres."""

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        name: str,
        per_second: float,
        per_hour: int,
    ):
        self.client_id = client_id
        self.client_secret = client_secret
        self.name = name
        self.limiter = RateLimiter(per_second, per_hour)
        self.n_requests = 0
        self.cooldown_until = 0.0  # renseigné après un 429
        self._token: str | None = None
        self._expires_at = 0.0

    def wait_time(self) -> float:
        cooldown = max(0.0, self.cooldown_until - time.monotonic())
        return max(self.limiter.wait_time(), cooldown)

    def invalidate_token(self) -> None:
        self._token = None

    def token(self, session: requests.Session, verbose: bool = True) -> str:
        if self._token and time.time() < self._expires_at - 60:
            return self._token

        resp = session.post(
            TOKEN_URL,
            data={
                "grant_type": "client_credentials",
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            },
            timeout=30,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"[{self.name}] échec de l'authentification ({resp.status_code}). "
                f"Vérifie les identifiants. Réponse : {resp.text[:200]}"
            )
        payload = resp.json()
        self._token = payload["access_token"]
        self._expires_at = time.time() + payload.get("expires_in", 7200)
        if verbose:
            print(f"[auth] {self.name} : token obtenu "
                  f"({payload.get('expires_in', 7200) // 60} min)")
        return self._token


class Client42:
    def __init__(
        self,
        credentials: list[tuple[str, str]] | None = None,
        client_id: str | None = None,
        client_secret: str | None = None,
        cache_dir: Path | str = "cache/http",
        per_second: float = 1.8,
        per_hour: int = 1140,
        verbose: bool = True,
    ):
        # accepte soit une liste, soit l'ancien couple id/secret
        pairs = credentials or ([(client_id, client_secret)] if client_id else [])
        pairs = [(i, s) for i, s in pairs if i and s]
        if not pairs:
            raise ValueError(
                "Aucun identifiant API. Copie .env.example vers .env et "
                "remplis FT_CLIENT_ID / FT_CLIENT_SECRET."
            )

        self.creds = [
            Credential(i, s, f"app{n}", per_second, per_hour)
            for n, (i, s) in enumerate(pairs, 1)
        ]
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.session = requests.Session()

        self.n_cache_hits = 0
        self.n_requests = 0

        if verbose and len(self.creds) > 1:
            print(f"[auth] {len(self.creds)} applications en rotation "
                  f"({len(self.creds) * per_hour} req/h cumulées)")

    # --------------------------------------------------------- rotation

    def _pick(self) -> Credential:
        """
        Retourne le credential disponible le plus tôt, après attente si
        nécessaire. Avec plusieurs applications, l'attente d'une seconde
        imposée à l'une est mise à profit par l'autre.
        """
        best = min(self.creds, key=lambda c: c.wait_time())
        wait = best.wait_time()
        if wait > 0:
            if wait > 30 and self.verbose:
                print(f"[rate] quotas épuisés, pause {wait / 60:.1f} min")
            time.sleep(wait)
        return best

    def quota_report(self) -> str:
        return " | ".join(
            f"{c.name}: {c.limiter.used()}/{c.limiter.per_hour}" for c in self.creds
        )

    # ------------------------------------------------------------ cache

    def _cache_path(self, path: str, params: dict[str, Any]) -> Path:
        key = json.dumps(
            {"path": path, "params": {k: str(v) for k, v in sorted(params.items())}},
            sort_keys=True,
        )
        digest = hashlib.sha256(key.encode()).hexdigest()
        sub = self.cache_dir / digest[:2]
        sub.mkdir(parents=True, exist_ok=True)
        return sub / f"{digest}.json"

    # --------------------------------------------------------- requêtes

    def get(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        use_cache: bool = True,
        max_retries: int = 6,
    ) -> Any:
        """GET sur un endpoint. Retourne le JSON parsé, ou None sur un 404."""
        params = dict(params or {})
        cache_file = self._cache_path(path, params)

        if use_cache and cache_file.exists():
            self.n_cache_hits += 1
            with cache_file.open(encoding="utf-8") as fh:
                return json.load(fh)

        url = path if path.startswith("http") else f"{API_ROOT}{path}"
        backoff = 2.0

        for attempt in range(max_retries):
            cred = self._pick()
            cred.limiter.record()
            headers = {"Authorization": f"Bearer {cred.token(self.session, self.verbose)}"}

            try:
                resp = self.session.get(url, headers=headers, params=params, timeout=60)
            except requests.RequestException as exc:
                if attempt == max_retries - 1:
                    raise
                print(f"[warn] erreur réseau ({exc}), retry dans {backoff:.0f}s")
                time.sleep(backoff)
                backoff = min(backoff * 2, 120)
                continue

            self.n_requests += 1
            cred.n_requests += 1

            if resp.status_code == 200:
                data = resp.json()
                tmp = cache_file.with_suffix(".tmp")
                with tmp.open("w", encoding="utf-8") as fh:
                    json.dump(data, fh)
                tmp.replace(cache_file)  # écriture atomique
                return data

            if resp.status_code == 404:
                return None

            if resp.status_code == 401:
                cred.invalidate_token()
                continue

            if resp.status_code == 429:
                # on met CETTE application en pause ; les autres continuent
                retry_after = float(resp.headers.get("Retry-After", 5))
                cred.cooldown_until = time.monotonic() + retry_after + 1
                if self.verbose:
                    print(f"[rate] 429 sur {cred.name}, "
                          f"pause {retry_after:.0f}s"
                          + (" (bascule)" if len(self.creds) > 1 else ""))
                continue

            if resp.status_code >= 500:
                print(f"[warn] {resp.status_code} serveur, retry dans {backoff:.0f}s")
                time.sleep(backoff)
                backoff = min(backoff * 2, 120)
                continue

            raise RuntimeError(f"GET {url} -> {resp.status_code} : {resp.text[:200]}")

        raise RateLimitError(f"Abandon après {max_retries} tentatives sur {url}")

    def get_all(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        page_size: int = 100,
        max_pages: int = 100,
        use_cache: bool = True,
    ) -> list[dict]:
        """
        Récupère toutes les pages d'un endpoint paginé.

        use_cache=False désactive le cache HTTP pour cet appel : utile
        quand la reprise est déjà assurée en aval (un fichier par
        utilisateur dans data/raw), auquel cas cacher les réponses
        brutes ne ferait que doubler l'espace disque.
        """
        params = dict(params or {})
        params["page[size]"] = page_size
        out: list[dict] = []

        for page in range(1, max_pages + 1):
            params["page[number]"] = page
            chunk = self.get(path, params, use_cache=use_cache)
            if not chunk:
                break
            if not isinstance(chunk, list):
                raise TypeError(f"{path} n'est pas un endpoint paginé ({type(chunk)})")
            out.extend(chunk)
            if len(chunk) < page_size:
                break
        else:
            print(f"[warn] {path} : max_pages={max_pages} atteint, tronqué")

        return out


def make_client(verbose: bool = True) -> Client42:
    """Construit le client à partir de config.py. À utiliser partout."""
    import config

    return Client42(
        credentials=config.CREDENTIALS,
        cache_dir=config.CACHE_DIR,
        per_second=config.REQ_PER_SECOND,
        per_hour=config.REQ_PER_HOUR,
        verbose=verbose,
    )
