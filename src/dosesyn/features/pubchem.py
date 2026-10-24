"""Minimal PubChem PUG-REST client with an on-disk JSON cache.

Only the endpoints needed for M2 are implemented.  Every response is cached
under ``<cache_dir>/pubchem`` so that the feature build is reproducible
offline after the first run.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, Optional

PROPERTIES = (
    "Title,SMILES,ConnectivitySMILES,InChIKey,MolecularFormula,MolecularWeight"
)


class PubChemError(RuntimeError):
    pass


class PubChemClient:
    def __init__(
        self,
        cache_dir: str | Path,
        base_url: str = "https://pubchem.ncbi.nlm.nih.gov/rest/pug",
        timeout: int = 30,
        sleep_seconds: float = 0.34,
        user_agent: str = "DoseSyn/0.2",
        max_retries: int = 3,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.sleep_seconds = sleep_seconds
        self.user_agent = user_agent
        self.max_retries = max_retries
        self._last_request = 0.0
        self.n_requests = 0
        self.n_cache_hits = 0

    # ------------------------------------------------------------------
    def _throttle(self) -> None:
        elapsed = time.time() - self._last_request
        if elapsed < self.sleep_seconds:
            time.sleep(self.sleep_seconds - elapsed)
        self._last_request = time.time()

    def _get_json(self, url: str) -> Dict[str, Any]:
        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            self._throttle()
            req = urllib.request.Request(url, headers={"User-Agent": self.user_agent})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    self.n_requests += 1
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                self.n_requests += 1
                body = exc.read().decode("utf-8", "replace")[:300]
                last_error = PubChemError(f"HTTP {exc.code} for {url}: {body}")
                if exc.code == 404:
                    break
                time.sleep(1.5 * attempt)
            except Exception as exc:  # network/timeout/JSON
                last_error = exc
                time.sleep(1.5 * attempt)
        raise PubChemError(f"PubChem request failed: {url}: {last_error}")

    def _cached(self, key: str, fetch) -> Dict[str, Any]:
        path = self.cache_dir / f"{key}.json"
        if path.exists():
            with open(path, "r", encoding="utf-8") as fh:
                self.n_cache_hits += 1
                return json.load(fh)
        payload = fetch()
        tmp = path.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        tmp.replace(path)
        return payload

    # ------------------------------------------------------------------
    def cid_by_name(self, name: str) -> int:
        safe = urllib.parse.quote(name, safe="")
        url = f"{self.base_url}/compound/name/{safe}/cids/JSON"
        data = self._cached(f"name_{safe}", lambda: self._get_json(url))
        cids = data.get("IdentifierList", {}).get("CID", [])
        if not cids:
            raise PubChemError(f"no CID for name {name!r}")
        return int(cids[0])

    def cid_by_inchikey(self, inchikey: str) -> int:
        safe = urllib.parse.quote(inchikey, safe="")
        url = f"{self.base_url}/compound/inchikey/{safe}/cids/JSON"
        data = self._cached(f"inchikey_{safe}", lambda: self._get_json(url))
        cids = data.get("IdentifierList", {}).get("CID", [])
        if not cids:
            raise PubChemError(f"no CID for InChIKey {inchikey!r}")
        return int(cids[0])

    def properties_by_cid(self, cid: int) -> Dict[str, Any]:
        url = f"{self.base_url}/compound/cid/{cid}/property/{PROPERTIES}/JSON"
        data = self._cached(f"cid_{cid}", lambda: self._get_json(url))
        table = data.get("PropertyTable", {}).get("Properties", [])
        if not table:
            raise PubChemError(f"no properties for CID {cid}")
        props = dict(table[0])
        props["CID"] = int(props["CID"])
        return props

    def fetch_by_name(self, name: str) -> Dict[str, Any]:
        cid = self.cid_by_name(name)
        props = self.properties_by_cid(cid)
        props["query_name"] = name
        props["query_type"] = "name"
        return props

    def fetch_by_inchikey(self, inchikey: str) -> Dict[str, Any]:
        cid = self.cid_by_inchikey(inchikey)
        props = self.properties_by_cid(cid)
        props["query_name"] = inchikey
        props["query_type"] = "inchikey"
        return props
