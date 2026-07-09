"""
test_api_key.py — samostalna provjera radi li tvoj Quiver API key.

Key je hardkodiran ispod (API_KEY) — samo pokreni:

    python test_api_key.py

Ne ovisi o ostatku aplikacije (quiver_client.py, app.py itd.) — radi sam za sebe,
treba mu samo `requests`.
"""
from __future__ import annotations

import sys

try:
    import requests
except ImportError:
    print("Nedostaje paket 'requests'. Instaliraj s:  python -m pip install requests")
    sys.exit(1)

# ---- tvoj key, hardkodiran radi brzog testa ------------------------------- #
API_KEY = "33d24d979f11ca7d75d1937cc15dd3ca02a23008"
# ---------------------------------------------------------------------------- #

BASE = "https://api.quiverquant.com/beta"
PROBE_PATH = "live/congresstrading"   # lagan, brz endpoint samo za provjeru ključa


def get_key() -> str:
    # Argument i dalje radi ako ikad želiš testirati drugi key bez editiranja file-a
    if len(sys.argv) > 1:
        return sys.argv[1].strip()
    return API_KEY.strip()


def try_scheme(key: str, scheme: str) -> tuple[int, str]:
    url = f"{BASE}/{PROBE_PATH}"
    headers = {"accept": "application/json", "Authorization": f"{scheme} {key}"}
    try:
        r = requests.get(url, headers=headers, timeout=15)
    except requests.RequestException as exc:
        return -1, f"mrežna greška: {exc}"
    if r.status_code == 200:
        try:
            n = len(r.json())
        except Exception:
            n = "?"
        return r.status_code, f"OK — vraćeno {n} zapisa"
    if r.status_code in (401, 403):
        return r.status_code, "key odbijen (neispravan / istekao)"
    if r.status_code == 429:
        return r.status_code, "rate-limited (previše zahtjeva — key je vjerojatno ispravan)"
    return r.status_code, f"neočekivan status: {r.text[:150]}"


def main() -> int:
    key = get_key()
    if not key:
        print("Nisi unio API key.")
        return 1

    print(f"\nTestiram key koji završava na …{key[-4:] if len(key) >= 4 else key}\n")

    for scheme in ("Token", "Bearer"):
        status, msg = try_scheme(key, scheme)
        symbol = "✅" if status == 200 else ("⏳" if status == 429 else "❌")
        print(f"  {symbol} Authorization: {scheme} <key>  →  [{status}] {msg}")
        if status == 200:
            print(f"\nKLJUČ RADI. Koristi auth shemu: {scheme}")
            return 0
        if status == 429:
            print(f"\nKLJUČ JE VJEROJATNO ISPRAVAN, ali trenutno si rate-limited "
                  f"(shema: {scheme}). Pričekaj malo pa probaj ponovno.")
            return 0

    print("\nKLJUČ NE RADI ni s jednom auth shemom.")
    print("Provjeri: jesi li zalijepio cijeli key bez razmaka, je li aktivan na "
          "https://api.quiverquant.com, i odgovara li tvoj plan ovom endpointu.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())