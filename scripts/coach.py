"""Lance l'application d'entrainement "zero to hero".

    python scripts/coach.py                 # http://127.0.0.1:8000
    python scripts/coach.py --port 8080
    python scripts/coach.py --store /tmp/essai.json

Le serveur ecoute par defaut sur 127.0.0.1 uniquement : l'application n'a
aucune authentification et n'est pas destinee a etre exposee sur un reseau.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--store", default=None, help="fichier du compte fictif")
    args = parser.parse_args()

    import uvicorn

    from trader.webapp.server import create_app

    if args.host not in {"127.0.0.1", "localhost"}:
        print(
            f"ATTENTION : ecoute sur {args.host}. L'application n'a aucune "
            "authentification ; ne l'exposez pas sur un reseau non maitrise.",
            file=sys.stderr,
        )

    print(f"\n  Coach Trading  →  http://{args.host}:{args.port}\n")
    uvicorn.run(create_app(args.store), host=args.host, port=args.port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
