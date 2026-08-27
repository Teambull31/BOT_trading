"""Point d'entree sans serveur pour l'envoi manuel vers Vercel.

Le depot contient deja un `api/index.py` a la racine : c'est celui qu'utilise un
projet Vercel relie a Git, ou les sources sont sur place. Ce fichier-ci sert au
cas ou elles ne le sont pas -- le canal d'envoi manuel disponible ici ne
transporte que quelques fichiers -- et va donc chercher l'arborescence
`src/trader` dans l'archive publique de la branche, une fois par instance.

Trois contraintes d'hebergement expliquent le reste :
  - le disque est en lecture seule sauf /tmp, d'ou le depliage dans /tmp ;
  - /tmp ne survit pas a l'instance : le compte de l'eleve appartient donc au
    navigateur, le serveur n'en garde qu'une copie de travail jetable ;
  - plusieurs instances tournent en parallele : chacune deplie sa propre copie.

L'archive n'etant pas figee sur un commit, l'application deployee suit la
branche : un envoi corrige la prochaine mise en route, sans redeploiement.
"""

from __future__ import annotations

import io
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

BRANCH = "claude/new-session-wqnqqb"
ARCHIVE = f"https://codeload.github.com/Teambull31/BOT_trading/tar.gz/refs/heads/{BRANCH}"

_TMP = Path(tempfile.gettempdir())
_SRC = _TMP / "coach-src"
_MARKER = _SRC / "trader" / "webapp" / "server.py"


def _download() -> None:
    """Deplie `src/trader` de l'archive publique dans /tmp."""
    with urllib.request.urlopen(ARCHIVE, timeout=30) as resp:  # noqa: S310
        archive = io.BytesIO(resp.read())
    with tarfile.open(fileobj=archive, mode="r:gz") as tar:
        keep = []
        for member in tar.getmembers():
            # <racine>/src/trader/... -> trader/...
            parts = member.name.split("/")
            if len(parts) > 3 and parts[1] == "src" and parts[2] == "trader":
                member.name = "/".join(parts[2:])
                keep.append(member)
        _SRC.mkdir(parents=True, exist_ok=True)
        tar.extractall(_SRC, members=keep, filter="data")


def _sources() -> Path:
    """Chemin d'import : les sources locales si elles sont la, sinon l'archive."""
    local = Path(__file__).resolve().parent.parent / "src"
    if (local / "trader" / "webapp" / "server.py").exists():
        return local
    if not _MARKER.exists():
        _download()
    if not _MARKER.exists():
        raise RuntimeError(f"sources introuvables apres depliage de {ARCHIVE}")
    return _SRC


sys.path.insert(0, str(_sources()))

from trader.webapp.server import create_app  # noqa: E402

app = create_app(accounts_dir=_TMP / "coach-accounts")
