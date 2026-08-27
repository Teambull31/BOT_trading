"""Construit le paquet deployable sur Vercel a partir des sources du depot.

Vercel ne recoit ici les fichiers qu'en un seul envoi, et le mode entrainement
pese ~140 Ko de source. Plutot que de recopier l'arborescence a la main -- avec
le risque d'oublier un fichier ou de deployer une version differente de celle du
depot -- ce script embarque l'arborescence reelle, compressee, dans le point
d'entree sans serveur. Le paquet est donc toujours le reflet exact du commit
courant, et regenerable a l'identique (horodatages neutralises).

    python scripts/build_vercel_payload.py <dossier_de_sortie>

Le dossier de sortie contient alors les quatre fichiers a envoyer a Vercel :
api/index.py, vercel.json, requirements.txt et .python-version.
"""

from __future__ import annotations

import base64
import io
import lzma
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Uniquement ce que le serveur web importe reellement. Le reste du depot
# (pandas, scikit-learn, ccxt...) appartient a la chaine de backtest.
BUNDLED = [
    "trader/__init__.py",
    "trader/logging_setup.py",
    "trader/equities/__init__.py",
    "trader/equities/symbols.py",
    "trader/coach/__init__.py",
    "trader/coach/quotes.py",
    "trader/coach/account.py",
    "trader/coach/curriculum.py",
    "trader/coach/advisor.py",
    "trader/coach/debrief.py",
    "trader/webapp/__init__.py",
    "trader/webapp/server.py",
    "trader/webapp/static/index.html",
    "trader/webapp/static/app.css",
    "trader/webapp/static/app.js",
]

LOADER = '''"""Point d'entree sans serveur -- GENERE par scripts/build_vercel_payload.py.

Ne pas editer a la main : relancer le script apres toute modification des
sources. L'arborescence `src/trader` du depot est embarquee ci-dessous sous
forme compressee, puis depliee dans /tmp au demarrage a froid de la fonction.

Deux contraintes d'hebergement expliquent le reste :
  - le disque est en lecture seule sauf /tmp, et /tmp ne survit pas a
    l'instance : le compte de l'eleve appartient donc au navigateur, le serveur
    n'en garde qu'une copie de travail jetable ;
  - plusieurs instances tournent en parallele : chacune deplie sa propre copie.
"""

from __future__ import annotations

import base64
import io
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path

_BUNDLE = """\
__BUNDLE__"""

# Filet de securite : si la charge embarquee ci-dessus etait abimee en route,
# la meme arborescence reste telechargeable depuis le depot public.
_FALLBACK_URL = "__FALLBACK__"

_TMP = Path(tempfile.gettempdir())
_SRC = _TMP / "coach-src"


def _from_bundle() -> None:
    """Source embarquee : aucun reseau, donc aucun point de panne au demarrage."""
    # b64decode ignore les retours a la ligne du litteral ci-dessus.
    raw = io.BytesIO(base64.b64decode(_BUNDLE))
    with tarfile.open(fileobj=raw, mode="r:xz") as tar:
        tar.extractall(_SRC, filter="data")


def _from_github() -> None:
    """Repli : archive publique de la branche, reduite au paquet `trader`."""
    with urllib.request.urlopen(_FALLBACK_URL, timeout=30) as resp:
        raw = io.BytesIO(resp.read())
    with tarfile.open(fileobj=raw, mode="r:gz") as tar:
        keep = []
        for member in tar.getmembers():
            parts = member.name.split("/")
            if len(parts) > 3 and parts[1] == "src" and parts[2] == "trader":
                member.name = "/".join(parts[2:])
                keep.append(member)
        tar.extractall(_SRC, members=keep, filter="data")


def _unpack() -> Path:
    """Deplie les sources, une fois par instance."""
    marker = _SRC / "trader" / "webapp" / "server.py"
    if marker.exists():
        return _SRC
    _SRC.mkdir(parents=True, exist_ok=True)
    try:
        _from_bundle()
    except Exception:
        _from_github()
    if not marker.exists():
        raise RuntimeError("sources introuvables apres depliage")
    return _SRC


sys.path.insert(0, str(_unpack()))

from trader.webapp.server import create_app  # noqa: E402

app = create_app(accounts_dir=_TMP / "coach-accounts")
'''

FALLBACK_URL = (
    "https://codeload.github.com/Teambull31/BOT_trading/tar.gz/"
    "refs/heads/claude/new-session-wqnqqb"
)

VERCEL_JSON = """{
  "$schema": "https://openapi.vercel.sh/vercel.json",
  "functions": {
    "api/index.py": {
      "maxDuration": 60
    }
  },
  "rewrites": [{ "source": "/(.*)", "destination": "/api/index" }]
}
"""


def build_bundle() -> str:
    """Archive tar.xz deterministe des sources hebergees, en base64."""
    buf = io.BytesIO()
    # mtime=0 et tri des noms : deux executions sur le meme commit donnent
    # exactement le meme paquet, donc le meme deploiement. xz plutot que gzip
    # parce que le paquet doit tenir dans un seul envoi vers Vercel.
    with (
        lzma.LZMAFile(buf, mode="wb", preset=9) as xz,
        tarfile.open(fileobj=xz, mode="w") as tar,
    ):
        for name in sorted(BUNDLED):
            data = (ROOT / "src" / name).read_bytes()
            info = tarfile.TarInfo(name)
            info.size = len(data)
            info.mtime = 0
            info.mode = 0o644
            info.uid = info.gid = 0
            info.uname = info.gname = ""
            tar.addfile(info, io.BytesIO(data))
    return base64.b64encode(buf.getvalue()).decode("ascii")


def main(out: Path) -> None:
    (out / "api").mkdir(parents=True, exist_ok=True)
    bundle = build_bundle()
    # Replie sur 120 colonnes : un litteral de 47 000 caracteres sur une seule
    # ligne est illisible et fragile a transporter.
    folded = "\n".join(bundle[i : i + 120] for i in range(0, len(bundle), 120))
    body = LOADER.replace("__BUNDLE__", folded + "\n").replace(
        "__FALLBACK__", FALLBACK_URL
    )
    (out / "api" / "index.py").write_text(body, encoding="utf-8")
    (out / "vercel.json").write_text(VERCEL_JSON, encoding="utf-8")
    (out / "requirements.txt").write_text(
        (ROOT / "requirements.txt").read_text(encoding="utf-8"), encoding="utf-8"
    )
    (out / ".python-version").write_text("3.12\n", encoding="utf-8")
    print(f"paquet ecrit dans {out} ({len(bundle)} caracteres de charge utile)")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve())
