# Envoyer l'application d'entraînement sur Vercel

Deux chemins, selon que le projet Vercel est relié au dépôt ou non.

## 1. Projet relié à Git (recommandé)

Dans le projet Vercel : *Settings → Git → Connect Git Repository*, choisir
`Teambull31/BOT_trading` et la branche `claude/new-session-wqnqqb` comme
branche de production. La configuration vit à la racine du dépôt :
`vercel.json`, `main.py`, `requirements.txt`, `.python-version`, plus
`.vercelignore` qui écarte le backtest (pandas, scikit-learn, ccxt…) pour que
la fonction sans serveur reste légère.

C'est le meilleur montage ici : l'application change toutes les heures, et
chaque `push` redéploie tout seul. C'est aussi ce qui empêche un nouveau projet
d'apparaître à chaque envoi : un projet relié se redéploie, il ne se duplique
pas.

Attention : ces fichiers ne sont que sur `claude/new-session-wqnqqb`, pas sur
`main`.

## 2. Envoi manuel (ce dossier)

Quand le canal d'envoi ne transporte que quelques fichiers, envoyer les
fichiers de ce dossier tels quels (`.python-version` compris). `main.py` va alors chercher `src/trader`
dans l'archive publique de la branche au premier démarrage de chaque instance,
et la déplie dans `/tmp`.

Conséquence utile : l'application déployée **suit la branche**. Un `push`
corrige la prochaine mise en route sans redéploiement — mais un `push` cassé
casse aussi l'application en ligne.

## Pourquoi le point d'entrée est à la racine, et sans réécriture

Vercel connaît deux montages Python, qui ne se mélangent pas :

- un fichier placé dans `api/` est une fonction **adressée par son chemin** :
  `api/index.py` ne répond qu'à l'adresse `/api/index`, et rien d'autre ;
- une application ASGI déclarée **à la racine** (`main.py`) reçoit tout le
  trafic, chemin d'origine compris, et c'est FastAPI qui choisit la route.

Le dépôt a longtemps combiné les deux : `api/index.py` *plus* une réécriture
attrape-tout `"/(.*)" → "/api/index"`. Or une réécriture **remplace** le chemin
au lieu de le conserver (c'est précisément ce que corrige la transformation
`request.path` de Vercel, qui n'aurait aucun objet sinon). L'application
recevait donc toujours `/api/index` : le préfixe `/api/` déclenchait le
contrôle d'en-tête de compte — d'où le `{"detail":"identifiant de compte absent
ou invalide"}` affiché sur **toutes** les adresses — et aucune route ne
correspondait jamais.

D'où la forme actuelle, celle que documente Vercel pour FastAPI : un `main.py`
à la racine déclaré sous `functions`, **aucune clé `rewrites`**. Ne pas
réintroduire de réécriture attrape-tout : elle recasserait tout.

## Ce que l'hébergement change

- Disque en lecture seule sauf `/tmp`, qui ne survit pas à l'instance : le
  compte est détenu par le navigateur (`localStorage`), et le bandeau de
  l'application propose *Enregistrer une sauvegarde* / *Restaurer une
  sauvegarde*.
- Plusieurs instances tournent en parallèle : chacune déplie sa propre copie
  des sources et travaille sur sa propre copie du compte.
- Les cours restent réels : le serveur interroge l'API de cotation à chaque
  demande. Si la sortie réseau était fermée, ce sont les cours qui
  tomberaient — l'entraînement en conditions réelles n'aurait plus de sens.

## Si l'adresse renvoie vers `vercel.com/sso-api`

C'est la protection des mises en ligne, pas un défaut de l'application :
*Settings → Deployment Protection → Vercel Authentication → Disabled*, puis
*Save*. Sur les projets actuels elle est déjà désactivée — vérifier avant de
soupçonner ce point.

Rien à craindre côté données de toute façon : l'argent est fictif, aucun ordre
n'est passé, aucun courtier n'est connecté, et le compte de l'élève appartient
à son navigateur (le serveur n'en garde qu'une copie de travail jetable dans
`/tmp`). Il n'y a donc aucun secret ni aucune donnée d'utilisateur à protéger
derrière l'authentification.
