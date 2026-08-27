# Envoyer l'application d'entraînement sur Vercel

Deux chemins, selon que le projet Vercel est relié au dépôt ou non.

## 1. Projet relié à Git (recommandé)

Dans le projet Vercel : *Settings → Git → Connect Git Repository*, choisir
`Teambull31/BOT_trading` et la branche `claude/new-session-wqnqqb` comme
branche de production. La configuration vit à la racine du dépôt :
`vercel.json`, `api/index.py`, `requirements.txt`, `.python-version`, plus
`.vercelignore` qui écarte le backtest (pandas, scikit-learn, ccxt…) pour que
la fonction sans serveur reste légère.

C'est le meilleur montage ici : l'application change toutes les heures, et
chaque `push` redéploie tout seul.

Attention : ces fichiers ne sont que sur `claude/new-session-wqnqqb`, pas sur
`main`.

## 2. Envoi manuel (ce dossier)

Quand le canal d'envoi ne transporte que quelques fichiers, envoyer les quatre
fichiers de ce dossier tels quels. `api/index.py` va alors chercher
`src/trader` dans l'archive publique de la branche au premier démarrage de
chaque instance, et la déplie dans `/tmp`.

Conséquence utile : l'application déployée **suit la branche**. Un `push`
corrige la prochaine mise en route sans redéploiement — mais un `push` cassé
casse aussi l'application en ligne.

## Après le premier déploiement : rendre l'adresse publique

Par défaut, ce compte protège **toutes** les mises en ligne : l'adresse
renvoie une redirection vers `vercel.com/sso-api` et personne d'autre que le
propriétaire ne peut ouvrir l'application.

Pour l'ouvrir : *Settings → Deployment Protection → Vercel Authentication →
Disabled*, puis *Save*.

Rien à craindre côté données : l'argent est fictif, aucun ordre n'est passé,
aucun courtier n'est connecté, et le compte de l'élève appartient à son
navigateur (le serveur n'en garde qu'une copie de travail jetable dans `/tmp`).
Il n'y a donc aucun secret ni aucune donnée d'utilisateur à protéger derrière
l'authentification.

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
