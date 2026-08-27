# Agent de trading autonome adaptatif

Systeme de trading crypto concu pour **survivre aux changements de regime de
marche**, pas pour maximiser un backtest. Trois principes non negociables :

1. **Aucune strategie n'est eternelle.** Un ensemble de strategies est pondere,
   active et desactive en continu selon le regime detecte et la performance
   mesuree.
2. **Le risk management a toujours le dernier mot.** Le module de risque est
   independant et non contournable : l'executeur refuse tout ordre qui ne porte
   pas un verdict de risque approuve.
3. **Le biais de confirmation est l'ennemi.** Chaque strategie doit produire les
   preuves CONTRE son propre signal, et un module dedie (DevilAdvocate), qui ne
   peut pas etre desactive, cherche activement les raisons de ne pas trader.

Le systeme demarre toujours en **paper trading**. Le passage en argent reel
exige une validation statistique explicite (voir la checklist go-live).

---

## Demarrage rapide

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"

pytest -q                                   # 359 tests, 91 % de couverture
python -m trader.main paper                 # paper trading
python -m trader.main status                # etat courant
python scripts/backtest.py --walk-forward   # backtest walk-forward
```

Avec Docker (trader + watchdog + Prometheus + Grafana) :

```bash
cp .env.example .env    # renseigner TRADER_ALERT_URLS et les cles exchange
docker compose up -d
docker compose logs -f trader
```

---

## Architecture

```
ORCHESTRATOR (boucle principale)
  |
  |-- DATA ENGINE      ingestion ccxt async, normalisation, 44 features
  |-- REGIME DETECTOR  HMM + clustering + regles, vote a 3 methodes
  |-- STRATEGY ENSEMBLE 4 strategies, ponderation dynamique, shadow mode
  |-- DEVIL ADVOCATE   11 controles a charge, non desactivable
  |-- RISK MANAGER     12 controles court-circuitants, limites en dur
  |-- EXECUTION        paper (slippage, frais, fills partiels) ou live ccxt
  |-- ADAPTATION       decay detection, retraining walk-forward
  |-- MONITORING       Prometheus, alertes, dashboard lecture seule
  |-- PERSISTENCE      SQLite/SQLAlchemy, audit trail complet
```

Ordre strict d'un cycle :

```
kill switch -> donnees -> features -> regime -> SORTIES -> ensemble
  -> devil advocate -> risk manager -> execution -> persistence -> adaptation
```

Les sorties sont traitees **avant** les entrees : proteger le capital deja
engage passe avant l'envie d'en engager davantage.

---

## Les limites en dur

Elles sont dans le **code** (`src/trader/config.py`), pas dans la configuration.
Une configuration qui tente de les depasser est refusee **au chargement**, pas a
l'execution :

| Limite | Valeur | Ou |
|---|---|---|
| Exposition par position | 2 % du capital | `HARD_MAX_POSITION_PCT` |
| Drawdown total | -15 % -> kill switch | `HARD_MAX_DRAWDOWN_TOTAL_PCT` |
| Exposition brute totale | 50 % | `HARD_MAX_EXPOSURE_PCT` |
| Poids d'une seule strategie | 40 % | `EnsembleConfig.max_weight_single` |
| DevilAdvocate | non desactivable | `DevilAdvocateConfig.enabled` |
| Nouvelles positions en crise | interdites | `CrisisRegimeConfig` |

Les modifier demande un changement de code source, revu et commite.

---

## Le kill switch

Quatre voies d'arret, dont **trois n'exigent pas que le process principal
fonctionne** :

```bash
touch /tmp/trader_kill                     # 1. sentinelle fichier
curl -X POST http://127.0.0.1:9091/kill    # 2. endpoint HTTP
python -m trader.main kill "raison"        # via la CLI
python -m trader.main watchdog             # 3. watchdog heartbeat + 4. drawdown
```

Le watchdog tourne dans un **conteneur separe** : si le trader se fige avec des
positions ouvertes, il le detecte et arme la sentinelle. Le desarmement est
manuel et exige une confirmation textuelle explicite — aucun code du systeme ne
le fait tout seul.

---

## Configuration

| Fichier | Role |
|---|---|
| `config/default.toml` | valeurs par defaut, mode paper |
| `config/paper.toml` | override paper (sandbox) |
| `config/live.toml` | override live, **verrouille par defaut**, plus conservateur |

Les secrets (cles API, tokens Telegram) passent **uniquement** par variables
d'environnement : `BINANCE_API_KEY`, `BINANCE_API_SECRET`, `TRADER_ALERT_URLS`.

Derriere un proxy sortant, `HTTPS_PROXY` et `SSL_CERT_FILE` sont detectes et
transmis explicitement aux clients ccxt : aiohttp, contrairement a curl, ne lit
pas ces variables tout seul. La verification TLS n'est jamais desactivee.

---

## Passage en live

Deux verrous independants, plus la checklist :

```bash
python scripts/go_live_checklist.py           # 14 criteres, echec en bloc
python -m trader.main live --i-understand-the-risk
```

Le mode live exige **a la fois** une configuration en `mode = "live"` et le
drapeau `--i-understand-the-risk`. Un fichier de configuration oublie ne peut
pas, a lui seul, envoyer de l'argent reel sur le marche.

Trois criteres de la checklist ne sont pas verifiables par du code et exigent
une attestation ecrite dans `artifacts/go_live_manual.json` :

```json
{
  "logs_audited": true,
  "capital_is_expendable": true,
  "alerts_tested": true
}
```

Signer cette attestation est volontairement plus engageant que passer un
drapeau `--force`.

---

## Exploitation

### Surveillance quotidienne

```bash
python -m trader.main status              # equity, drawdown, regime, evenements
curl -s localhost:9092/status | jq        # etat complet en JSON
curl -s localhost:9092/health | jq        # vivacite + heartbeat
curl -s localhost:9090/metrics            # metriques Prometheus
```

Grafana est expose sur `localhost:3000`. Les regles d'alerte Prometheus
(`deploy/alerts.yml`) doublent les alertes internes : si le trader est incapable
d'alerter parce qu'il est fige, Prometheus le voit de l'exterieur.

### Que faire quand...

| Situation | Action |
|---|---|
| Le kill switch s'est arme | Lire `cat /tmp/trader_kill`, auditer les logs, corriger la cause, puis desarmer manuellement |
| Une strategie passe DEAD | Rien d'urgent : son poids est deja a zero et elle continue en shadow. Le retraining se declenche seul |
| Le regime est UNCERTAIN longtemps | Normal en marche indecis. Seules breakout et sentiment tradent, a taille reduite de moitie |
| Aucun trade depuis des jours | Verifier `blocked_reasons` dans les logs : c'est presque toujours un quorum de strategies non atteint, ce qui est le comportement voulu |
| Le slippage reel depasse l'estime | Le modele de cout est faux : le backtest est trop optimiste. Ne pas passer en live |
| CI ou tests rouges | Ne jamais deployer. Le systeme manipule de l'argent |

### Journaux

Logs structures JSON (`logs/trader.log`), et audit trail complet en base :
chaque decision est reconstituable a posteriori (table `decisions` : etape
ensemble, devil advocate, risque, avec le detail chiffre de chaque controle).

---

## Tests

```bash
pytest -q                                  # tout
pytest tests/unit -q                       # unitaires (rapides)
pytest tests/integration -q                # end-to-end paper, sans reseau
pytest tests/backtest -q                   # walk-forward et regimes (lent)
pytest --cov=src/trader --cov-report=term  # couverture
```

Les tests d'integration n'appellent jamais le reseau : un faux exchange sert
des bougies deterministes.

Test le plus important du depot : `test_no_lookahead_on_any_feature`. Il
recalcule les features sur un prefixe des donnees et verifie que les valeurs
sont identiques — la seule preuve empirique qu'aucune feature ne lit le futur.

---

## Limites connues

Enumerees ici plutot que decouvertes en production :

- **Le systeme trade peu.** Le quorum de deux strategies, le seuil de consensus
  et le DevilAdvocate se cumulent. Sur des donnees synthetiques sans structure,
  cela donne ~1 trade par semaine. Atteindre les 50 trades exiges par la
  checklist go-live peut demander bien plus de 30 jours.
- **La strategie `sentiment` est muette sans donnees de derives.** Sans funding
  rate ni open interest, elle reste NEUTRAL, ce qui reduit le pool effectif a
  trois strategies et rend le quorum plus difficile en regime UNCERTAIN.
- **Aucun edge n'est demontre.** Le systeme est un cadre de gestion du risque et
  d'adaptation ; les strategies elles-memes sont classiques et n'ont pas ete
  validees sur donnees reelles. Les backtests fournis tournent sur des series
  synthetiques et ne prouvent rien sur la rentabilite.
- **Un seul exchange a la fois pour l'execution.** L'arbitrage cross-exchange
  est prepare cote donnees (`fetch_cross_exchange_prices`) mais la strategie
  correspondante n'est pas implementee.
- **Pas de gestion multi-actifs correlee.** L'exposition totale est plafonnee,
  mais aucun controle de correlation entre positions ouvertes n'est applique.
- **Binance est geo-bloque depuis certains hebergeurs** (reponse HTTP 451).
  Le systeme fonctionne avec n'importe quel exchange supporte par ccxt : il
  suffit de changer `exchanges.primary` dans la configuration. Kraken a ete
  valide en conditions reelles.

---

## Structure du depot

```
config/           TOML par defaut / paper / live
deploy/           Prometheus, regles d'alerte
scripts/          backtest, paper_trade, go_live_checklist
src/trader/
  config.py       settings Pydantic + LIMITES EN DUR
  models.py       types de domaine partages
  orchestrator.py boucle principale
  main.py         CLI et assemblage
  portfolio.py    cash, positions, drawdowns
  data/           ingester, normalizer, features, store, snapshot
  regime/         detector, volatility, trend
  strategy/       base, ensemble, momentum, mean_revert, breakout, sentiment
  risk/           manager, position_sizer, circuit_breaker, kill_switch
  execution/      executor, paper, slippage
  adaptation/     devil_advocate, decay_detector, evaluator, retrainer
  monitoring/     metrics, alerter, dashboard
  backtest/       engine, walk_forward
tests/            unit, integration, backtest
```

---

*Le capital engage doit etre de l'argent que vous pouvez perdre integralement.*

---

## Simulation actions (module `equities`)

Backtest de suivi de tendance sur actions, independant du moteur crypto mais
soumis aux memes exigences : aucune lecture du futur, frais systematiques,
comparaison au buy & hold.

```bash
python scripts/equity_sim.py --walk-forward         # simulation + validation
python scripts/equity_sim.py --profile offensif     # curseur de risque
python scripts/equity_sim.py --compare-profiles     # tous les profils, memes dates
python scripts/equity_sim.py --stress-universe      # fragilite au choix des titres
python scripts/equity_sim.py --extra NVDA,XOM       # imposer d'autres titres
```

### Profils de risque

| profil | exposition max | intention |
|---|---|---|
| `budget_risque` | 1 % de perte max par trade | borner la perte unitaire, quitte a rester en cash |
| `defensif` | 60 % (3 x 20 %) | traverser les mauvaises annees sans degats |
| `equilibre` | 99 % (3 x 33 %) | capter les tendances en repartissant le risque |
| `offensif` | 100 % (2 x 50 %) | concentrer sur les deux meilleures tendances |
| `maximal` | 100 % (1 x 100 %) | un seul titre a la fois, aucune diversification |

Aucun profil n'utilise de levier. `--stress-universe` mesure la part du resultat
qui tient au choix des titres plutot qu'a la strategie : c'est ce qui distingue
un profil exploitable d'un pari deguise.

### Diagnostic de marche

Affiche a chaque execution : part des titres en tendance, recul depuis les plus
hauts, regime de volatilite, correlation moyenne, et position du marche large
(SPY) — pour distinguer un probleme sectoriel d'une correction generale. Le
score de prudence et le profil coherent avec l'etat constate sont explicites.

**Methode.** L'univers est selectionne sur des donnees strictement anterieures a
la fenetre evaluee, avec un score qui combine decorrelation et tendancialite —
un titre decorrele mais qui ne tend jamais est inutile a un systeme de tendance.
La configuration a ete arretee sur 2023-10 → 2025-12 (in-sample) puis appliquee
telle quelle a la fenetre evaluee, qui n'a servi a aucun choix de parametre.

**Trois garde-fous verifies a chaque execution** : test de causalite des
indicateurs (recalcul sur prefixe), execution a l'ouverture de la seance
suivante, frais et slippage sur chaque ordre.

## Application d'entrainement "zero to hero" (`scripts/coach.py`)

```bash
python scripts/coach.py            # http://127.0.0.1:8000
```

Interface web locale pour s'entrainer sur des cours REELS avec de l'argent
FICTIF. Aucun ordre reel, aucun courtier connecte, un fichier JSON sur disque.
Le serveur n'a pas d'authentification : il ecoute sur 127.0.0.1 et ne doit pas
etre expose sur un reseau.

- **Capital saisi manuellement.** Chaque apport est un evenement date et
  conserve : voir "j'ai remis 500 EUR apres m'etre fait sortir" est la lecon la
  plus utile qu'un compte d'entrainement puisse donner.
- **Cours en temps reel** via l'API de quotation Nasdaq, avec le statut de
  marche affiche — hors seance, l'interface previent que les prix sont
  indicatifs plutot que de laisser croire qu'on peut trader ce prix.
- **Conseils avant ouverture** : taille, distance au stop, rapport gain/perte,
  concentration, ecart achat/vente. Deux verifications sont BLOQUANTES : un
  risque au stop superieur a 4 % du capital, et une position depassant 60 % du
  compte — parce que le risque "au stop" suppose que le stop tienne, ce qu'un
  trou de cotation dement.
- **Debrief automatique a chaque cloture** : ce qui etait sous controle, puis
  ce qu'une autre decision aurait donne, chiffres a l'appui. Les habitudes
  repetees (stop recule, position surdimensionnee, sorties le jour meme) sont
  detectees sur les vingt derniers trades.
- **Parcours en sept paliers.** Aucun ne recompense le fait de GAGNER : sur dix
  trades le resultat est surtout du hasard, et progresser au resultat
  apprendrait a confondre chance et competence. Ce qui est evalue est le
  process — stop defini avant l'entree, dimensionnement, discipline, patience,
  asymetrie, resilience, regularite sur trente trades.

### Mise en ligne (Vercel)

L'application peut tourner en fonction sans serveur : `api/index.py` en est le
point d'entree, `vercel.json` la configuration, `requirements.txt` les seules
dependances embarquees (FastAPI, httpx, pydantic, structlog — ni pandas ni
scikit-learn, que le serveur web n'importe jamais).

Deux contraintes de l'hebergement ont impose une conception differente du mode
local, et il faut les avoir en tete avant de mettre l'application en ligne :

- **Pas de disque durable.** Le fichier JSON du mode local disparaitrait a tout
  moment, et avec lui l'historique — or le dernier palier du parcours demande
  trente trades. C'est donc le NAVIGATEUR qui detient le compte de reference,
  dans son `localStorage` ; le serveur n'en garde qu'une copie de travail
  jetable. Chaque requete porte l'identifiant du compte (`X-Coach-Account`) et
  son numero de revision (`X-Coach-Rev`) ; si la copie serveur est perimee, le
  serveur repond 409 et le navigateur reinjecte son instantane avant de
  reessayer.
- **Pas d'authentification.** Un compte partage laisserait chaque visiteur
  trader l'argent des autres. L'identifiant tire par le navigateur isole les
  comptes ; il est *verifie* (`^[A-Za-z0-9_-]{8,64}$`) et non assaini, parce
  qu'il devient un nom de fichier.

Consequence a dire honnetement a l'utilisateur, et c'est ce que fait le bandeau
affiche en ligne : effacer les donnees du site ou changer de navigateur efface
l'entrainement. Les boutons « Enregistrer / Restaurer une sauvegarde » exportent
et reimportent le compte en JSON, pour que l'avertissement soit suivi d'un moyen
d'agir.

Le mode local, lui, ne change pas : sans `accounts_dir`, `create_app()` sert un
compte unique sur disque et n'exige aucun en-tete.

### Probabilite de reussite et effet de levier (`signal_probability.py`)

```bash
python scripts/signal_probability.py --horizon 10 --leverages 1,2,5,10,20,30
```

Repond a deux questions distinctes, mesurees sur 2019-2025, hors fenetre 2026.

**Un faisceau de signaux concordants annonce-t-il une hausse ?** Sept signaux
independants (tendance, momentum, MACD, RSI, canal, volume, volatilite) donnent
un score de -7 a +7. Le module produit deux tables : une DESCRIPTIVE (calculee
sur tout l'historique, a ne jamais utiliser pour decider) et une CAUSALE, ou
l'estimation en `t` n'utilise que les signaux dont le resultat etait deja connu
en `t`. L'ecart entre les deux mesure ce que l'on croirait gagner en trichant.

Le point de comparaison est le TAUX DE BASE, pas 50 % : une fenetre de dix
seances monte deja 56.9 % du temps sans rien faire. Un signal a 58 % de
reussite n'est pas "+8 points au-dessus du hasard", il est a +1 point du taux
de base.

**Que donne ce signal a levier ?** Le simulateur modelise ce qui decide du
resultat a fort levier : liquidation sur marge de maintenance, gap d'ouverture
(fonds propres pouvant devenir NEGATIFS — une dette, pas un compte a zero),
cout de portage sur la part empruntee, et frictions calculees sur le NOTIONNEL.

### Pistes mesurees puis ecartees

Consignees ici pour ne pas etre re-testees indefiniment. Une piste rejetee est
un resultat, pas un echec.

| Piste | Mesure in-sample | Verdict |
| --- | --- | --- |
| Delai de carence apres stop (`reentry_cooldown_days`) | 7 valeurs x 5 univers sur 2023-10 → 2025-12 | **Rejetee.** Courbe non monotone (0 bon, 3 a 20 moins bons, 30 meilleur) : signature du bruit. Le delai de 30 jours ne gagne que 2 semestres sur 5, tout son avantage venant d'un seul marche sans direction. Le parametre reste disponible, desactive par defaut. |
| Score de signaux concordants comme predicteur | 8 titres, 2019-2025, 12 000 observations | **Rejetee.** Score +7 (les sept signaux haussiers) : 57.8 % de hausse contre 56.9 % de taux de base, soit +0.9 point. Les scores +3 et +5 font MOINS BIEN que le taux de base. Aucun ecart ne tient d'une annee sur l'autre. |
| Effet contrarien du score -7 | idem | **Rejetee.** Paraissait valoir +11.7 points sur 2022-2025 ; tombe a +0.7 en ajoutant 2019-2021. En 2022, seule annee reellement baissiere de l'echantillon, l'ecart est NEGATIF (-3.0). Un edge qui disparait quand le marche baisse est de l'exposition au marche, pas un signal. |
| Levier 10x et au-dela | 8 titres, signal causal, liquidation et gaps modelises | **Rejetee.** Le rendement median culmine vers 2x puis s'effondre. A 30x, 7 titres sur 8 sont liquides et le pire cas laisse une dette de 122 % du capital. Le portage seul coute environ 145 % des fonds propres par an. |

**Limites.** Cours non ajustes des dividendes ; risque de change EUR/USD non
modelise ; fractions d'actions supposees disponibles (indispensable avec 1000 €
sur des titres a plusieurs centaines de dollars).
