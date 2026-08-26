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

pytest -q                                   # 254 tests
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
