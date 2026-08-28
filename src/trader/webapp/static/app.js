/* Coach Trading — logique d'interface, sans framework ni dependance. */

const $ = (id) => document.getElementById(id);
const euro = (v) => (v ?? 0).toLocaleString('fr-FR', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const signed = (v) => (v >= 0 ? '+' : '') + euro(v);
const cls = (v) => (v >= 0 ? 'pos' : 'neg');

let state = null;
let riskPct = 1;
let pendingPlan = null;

/* ------------------------------------------------- compte detenu localement

   En ligne, le serveur n'a pas de disque durable et ne demande aucun mot de
   passe : s'il gardait la reference, tous les visiteurs partageraient le meme
   compte et le perdraient a chaque redemarrage. C'est donc CE navigateur qui
   detient le compte ; le serveur n'en manipule qu'une copie de travail.

   Consequence a assumer : effacer les donnees du site efface l'entrainement.
   En local, le fichier JSON reste la reference et tout ce bloc ne fait rien de
   plus que recopier un etat que le serveur possede deja. */

const STORE_KEY = 'coach.snapshot';
const ID_KEY = 'coach.account';

function accountId() {
  let id = localStorage.getItem(ID_KEY);
  if (!id) {
    id = (crypto.randomUUID?.() || String(Date.now()) + Math.random().toString(36).slice(2))
      .replace(/[^A-Za-z0-9_-]/g, '');
    localStorage.setItem(ID_KEY, id);
  }
  return id;
}

function savedSnapshot() {
  try {
    return JSON.parse(localStorage.getItem(STORE_KEY) || 'null');
  } catch {
    return null;
  }
}

function keepSnapshot(snapshot) {
  if (!snapshot) return;
  try {
    localStorage.setItem(STORE_KEY, JSON.stringify(snapshot));
  } catch (error) {
    // Quota plein ou stockage refuse : on ne casse pas la session en cours,
    // mais l'utilisateur doit savoir que rien ne sera conserve.
    console.warn('sauvegarde locale impossible', error);
  }
}

async function send(path, options) {
  const snapshot = savedSnapshot();
  return fetch(path, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      'X-Coach-Account': accountId(),
      'X-Coach-Rev': String(snapshot?.rev ?? 0),
      ...(options.headers || {}),
    },
  });
}

async function api(path, options = {}) {
  let response = await send(path, options);
  if (response.status === 409) {
    // Le serveur a perdu sa copie de travail. On lui rend la notre, puis on
    // rejoue l'appel : une seule fois, pour qu'une desynchronisation
    // persistante remonte en erreur au lieu de boucler en silence.
    const snapshot = savedSnapshot();
    if (snapshot) {
      await send('/api/restore', { method: 'POST', body: JSON.stringify({ snapshot }) });
      response = await send(path, options);
    }
  }
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || `erreur ${response.status}`);
  if (body.snapshot) keepSnapshot(body.snapshot);
  return body;
}

/* ------------------------------------------------------------------ rendu */

function renderStats(p) {
  const rows = [
    { label: 'Valeur du compte', value: euro(p.equity) + ' €', sub: `dont ${euro(p.cash)} € en liquide` },
    {
      label: 'Résultat',
      value: signed(p.pnl) + ' €',
      sub: `${p.pnl_pct >= 0 ? '+' : ''}${p.pnl_pct.toFixed(2)} % du capital versé`,
      klass: cls(p.pnl),
    },
    { label: 'Capital versé', value: euro(p.deposited) + ' €', sub: `${p.closed_trades} trade(s) clôturé(s)` },
    {
      label: 'Exposition',
      value: p.exposure_pct.toFixed(0) + ' %',
      sub: `${p.open_positions} position(s) ouverte(s)`,
    },
  ];
  if (p.closed_trades >= 3) {
    rows.push({
      label: 'Réussite',
      value: (p.hit_rate * 100).toFixed(0) + ' %',
      sub: `gain moy. ${euro(p.avg_win)} / perte moy. ${euro(p.avg_loss)}`,
    });
  }
  $('stats').innerHTML = rows
    .map(
      (r) => `<div class="stat">
        <div class="stat-label">${r.label}</div>
        <div class="stat-value ${r.klass || ''}">${r.value}</div>
        <div class="stat-sub">${r.sub}</div>
      </div>`
    )
    .join('');
}

function renderMission(progress) {
  const current = progress.current;
  $('rank-name').textContent = progress.rank;
  $('rank-progress').textContent = `${progress.completed}/${progress.total} paliers`;
  $('mission-title').textContent = `Palier ${current.number} — ${current.title}`;
  $('mission-goal').textContent = current.goal;
  $('mission-why').textContent = current.why;
  $('mission-detail').textContent = current.detail;
}

function stopColor(pct) {
  if (pct < 3) return 'var(--loss)';
  if (pct < 8) return 'var(--warn)';
  return 'var(--win)';
}

function renderPositions(positions) {
  $('count-positions').textContent = positions.length || '';
  if (!positions.length) {
    $('positions-list').innerHTML =
      '<div class="card empty">Aucune position ouverte.<br>Passez par l\'onglet <strong>Trader</strong> pour en préparer une.</div>';
    return;
  }
  $('positions-list').innerHTML = positions
    .map((p) => {
      const margin = Math.max(0, Math.min(100, p.distance_to_stop_pct * 5));
      const near = p.distance_to_stop_pct < 3;
      // Un risque negatif n'est pas une perte : le stop est passe au-dessus du
      // prix de revient et le trade ne peut plus rien couter. C'est le seul
      // resultat que la gestion du risque produise a coup sur, et il meritait
      // d'etre annonce plutot que laisse a deduire de deux prix.
      const locked = p.risk_at_stop <= 0;
      return `<div class="position">
        <div class="pos-head">
          <div>
            <div class="pos-sym">${p.symbol}</div>
            <div class="muted small">${p.shares.toFixed(4)} titres · entrée ${euro(p.entry_price)} ${p.live ? '' : '· cours différé'}</div>
          </div>
          <div class="pos-pnl ${cls(p.unrealised)}">
            ${signed(p.unrealised)} €<div class="muted small" style="text-align:right">${p.unrealised_pct >= 0 ? '+' : ''}${p.unrealised_pct.toFixed(2)} %</div>
          </div>
        </div>
        ${near ? `<div class="alert-stop">⚠ Le stop est à ${p.distance_to_stop_pct.toFixed(2)} % — une séance ordinaire suffit à le déclencher.</div>` : ''}
        <div class="stop-bar"><div class="stop-fill" style="width:${margin}%;background:${stopColor(p.distance_to_stop_pct)}"></div></div>
        <div class="muted small" style="margin-bottom:12px">Marge avant le stop : ${p.distance_to_stop_pct.toFixed(2)} %</div>
        ${locked ? `<div class="alert-locked">✓ Ce trade ne peut plus coûter d'argent : au stop, il rapporterait encore ${euro(-p.risk_at_stop)} €. Le laisser courir ne risque plus que le gain, jamais le capital.</div>` : ''}
        <div class="pos-grid">
          <div><div class="pg-label">Cours</div><div class="pg-value">${euro(p.price)}</div></div>
          <div><div class="pg-label">Stop</div><div class="pg-value">${euro(p.stop)}</div></div>
          <div><div class="pg-label">Objectif</div><div class="pg-value">${p.target ? euro(p.target) : '—'}</div></div>
          <div><div class="pg-label">Suiveur</div><div class="pg-value">${p.trailing_pct ? p.trailing_pct.toFixed(1) + ' %' : '—'}</div></div>
          <div><div class="pg-label">Valeur</div><div class="pg-value">${euro(p.value)} €</div></div>
          <div><div class="pg-label">${locked ? 'Gain verrouillé' : 'Perte si le stop tombe'}</div><div class="pg-value ${locked ? 'pos' : 'neg'}">${euro(Math.abs(p.risk_at_stop))} €</div></div>
        </div>
        ${p.rationale ? `<div class="muted small" style="margin-bottom:12px">« ${p.rationale} »</div>` : ''}
        <div class="pos-actions">
          <input type="number" step="0.01" placeholder="nouveau stop" id="stop-${p.id}">
          <button class="ghost" onclick="moveStop('${p.id}')">Déplacer le stop</button>
          <button class="primary" onclick="closePosition('${p.id}')">Clôturer</button>
        </div>
        <div class="pos-actions trailing-row">
          <input type="number" step="0.1" min="0.1" max="99" placeholder="suiveur %"
                 id="trail-${p.id}" value="${p.trailing_pct || ''}">
          <button class="ghost" onclick="setTrailing('${p.id}')">${p.trailing_pct ? 'Ajuster le suiveur' : 'Activer le suiveur'}</button>
          ${p.trailing_pct ? `<button class="ghost" onclick="setTrailing('${p.id}', true)">Retirer</button>` : ''}
        </div>
      </div>`;
    })
    .join('');
}

function renderLevels(progress) {
  $('levels').innerHTML = progress.levels
    .map((l) => {
      const isCurrent = !l.achieved && l.number === progress.current.number;
      const klass = l.achieved ? 'done' : isCurrent ? 'current' : '';
      return `<div class="level ${klass}">
        <div class="level-num">${l.achieved ? '✓' : l.number}</div>
        <div>
          <div class="level-title">${l.title}</div>
          <div class="level-goal">${l.goal}</div>
          <div class="level-why">${l.why}</div>
          <div class="level-detail" style="color:${l.achieved ? 'var(--win)' : 'var(--muted)'}">${l.detail}</div>
        </div>
      </div>`;
    })
    .join('');
}

function renderPatterns(patterns) {
  const card = $('patterns-card');
  if (!patterns.length) {
    card.classList.add('hidden');
    return;
  }
  card.classList.remove('hidden');
  card.innerHTML =
    '<h3>Habitudes détectées sur vos derniers trades</h3>' +
    patterns
      .map(
        (l) => `<div class="lesson ${l.kind}">
          <div class="lesson-title">${l.title}</div>
          <div class="lesson-msg">${l.message}</div>
        </div>`
      )
      .join('');
}

/** Ce que l'eleve avait mis en face de son stop, avant d'entrer. */
function planLabel(t) {
  if (t.plan === 'suiveur') return `suiveur ${t.trailing_pct} %`;
  if (t.plan === 'aucun') return 'sans objectif ni suiveur';
  // `planned_ratio` est nul quand le gain visé n'a pas de plafond chiffrable :
  // stop déjà remonté au-dessus de l'entrée, il n'y a plus rien à diviser.
  if (t.planned_ratio === null) return 'objectif sans risque au stop';
  return `objectif ${t.planned_ratio.toFixed(1)} × le risque`;
}

async function renderHistory() {
  const { trades } = await api('/api/history');
  if (!trades.length) {
    $('history-list').innerHTML = '<div class="empty">Aucun trade clôturé pour l\'instant.</div>';
    return;
  }
  $('history-list').innerHTML = trades
    .map((t) => {
      const flags = [];
      if (t.stop_moved_against) flags.push('stop élargi');
      if (!t.respected_stop) flags.push('perte hors enveloppe');
      // Le palier « couper court, laisser courir » compte les trades sans plan
      // de sortie ; l'historique doit dire lesquels, sinon le reproche est
      // abstrait. Le seuil reste au serveur (`planned_ok`) : le recopier ici
      // laisserait les deux versions diverger en silence.
      if (!t.planned_ok) flags.push(planLabel(t));
      return `<div class="hist-row" onclick="showDebrief('${t.id}')">
        <div class="wl-sym">${t.symbol}</div>
        <div>
          <div class="small">${euro(t.entry_price)} → ${euro(t.exit_price)} · ${t.holding_days} j · ${planLabel(t)}</div>
          <div class="hist-flags">${flags.length ? '⚠ ' + flags.join(' · ') : t.exit_reason}</div>
        </div>
        <div class="hist-pnl ${cls(t.pnl)}">${signed(t.pnl)} €</div>
        <div class="hist-pnl ${cls(t.pnl)} small">${t.return_pct >= 0 ? '+' : ''}${t.return_pct.toFixed(1)} %</div>
      </div>`;
    })
    .join('');
}

async function renderWatchlist() {
  try {
    const { quotes } = await api('/api/quotes');
    if (!quotes.length) return;
    $('market-status').textContent = `Marché : ${quotes[0].market_status}${quotes[0].is_tradable_session ? '' : ' — hors séance, prix indicatifs'}`;
    $('watchlist').innerHTML = quotes
      .map(
        (q) => `<div class="wl-row" onclick="pickSymbol('${q.symbol}', ${q.price})">
          <div class="wl-sym">${q.symbol}</div>
          <div class="wl-name">${q.company || ''}</div>
          <div>
            <div class="wl-price">${euro(q.price)}</div>
            <div class="wl-chg ${cls(q.change_pct)}" style="text-align:right">${q.change_pct >= 0 ? '+' : ''}${q.change_pct.toFixed(2)} %</div>
          </div>
        </div>`
      )
      .join('');
  } catch (error) {
    $('market-status').textContent = 'Cours indisponibles : ' + error.message;
  }
}

/* --------------------------------------------------------------- actions */

window.pickSymbol = (symbol, price) => {
  $('t-symbol').value = symbol;
  $('price-hint').textContent = `${symbol} cote ${euro(price)} — un stop à −5 % serait ${euro(price * 0.95)}`;
  if (!$('t-stop').value) $('t-stop').value = (price * 0.95).toFixed(2);
};

window.moveStop = async (id) => {
  const value = parseFloat($(`stop-${id}`).value);
  if (!value) return;
  try {
    const result = await api('/api/stop', { method: 'POST', body: JSON.stringify({ position_id: id, stop: value }) });
    if (result.triggers_now) {
      // Un stop pose au-dessus du cours est deja touche : ce n'est plus une
      // protection, c'est une vente. Le dire ici, pendant que la position
      // existe encore, est la seule occasion de corriger la confusion.
      alert(
        `Ce niveau (${euro(result.stop)} €) est au-dessus du cours actuel (${euro(result.price)} €).\n\n` +
          "Un stop au-dessus du cours est un stop déjà touché : ce n'est pas une protection, " +
          "c'est un ordre de vente. La position sera soldée au prochain rafraîchissement, au " +
          "cours du moment — pas au niveau que vous venez de poser.\n\nPour verrouiller un gain " +
          'sans vendre, placez le stop SOUS le cours, ou utilisez le stop suiveur.'
      );
    } else if (result.widened) {
      alert(
        "Stop élargi.\n\nC'est enregistré et cela apparaîtra dans le débrief. Reculer un stop " +
          'transforme une petite perte prévue en grande perte subie — si le stop vous paraît ' +
          'trop serré, la bonne réponse est de réduire la taille avant d\'entrer.'
      );
    }
    await refresh();
  } catch (error) {
    alert(error.message);
  }
};

window.setTrailing = async (id, remove = false) => {
  // Le suiveur ne descend jamais : le retirer laisse en place le stop qu'il a
  // deja fait monter. On le dit, sinon l'utilisateur croit revenir en arriere.
  const value = remove ? null : parseFloat($(`trail-${id}`).value);
  if (!remove && !(value > 0)) return;
  try {
    const result = await api('/api/trailing', {
      method: 'POST',
      body: JSON.stringify({ position_id: id, trailing_pct: value }),
    });
    if (remove) {
      alert(
        `Suiveur retiré. Le stop reste à ${result.stop.toFixed(2)} : un stop suiveur ne rend ` +
          'jamais le terrain gagné. Pour élargir la marge, il faut déplacer le stop, et cela ' +
          'sera enregistré comme un élargissement.'
      );
    }
    await refresh();
  } catch (error) {
    alert(error.message);
  }
};

window.closePosition = async (id) => {
  try {
    const { debrief } = await api('/api/close', { method: 'POST', body: JSON.stringify({ position_id: id }) });
    showDebriefData(debrief);
    await refresh();
  } catch (error) {
    alert(error.message);
  }
};

window.showDebrief = async (tradeId) => {
  try {
    showDebriefData(await api(`/api/debrief/${tradeId}`));
  } catch (error) {
    alert(error.message);
  }
};

/* File d'attente : plusieurs stops peuvent sauter au même rafraîchissement,
   et empiler les fenêtres les rendrait illisibles. On les montre l'une après
   l'autre — une sortie subie mérite d'être lue, pas balayée. */
let debriefQueue = [];

function queueDebriefs(list) {
  if (!list?.length) return;
  debriefQueue.push(...list);
  if ($('debrief-modal').classList.contains('hidden')) showNextDebrief();
}

function showNextDebrief() {
  const next = debriefQueue.shift();
  if (next) showDebriefData(next, true);
}

function renderTargets(targets) {
  const box = $('targets-banner');
  if (!targets?.length) {
    box.classList.add('hidden');
    return;
  }
  box.classList.remove('hidden');
  box.innerHTML = targets
    .map(
      (t) => `<div class="target-row">
        <div>
          <strong>${t.symbol} a atteint votre objectif</strong> (${euro(t.target)})
          — actuellement ${euro(t.price)}, ${signed(t.unrealised)} €.
        </div>
        <div class="muted small">
          À vous de décider : encaisser, ou remonter le stop sous le cours et laisser courir.
          L'app ne clôture pas à votre place — c'est précisément l'arbitrage à travailler.
        </div>
      </div>`
    )
    .join('');
}

function showDebriefData(d, forced = false) {
  $('debrief-title').textContent = forced
    ? `Stop déclenché — ${d.symbol}`
    : `Débrief — ${d.symbol}`;
  $('debrief-result').innerHTML = `<span class="${cls(d.pnl)}">${signed(d.pnl)} € (${d.return_pct >= 0 ? '+' : ''}${d.return_pct.toFixed(2)} %)</span>
    <span class="muted" style="font-size:14px;font-weight:400"> · ${d.holding_days} jour(s)</span>`;
  $('debrief-verdict').textContent = d.verdict;
  $('debrief-lessons').innerHTML = d.lessons
    .map(
      (l) => `<div class="lesson ${l.kind}">
        <div class="lesson-title">${l.title}</div>
        <div class="lesson-msg">${l.message}</div>
      </div>`
    )
    .join('');
  $('debrief-ok').textContent = debriefQueue.length
    ? `J'ai compris (${debriefQueue.length} autre(s))`
    : "J'ai compris";
  $('debrief-modal').classList.remove('hidden');
}

/* ------------------------------------------------------------ preparation */

async function calcSize() {
  const symbol = $('t-symbol').value.trim().toUpperCase();
  const stop = parseFloat($('t-stop').value);
  if (!symbol || !stop) return alert('Renseignez le titre et le stop.');
  try {
    const r = await api('/api/suggest-size', {
      method: 'POST',
      body: JSON.stringify({ symbol, stop, risk_pct: riskPct }),
    });
    $('t-shares').value = r.shares.toFixed(4);
    let hint = `${r.symbol} à ${euro(r.price)} — ${r.shares.toFixed(4)} titres = ${euro(r.notional)} € investis, ${euro(r.risk_amount)} € risqués (${riskPct} %)`;
    // L'objectif se deduit du stop, comme la quantite. On le propose sans jamais
    // ecraser une valeur saisie : la cible reste la decision de l'utilisateur.
    if (r.suggested_target !== null && r.suggested_target !== undefined) {
      if (!$('t-target').value) $('t-target').value = r.suggested_target.toFixed(2);
      // Formulation deliberement non predictive : rien ici ne dit que le cours
      // atteindra ce niveau, seulement ce qu'il faut viser pour que le risque
      // deja accepte ait une contrepartie suffisante.
      hint += `. Pour viser ${r.suggested_target_ratio.toFixed(1)} fois cette perte, l'objectif se pose à ${euro(r.suggested_target)} € — c'est la contrepartie à demander, pas une prévision`;
    }
    $('price-hint').textContent = hint;
  } catch (error) {
    $('trade-error').textContent = error.message;
  }
}

async function reviewTrade() {
  $('trade-error').textContent = '';
  const plan = {
    symbol: $('t-symbol').value.trim().toUpperCase(),
    shares: parseFloat($('t-shares').value),
    stop: parseFloat($('t-stop').value),
    target: parseFloat($('t-target').value) || null,
    // Le suiveur ne part que si la case est cochee : un champ pre-rempli mais
    // decoche ne doit pas armer un stop que l'utilisateur n'a pas demande.
    trailing_pct: $('t-trailing-on').checked ? parseFloat($('t-trailing').value) || null : null,
  };
  if (!plan.symbol || !plan.shares || !plan.stop) {
    $('trade-error').textContent = 'Titre, quantité et stop sont nécessaires.';
    return;
  }
  try {
    const review = await api('/api/review', { method: 'POST', body: JSON.stringify(plan) });
    pendingPlan = { ...plan, rationale: $('t-rationale').value };
    // Le stop en vigueur n'est pas toujours celui qui a ete saisi : un suiveur
    // plus serre le remonte des l'entree. La decision se prend ici, pas dans
    // l'alerte qui suit la confirmation — c'est donc ici qu'il faut le lire.
    const stopLabel = review.trailing_overrides_stop ? 'Stop en vigueur (suiveur)' : 'Stop en vigueur';
    $('review-metrics').innerHTML = [
      { l: 'Risque', v: review.risk_pct.toFixed(2) + ' %' },
      { l: 'Perte au stop', v: euro(review.risk_amount) + ' €' },
      { l: 'Part du compte', v: review.position_pct.toFixed(0) + ' %' },
      { l: stopLabel, v: euro(review.effective_stop) + ' €' },
      { l: 'Distance au stop', v: review.stop_distance_pct.toFixed(1) + ' %' },
      // `0` est un rapport gain/perte — objectif pose au prix d'entree — et non
      // une absence d'objectif. Le tester comme un booleen l'afficherait « — ».
      { l: 'Gain / perte', v: review.reward_risk === null ? '—' : review.reward_risk.toFixed(2) },
    ]
      .map(
        (m) =>
          `<div class="rm${m.l === stopLabel && review.trailing_overrides_stop ? ' rm-alert' : ''}">`
          + `<div class="rm-label">${m.l}</div><div class="rm-value">${m.v}</div></div>`
      )
      .join('');
    $('advices').innerHTML = review.advices
      .map(
        (a) => `<div class="advice ${a.severity}">
          <div class="advice-title">${a.title}</div>
          <div class="advice-msg">${a.message}</div>
        </div>`
      )
      .join('');
    const confirm = $('confirm-btn');
    confirm.disabled = !review.can_proceed;
    confirm.textContent = review.can_proceed ? 'Ouvrir la position' : 'Corrigez les points bloquants';
    confirm.style.opacity = review.can_proceed ? '1' : '0.5';
    $('review-card').classList.remove('hidden');
    $('review-card').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  } catch (error) {
    $('trade-error').textContent = error.message;
  }
}

async function confirmTrade() {
  if (!pendingPlan) return;
  try {
    const opened = await api('/api/open', { method: 'POST', body: JSON.stringify(pendingPlan) });
    // Le stop reellement en vigueur peut differer de celui saisi : le suiveur
    // le resserre des l'entree s'il est plus proche. Le taire laisserait
    // l'utilisateur croire qu'il risque plus qu'il ne risque vraiment.
    if (opened.trailing_pct && opened.stop > pendingPlan.stop) {
      alert(
        `Le suiveur à ${opened.trailing_pct} % place le stop à ${euro(opened.stop)} €, `
          + `au-dessus du ${euro(pendingPlan.stop)} € que vous aviez saisi. C'est ce niveau-là qui protège la position.`
      );
    }
    $('review-card').classList.add('hidden');
    ['t-symbol', 't-stop', 't-shares', 't-target', 't-rationale'].forEach((id) => ($(id).value = ''));
    $('t-trailing-on').checked = false;
    $('t-trailing-row').classList.add('hidden');
    $('price-hint').textContent = '';
    pendingPlan = null;
    await refresh();
    document.querySelector('[data-tab="positions"]').click();
  } catch (error) {
    $('trade-error').textContent = error.message;
  }
}

/* ------------------------------------------------------------------ cycle */

async function refresh() {
  state = await api('/api/state');
  if (!state.has_capital) {
    $('onboarding').classList.remove('hidden');
    $('app').classList.add('hidden');
    $('local-notice').classList.toggle('hidden', !state.hosted);
    return;
  }
  $('onboarding').classList.add('hidden');
  $('app').classList.remove('hidden');
  $('local-notice').classList.toggle('hidden', !state.hosted);
  renderStats(state.performance);
  renderMission(state.progress);
  renderPositions(state.positions);
  renderLevels(state.progress);
  renderPatterns(state.patterns);
  renderTargets(state.targets_reached);
  await renderHistory();
  queueDebriefs(state.stopped);
}

async function deposit(amount) {
  $('deposit-error').textContent = '';
  try {
    await api('/api/deposit', { method: 'POST', body: JSON.stringify({ amount }) });
    await refresh();
    await renderWatchlist();
  } catch (error) {
    $('deposit-error').textContent = error.message;
  }
}

/* ------------------------------------------------------- sauvegarde fichier

   Prevenir que le compte vit dans le navigateur ne suffit pas : sans moyen
   d'en sortir une copie, l'avertissement ne fait que decrire la perte a
   venir. L'export produit exactement l'instantane que le serveur sait relire. */

function exportAccount() {
  const snapshot = savedSnapshot();
  if (!snapshot) {
    alert("Rien à sauvegarder pour l'instant : versez d'abord un capital d'entraînement.");
    return;
  }
  const url = URL.createObjectURL(
    new Blob([JSON.stringify(snapshot, null, 2)], { type: 'application/json' })
  );
  const link = document.createElement('a');
  link.href = url;
  link.download = `coach-trading-${new Date().toISOString().slice(0, 10)}.json`;
  link.click();
  URL.revokeObjectURL(url);
}

async function importAccount(file) {
  try {
    const snapshot = JSON.parse(await file.text());
    if (typeof snapshot?.cash !== 'number' || !Array.isArray(snapshot?.history)) {
      throw new Error("ce fichier n'est pas une sauvegarde de compte");
    }
    // La revision reprend au-dessus de celle deja connue, sinon le serveur
    // prendrait la sauvegarde restauree pour un etat perime.
    snapshot.rev = Math.max(snapshot.rev || 0, (savedSnapshot()?.rev || 0) + 1);
    keepSnapshot(snapshot);
    await api('/api/restore', { method: 'POST', body: JSON.stringify({ snapshot }) });
    await refresh();
    await renderWatchlist();
  } catch (error) {
    alert(`Restauration impossible : ${error.message}`);
  }
}

/* ---------------------------------------------------------------- ecouteurs */

$('export-account').addEventListener('click', exportAccount);
$('import-account').addEventListener('click', () => $('import-file').click());
$('import-file').addEventListener('change', (event) => {
  const file = event.target.files?.[0];
  if (file) importAccount(file);
  event.target.value = '';
});

document.querySelectorAll('.chip[data-amount]').forEach((chip) =>
  chip.addEventListener('click', () => deposit(parseFloat(chip.dataset.amount)))
);
$('deposit-btn').addEventListener('click', () => {
  const amount = parseFloat($('deposit-amount').value);
  if (amount > 0) deposit(amount);
  else $('deposit-error').textContent = 'Saisissez un montant positif.';
});

document.querySelectorAll('.tab').forEach((tab) =>
  tab.addEventListener('click', () => {
    document.querySelectorAll('.tab').forEach((t) => t.classList.remove('active'));
    document.querySelectorAll('.panel').forEach((p) => p.classList.remove('active'));
    tab.classList.add('active');
    $('tab-' + tab.dataset.tab).classList.add('active');
  })
);

document.querySelectorAll('.chip[data-risk]').forEach((chip) =>
  chip.addEventListener('click', () => {
    document.querySelectorAll('.chip[data-risk]').forEach((c) => c.classList.remove('active'));
    chip.classList.add('active');
    riskPct = parseFloat(chip.dataset.risk);
  })
);

$('t-trailing-on').addEventListener('change', (event) =>
  $('t-trailing-row').classList.toggle('hidden', !event.target.checked)
);

$('calc-size').addEventListener('click', calcSize);
$('review-btn').addEventListener('click', reviewTrade);
$('confirm-btn').addEventListener('click', confirmTrade);
$('cancel-btn').addEventListener('click', () => $('review-card').classList.add('hidden'));
function closeDebrief() {
  $('debrief-modal').classList.add('hidden');
  if (debriefQueue.length) setTimeout(showNextDebrief, 250);
}
$('debrief-close').addEventListener('click', closeDebrief);
$('debrief-ok').addEventListener('click', closeDebrief);

refresh().then(renderWatchlist);
setInterval(renderWatchlist, 30000);
setInterval(() => state?.has_capital && refresh(), 60000);
