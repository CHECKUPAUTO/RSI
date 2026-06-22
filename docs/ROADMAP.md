# Feuille de route RSI — Objectifs

> État actuel (v0.10) : cœur `std`-only sans dépendance (surface Σ_I, état
> S=(D,M,R,A,C,V), substrat P_eff, dynamique contrainte, méta-révision ℳ,
> criticité §7, audit hash-chaîné), backends réels en features
> (Forge, OctaSoma, CCOS), corpus de tâches ancré, source `D` (PAPERS en
> sous-processus). 65/70 tests, 0 warning.

## Objectif phare — ⚙️ LOOP ENGINEERING (travail massif)

La boucle d'auto-amélioration est aujourd'hui un `agent.step()` à cadence fixe.
Pour passer d'un *modèle* à un véritable **moteur de boucle RSI**, on engage un
chantier transverse et prioritaire sur l'**ingénierie de la boucle** : sa
structure, son contrôle, sa sûreté, son observabilité et son pilotage.

### Pourquoi
Dans un système récursif, *la boucle EST le produit*. La qualité du RSI dépend
moins d'un pas isolé que de : quand s'arrêter, à quelle cadence méta-réviser,
comment détecter un plateau / une divergence, comment reprendre après incident,
comment imbriquer les échelles de temps, et comment un humain (ou un agent)
peut piloter, mettre en pause, rejouer et reconfigurer la boucle en sûreté.

### Chantiers (L1–L9)

**L1 — Pilote de boucle & critères d'arrêt.** `LoopController` configurable
au-dessus de `step()` : budgets (pas / temps / évaluations), arrêt sur
**convergence** (pente de SI ≈ 0), plateau, ou cible atteinte ; `LoopConfig`,
`StopReason`. *Livrable* : `loop_ctrl.rs` + `RSIAgent::run_until(cfg)`.
*Acceptation* : arrêt déterministe et motivé sur 3 critères, testé.

**L2 — Détection de convergence / divergence / attracteur.** Estimateurs en
ligne de la pente de `SI_global`/`SI_safe`, détection de plateau (attracteur
substrate-limited), de divergence (oscillation, explosion de `risk`), et de
stagnation. *Livrable* : `convergence.rs`. *Acceptation* : détecte le plateau de
la démo et signale une divergence injectée.

**L3 — Boucles multi-échelles (nested loops).** Formaliser les cadences :
boucle interne (apprentissage ΔS, chaque pas), boucle méta (ℳ, tous les *k*),
boucle substrat/architecture (plus lente), boucle **méta-méta** (révise les
cadences elles-mêmes). *Livrable* : `schedule.rs` (ordonnanceur de cadences) ;
généralise `meta_interval`. *Acceptation* : cadences indépendantes vérifiées,
invariants préservés.

**L4 — Disjoncteurs de sûreté au niveau boucle.** Élever la criticité §7 au rang
de **circuit breakers** : rollback au dernier checkpoint sain, gel/limitation de
cadence sous criticité, **kill-switch**, portes *human-in-the-loop*. *Livrable* :
intégration `criticality` ↔ `LoopController`. *Acceptation* : un pic de
criticité déclenche rollback/halt traçé dans l'audit.

**L5 — Checkpoint / reprise / replay.** Sérialiser l'état complet (S, substrat,
stratégie, RNG, mémoire, tête d'audit) ; reprise après crash ; **replay
déterministe** d'une trajectoire (adossé à CCOS). *Livrable* : `checkpoint.rs`
(via `json`). *Acceptation* : run → checkpoint → reprise donne une trajectoire
bit-identique.

**L6 — Plan de contrôle observable.** Hooks par phase (`on_step`, `on_meta`,
`on_critical`), métriques en flux, journal structuré. *Livrable* : trait
`LoopObserver` + sorties CSV/JSON/SVG en flux. *Acceptation* : un observateur
externe reçoit chaque transition sans modifier le cœur.

**L7 — Pilotage interactif & live-reconfig.** `start/pause/step/resume/stop` et
reconfiguration à chaud (λ, ε, cadences, κ) — exposé via **MCP** pour qu'un agent
LLM pilote la boucle. *Livrable* : commandes API/MCP `loop_*`. *Acceptation* :
session MCP qui met en pause, reconfigure, reprend.

**L8 — Parallélisme & portefeuille de boucles.** Population d'agents / boucles
concurrentes, sélection de portefeuille, redémarrages (restart strategies),
boucles asynchrones. *Livrable* : `swarm.rs` (orchestration multi-agents).
*Acceptation* : N boucles en parallèle, agrégation du meilleur, déterministe par
graine.

**L9 — Banc d'essai de boucle.** Mesures de **vitesse de convergence**,
**efficacité d'échantillonnage**, AUC, stabilité, coût ; ablations
(avec/sans L1–L8). *Livrable* : `bin/rsi-loopbench`. *Acceptation* : tableau +
SVG comparatifs reproductibles.

### Phasage
1. **Socle** : L1 (pilote/arrêt) → L2 (convergence) → L5 (checkpoint/replay).
2. **Sûreté & échelles** : L4 (disjoncteurs) → L3 (multi-échelles).
3. **Pilotage & passage à l'échelle** : L6 (observabilité) → L7 (MCP) → L8 (swarm).
4. **Mesure** : L9 (banc d'essai + ablations).

### Invariants à préserver
Cœur `std`-only sans dépendance par défaut ; garde-fous `‖ΔS‖<λ` et
non-régression ; déterminisme par graine ; auditabilité (CCOS) de toute
transition de boucle.

---

## Autres objectifs

- **Validation empirique** : ✅ banc d'ablation `rsi-ablate` (cœur pur) +
  corpus élargi (Ω=40) — voir [`docs/VALIDATION.md`](VALIDATION.md). Constats :
  non-régression préservée partout, connaissances ↑ SI, substrat ↑ vitesse,
  ε adaptatif ↓ risque. À étendre : corpus issu d'un **benchmark public** réel
  (chargeable via `TaskCorpus::from_json`).
- **Connaissances `D`** : ✅ port + `CorpusKnowledge` + `PapersKnowledge`
  (sous-processus). Étendre : ingestion incrémentale, dédup sémantique.
- **Substrat** : ✅ `MeasuredSubstrate` natif + Forge. Étendre : domaines réels
  (GPU/SIMD via Forge côté toolchain), efficacité matérielle `H` mesurée.
- **Surface** : ✅ corpus ancré. Étendre : tâches *exécutées* (compétence =
  succès réel d'un solveur), import de jeux de tâches publics.
- **GPU / temps réel** : documenté ([`docs/VALIDATION.md`](VALIDATION.md)) — non
  bloquant par conception (seam `SubstrateImprover` + Forge SIMD/CUDA côté
  toolchain) ; `MeasuredSubstrate` (CPU) couvre l'environnement sans GPU.
