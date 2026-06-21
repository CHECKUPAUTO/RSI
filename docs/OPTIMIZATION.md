# Optimisation du moteur RSI pour ses intégrations + §7 criticité (AMDEC)

Ce document décrit comment le moteur a été rendu **optimal avec ses
intégrations** (OctaSoma, Forge, et le futur CCOS), et l'ajout d'une **analyse
des modes de défaillance et de criticité** (AMDEC / FMECA) au système
mathématique.

## Diagnostic initial

Les intégrations étaient *branchées mais passives/cloisonnées* :

- la mémoire OctaSoma était **écrite mais jamais relue** → composante `C`
  décorative ;
- le substrat Forge **écrasait** le facteur logiciel analytique → le
  `software_edit` de `ℳ` devenait un no-op (canaux qui se neutralisent) ;
- chaque révision Forge repartait **à froid** (campagne re-seedée, fitness
  recalculée) ;
- aucune **théorie des défaillances** : seulement deux garde-fous ponctuels
  (`‖ΔS‖<λ`, non-régression `ε`).

## A — Mémoire active (recall → ℳ)

`MetaSearch` reçoit une méthode `warm_start(&[MetaStrategy])` (no-op par
défaut). À chaque pas, l'agent **rappelle** les contextes proches
(`recall_similar`), **décode** les stratégies gagnantes mémorisées
(`decode_strategy_payload`) et les **réinjecte comme graines** de la révision.
Les trois optimiseurs (`MetaOptimizer`, `CmaEsMeta`, `ForgeMetaSearch`) les
exploitent (évaluation directe / centre d'exploration / warm-start de campagne).
OctaSoma passe ainsi de *journal* à *accélérateur*. *(`memory.rs`, `meta.rs`,
`agent.rs`)*

## B — Canal substrat unifié

`software_efficiency()` combine désormais l'analytique σ(OᵀB O) et l'efficience
mesurée par **maximum** au lieu d'un écrasement :
`software_efficiency = max(σ(OᵀB O), measured)`. Les deux leviers coopèrent :
améliorer `O` reste utile, et la mesure Forge agit comme **plancher**.
*(`substrate.rs`)*

## C — Campagnes Forge amorties

- `RSIAgent::meta_interval` : la révision `ℳ` n'est exécutée que tous les `k`
  pas (`with_meta_interval`).
- `RsiDomain.si_cache` : `SI_global` est **mémoïsé par id de candidat**
  (`Mutex<HashMap>`), évitant de recalculer la fitness des candidats réapparus.
- Warm-start implicite : la campagne est centrée sur la meilleure stratégie
  connue (courante ou graine mémoire). *(`agent.rs`, `forge_meta.rs`)*

## D — Routage par criticité

L'améliorateur de substrat (coûteux) n'est invoqué **que lorsque le substrat est
la contrainte qui bride réellement** : goulot substrat ≥ `route_threshold` **ou**
mode le plus critique = effondrement du substrat. Généralisation du routage par
goulot vers un routage piloté par l'AMDEC. *(`agent.rs`)*

## §7 — Modes de défaillance & criticité (AMDEC / FMECA)

Module cœur [`criticality.rs`](../src/criticality.rs), sans dépendance.

Pour chaque mode `f` : `RPN_f = Sévérité_f · Occurrence_f · Détection_f`
(facteurs ∈ [0,1], `Détection` = *difficulté* de détection). Modes couverts :

| Mode | Occurrence (dynamique) | Détecteur |
|------|------------------------|-----------|
| régression de compétence | `−ΔSI / ε` | garde-fou ε |
| instabilité / divergence | `‖ΔS‖ / λ` | garde-fou λ |
| dérive des valeurs | `A − V` | (AMDEC) |
| effondrement du substrat | `(1−P_eff)·%substrat` | bottleneck |
| Goodhart / sur-ajustement | `backtracks / 5` | line-search |
| empoisonnement mémoire | base si mémoire active | (CCOS à venir) |
| wireheading | `max(0, mesuré − analytique)` | Forge `verify` |

Agrégats : `Risk_global = moyenne_f RPN_f` et **intelligence ajustée au
risque** `SI_safe = SI_global − κ · Risk_global`. Un **garde-fou de criticité**
(`RiskConfig { kappa, rpn_max, risk_delta }`) déclenche un **pas conservateur**
(atténuation du gain de `ℳ`) quand `max RPN > rpn_max`.

Chaque `StepReport` expose désormais `risk_global`, `max_rpn`, `most_critical`,
`si_safe` (présents aussi dans l'export CSV/JSON et l'API/MCP). Les garde-fous
λ/ε de §4 deviennent des cas particuliers de la maîtrise du risque (modes
régression/instabilité).

### Place de CCOS dans ce cadre

CCOS trouve ici sa fonction naturelle : **détecteur + forensics**. Son journal
hash-chaîné et son *replay* améliorent la *Détectabilité* des modes (surtout
l'empoisonnement mémoire et le wireheading) et rendent chaque pas de `ℳ`
**auditable et rejouable** — l'étape suivante recommandée.

## Effet observé

Invariants préservés (‖ΔS‖<λ, non-régression ε) ; l'agent progresse toujours ;
`SI_safe ≤ SI_global` ; routage et warm-start actifs. Tests : 44 (défaut) /
48 (forge+octasoma), aucun warning.

## Configuration

```rust
let agent = RSIAgent::new(state, substrate, surface, cfg, meta)
    .with_memory(Box::new(LinearContextMemory::new()))   // §A (ou OctaSomaMemory)
    .with_meta_interval(3)                                // §C
    .with_route_threshold(0.5)                            // §D
    .with_risk_config(RiskConfig::default());             // §7
```
