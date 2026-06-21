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

### §7bis — Port d'audit & déterminisme (CCOS)

Implémenté : le module cœur [`audit.rs`](../src/audit.rs) fournit le trait
`AuditLog` et `HashChainLog`, un **journal hash-chaîné SHA-256** (SHA-256 en
Rust pur, [`sha256.rs`](../src/sha256.rs), validé contre les vecteurs NIST) du
**même schéma que l'`EventLog` de CCOS** (`TraceEvent { sequence_number,
prev_hash, hash, event_type, payload }`). Chaque pas de `ℳ` est enregistré :

- **traçabilité** : `record(AuditEvent)` chaîne SHA-256 chaque pas ;
- **intégrité** : `verify()` détecte toute altération (lien rompu / contenu
  falsifié) ;
- **déterminisme** : même trajectoire ⇒ même `head()` (hash de tête
  reproductible) ;
- **replay** : `replay(from, to)` rejoue une sous-séquence ;
- **pont CCOS** : `to_ccos_json()` exporte un flux ingestable par
  `ExternalMemory::ingest_source` de CCOS pour la forensique avancée.

Branchement : `RSIAgent::with_audit(Box::new(HashChainLog::new()))`, puis
`audit_head()`, `audit_verify()`, `audit_len()`.

> **Pourquoi pas une dépendance directe à CCOS ?** Le port d'audit reproduit
> nativement le schéma hash-chaîné de CCOS (zéro dépendance) et **exporte au
> format CCOS**.

### Adaptateur CCOS prêt à activer

[`src/ccos_audit.rs`](../src/ccos_audit.rs) fournit `CcosAudit`, qui implémente
`AuditLog` en déléguant à l'`EventLog` de CCOS (`EventType::AgentAction` +
`EventPayload::Custom`, `verify_integrity`, `replay_events`). Il est écrit contre
l'API publique vérifiée de CCOS mais **non câblé par défaut** car le dépôt CCOS
n'est pas encore consommable par cargo :

1. CCOS doit recevoir un `LICENSE.md` (PolyForm Noncommercial, cf. licence du
   projet) ;
2. CCOS a un **sous-module git mal configuré** (`no URL configured for submodule
   'CCOS'`) qui empêche `cargo` de le fetch — même en dépendance optionnelle,
   cela casse *tous* les builds, d'où le dé-câblage.

Les 3 étapes d'activation (licence + sous-module côté CCOS, puis dep/feature/
module côté RSI) sont détaillées dans l'en-tête de `src/ccos_audit.rs`. En
attendant, `HashChainLog` fournit l'auditabilité complète sans dépendance.

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
