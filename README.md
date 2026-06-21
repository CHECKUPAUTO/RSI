# RSI — Recursive Self-Improvement (Rust)

Implémentation **100 % Rust** (std-only, aucune dépendance externe) du
**système mathématique d'auto-amélioration récursive** — formulation
géométrique unifiée, v9.

Le système modélise un **agent cognitif** dont la *surface de compétence*
`Σ_I(t)` se déforme sous l'effet de l'apprentissage, du substrat
matériel/logiciel et d'une **méta-optimisation récursive**, le tout sous des
**garde-fous de stabilité**.

```
SI_global : 0.1385 → 0.6760   (+388 %)   en 120 pas
goulot : cognitif (0 %) ──────────────► substrat (100 %)
```

## Correspondance équations ↔ code

| Section | Équation | Module |
|---------|----------|--------|
| **§1** Surface d'intelligence | `Σ_I(t)={(x,C_réel)}`, `C_réel=min(Φ_x(S),g_x(P_eff))`, `SI_global=∫_Ω C_réel dμ` | [`src/surface.rs`](src/surface.rs) |
| **§2** Vecteur d'état cognitif | `S=(D,M,R,A,C,V)` | [`src/state.rs`](src/state.rs) |
| **§3** Substrat physique/logiciel | `P_eff=σ(HᵀAH)·σ(OᵀBO)·σ(HᵀCO)` | [`src/substrate.rs`](src/substrate.rs) |
| **§4** Dynamique + stabilité | `dS/dt=η·[L+E+U]−P`, `‖ΔS‖<λ`, `SI(t+1)≥SI(t)−ε` | [`src/dynamics.rs`](src/dynamics.rs) |
| **§5** Boucle discrète + méta | `S_{t+1}=S_t+ℳ(…)+ΔS_appr`, `ℳ_{t+1}=argmax_ℳ SI_global(ℳ(S_t))` | [`src/meta.rs`](src/meta.rs) |
| **§6** Forme compacte (onde) | `Σ_I(t+1)=Σ_I(t)+η·ℳ−P` | [`src/agent.rs`](src/agent.rs) |

## Modèle

1. **Espace des tâches** `(Ω,𝒜,μ)` : Ω est échantillonné une fois par
   Monte-Carlo (profils de besoins via Dirichlet sur les 6 composantes),
   μ pondère l'importance des tâches. Le même échantillon sert à toutes les
   évaluations ⇒ `SI_global` comparable d'un pas à l'autre.

2. **Compétence réelle** `C_réel = min(Φ, g)` : minimum entre la compétence
   *cognitive* `Φ_x(S)` (ce que l'agent sait faire) et le plafond *physique*
   `g_x(P_eff)` (ce que le substrat permet). C'est un **goulot
   d'étranglement** : être limité par le cerveau OU par le matériel.

3. **Efficacité multiplicative** `P_eff` : un matériel puissant est inutile
   sans logiciel capable de l'exploiter — d'où le produit de trois sigmoïdes,
   dont un terme de **couplage** `σ(HᵀCO)`.

4. **Dynamique contrainte** : le pas d'apprentissage brut `η·[L+E+U]−P` est
   borné en amplitude (`‖ΔS‖<λ`, projection radiale) puis atténué par
   *line search* tant qu'il ferait régresser `SI_global` de plus de `ε`.

5. **Méta-révision récursive** : à chaque pas, l'agent explore un voisinage de
   stratégies d'auto-modification `ℳ` et retient celle qui **maximise le
   `SI_global` projeté** — y compris en réécrivant son propre logiciel `O`
   (auto-amélioration du substrat, cœur du RSI).

## Utilisation

```bash
# tests (unitaires + intégration + doctest)
cargo test

# simulation de démonstration : cargo run -- [n_pas] [graine]
cargo run --release --bin rsi-demo -- 120 2026
```

### Comme bibliothèque

```rust
use rsi::RSIAgent;

let mut agent = RSIAgent::demo(2026);
let start = agent.si_global();
let reports = agent.run(120);          // trajectoire complète
let end = reports.last().unwrap();
println!("SI_global : {start:.4} → {:.4}", end.si_global);
```

Pour assembler un agent sur mesure :

```rust
use rsi::*;

let mut rng = Rng::new(7);
let state     = CognitiveState::random(Dims::uniform(6), &mut rng, 0.08);
let substrate = Substrate::default_with(/*hw*/ 4, /*sw*/ 4, &mut rng);
let surface   = IntelligenceSurface::sample(1024, &mut rng);
let meta      = MetaOptimizer::new(/*candidats*/ 48, /*exploration*/ 0.12, 999);

let mut agent = RSIAgent::new(state, substrate, surface,
                              StabilityConfig::default(), meta);
agent.run(150);
```

## Interprétation des sorties

Chaque [`StepReport`](src/agent.rs) expose :

- `si_global` — volume sous Σ_I (intelligence globale, ∈ [0,1]) ;
- `delta_si` — variation par pas (garantie `≥ −ε`) ;
- `p_eff` — efficacité du substrat (monte si l'agent améliore son logiciel) ;
- `appr.delta_norm` — `‖ΔS‖` du pas d'apprentissage (garanti `< λ`) ;
- `frac_limited_by_substrate` — fraction des tâches bridées par le matériel
  plutôt que par le cognitif (le **basculement** vers 100 % est la signature
  d'un agent devenu substrate-limited) ;
- `capabilities` — niveaux `(D,M,R,A,C,V)`.

## Extensions

- **Modèles `Φ_x` / `g_x` configurables** via les traits `CapabilityModel` /
  `CeilingModel` (`surface.rs`) — branchez n'importe quelle loi de
  compétence/plafond sans toucher au reste.
- **Méta-optimiseur sep-CMA-ES** (`cma.rs`) en plus de la recherche aléatoire,
  derrière le trait `MetaSearch` ; sélectionnable par `optimizer: random|cma`.
- **Export CSV / JSON** de la trajectoire (`report.rs`, flags `--csv`/`--json`).
- **API** orientée commandes JSON (`api.rs`, `RsiApi`).
- **Serveur MCP** (`rsi-mcp`) pour piloter le système depuis un agent IA / LLM.
- **Auto-connexion** (`rsi-connect` + `scripts/auto-connect.sh`) aux runtimes
  d'agents (openclaw, hermes-agent, …) sans intervention humaine.
- **Méta-optimiseur Forge** (feature optionnelle `forge`) : la méta-révision
  `ℳ` devient une recherche évolutionnaire *exécutée* (moteur `forge-core`)
  dont la fitness est `SI_global`. Cœur RSI inchangé et sans dépendance par
  défaut. Voir l'étude d'intégration : [`docs/INTEGRATION_STUDY.md`](docs/INTEGRATION_STUDY.md).

  ```bash
  cargo build --features forge      # branche le méta-optimiseur Forge
  ```

### Intégration agent IA / LLM

```bash
cargo build --release --bins
./scripts/auto-connect.sh            # build + enregistrement MCP automatique
./target/release/rsi-mcp             # serveur MCP (JSON-RPC 2.0 sur stdio)
```

Voir le guide complet : [`docs/INTEGRATION.md`](docs/INTEGRATION.md)
(API, outils MCP, schémas, boucle d'auto-amélioration côté LLM, auto-connexion).

### Paper scientifique

Formalisation, implémentation et résultats expérimentaux :
[`paper/rsi.md`](paper/rsi.md) (Markdown) et [`paper/rsi.tex`](paper/rsi.tex)
(source LaTeX).

## Architecture

```
src/
├── lib.rs          exports + table de correspondance équations/modules
├── rng.rs          PRNG xoshiro256** + Dirichlet/normale (std-only)
├── linalg.rs       vecteurs, matrices denses, formes quadratiques, σ
├── state.rs        §2  S = (D,M,R,A,C,V)
├── substrate.rs    §3  P_eff = σ(HᵀAH)·σ(OᵀBO)·σ(HᵀCO)
├── surface.rs      §1  Σ_I, C_réel = min(Φ,g), SI_global + traits Φ/g
├── dynamics.rs     §4  dS/dt + contraintes ‖ΔS‖<λ et non-régression ε
├── meta.rs         §5  ℳ, trait MetaSearch, recherche aléatoire + CMA-ES
├── cma.rs          §5  sep-CMA-ES (covariance diagonale)
├── agent.rs        §5/§6  boucle discrète complète
├── json.rs         (dé)sérialisation JSON std-only
├── report.rs       export CSV / JSON de la trajectoire
├── api.rs          façade RsiApi (commandes JSON in/out)
├── main.rs         binaire de démonstration (rsi-demo)
└── bin/
    ├── rsi_mcp.rs      serveur MCP (JSON-RPC 2.0 / stdio)
    └── rsi_connect.rs  auto-enregistrement MCP (openclaw, hermes-agent, …)
scripts/
└── auto-connect.sh  build + connexion MCP automatique
docs/
└── INTEGRATION.md   guide API / MCP / LLM / auto-connexion
paper/
├── rsi.md           paper scientifique (anglais)
└── rsi.tex          source LaTeX
tests/
└── integration.rs   trajectoire stable & croissante, déterminisme, substrat
```

## Licence

MIT (voir [LICENSE](LICENSE)).
