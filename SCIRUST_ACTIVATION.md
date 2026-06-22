# Moteur `scirust-rsi` — activation

RSI consomme le contrat de `scirust-rsi` (`propose → évalue → garde si meilleur →
répète`, élitiste/borné/reproductible) de **deux** façons :

| Mode | Module | Dépendance | Statut |
|------|--------|-----------|--------|
| **Stand-in intégré** | [`src/ascent.rs`](src/ascent.rs) | aucune (std) | ✅ actif par défaut, testé |
| **Moteur réel** | [`src/scirust_bridge.rs`](src/scirust_bridge.rs) | `scirust-rsi` (vendorisé) | ✅ `--features scirust` |

## Activer le moteur réel

Le crate `scirust-rsi` est **vendorisé** dans [`vendor/scirust-rsi`](vendor/scirust-rsi)
en dépendance `path` (aucun accès réseau requis ; ne tire que `rand`, déjà dans
l'arbre de RSI). La feature est déjà câblée dans `Cargo.toml` :

```toml
[features]
scirust = ["dep:scirust-rsi", "dep:rand"]

[dependencies]
scirust-rsi = { path = "vendor/scirust-rsi", optional = true }
```

Il suffit donc de :

```bash
cargo run    --release --features scirust --example self_improve_real
cargo test   --features scirust
cargo clippy --features scirust --all-targets
```

> ⚠️ **`vendor/scirust-rsi` est une reconstruction API-compatible** de l'API
> publiée (cf. en-tête + `INTEGRATION.md`), faite pour débloquer le build
> hors-ligne. Elle respecte exactement le contrat public mais n'est **pas** les
> octets amont. Pour passer au crate amont (recevoir les mises à jour), voir
> ci-dessous.

## API ciblée (vérifiée sur l'en-tête réel)

- `pub type Fitness = f64;` (plus grand = mieux) → aucun constructeur, le score
  scalaire **est** la fitness.
- `trait RefineTask { type Solution: Clone; fn initial(&self, &mut StdRng); fn
  score(&self, &Solution) -> Fitness; fn refine(&self, &Solution, &mut StdRng)
  -> Solution; }`
- `SelfRefiner::new(seed).run(&task, &guard) -> (Solution, Report)`.
- `Report { iterations, accepted, best_fitness, history, stop_reason }`,
  `Report::is_monotone()`, `Report::total_gain()`.
- Aussi exposés : `ascend(...)` (fonction libre), `bench::{sphere, rastrigin,
  rosenbrock}`, et les pilotes `star::Star`, `expert_iteration::ExpertIteration`,
  `pbt::Pbt`, `evo::OnePlusLambda`.

## Passer au crate amont (`CHECKUPAUTO/scirust`)

Quand le dépôt + le réseau sont ouverts dans l'environnement web, remplacer la
dépendance `path` par l'amont — l'API est identique, aucun changement de code :

```toml
# git-dependency
scirust-rsi = { git = "https://github.com/CHECKUPAUTO/scirust", branch = "master", optional = true }
# ou installeur vendorisé amont : bash vendor_scirust_rsi.sh   (recrée vendor/scirust-rsi)
# variante umbrella : scirust = { git = … } puis `use scirust::rsi::{…}`
```

## Garde-fous (identiques dans les deux modes)

- **Sandbox** : le candidat est un AST `Expr` évalué par notre interpréteur
  (`Expr::eval`) — le moteur n'exécute jamais de code généré, ne se modifie pas.
- **Non-régression** : adoption élitiste ⇒ `report.is_monotone()`.
- **Terminaison** : `Guard::max_iters` borne chaque run ; `patience`/`target`
  arrêtent proprement.
- **Reproductible** : même graine ⇒ même run.
