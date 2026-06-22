# Activer le moteur réel `scirust-rsi`

RSI consomme le contrat de `scirust-rsi` (`propose → évalue → garde si meilleur →
répète`, élitiste/borné/reproductible) de **deux** façons :

| Mode | Module | Dépendance réseau | Statut |
|------|--------|-------------------|--------|
| **Stand-in hors-ligne** | [`src/ascent.rs`](src/ascent.rs) | aucune | ✅ actif par défaut, testé |
| **Moteur réel** | [`src/scirust_bridge.rs`](src/scirust_bridge.rs) | `git fetch` de `CHECKUPAUTO/scirust` | 🔌 prêt, à activer |

Le pont (`scirust_bridge.rs`) implémente le **vrai** trait
`scirust_rsi::refine::RefineTask` (`type Solution`, `initial`/`score → Fitness`/
`refine(&self, _, &mut StdRng)`) et pilote la boucle avec `SelfRefiner::new(seed)
.run(&task, &Guard)`, exactement comme `scirust-rsi/INTEGRATION.md`.

## Pourquoi ce n'est pas activé par défaut

Une dépendance git **non joignable casse `cargo build` pour tout le monde** (la
résolution du lockfile échoue, même pour une dépendance `optional`). Dans
l'environnement d'exécution actuel, `CHECKUPAUTO/scirust` n'est pas autorisé :

```
$ cargo generate-lockfile        # avec scirust-rsi = { git = ... }
error: failed to fetch ... CHECKUPAUTO/scirust
  failed to authenticate when downloading repository
```

On garde donc le dépôt **compilable hors-ligne** et on active le moteur réel
quand l'accès est ouvert.

## Les 3 étapes (dans un environnement où scirust est autorisé)

### 1. `Cargo.toml` — pointer la feature sur la dépendance

```toml
[features]
# remplacer `scirust = []` par :
scirust = ["dep:scirust-rsi", "dep:rand"]

[dependencies]
# ajouter :
scirust-rsi = { git = "https://github.com/CHECKUPAUTO/scirust", branch = "master", optional = true }
```

> `rand` est déjà une dépendance optionnelle du projet ; `scirust-rsi` réutilise
> son `StdRng`. (Variante : `scirust = { git = ... }` puis `scirust::rsi`.)

### 2. `src/lib.rs` — aucune action

Le module est déjà déclaré sous la feature :

```rust
#[cfg(feature = "scirust")]
pub mod scirust_bridge;
```

### 3. Compiler / exécuter / tester

```bash
cargo build  --features scirust
cargo run    --features scirust --release --example self_improve_real
cargo test   --features scirust
cargo clippy --features scirust --all-targets
```

## Seul point à vérifier dans le pont

`scirust_bridge::fit()` suppose `Fitness: From<f64>`. Si l'API réelle expose un
autre constructeur (p. ex. `Fitness::new(f64)`), c'est **l'unique ligne** à
ajuster dans `src/scirust_bridge.rs`.

## Garde-fous (identiques dans les deux modes)

- **Sandbox** : le candidat est un AST `Expr` évalué par notre interpréteur
  (`Expr::eval`) — le moteur n'exécute jamais de code généré, ne se modifie pas.
- **Non-régression** : adoption élitiste ⇒ `report.is_monotone()`.
- **Terminaison** : `Guard::max_iters` borne chaque run ; `patience`/`target`
  arrêtent proprement.
- **Reproductible** : même graine ⇒ même run.
