# Environnement Claude Code on the web — recréation & déblocage réseau

Ce dépôt se développe en sessions cloud (`claude.ai/code`). Cette page persiste
la configuration d'environnement requise et le plan des briques qui nécessitent
le réseau.

## Diagnostic

Si `cargo` ne peut récupérer aucune dépendance et que l'API Anthropic est
injoignable, l'environnement est sur **`None`** (aucun accès réseau sortant).

## Débloquer : passer en accès réseau « Trusted »

Le niveau **`Trusted`** (le défaut) autorise déjà tout ce dont RSI a besoin :

| Besoin | Domaine couvert par `Trusted` |
|---|---|
| backend Claude (`llm-claude`) | `api.anthropic.com` |
| dépendances cargo (`ureq`, `wide`, `wasmtime`, `tracing`…) | crates.io |
| git-deps (`CHECKUPAUTO/scirust`, `forge`, `octasoma`, `CCOS`) | `github.com`, `codeload.github.com`, `api.github.com` |
| images Docker | Docker Hub / ghcr / gcr |

Niveaux : **None** (aucun) · **Trusted** (liste blanche) · **Full** (tout) ·
**Custom** (liste blanche perso, + option d'inclure les défauts).

### Étapes (interface web)
1. Icône **cloud** (nom de l'environnement) → sélecteur.
2. Survoler l'environnement → icône **réglages** (engrenage).
3. Sélecteur **Network access** → **Trusted** (ou **Custom** + cocher
   « Also include default list of common package managers »).
4. Renseigner les **Environment variables** (cf. `.env.example`) et le
   **Setup script** (cf. `scripts/web-setup.sh`).
5. Enregistrer. *(Changer le réseau ou le setup relance le cache au prochain
   démarrage — normal.)*

> Variables d'env : format `.env`, `KEY=value` sans guillemets. **Pas de coffre
> à secrets** — visibles par les éditeurs de l'environnement.

## Plan prêt-à-appliquer : transport TLS pour Claude

La logique Claude (requête Messages API + parsing) est déjà livrée et **testée
hors-ligne** (`src/llm.rs`, feature `llm-claude`, transport injecté via
`ClaudeTransport`). Une fois en `Trusted`, il ne reste qu'à brancher une pile
HTTPS réelle. Approche recommandée : une feature `llm-claude-ureq` tirant `ureq`
(client bloquant, sans tokio).

**À appliquer puis vérifier en environnement réseau** (non committé tant que
non compilable hors-ligne) :

1. `Cargo.toml` :
   ```toml
   [features]
   llm-claude-ureq = ["llm-claude", "dep:ureq"]

   [dependencies]
   ureq = { version = "2", optional = true }
   ```

2. `src/llm.rs` (gated `#[cfg(feature = "llm-claude-ureq")]`) :
   ```rust
   /// Transport HTTPS réel pour Claude via `ureq` (bloquant, rustls).
   pub struct UreqTransport;

   impl ClaudeTransport for UreqTransport {
       fn post_json(&self, url: &str, headers: &[(String, String)], body: &str)
           -> Result<String, String> {
           let mut req = ureq::post(url);
           for (k, v) in headers {
               req = req.set(k, v);
           }
           match req.send_string(body) {
               Ok(resp) => resp.into_string().map_err(|e| e.to_string()),
               // 4xx/5xx : ureq renvoie Err(Status) avec un corps JSON d'erreur
               // Anthropic — on le transmet à parse_claude_response via Ok.
               Err(ureq::Error::Status(_, resp)) => {
                   resp.into_string().map_err(|e| e.to_string())
               }
               Err(e) => Err(e.to_string()),
           }
       }
   }
   ```

3. Vérifier : `cargo build --features llm-claude-ureq` puis un test
   d'intégration réel (clé API via `ANTHROPIC_API_KEY`), p. ex. brancher
   `ClaudeClient::new(UreqTransport, key, "claude-sonnet-4-6")` dans
   `ascend_llm` sur le domaine `prompt`.

## Suite (ordre de valeur, une fois en Trusted)
1. Transport TLS Claude (ci-dessus).
2. Vraie git-dep `CHECKUPAUTO/scirust` (remplacer la reconstruction vendorisée).
3. SIMD de `si_global` via la crate `wide` (sans `unsafe`).
4. Sandbox WASM (`wasmtime`) — 4ᵉ domaine (exécution réelle isolée).
5. Observabilité `tracing` + export Prometheus.

## Référence
Doc officielle : https://code.claude.com/docs/en/claude-code-on-the-web
