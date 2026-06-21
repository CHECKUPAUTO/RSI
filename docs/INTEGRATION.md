# Intégration RSI — API, MCP & connexion automatique aux agents IA

Ce document décrit comment piloter le système RSI depuis un **agent IA / LLM**,
de la façade programmatique jusqu'à la connexion automatique aux runtimes
d'agents (openclaw, hermes-agent, …).

Trois niveaux d'intégration, du plus bas au plus automatique :

1. **API Rust** — appel direct de la bibliothèque (`rsi::RsiApi`).
2. **Serveur MCP** — binaire `rsi-mcp`, JSON-RPC 2.0 sur stdio.
3. **Auto-connexion** — binaire `rsi-connect` + `scripts/auto-connect.sh`.

---

## 1. API (JSON in / JSON out)

La façade [`RsiApi`](../src/api.rs) gère des *sessions* d'agents identifiées par
`id` et répond à des commandes JSON. Elle ne dépend d'aucun transport : on peut
l'embarquer dans un service HTTP, une lambda, un bot, etc.

```rust
use rsi::{RsiApi, Json};

let mut api = RsiApi::new();

// 1) créer une session
let mut cfg = Json::parse(r#"{"id":"a","optimizer":"cma","seed":7}"#).unwrap();
api.handle("create", &cfg).unwrap();

// 2) faire évoluer l'agent
let run = Json::parse(r#"{"id":"a","steps":100}"#).unwrap();
let res = api.handle("run", &run).unwrap();
println!("gain SI_global = {:?}", res.get("gain"));

// 3) inspecter / exporter
let state = api.handle("state", &Json::parse(r#"{"id":"a"}"#).unwrap()).unwrap();
let csv   = api.handle("export", &Json::parse(r#"{"id":"a","format":"csv"}"#).unwrap()).unwrap();
```

### Commandes disponibles

| Commande         | Paramètres                                                                 | Résultat |
|------------------|----------------------------------------------------------------------------|----------|
| `describe`       | —                                                                          | modèle + catalogue |
| `create`         | `id, seed, optimizer(random\|cma), dim, n_tasks, n_hardware, n_software, lambda, epsilon, eta0, forgetting, phi_slope, phi_bias` (+ `candidates/explore_scale` ou `population/generations/sigma0`) | état initial |
| `step`           | `id`                                                                       | rapport de pas |
| `run`            | `id, steps`                                                                | résumé + dernier pas |
| `state`          | `id`                                                                       | `si_global, p_eff, capacités, goulot` |
| `export`         | `id, format(csv\|json)`                                                    | trajectoire sérialisée |
| `reset`          | `id`                                                                       | état réinitialisé |
| `list_sessions`  | —                                                                          | sessions actives |

Appelez `describe` à tout moment : le système est **auto-documenté**, ce qui
permet à un LLM de découvrir seul les paramètres.

---

## 2. Serveur MCP (`rsi-mcp`)

Le binaire `rsi-mcp` parle **JSON-RPC 2.0** sur stdio (transport MCP standard,
messages délimités par des sauts de ligne). Il expose la façade API sous forme
d'outils MCP, directement consommables par un client compatible (Claude
Desktop, agents MCP, etc.).

```bash
cargo build --release --bin rsi-mcp
./target/release/rsi-mcp        # attend du JSON-RPC sur stdin
```

### Méthodes JSON-RPC

- `initialize` → handshake (`protocolVersion`, `capabilities`, `serverInfo`)
- `tools/list` → catalogue des 8 outils avec leur *JSON Schema*
- `tools/call` → exécution d'un outil (`{"name":"rsi_run","arguments":{…}}`)
- `ping` → `{}`
- `notifications/initialized` → notification (ignorée)

### Outils exposés

`rsi_describe`, `rsi_create`, `rsi_step`, `rsi_run`, `rsi_state`,
`rsi_export`, `rsi_reset`, `rsi_list_sessions`.

### Exemple d'échange

```jsonc
// → requête
{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}
// ← réponse
{"jsonrpc":"2.0","id":1,"result":{"protocolVersion":"2024-11-05",
  "capabilities":{"tools":{}},"serverInfo":{"name":"rsi-mcp","version":"0.9.0"}}}

// → créer + faire tourner un agent
{"jsonrpc":"2.0","id":2,"method":"tools/call",
 "params":{"name":"rsi_create","arguments":{"id":"a","optimizer":"cma"}}}
{"jsonrpc":"2.0","id":3,"method":"tools/call",
 "params":{"name":"rsi_run","arguments":{"id":"a","steps":100}}}
```

Le résultat d'un outil est renvoyé au format MCP `content` :

```jsonc
{"jsonrpc":"2.0","id":3,"result":{"content":[{"type":"text",
  "text":"{\"ok\":true,\"si_start\":0.146,\"si_end\":0.71,\"gain\":0.564,...}"}]}}
```

---

## 3. Connexion automatique (sans intervention humaine)

Le binaire `rsi-connect` **auto-enregistre** le serveur MCP dans la
configuration des runtimes d'agents, et `scripts/auto-connect.sh` enchaîne
build + enregistrement. Tout est **idempotent** et **fusionne** avec les
serveurs MCP déjà déclarés (rien n'est écrasé hormis la clé `rsi`).

```bash
# tout-en-un : compile puis enregistre auprès de toutes les cibles
./scripts/auto-connect.sh

# ou manuellement
cargo build --release --bins
./target/release/rsi-connect            # enregistre
./target/release/rsi-connect --print    # affiche seulement le descripteur
```

### Cibles & résolution des chemins

| Runtime        | Variable d'env          | Chemin par défaut                          |
|----------------|-------------------------|--------------------------------------------|
| openclaw       | `OPENCLAW_CONFIG`       | `~/.openclaw/mcp.json`                     |
| hermes-agent   | `HERMES_AGENT_CONFIG`   | `~/.config/hermes-agent/mcp.json`          |
| générique MCP  | `MCP_CONFIG`            | `~/.config/mcp/servers.json`               |

Le chemin du binaire MCP est résolu dans l'ordre : `--bin`, puis
`RSI_MCP_BIN`, puis `target/release|debug/rsi-mcp`, puis à côté de
`rsi-connect`, puis le `PATH`.

### Descripteur écrit

```json
{
  "mcpServers": {
    "rsi": {
      "command": "/chemin/absolu/rsi-mcp",
      "args": [],
      "env": { "RSI_DEFAULT_OPTIMIZER": "random" },
      "transport": "stdio",
      "description": "RSI — agent cognitif auto-améliorant …"
    }
  }
}
```

### Déclenchement automatique au démarrage

Pour une connexion **réellement sans intervention**, branchez le script sur le
démarrage de l'environnement :

```bash
# entrypoint Docker / hook SessionStart / systemd / cron @reboot
/chemin/RSI/scripts/auto-connect.sh
```

Au prochain démarrage du runtime (openclaw / hermes-agent), le serveur RSI est
découvert et chargé automatiquement — l'agent IA dispose alors des outils
`rsi_*` sans aucune configuration manuelle.

> **Note sur les schémas de configuration.** Le descripteur émis suit le
> format de facto `mcpServers` (Claude Desktop & la majorité des clients MCP).
> Si openclaw ou hermes-agent attendent un schéma différent, ajustez la
> fonction `server_entry` dans [`src/bin/rsi_connect.rs`](../src/bin/rsi_connect.rs)
> (ou ouvrez une issue) — le reste du mécanisme (fusion, idempotence,
> résolution de chemins) reste inchangé.

---

## Boucle d'auto-amélioration côté agent LLM

Schéma d'usage typique par un agent autonome :

1. `rsi_describe` — comprendre le modèle et les leviers.
2. `rsi_create` — instancier un agent (choix de l'optimiseur, du substrat…).
3. boucle : `rsi_run` (par lots) → `rsi_state` → décider d'ajuster la config
   (ex. augmenter `n_hardware` si `frac_limited_by_substrate → 1`).
4. `rsi_export` — récupérer la trajectoire pour analyse / visualisation.

Le diagnostic `bottleneck` (cognitif vs substrat) renvoyé par `state` permet à
l'agent de **raisonner sur ses propres goulots d'étranglement** et d'orienter
la méta-optimisation — c'est le ressort récursif du système.
