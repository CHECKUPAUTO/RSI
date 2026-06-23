//! Serveur **MCP** (Model Context Protocol) pour le système RSI.
//!
//! Transport stdio, messages JSON-RPC 2.0 délimités par des sauts de ligne.
//! Expose la [`RsiApi`] sous forme d'outils MCP, permettant à un agent IA / LLM
//! de créer un agent auto-améliorant, de l'avancer, d'inspecter son état et
//! d'exporter sa trajectoire.
//!
//! Méthodes JSON-RPC gérées :
//! - `initialize`               → handshake (capacités + infos serveur)
//! - `notifications/initialized`→ notification (ignorée)
//! - `ping`                     → `{}`
//! - `tools/list`               → catalogue d'outils (JSON Schema)
//! - `tools/call`               → exécute un outil et renvoie son résultat
//!
//! Lancement : `cargo run --release --bin rsi-mcp` puis dialogue JSON-RPC
//! ligne par ligne sur stdin/stdout.

use std::io::{self, BufRead, Read, Write};

use rsi::api::RsiApi;
use rsi::json::Json;

const PROTOCOL_VERSION: &str = "2024-11-05";

/// Plafond de taille d'une requête (une ligne JSON-RPC). Un client hostile
/// pourrait sinon envoyer une ligne de plusieurs Go et provoquer un OOM avant
/// même le parsing.
const MAX_LINE_BYTES: usize = 16 * 1024 * 1024; // 16 Mio

fn main() {
    let stdin = io::stdin();
    let stdout = io::stdout();
    let mut out = stdout.lock();
    let mut reader = stdin.lock();
    let mut api = RsiApi::new();

    let mut buf = Vec::with_capacity(4096);
    loop {
        buf.clear();
        // Lecture *bornée* d'une ligne : `take` plafonne CETTE lecture à
        // MAX_LINE_BYTES octets (anti-OOM sur ligne géante).
        let n = match (&mut reader).take(MAX_LINE_BYTES as u64).read_until(b'\n', &mut buf) {
            Ok(0) => break, // EOF
            Ok(n) => n,
            Err(_) => break,
        };
        // Limite atteinte sans fin de ligne ⇒ requête trop grande : on refuse et
        // on coupe proprement (drainer le reste risquerait l'OOM qu'on évite).
        if n >= MAX_LINE_BYTES && buf.last() != Some(&b'\n') {
            let resp = error_response(&Json::Null, -32600, "request line exceeds 16 MiB limit");
            let _ = writeln!(out, "{}", resp.to_string());
            let _ = out.flush();
            break;
        }

        let line = String::from_utf8_lossy(&buf);
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let response = match Json::parse(line) {
            Ok(req) => handle_request(&mut api, &req),
            Err(e) => Some(error_response(&Json::Null, -32700, &format!("parse error: {e}"))),
        };

        if let Some(resp) = response {
            let _ = writeln!(out, "{}", resp.to_string());
            let _ = out.flush();
        }
    }
}

/// Dispatche une requête JSON-RPC. Renvoie `None` pour les notifications.
fn handle_request(api: &mut RsiApi, req: &Json) -> Option<Json> {
    let id = req.get("id").cloned().unwrap_or(Json::Null);
    let method = req.get("method").and_then(|v| v.as_str()).unwrap_or("");
    let is_notification = req.get("id").is_none();

    let result: Result<Json, (i64, String)> = match method {
        "initialize" => Ok(initialize_result()),
        "notifications/initialized" | "initialized" => return None,
        "ping" => Ok(Json::obj()),
        "tools/list" => Ok(tools_list()),
        "tools/call" => tools_call(api, req.get("params")),
        other => Err((-32601, format!("méthode inconnue : '{other}'"))),
    };

    if is_notification {
        return None;
    }

    Some(match result {
        Ok(value) => success_response(&id, value),
        Err((code, msg)) => error_response(&id, code, &msg),
    })
}

fn initialize_result() -> Json {
    let mut caps = Json::obj();
    caps.set("tools", Json::obj());

    let mut server = Json::obj();
    server
        .set("name", Json::Str("rsi-mcp".into()))
        .set("version", Json::Str(env!("CARGO_PKG_VERSION").into()));

    let mut out = Json::obj();
    out.set("protocolVersion", Json::Str(PROTOCOL_VERSION.into()))
        .set("capabilities", caps)
        .set("serverInfo", server)
        .set(
            "instructions",
            Json::Str(
                "Système d'auto-amélioration récursive (RSI). Utilisez \
'rsi_create' pour instancier un agent, 'rsi_run'/'rsi_step' pour le faire \
évoluer, 'rsi_state' pour l'inspecter, 'rsi_export' pour récupérer la \
trajectoire. 'rsi_describe' documente le modèle mathématique."
                    .into(),
            ),
        );
    out
}

/// Description d'un outil MCP (name, description, inputSchema JSON Schema).
fn tool(name: &str, description: &str, properties: Json, required: &[&str]) -> Json {
    let mut schema = Json::obj();
    schema.set("type", Json::Str("object".into())).set("properties", properties);
    if !required.is_empty() {
        let req: Vec<Json> = required.iter().map(|s| Json::Str((*s).into())).collect();
        schema.set("required", Json::Arr(req));
    }
    let mut t = Json::obj();
    t.set("name", Json::Str(name.into()))
        .set("description", Json::Str(description.into()))
        .set("inputSchema", schema);
    t
}

fn prop(ty: &str, desc: &str) -> Json {
    let mut p = Json::obj();
    p.set("type", Json::Str(ty.into())).set("description", Json::Str(desc.into()));
    p
}

fn props(pairs: &[(&str, Json)]) -> Json {
    let mut o = Json::obj();
    for (k, v) in pairs {
        o.set(k, v.clone());
    }
    o
}

fn tools_list() -> Json {
    let id = || ("id", prop("string", "Identifiant de session (défaut: 'default')."));

    let tools = vec![
        tool(
            "rsi_describe",
            "Décrit le système RSI (modèle mathématique) et le catalogue de commandes.",
            Json::obj(),
            &[],
        ),
        tool(
            "rsi_create",
            "Crée (ou remplace) une session d'agent RSI auto-améliorant.",
            props(&[
                id(),
                ("seed", prop("integer", "Graine reproductible (défaut 2026).")),
                ("optimizer", prop("string", "Méta-optimiseur: 'random' ou 'cma' (sep-CMA-ES).")),
                ("dim", prop("integer", "Dimension de chaque composante de S (défaut 6).")),
                ("n_tasks", prop("integer", "Taille de l'échantillon de tâches |Ω| (défaut 1024).")),
                ("n_hardware", prop("integer", "Dimension du vecteur matériel H (défaut 4).")),
                ("n_software", prop("integer", "Dimension du vecteur logiciel O (défaut 4).")),
                ("lambda", prop("number", "Borne de stabilité ‖ΔS‖<λ (défaut 0.5).")),
                ("epsilon", prop("number", "Régression tolérée sur SI_global (défaut 1e-3).")),
            ]),
            &[],
        ),
        tool(
            "rsi_step",
            "Avance la boucle RSI d'un pas et renvoie le rapport du pas.",
            props(&[id()]),
            &[],
        ),
        tool(
            "rsi_run",
            "Avance la boucle RSI de plusieurs pas et renvoie un résumé (gain de SI_global).",
            props(&[id(), ("steps", prop("integer", "Nombre de pas (défaut 100)."))]),
            &[],
        ),
        tool(
            "rsi_run_until",
            "Pilote la boucle RSI jusqu'à un critère d'arrêt motivé (budget, cible, plateau, disjoncteur de criticité).",
            props(&[
                id(),
                ("max_steps", prop("integer", "Budget de pas (défaut 500).")),
                ("target_si", prop("number", "Arrêt si SI_global ≥ cette valeur.")),
                ("plateau_window", prop("integer", "Fenêtre de détection de plateau (défaut 12).")),
                ("plateau_eps", prop("number", "Seuil de pente sous lequel = plateau.")),
                ("breaker_rpn", prop("number", "Disjoncteur : arrêt/rollback si max_rpn dépasse ce seuil.")),
                ("max_seconds", prop("number", "Budget de temps (s).")),
            ]),
            &[],
        ),
        tool(
            "rsi_state",
            "Renvoie un instantané: SI_global, P_eff, capacités (D,M,R,A,C,V), goulot d'étranglement.",
            props(&[id()]),
            &[],
        ),
        tool(
            "rsi_export",
            "Exporte la trajectoire accumulée de la session au format csv ou json.",
            props(&[id(), ("format", prop("string", "'csv' ou 'json' (défaut json)."))]),
            &[],
        ),
        tool("rsi_reset", "Réinitialise la session à partir de sa configuration.", props(&[id()]), &[]),
        tool("rsi_list_sessions", "Liste les sessions actives.", Json::obj(), &[]),
        tool(
            "rsi_refine_new",
            "Crée une session de raffinement piloté par LLM. Le LLM propose des candidats (texte) ; \
             le serveur les évalue en sandbox et n'adopte que les strictement meilleurs et sûrs — \
             le LLM ne contrôle aucun garde-fou. Deux domaines : 'synthesis' (expressions \
             arithmétiques) et 'tuning' (hyperparamètres JSON).",
            props(&[
                id(),
                ("domain", prop("string", "Domaine: 'synthesis' (défaut) ou 'tuning'.")),
                ("target", prop("string", "[synthesis] cible: 'quadratic' (x²+1), 'linear' (2x-1), 'cubic' (x³-x).")),
                ("lo", prop("number", "[synthesis] borne basse d'échantillonnage (défaut -3).")),
                ("hi", prop("number", "[synthesis] borne haute (défaut 3).")),
                ("n", prop("integer", "[synthesis] nombre de points (défaut 30 ; ~30% en held-out).")),
                ("seed", prop("integer", "[synthesis] graine reproductible (défaut 2026).")),
            ]),
            &[],
        ),
        tool(
            "rsi_incumbent",
            "Renvoie l'incumbent courant d'une session de raffinement : expression, score \
             (fraction de cas réussis), score held-out (généralisation), taille, compteurs.",
            props(&[id()]),
            &[],
        ),
        tool(
            "rsi_evaluate",
            "Évalue un candidat SANS l'adopter (sonde pour le raisonnement). Renvoie score, \
             held-out, sûreté et s'il serait adopté. N'altère pas l'incumbent.",
            props(&[
                id(),
                ("candidate", prop("string", "Candidat texte. synthesis: expression sur x (ex: 'x*x + 1'). tuning: config JSON (ex: '{\"top_k\":50,\"chunk\":1024,\"threshold\":0.4}').")),
            ]),
            &["candidate"],
        ),
        tool(
            "rsi_propose",
            "Soumet une ou plusieurs propositions d'expressions. Le serveur parse, vérifie la \
             sûreté (taille bornée), score, et n'adopte qu'élitistement (strictement meilleur ET \
             sûr). Renvoie le détail par proposition et le nouvel incumbent.",
            props(&[
                id(),
                ("proposals", prop("array", "Tableau de candidats (chaînes) : expressions (synthesis) ou configs JSON (tuning).")),
                ("candidate", prop("string", "Alternative: un ou plusieurs candidats, un par ligne.")),
            ]),
            &[],
        ),
    ];

    let mut out = Json::obj();
    out.set("tools", Json::Arr(tools));
    out
}

/// Mappe `tools/call` vers une commande de l'API.
fn tools_call(api: &mut RsiApi, params: Option<&Json>) -> Result<Json, (i64, String)> {
    let params = params.ok_or((-32602, "params manquants".into()))?;
    let name = params
        .get("name")
        .and_then(|v| v.as_str())
        .ok_or((-32602, "nom d'outil manquant".into()))?;
    let empty = Json::obj();
    let args = params.get("arguments").unwrap_or(&empty);

    // outils préfixés 'rsi_' → commandes de l'API
    let command = name.strip_prefix("rsi_").unwrap_or(name);

    match api.handle(command, args) {
        Ok(value) => Ok(tool_text_result(&value, false)),
        Err(e) => Ok(tool_text_result(&Json::Str(e), true)),
    }
}

/// Emballe un résultat dans le format `content` attendu par MCP.
fn tool_text_result(value: &Json, is_error: bool) -> Json {
    let mut text = Json::obj();
    text.set("type", Json::Str("text".into()))
        .set("text", Json::Str(value.to_string()));
    let mut out = Json::obj();
    out.set("content", Json::Arr(vec![text]));
    if is_error {
        out.set("isError", Json::Bool(true));
    }
    out
}

fn success_response(id: &Json, result: Json) -> Json {
    let mut out = Json::obj();
    out.set("jsonrpc", Json::Str("2.0".into()))
        .set("id", id.clone())
        .set("result", result);
    out
}

fn error_response(id: &Json, code: i64, message: &str) -> Json {
    let mut err = Json::obj();
    err.set("code", Json::Num(code as f64)).set("message", Json::Str(message.into()));
    let mut out = Json::obj();
    out.set("jsonrpc", Json::Str("2.0".into()))
        .set("id", id.clone())
        .set("error", err);
    out
}
