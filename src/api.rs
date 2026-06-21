//! Façade **API** du système RSI : commandes JSON in / JSON out.
//!
//! C'est le socle d'intégration neutre (sans transport) réutilisé par le
//! serveur MCP (binaire `rsi-mcp`) et utilisable directement depuis n'importe
//! quel code hôte. Une [`RsiApi`] gère plusieurs *sessions* d'agents,
//! identifiées par `id`, et répond à un petit jeu de commandes :
//!
//! | Commande         | Effet                                                        |
//! |------------------|--------------------------------------------------------------|
//! | `describe`       | Décrit le système et le catalogue de commandes               |
//! | `create`         | Crée/replace une session d'agent (config JSON)               |
//! | `step`           | Avance d'un pas la boucle RSI                                |
//! | `run`            | Avance de `steps` pas, renvoie un résumé                     |
//! | `state`          | Instantané (SI_global, P_eff, capacités, goulot)            |
//! | `export`         | Exporte la trajectoire (`format`: `csv` \| `json`)          |
//! | `reset`          | Réinitialise la session à partir de sa config               |
//! | `list_sessions`  | Liste les sessions actives                                  |

use std::collections::BTreeMap;

use crate::agent::{RSIAgent, StepReport};
use crate::dynamics::StabilityConfig;
use crate::json::Json;
use crate::meta::{CmaEsMeta, MetaOptimizer, MetaSearch};
use crate::rng::Rng;
use crate::state::{CognitiveState, Dims};
use crate::substrate::Substrate;
use crate::surface::IntelligenceSurface;
use crate::{report, surface};

/// Résultat d'une commande : un JSON, ou un message d'erreur.
pub type ApiResult = Result<Json, String>;

struct Session {
    agent: RSIAgent,
    history: Vec<StepReport>,
    config: Json,
}

/// Gestionnaire de sessions d'agents RSI piloté par commandes JSON.
#[derive(Default)]
pub struct RsiApi {
    sessions: BTreeMap<String, Session>,
}

impl RsiApi {
    pub fn new() -> Self {
        RsiApi { sessions: BTreeMap::new() }
    }

    /// Point d'entrée unique : dispatche `command` avec ses `params`.
    pub fn handle(&mut self, command: &str, params: &Json) -> ApiResult {
        match command {
            "describe" => Ok(describe()),
            "create" => self.cmd_create(params),
            "step" => self.cmd_step(params),
            "run" => self.cmd_run(params),
            "state" => self.cmd_state(params),
            "export" => self.cmd_export(params),
            "reset" => self.cmd_reset(params),
            "list_sessions" => Ok(self.cmd_list()),
            other => Err(format!("commande inconnue : '{other}'")),
        }
    }

    // ------------------------------------------------------------------ //
    fn session_id(params: &Json) -> String {
        params
            .get("id")
            .and_then(|v| v.as_str())
            .unwrap_or("default")
            .to_string()
    }

    fn cmd_create(&mut self, params: &Json) -> ApiResult {
        let id = Self::session_id(params);
        let agent = build_agent(params)?;
        let info = snapshot_json(&agent, 0);
        self.sessions.insert(
            id.clone(),
            Session { agent, history: Vec::new(), config: params.clone() },
        );
        let mut out = Json::obj();
        out.set("ok", Json::Bool(true))
            .set("id", Json::Str(id))
            .set("initial", info);
        Ok(out)
    }

    fn cmd_reset(&mut self, params: &Json) -> ApiResult {
        let id = Self::session_id(params);
        let cfg = self
            .sessions
            .get(&id)
            .map(|s| s.config.clone())
            .ok_or_else(|| format!("session inconnue : '{id}'"))?;
        let agent = build_agent(&cfg)?;
        let info = snapshot_json(&agent, 0);
        self.sessions
            .insert(id.clone(), Session { agent, history: Vec::new(), config: cfg });
        let mut out = Json::obj();
        out.set("ok", Json::Bool(true)).set("id", Json::Str(id)).set("initial", info);
        Ok(out)
    }

    fn session_mut(&mut self, id: &str) -> Result<&mut Session, String> {
        self.sessions
            .get_mut(id)
            .ok_or_else(|| format!("session inconnue : '{id}' (appelez d'abord 'create')"))
    }

    fn cmd_step(&mut self, params: &Json) -> ApiResult {
        let id = Self::session_id(params);
        let s = self.session_mut(&id)?;
        let report = s.agent.step();
        s.history.push(report.clone());
        Ok(step_report_json(&report))
    }

    fn cmd_run(&mut self, params: &Json) -> ApiResult {
        let id = Self::session_id(params);
        let steps = params.get("steps").and_then(|v| v.as_usize()).unwrap_or(100);
        let s = self.session_mut(&id)?;

        let si_start = s.agent.si_global();
        let reports = s.agent.run(steps);
        let last = reports.last().cloned();
        s.history.extend(reports);

        let mut out = Json::obj();
        out.set("ok", Json::Bool(true))
            .set("id", Json::Str(id))
            .set("steps", Json::Num(steps as f64))
            .set("si_start", Json::Num(si_start))
            .set("total_steps", Json::Num(s.agent.t as f64));
        if let Some(r) = last {
            out.set("si_end", Json::Num(r.si_global))
                .set("gain", Json::Num(r.si_global - si_start))
                .set("last", step_report_json(&r));
        }
        Ok(out)
    }

    fn cmd_state(&mut self, params: &Json) -> ApiResult {
        let id = Self::session_id(params);
        let s = self.session_mut(&id)?;
        Ok(snapshot_json(&s.agent, s.agent.t))
    }

    fn cmd_export(&mut self, params: &Json) -> ApiResult {
        let id = Self::session_id(params);
        let format = params
            .get("format")
            .and_then(|v| v.as_str())
            .unwrap_or("json")
            .to_string();
        let s = self.session_mut(&id)?;
        let data = match format.as_str() {
            "csv" => report::to_csv(&s.history),
            "json" => report::to_json(&s.history),
            other => return Err(format!("format inconnu : '{other}' (csv|json)")),
        };
        let mut out = Json::obj();
        out.set("ok", Json::Bool(true))
            .set("format", Json::Str(format))
            .set("rows", Json::Num(s.history.len() as f64))
            .set("data", Json::Str(data));
        Ok(out)
    }

    fn cmd_list(&self) -> Json {
        let arr: Vec<Json> = self
            .sessions
            .iter()
            .map(|(id, s)| {
                let mut o = Json::obj();
                o.set("id", Json::Str(id.clone()))
                    .set("t", Json::Num(s.agent.t as f64))
                    .set("si_global", Json::Num(s.agent.si_global()))
                    .set("history_len", Json::Num(s.history.len() as f64));
                o
            })
            .collect();
        let mut out = Json::obj();
        out.set("sessions", Json::Arr(arr));
        out
    }
}

// ---------------------------------------------------------------------- //
// Construction d'un agent depuis une config JSON
// ---------------------------------------------------------------------- //

fn build_agent(cfg: &Json) -> Result<RSIAgent, String> {
    let seed = cfg.get("seed").and_then(|v| v.as_u64()).unwrap_or(2026);
    let dim = cfg.get("dim").and_then(|v| v.as_usize()).unwrap_or(6).max(1);
    let n_tasks = cfg.get("n_tasks").and_then(|v| v.as_usize()).unwrap_or(1024).max(16);
    let n_hw = cfg.get("n_hardware").and_then(|v| v.as_usize()).unwrap_or(4).max(1);
    let n_sw = cfg.get("n_software").and_then(|v| v.as_usize()).unwrap_or(4).max(1);

    let mut rng = Rng::new(seed);
    let state = CognitiveState::random(Dims::uniform(dim), &mut rng, 0.08);
    let substrate = Substrate::default_with(n_hw, n_sw, &mut rng);

    // modèle de compétence configurable (slope / bias)
    let slope = cfg.get("phi_slope").and_then(|v| v.as_f64()).unwrap_or(4.0);
    let bias = cfg.get("phi_bias").and_then(|v| v.as_f64()).unwrap_or(0.5);
    let capability = Box::new(surface::SigmoidCapability { slope, bias });
    let ceiling = Box::new(surface::PowerCeiling);
    let surface = IntelligenceSurface::sample_with(n_tasks, &mut rng, capability, ceiling);

    let mut stab = StabilityConfig::default();
    if let Some(v) = cfg.get("lambda").and_then(|v| v.as_f64()) {
        stab.lambda = v;
    }
    if let Some(v) = cfg.get("epsilon").and_then(|v| v.as_f64()) {
        stab.epsilon = v;
    }
    if let Some(v) = cfg.get("eta0").and_then(|v| v.as_f64()) {
        stab.eta0 = v;
    }
    if let Some(v) = cfg.get("forgetting").and_then(|v| v.as_f64()) {
        stab.forgetting = v;
    }

    let optimizer = cfg.get("optimizer").and_then(|v| v.as_str()).unwrap_or("random");
    let meta: Box<dyn MetaSearch> = match optimizer {
        "cma" | "cma-es" | "sep-cma-es" => {
            let pop = cfg.get("population").and_then(|v| v.as_usize()).unwrap_or(0);
            let gen = cfg.get("generations").and_then(|v| v.as_usize()).unwrap_or(10);
            let sigma0 = cfg.get("sigma0").and_then(|v| v.as_f64()).unwrap_or(0.3);
            Box::new(CmaEsMeta::new(pop, gen, sigma0, seed ^ 0xC3A))
        }
        "random" | "neighborhood" => {
            let cand = cfg.get("candidates").and_then(|v| v.as_usize()).unwrap_or(48);
            let scale = cfg.get("explore_scale").and_then(|v| v.as_f64()).unwrap_or(0.12);
            Box::new(MetaOptimizer::new(cand, scale, seed ^ 0xABCD))
        }
        other => return Err(format!("optimiseur inconnu : '{other}' (random|cma)")),
    };

    Ok(RSIAgent::new(state, substrate, surface, stab, meta))
}

// ---------------------------------------------------------------------- //
// Conversion en JSON
// ---------------------------------------------------------------------- //

fn step_report_json(r: &StepReport) -> Json {
    let mut appr = Json::obj();
    appr.set("si_before", Json::Num(r.appr.si_before))
        .set("si_after", Json::Num(r.appr.si_after))
        .set("delta_norm", Json::Num(r.appr.delta_norm))
        .set("clamped_to_lambda", Json::Bool(r.appr.clamped_to_lambda))
        .set("backtracks", Json::Num(r.appr.backtracks as f64))
        .set("step_factor", Json::Num(r.appr.step_factor));

    let mut out = Json::obj();
    out.set("t", Json::Num(r.t as f64))
        .set("si_global", Json::Num(r.si_global))
        .set("delta_si", Json::Num(r.delta_si))
        .set("p_eff", Json::Num(r.p_eff))
        .set("state_norm", Json::Num(r.state_norm))
        .set("meta_delta_norm", Json::Num(r.meta_delta_norm))
        .set("frac_limited_by_substrate", Json::Num(r.frac_limited_by_substrate))
        .set("appr", appr)
        .set("capabilities", capabilities_json(&r.capabilities));
    out
}

fn capabilities_json(caps: &[f64; 6]) -> Json {
    let mut o = Json::obj();
    o.set("D", Json::Num(caps[0]))
        .set("M", Json::Num(caps[1]))
        .set("R", Json::Num(caps[2]))
        .set("A", Json::Num(caps[3]))
        .set("C", Json::Num(caps[4]))
        .set("V", Json::Num(caps[5]));
    o
}

fn snapshot_json(agent: &RSIAgent, t: usize) -> Json {
    let b = agent.surface.bottleneck(&agent.state, &agent.substrate);
    let mut bottleneck = Json::obj();
    bottleneck
        .set("frac_limited_by_substrate", Json::Num(b.frac_limited_by_substrate))
        .set("frac_limited_by_cognition", Json::Num(b.frac_limited_by_cognition))
        .set("mean_phi", Json::Num(b.mean_phi))
        .set("mean_g", Json::Num(b.mean_g));

    let mut out = Json::obj();
    out.set("t", Json::Num(t as f64))
        .set("si_global", Json::Num(agent.si_global()))
        .set("p_eff", Json::Num(agent.substrate.effective_power()))
        .set("state_norm", Json::Num(agent.state.norm()))
        .set("capabilities", capabilities_json(&agent.state.capability_array()))
        .set("bottleneck", bottleneck);
    out
}

/// Description auto-documentée du système et des commandes (pour les agents IA).
pub fn describe() -> Json {
    let mut out = Json::obj();
    out.set("name", Json::Str("RSI — Recursive Self-Improvement".into()))
        .set("version", Json::Str(env!("CARGO_PKG_VERSION").into()))
        .set(
            "summary",
            Json::Str(
                "Modèle dynamique d'un agent cognitif dont la surface de \
compétence Σ_I se déforme sous l'effet de l'apprentissage, du substrat \
matériel/logiciel et d'une méta-optimisation récursive, sous garde-fous \
de stabilité (‖ΔS‖<λ et non-régression de SI_global)."
                    .into(),
            ),
        );

    let commands = [
        ("describe", "Décrit le système et les commandes."),
        ("create", "Crée une session. Params: id, seed, optimizer(random|cma), dim, n_tasks, n_hardware, n_software, lambda, epsilon, eta0, forgetting, phi_slope, phi_bias, (+ candidates/explore_scale ou population/generations/sigma0)."),
        ("step", "Un pas de boucle RSI. Params: id."),
        ("run", "Avance de N pas. Params: id, steps."),
        ("state", "Instantané: si_global, p_eff, capacités, goulot. Params: id."),
        ("export", "Exporte la trajectoire. Params: id, format(csv|json)."),
        ("reset", "Réinitialise la session. Params: id."),
        ("list_sessions", "Liste les sessions actives."),
    ];
    let arr: Vec<Json> = commands
        .iter()
        .map(|(name, desc)| {
            let mut o = Json::obj();
            o.set("name", Json::Str((*name).into()))
                .set("description", Json::Str((*desc).into()));
            o
        })
        .collect();
    out.set("commands", Json::Arr(arr));
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_run_export_cycle() {
        let mut api = RsiApi::new();
        let mut cfg = Json::obj();
        cfg.set("id", Json::Str("s1".into()))
            .set("seed", Json::Num(7.0))
            .set("optimizer", Json::Str("random".into()));
        api.handle("create", &cfg).unwrap();

        let mut run = Json::obj();
        run.set("id", Json::Str("s1".into())).set("steps", Json::Num(40.0));
        let res = api.handle("run", &run).unwrap();
        let gain = res.get("gain").and_then(|v| v.as_f64()).unwrap();
        assert!(gain > 0.0, "gain attendu positif, obtenu {gain}");

        let mut exp = Json::obj();
        exp.set("id", Json::Str("s1".into())).set("format", Json::Str("csv".into()));
        let res = api.handle("export", &exp).unwrap();
        assert_eq!(res.get("rows").and_then(|v| v.as_f64()), Some(40.0));
    }

    #[test]
    fn cma_optimizer_via_api() {
        let mut api = RsiApi::new();
        let mut cfg = Json::obj();
        cfg.set("optimizer", Json::Str("cma".into())).set("seed", Json::Num(3.0));
        api.handle("create", &cfg).unwrap();
        let res = api.handle("run", &{
            let mut r = Json::obj();
            r.set("steps", Json::Num(30.0));
            r
        }).unwrap();
        assert!(res.get("si_end").and_then(|v| v.as_f64()).unwrap() > 0.0);
    }

    #[test]
    fn unknown_command_errors() {
        let mut api = RsiApi::new();
        assert!(api.handle("nope", &Json::obj()).is_err());
    }
}
