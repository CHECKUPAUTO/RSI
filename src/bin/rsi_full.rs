//! Démo **complète** : un agent RSI « tout intégré » réunissant les quatre
//! backends réels et la criticité §7 :
//!
//! - **ℳ** méta-optimiseur exécuté par **Forge** ;
//! - **P_eff** substrat mesuré par **Forge** (kernel réel) ;
//! - **C** mémoire contextuelle par **OctaSoma** (k-NN fractal) ;
//! - **audit** hash-chaîné délégué à **CCOS** (`EventLog`) ;
//! - **criticité** AMDEC (risk_global, SI_safe, routage par criticité).
//!
//! Lancement :
//! ```text
//! cargo run --release --bin rsi-full --features "forge octasoma ccos" -- [pas] [graine]
//! ```

use rsi::{
    CcosAudit, CognitiveState, Dims, ForgeMetaSearch, ForgeSubstrate, IntelligenceSurface,
    OctaSomaMemory, RSIAgent, Rng, StabilityConfig, Substrate,
};

fn short(mode: &str) -> &str {
    match mode {
        "regression_competence" => "régression",
        "instabilite_divergence" => "instabilité",
        "derive_valeurs" => "dérive-V",
        "effondrement_substrat" => "substrat",
        "goodhart_surajustement" => "goodhart",
        "empoisonnement_memoire" => "mémoire",
        "wireheading" => "wirehead",
        other => other,
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let steps: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(30);
    let seed: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(2026);

    // --- assemblage de l'agent « tout intégré » ---------------------------- //
    let mut rng = Rng::new(seed);
    let dims = Dims::uniform(6);
    let state = CognitiveState::random(dims, &mut rng, 0.08);
    let substrate = Substrate::default_with(4, 4, &mut rng);
    let surface = IntelligenceSurface::sample(512, &mut rng);
    let state_dim = state.size();

    let mut agent = RSIAgent::new(
        state,
        substrate,
        surface,
        StabilityConfig::default(),
        Box::new(ForgeMetaSearch::new(4, 12, 0.15, seed ^ 0xF0)), // ℳ via Forge
    )
    .with_substrate_improver(Box::new(ForgeSubstrate::new(96, 2, 6, seed ^ 0x5B))) // P_eff réel
    .with_memory(Box::new(OctaSomaMemory::new(state_dim, seed))) // C via OctaSoma
    .with_audit(Box::new(CcosAudit::new(format!("rsi-{seed}")))) // audit via CCOS
    .with_meta_interval(1)
    .with_route_threshold(0.5);

    println!("╔════════════════════════════════════════════════════════════════════════════╗");
    println!("║   RSI — AGENT TOUT INTÉGRÉ   (Forge ℳ + Forge P_eff + OctaSoma C + CCOS audit)║");
    println!("╚════════════════════════════════════════════════════════════════════════════╝");
    println!("seed={seed}  pas={steps}  |Ω|={}  dim(S)={state_dim}", agent.surface.tasks.len());
    println!();
    println!(
        "{:>3} │ {:>7} │ {:>7} │ {:>6} │ {:>6} │ {:>6} │ {:>10} │ {:>5} │ {:>4}",
        "t", "SI", "SI_safe", "P_eff", "risk", "maxRPN", "critique", "%sub", "mém"
    );
    println!("{}", "─".repeat(78));

    let start_si = agent.si_global();
    let reports = agent.run(steps);

    let stride = (steps / 18).max(1);
    for (i, r) in reports.iter().enumerate() {
        if i % stride != 0 && i + 1 != reports.len() {
            continue;
        }
        println!(
            "{:>3} │ {:>7.4} │ {:>7.4} │ {:>6.4} │ {:>6.4} │ {:>6.4} │ {:>10} │ {:>4.0}% │ {:>4}",
            r.t,
            r.si_global,
            r.si_safe,
            r.p_eff,
            r.risk_global,
            r.max_rpn,
            short(r.most_critical),
            r.frac_limited_by_substrate * 100.0,
            agent.memory_len(),
        );
    }
    let end = reports.last().unwrap();
    println!("{}", "─".repeat(78));
    println!(
        "SI_global : {start_si:.4} → {:.4}  ({:+.1} %)   |   SI_safe final : {:.4}   |   P_eff : {:.4}",
        end.si_global,
        (end.si_global - start_si) / start_si * 100.0,
        end.si_safe,
        end.p_eff,
    );

    // --- 1) audit CCOS : intégrité + déterminisme -------------------------- //
    println!("\n▸ Audit CCOS (journal hash-chaîné)");
    println!("   événements enregistrés : {}", agent.audit_len());
    println!("   intégrité de la chaîne : {}", if agent.audit_verify() { "✓ valide" } else { "✗ ALTÉRÉE" });
    if let Some(head) = agent.audit_head() {
        println!("   hash de tête : {}…", &head[..head.len().min(32)]);
    }

    // --- 2) mémoire OctaSoma : rappel d'un contexte proche ----------------- //
    println!("\n▸ Mémoire OctaSoma (rappel du contexte le plus proche de l'état final)");
    let recalled = agent.recall_similar(1);
    if let Some(payload) = recalled.first() {
        if let Some((si, strat)) = rsi::meta::decode_strategy_payload(payload) {
            println!("   contexte rappelé : SI={si:.4}, gain ℳ={:.3}", strat.gain);
            println!("   focus (D,M,R,A,C,V) = {:?}", strat.focus.map(|x| (x * 100.0).round() / 100.0));
        }
    }

    // --- 3) export de la trajectoire --------------------------------------- //
    println!("\n▸ Export de la trajectoire");
    match rsi::report::write_csv(&reports, "rsi_full_trajectory.csv") {
        Ok(()) => println!("   ✓ CSV  : rsi_full_trajectory.csv ({} lignes)", reports.len()),
        Err(e) => println!("   ✗ CSV  : {e}"),
    }
    match rsi::report::write_json(&reports, "rsi_full_trajectory.json") {
        Ok(()) => println!("   ✓ JSON : rsi_full_trajectory.json"),
        Err(e) => println!("   ✗ JSON : {e}"),
    }

    println!("\nLecture : la criticité (risk/maxRPN) et le mode 'critique' montrent où le");
    println!("risque se concentre ; le goulot %sub bascule vers le substrat ; SI_safe ≤ SI ;");
    println!("le journal CCOS rend toute la trajectoire vérifiable et rejouable.");
}
