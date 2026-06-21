//! Démonstration de la boucle RSI : simule un agent auto-améliorant et
//! affiche la trajectoire de son intelligence globale `SI_global`.
//!
//! Usage :
//! ```text
//! cargo run --release --bin rsi-demo -- [n_steps] [seed]
//! ```

use rsi::RSIAgent;

fn main() {
    let mut args = std::env::args().skip(1);
    let n_steps: usize = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(120);
    let seed: u64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(2026);

    let mut agent = RSIAgent::demo(seed);

    println!("╔══════════════════════════════════════════════════════════════════════╗");
    println!("║   RSI — Auto-amélioration récursive (formulation géométrique v9)       ║");
    println!("╚══════════════════════════════════════════════════════════════════════╝");
    println!("seed = {seed}   |   pas = {n_steps}   |   tâches |Ω| = {}", agent.surface.tasks.len());
    println!();
    println!(
        "{:>4} │ {:>8} │ {:>8} │ {:>7} │ {:>7} │ {:>6} │ {}",
        "t", "SI_glob", "ΔSI", "P_eff", "‖ΔS‖", "%subst", "capacités (D M R A C V)"
    );
    println!("{}", "─".repeat(96));

    let start_si = agent.si_global();
    let reports = agent.run(n_steps);

    // n'affiche qu'un échantillon de lignes pour rester lisible
    let stride = (n_steps / 24).max(1);
    for (i, r) in reports.iter().enumerate() {
        if i % stride != 0 && i + 1 != reports.len() {
            continue;
        }
        let caps = r
            .capabilities
            .iter()
            .map(|c| format!("{c:.2}"))
            .collect::<Vec<_>>()
            .join(" ");
        println!(
            "{:>4} │ {:>8.4} │ {:>+8.4} │ {:>7.4} │ {:>7.4} │ {:>5.0}% │ {}",
            r.t,
            r.si_global,
            r.delta_si,
            r.p_eff,
            r.appr.delta_norm,
            r.frac_limited_by_substrate * 100.0,
            caps,
        );
    }

    let end = reports.last().unwrap();
    println!("{}", "─".repeat(96));
    println!(
        "SI_global : {start_si:.4} → {:.4}   (gain absolu {:+.4}, +{:.1} %)",
        end.si_global,
        end.si_global - start_si,
        if start_si > 0.0 {
            (end.si_global - start_si) / start_si * 100.0
        } else {
            0.0
        }
    );
    println!(
        "P_eff final : {:.4}   |   goulot substrat : {:.0} %   |   ‖S‖ : {:.3}",
        end.p_eff,
        end.frac_limited_by_substrate * 100.0,
        end.state_norm,
    );
}
