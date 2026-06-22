//! ⚙️ Loop Engineering — **L1 : pilote de boucle & critères d'arrêt**.
//!
//! Au-dessus de `RSIAgent::step`, un pilote configurable qui exécute la boucle
//! RSI jusqu'à un critère d'arrêt **motivé** : budget de pas, budget de temps,
//! cible de `SI_global` atteinte, **plateau** (convergence) ou **divergence**
//! (via [`crate::convergence`]). Premier maillon du chantier « Loop
//! Engineering » (cf. `docs/ROADMAP.md`).

use std::time::Instant;

use crate::agent::{RSIAgent, StepReport};
use crate::convergence::{ConvergenceDetector, Trend};

/// Critères d'arrêt du pilote de boucle.
#[derive(Clone, Copy, Debug)]
pub struct LoopConfig {
    /// budget maximal de pas.
    pub max_steps: usize,
    /// arrêt si `SI_global ≥ target` (si défini).
    pub target_si: Option<f64>,
    /// fenêtre d'analyse de tendance (plateau/divergence).
    pub plateau_window: usize,
    /// seuil de pente |slope| sous lequel on déclare un **plateau** (par pas).
    pub plateau_eps: f64,
    /// arrête sur divergence (pente fortement négative) si vrai.
    pub stop_on_divergence: bool,
    /// budget de temps optionnel (secondes).
    pub max_seconds: Option<f64>,
    /// **disjoncteur de criticité (L4)** : si `max_rpn` d'un pas dépasse ce
    /// seuil, la boucle s'arrête (`CircuitBreaker`). `None` = désactivé.
    pub breaker_rpn: Option<f64>,
    /// en cas de déclenchement du disjoncteur, restaure le dernier état sain
    /// (rollback) avant d'arrêter.
    pub rollback_on_breach: bool,
}

impl Default for LoopConfig {
    fn default() -> Self {
        LoopConfig {
            max_steps: 1000,
            target_si: None,
            plateau_window: 12,
            plateau_eps: 1e-4,
            stop_on_divergence: true,
            max_seconds: None,
            breaker_rpn: None,
            rollback_on_breach: false,
        }
    }
}

/// Raison de l'arrêt de la boucle.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StopReason {
    MaxSteps,
    TargetReached,
    Plateau,
    Diverged,
    Timeout,
    /// disjoncteur de criticité déclenché (L4).
    CircuitBreaker,
}

/// Résultat d'un run piloté.
#[derive(Clone, Debug)]
pub struct LoopOutcome {
    pub reports: Vec<StepReport>,
    pub reason: StopReason,
    pub steps: usize,
    /// pente finale de `SI_global` sur la fenêtre.
    pub final_slope: f64,
}

impl RSIAgent {
    /// Exécute la boucle jusqu'à un critère d'arrêt (L1). Retourne la
    /// trajectoire et la **raison** motivée de l'arrêt.
    pub fn run_until(&mut self, cfg: &LoopConfig) -> LoopOutcome {
        let start = Instant::now();
        let mut det = ConvergenceDetector::new(cfg.plateau_window);
        let mut reports = Vec::new();
        let mut reason = StopReason::MaxSteps;
        // dernier état sain (pour rollback du disjoncteur L4)
        let mut last_good = self.snapshot();

        for _ in 0..cfg.max_steps {
            let r = self.step();
            det.push(r.si_global);
            let si = r.si_global;
            let max_rpn = r.max_rpn;
            reports.push(r);

            // §L4 — disjoncteur de criticité
            if let Some(thr) = cfg.breaker_rpn {
                if max_rpn > thr {
                    if cfg.rollback_on_breach {
                        self.restore(&last_good); // retour au dernier état sain
                    }
                    reason = StopReason::CircuitBreaker;
                    break;
                }
                // mémorise l'état comme « sain » tant qu'on est loin du seuil
                if max_rpn < 0.5 * thr {
                    last_good = self.snapshot();
                }
            }

            if let Some(t) = cfg.target_si {
                if si >= t {
                    reason = StopReason::TargetReached;
                    break;
                }
            }
            if let Some(secs) = cfg.max_seconds {
                if start.elapsed().as_secs_f64() >= secs {
                    reason = StopReason::Timeout;
                    break;
                }
            }
            if det.filled() {
                match det.trend(cfg.plateau_eps) {
                    Trend::Plateau => {
                        reason = StopReason::Plateau;
                        break;
                    }
                    Trend::Diverging if cfg.stop_on_divergence => {
                        reason = StopReason::Diverged;
                        break;
                    }
                    _ => {}
                }
            }
        }

        LoopOutcome { steps: reports.len(), final_slope: det.slope(), reports, reason }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stops_on_target() {
        let mut agent = RSIAgent::demo(2026);
        let start = agent.si_global();
        let cfg = LoopConfig {
            max_steps: 500,
            target_si: Some(start + 0.05),
            ..LoopConfig::default()
        };
        let out = agent.run_until(&cfg);
        assert_eq!(out.reason, StopReason::TargetReached);
        assert!(out.reports.last().unwrap().si_global >= start + 0.05);
        assert!(out.steps < 500);
    }

    #[test]
    fn stops_on_plateau_before_max() {
        // l'agent converge vers l'attracteur → plateau détecté avant max_steps
        let mut agent = RSIAgent::demo(7);
        let cfg = LoopConfig {
            max_steps: 2000,
            plateau_window: 15,
            plateau_eps: 1e-4,
            ..LoopConfig::default()
        };
        let out = agent.run_until(&cfg);
        assert_eq!(out.reason, StopReason::Plateau);
        assert!(out.steps < 2000, "doit s'arrêter au plateau, pas au budget");
        assert!(out.final_slope.abs() < 1e-3);
    }

    #[test]
    fn circuit_breaker_trips_and_rolls_back() {
        let mut agent = RSIAgent::demo(2026);
        let cfg = LoopConfig {
            max_steps: 400,
            breaker_rpn: Some(0.1), // seuil bas → déclenchement garanti
            rollback_on_breach: true,
            plateau_window: 9999, // neutralise l'arrêt plateau
            ..LoopConfig::default()
        };
        let out = agent.run_until(&cfg);
        assert_eq!(out.reason, StopReason::CircuitBreaker);
        // rollback : l'état de l'agent est revenu en deçà du dernier pas exécuté
        assert!(agent.t < out.steps, "rollback doit ramener t en arrière");
    }

    #[test]
    fn respects_max_steps() {
        let mut agent = RSIAgent::demo(1);
        let cfg = LoopConfig {
            max_steps: 10,
            plateau_window: 50, // jamais rempli → pas d'arrêt plateau
            target_si: None,
            ..LoopConfig::default()
        };
        let out = agent.run_until(&cfg);
        assert_eq!(out.reason, StopReason::MaxSteps);
        assert_eq!(out.steps, 10);
    }
}
