//! §5 / §6 — AGENT RSI : BOUCLE DISCRÈTE & ÉQUATION D'ONDES DE LA SURFACE
//!
//! ```text
//! S_{t+1}  = S_t + ℳ(S_t, V_t, H, O) + ΔS_appr            (§5)
//! ℳ_{t+1}  = arg max_ℳ SI_global( ℳ(S_t) )                (méta-révision)
//! Σ_I(t+1) = Σ_I(t) + η · ℳ(Σ_I, S, H, O, V) − P           (§6, forme compacte)
//! ```
//!
//! Un pas de l'agent enchaîne :
//!   1. méta-révision : choisir la meilleure politique ℳ (argmax SI) ;
//!   2. proposition d'auto-modification ℳ(S_t) (état + réécriture logicielle) ;
//!   3. apprentissage ΔS_appr via la dynamique continue contrainte (§4) ;
//!   4. application des garde-fous de stabilité ‖ΔS‖ < λ et non-régression.
//!
//! La surface Σ_I n'est pas recalculée explicitement : `SI_global` en est le
//! résumé scalaire (volume sous Σ_I), suivi à chaque pas.

use crate::dynamics::{Dynamics, StabilityConfig, StepInfo};
use crate::meta::{CmaEsMeta, MetaOptimizer, MetaSearch, MetaStrategy};
use crate::state::{delta_norm, CognitiveState, Dims};
use crate::substrate::{Substrate, SubstrateImprover};
use crate::surface::IntelligenceSurface;

/// Rapport d'un pas de la boucle RSI.
#[derive(Clone, Debug)]
pub struct StepReport {
    pub t: usize,
    pub si_global: f64,
    pub delta_si: f64,
    pub p_eff: f64,
    pub state_norm: f64,
    pub meta_delta_norm: f64,
    pub appr: StepInfo,
    pub frac_limited_by_substrate: f64,
    pub capabilities: [f64; 6], // (D,M,R,A,C,V)
}

/// Agent cognitif auto-améliorant.
///
/// La stratégie de méta-recherche est polymorphe ([`MetaSearch`]) : on peut y
/// brancher [`MetaOptimizer`] (recherche aléatoire) ou [`CmaEsMeta`]
/// (sep-CMA-ES) sans changer la boucle.
pub struct RSIAgent {
    pub state: CognitiveState,
    pub substrate: Substrate,
    pub surface: IntelligenceSurface,
    pub strategy: MetaStrategy,
    pub dynamics_cfg: StabilityConfig,
    pub meta: Box<dyn MetaSearch>,
    /// Améliorateur de substrat optionnel (Phase 2 — P_eff *mesuré*). `None`
    /// par défaut → boucle d'origine inchangée.
    pub substrate_opt: Option<Box<dyn SubstrateImprover>>,
    pub t: usize,
}

impl RSIAgent {
    /// Construit un agent à partir de ses sous-systèmes.
    pub fn new(
        state: CognitiveState,
        substrate: Substrate,
        surface: IntelligenceSurface,
        dynamics_cfg: StabilityConfig,
        meta: Box<dyn MetaSearch>,
    ) -> Self {
        let strategy = MetaStrategy::neutral(substrate.o.len());
        RSIAgent {
            state,
            substrate,
            surface,
            strategy,
            dynamics_cfg,
            meta,
            substrate_opt: None,
            t: 0,
        }
    }

    /// Branche un améliorateur de substrat (Phase 2 : P_eff *mesuré* par une
    /// optimisation exécutée, p. ex. Forge). Builder fluide.
    pub fn with_substrate_improver(mut self, improver: Box<dyn SubstrateImprover>) -> Self {
        self.substrate_opt = Some(improver);
        self
    }

    /// Sous-systèmes communs d'un agent de démonstration (reproductible).
    fn demo_parts(seed: u64) -> (CognitiveState, Substrate, IntelligenceSurface) {
        use crate::rng::Rng;
        let mut rng = Rng::new(seed);
        let dims = Dims::uniform(6);
        let state = CognitiveState::random(dims, &mut rng, 0.08);
        let substrate = Substrate::default_with(4, 4, &mut rng);
        let surface = IntelligenceSurface::sample(1024, &mut rng);
        (state, substrate, surface)
    }

    /// Agent de démonstration avec méta-révision par **recherche aléatoire**.
    pub fn demo(seed: u64) -> Self {
        let (state, substrate, surface) = Self::demo_parts(seed);
        let meta = Box::new(MetaOptimizer::new(48, 0.12, seed ^ 0xABCD));
        RSIAgent::new(state, substrate, surface, StabilityConfig::default(), meta)
    }

    /// Agent de démonstration avec méta-révision par **sep-CMA-ES**.
    pub fn demo_cma(seed: u64) -> Self {
        let (state, substrate, surface) = Self::demo_parts(seed);
        let meta = Box::new(CmaEsMeta::new(0, 10, 0.3, seed ^ 0xC3A));
        RSIAgent::new(state, substrate, surface, StabilityConfig::default(), meta)
    }

    /// SI_global courant (volume sous Σ_I).
    pub fn si_global(&self) -> f64 {
        self.surface.si_global(&self.state, &self.substrate)
    }

    /// Un pas de la boucle discrète RSI.
    pub fn step(&mut self) -> StepReport {
        let si_before = self.si_global();

        // 1) méta-révision : ℳ_{t+1} = argmax_ℳ SI_global(ℳ(S_t))
        let (best_strategy, _proj_si) =
            self.meta
                .revise(&self.strategy, &self.state, &self.substrate, &self.surface);
        self.strategy = best_strategy;

        // 2) ℳ(S_t, V_t, H, O) : proposition d'auto-modification (état + logiciel)
        let (meta_delta, new_substrate) = self.strategy.apply(&self.state, &self.substrate);
        let state_after_meta = self.state.add(&meta_delta).clipped(0.0, 1.0);
        let meta_delta_norm = delta_norm(&self.state, &state_after_meta);

        // La réécriture logicielle n'est acceptée que si elle n'abaisse pas P_eff
        // (garde-fou : l'auto-amélioration du substrat ne doit pas régresser).
        let mut substrate = if new_substrate.effective_power() >= self.substrate.effective_power() {
            new_substrate
        } else {
            self.substrate.clone()
        };

        // 2bis) amélioration du substrat exécutée (Phase 2 : P_eff mesuré).
        // Même garde-fou de non-régression de P_eff.
        if let Some(opt) = self.substrate_opt.as_mut() {
            let improved = opt.improve(&substrate);
            if improved.effective_power() >= substrate.effective_power() {
                substrate = improved;
            }
        }

        // 3) ΔS_appr : apprentissage via la dynamique continue contrainte (§4)
        let dynamics = Dynamics::new(&self.surface, self.dynamics_cfg);
        let (next_state, appr) = dynamics.constrained_step(&state_after_meta, &substrate, 1.0);

        // 4) commit de l'état
        self.state = next_state;
        self.substrate = substrate;
        self.t += 1;

        let si_after = self.si_global();
        let bottleneck = self.surface.bottleneck(&self.state, &self.substrate);

        StepReport {
            t: self.t,
            si_global: si_after,
            delta_si: si_after - si_before,
            p_eff: self.substrate.effective_power(),
            state_norm: self.state.norm(),
            meta_delta_norm,
            appr,
            frac_limited_by_substrate: bottleneck.frac_limited_by_substrate,
            capabilities: self.state.capability_array(),
        }
    }

    /// Exécute `n` pas et retourne la trajectoire complète des rapports.
    pub fn run(&mut self, n: usize) -> Vec<StepReport> {
        (0..n).map(|_| self.step()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn si_is_monotone_within_epsilon() {
        let mut agent = RSIAgent::demo(2026);
        let eps = agent.dynamics_cfg.epsilon;
        let reports = agent.run(60);
        for r in &reports {
            // garde-fou de non-régression appliqué à l'étape d'apprentissage
            assert!(r.appr.si_after >= r.appr.si_before - eps - 1e-9);
        }
    }

    #[test]
    fn agent_improves_over_time() {
        let mut agent = RSIAgent::demo(7);
        let start = agent.si_global();
        agent.run(80);
        let end = agent.si_global();
        assert!(end > start, "SI start={start} end={end}");
    }

    #[test]
    fn delta_s_bounded_by_lambda() {
        let mut agent = RSIAgent::demo(99);
        let lam = agent.dynamics_cfg.lambda;
        for r in agent.run(50) {
            assert!(r.appr.delta_norm <= lam + 1e-9);
        }
    }
}
