//! §4 — DYNAMIQUE CONTINUE AVEC CONTRAINTE DE STABILITÉ
//!
//! ```text
//! dS/dt = η(S,H,O) · [ L(D) + E(A,V) + U(H,O) ] − P(S)
//!
//! Contraintes (au pas discret) :
//!     ‖ΔS‖ < λ
//!     SI_global(t+1) ≥ SI_global(t) − ε
//! ```
//!
//! - `η(S,H,O)` : taux d'apprentissage effectif (modulé par substrat et état) ;
//! - `L(D)`     : apprentissage tiré des connaissances ;
//! - `E(A,V)`   : exploration dirigée par autonomie A, alignée sur valeurs V ;
//! - `U(H,O)`   : uplift apporté par le substrat ;
//! - `P(S)`     : pénalité dissipative (oubli, coût d'entretien).
//!
//! Les deux contraintes sont des garde-fous appliqués *après* le pas brut :
//! on borne l'amplitude (`‖ΔS‖ < λ`) puis on atténue (line search) tout pas
//! qui ferait régresser SI_global au-delà de ε.

use crate::linalg::mean;
use crate::state::{delta_norm, CognitiveState};
use crate::substrate::Substrate;
use crate::surface::IntelligenceSurface;

/// Garde-fous et hyperparamètres de la dynamique (§4).
#[derive(Clone, Copy, Debug)]
pub struct StabilityConfig {
    /// λ : borne maximale sur ‖ΔS‖.
    pub lambda: f64,
    /// ε : régression tolérée sur SI_global.
    pub epsilon: f64,
    /// η₀ : taux d'apprentissage de base.
    pub eta0: f64,
    /// Coefficient de P(S) (dissipation / oubli).
    pub forgetting: f64,
    /// ε adaptatif : si vrai, la tolérance de non-régression devient
    /// `ε + epsilon_z · stderr(SI_global)`, pour ne pas pénaliser une variation
    /// sous le bruit d'échantillonnage Monte-Carlo (§2). `false` par défaut.
    pub adaptive_epsilon: bool,
    /// multiplicateur z de l'erreur-type dans l'ε adaptatif.
    pub epsilon_z: f64,
}

impl Default for StabilityConfig {
    fn default() -> Self {
        StabilityConfig {
            lambda: 0.5,
            epsilon: 1e-3,
            eta0: 0.15,
            forgetting: 0.02,
            adaptive_epsilon: false,
            epsilon_z: 2.0,
        }
    }
}

/// Rapport sur l'application d'un pas contraint.
#[derive(Clone, Copy, Debug)]
pub struct StepInfo {
    pub si_before: f64,
    pub si_after: f64,
    pub delta_norm: f64,
    pub clamped_to_lambda: bool,
    pub backtracks: u32,
    pub step_factor: f64,
}

/// Opérateur de dynamique : calcule dS/dt et applique le pas sous contraintes.
pub struct Dynamics<'a> {
    pub surface: &'a IntelligenceSurface,
    pub config: StabilityConfig,
}

impl<'a> Dynamics<'a> {
    pub fn new(surface: &'a IntelligenceSurface, config: StabilityConfig) -> Self {
        Dynamics { surface, config }
    }

    /// η(S,H,O) ∈ (0, η₀] — meilleur substrat & mémoire ⇒ apprentissage plus
    /// rapide ; la saturation des compétences le freine.
    ///
    /// La saturation ne porte que sur les **compétences cognitives saturantes**
    /// (D, M, R) — pas sur l'autonomie A ni les valeurs V, qui ne sont pas des
    /// capacités dont la progression ralentirait l'apprentissage. On utilise
    /// `capability_array()` (tableau stack `[f64;6]`, zero-alloc) au lieu de
    /// `to_vector()` qui allouait un `Vec<f64>` à chaque appel (or `eta` est
    /// appelé dans `velocity` → `constrained_step`, lui-même dans une boucle
    /// de line search jusqu'à 20× par pas).
    pub fn eta(&self, state: &CognitiveState, substrate: &Substrate) -> f64 {
        let p_eff = substrate.effective_power();
        let caps = state.capability_array(); // [D, M, R, A, C, V], zero-alloc
        let memory = caps[4]; // C
        let saturation = (1.0 - (caps[0] + caps[1] + caps[2]) / 3.0).clamp(0.0, 1.0);
        self.config.eta0 * p_eff * (0.5 + 0.5 * memory) * saturation
    }

    /// L(D) — apprentissage tiré des connaissances ; irrigue D, M, R.
    pub fn term_l(&self, state: &CognitiveState) -> CognitiveState {
        let drive = mean(&state.d) + 0.1;
        let mut out = CognitiveState::zeros(state.dims());
        out.d.iter_mut().for_each(|x| *x = drive);
        out.m.iter_mut().for_each(|x| *x = 0.7 * drive);
        out.r.iter_mut().for_each(|x| *x = 0.8 * drive);
        out
    }

    /// E(A,V) — exploration dirigée par l'autonomie, alignée sur les valeurs.
    /// Le produit A·V garantit qu'autonomie sans valeurs (ou l'inverse) ne
    /// produit pas d'exploration utile.
    pub fn term_e(&self, state: &CognitiveState) -> CognitiveState {
        let autonomy = mean(&state.a);
        let alignment = mean(&state.v);
        let drive = autonomy * alignment;
        let mut out = CognitiveState::zeros(state.dims());
        out.r.iter_mut().for_each(|x| *x = drive);
        out.a.iter_mut().for_each(|x| *x = 0.5 * autonomy);
        out.c.iter_mut().for_each(|x| *x = 0.6 * drive);
        out
    }

    /// U(H,O) — uplift apporté par le substrat ; irrigue M et D.
    pub fn term_u(&self, state: &CognitiveState, substrate: &Substrate) -> CognitiveState {
        let p_eff = substrate.effective_power();
        let mut out = CognitiveState::zeros(state.dims());
        out.m.iter_mut().for_each(|x| *x = p_eff);
        out.d.iter_mut().for_each(|x| *x = 0.4 * p_eff);
        out
    }

    /// P(S) — pénalité dissipative proportionnelle à l'état (empêche la
    /// divergence et force l'amélioration continue).
    pub fn term_p(&self, state: &CognitiveState) -> CognitiveState {
        state.scaled(self.config.forgetting)
    }

    /// Champ de vitesse dS/dt = η · [ L + E + U ] − P.
    pub fn velocity(&self, state: &CognitiveState, substrate: &Substrate) -> CognitiveState {
        let eta = self.eta(state, substrate);
        let growth = self
            .term_l(state)
            .add(&self.term_e(state))
            .add(&self.term_u(state, substrate));
        growth.scaled(eta).sub(&self.term_p(state))
    }

    /// Pas discret sous garde-fous ‖ΔS‖ < λ et non-régression de SI_global.
    pub fn constrained_step(
        &self,
        state: &CognitiveState,
        substrate: &Substrate,
        dt: f64,
    ) -> (CognitiveState, StepInfo) {
        let cfg = self.config;
        // ε effectif : adaptatif au bruit Monte-Carlo si demandé (§2).
        let (si_before, eps) = if cfg.adaptive_epsilon {
            let (si, se) = self.surface.si_global_stats(state, substrate);
            (si, cfg.epsilon + cfg.epsilon_z * se)
        } else {
            (self.surface.si_global(state, substrate), cfg.epsilon)
        };

        // 1) pas brut issu de la dynamique continue
        let mut delta = self.velocity(state, substrate).scaled(dt);

        // 2) contrainte d'amplitude : ‖ΔS‖ < λ (projection radiale)
        let dn = delta.norm();
        let mut clamped = false;
        if dn >= cfg.lambda {
            delta = delta.scaled((cfg.lambda * 0.999) / (dn + 1e-12));
            clamped = true;
        }

        let mut candidate = state.add(&delta).clipped(0.0, 1.0);

        // 3) garde-fou de non-régression : SI(t+1) ≥ SI(t) − ε  (line search)
        let mut si_after = self.surface.si_global(&candidate, substrate);
        let mut backtracks = 0u32;
        let mut factor = 1.0;
        while si_after < si_before - eps && backtracks < 20 {
            factor *= 0.5;
            candidate = state.add(&delta.scaled(factor)).clipped(0.0, 1.0);
            si_after = self.surface.si_global(&candidate, substrate);
            backtracks += 1;
        }

        // sécurité : si même un pas infinitésimal régresse, on reste sur place
        if si_after < si_before - eps {
            candidate = state.clone();
            si_after = si_before;
            factor = 0.0;
        }

        let info = StepInfo {
            si_before,
            si_after,
            delta_norm: delta_norm(state, &candidate),
            clamped_to_lambda: clamped,
            backtracks,
            step_factor: factor,
        };
        (candidate, info)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;
    use crate::state::Dims;

    #[test]
    fn step_respects_lambda() {
        let mut rng = Rng::new(2);
        let surf = IntelligenceSurface::sample(256, &mut rng);
        let cfg = StabilityConfig::default();
        let dyn_ = Dynamics::new(&surf, cfg);
        let state = CognitiveState::random(Dims::uniform(4), &mut rng, 0.3);
        let sub = Substrate::default_with(4, 4, &mut rng);
        let (_next, info) = dyn_.constrained_step(&state, &sub, 1.0);
        assert!(info.delta_norm <= cfg.lambda + 1e-9, "‖ΔS‖={}", info.delta_norm);
    }

    #[test]
    fn step_never_regresses_beyond_epsilon() {
        let mut rng = Rng::new(11);
        let surf = IntelligenceSurface::sample(256, &mut rng);
        let cfg = StabilityConfig::default();
        let dyn_ = Dynamics::new(&surf, cfg);
        let mut state = CognitiveState::random(Dims::uniform(4), &mut rng, 0.3);
        let sub = Substrate::default_with(4, 4, &mut rng);
        for _ in 0..50 {
            let (next, info) = dyn_.constrained_step(&state, &sub, 1.0);
            assert!(info.si_after >= info.si_before - cfg.epsilon - 1e-9);
            state = next;
        }
    }
}
