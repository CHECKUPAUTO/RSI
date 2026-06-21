//! §1 — MORPHOLOGIE DE LA SURFACE D'INTELLIGENCE
//!
//! ```text
//! (Ω, 𝒜, μ)                        espace probabilisé des tâches/contextes
//! Σ_I(t) = { (x, C_réel(x,t)) | x ∈ Ω }     surface (graphe dans Ω×[0,1])
//! C_réel(x,t) = min( Φ_x(S(t)), g_x(P_eff) )
//! SI_global(t) = ∫_Ω C_réel(x,t) dμ(x)      volume sous la surface
//! ```
//!
//! - `x ∈ Ω` : tâche/contexte (profil de besoins sur D,M,R,A,C,V) ;
//! - `μ` : mesure de probabilité (importance/fréquence des tâches) ;
//! - `Φ_x(S)` : compétence *cognitive* de l'agent sur x ;
//! - `g_x(P_eff)` : plafond *physique* imposé par le substrat ;
//! - `C_réel = min(Φ, g)` : goulot d'étranglement cognitif OU matériel ;
//! - `SI_global` : volume sous la surface, estimé par Monte-Carlo sur μ.

use crate::linalg::dot;
use crate::rng::Rng;
use crate::state::CognitiveState;
use crate::substrate::Substrate;

/// Compétence cognitive Φ_x(S) ∈ [0, 1] pour une tâche `task` (profil sur 6
/// composantes) et les niveaux de capacité `caps` = (D,M,R,A,C,V).
///
/// La tâche projette les capacités sur ses besoins ; la compétence est une
/// sigmoïde décalée du produit scalaire (faible si capacités insuffisantes).
fn phi(task: &[f64; 6], caps: &[f64; 6]) -> f64 {
    let raw = dot(task, caps);
    1.0 / (1.0 + (-(raw - 0.5) * 4.0).exp())
}

/// Plafond physique g_x(P_eff) ∈ [0, 1].
///
/// `demand` ∈ [0,1] est l'exigence calculatoire normalisée de la tâche. Une
/// tâche lourde est davantage bridée par un substrat faible : g = P_eff^demand.
fn g(p_eff: f64, demand: f64) -> f64 {
    p_eff.powf(demand)
}

/// Surface d'intelligence Σ_I et fonctionnelle SI_global.
///
/// Ω est échantillonné une seule fois (Monte-Carlo selon μ) ; le même
/// échantillon sert à toutes les évaluations, ce qui rend SI_global
/// comparable d'un pas à l'autre (estimateur cohérent — essentiel pour le
/// garde-fou de non-régression §4 et le `argmax` méta §5).
#[derive(Clone, Debug)]
pub struct IntelligenceSurface {
    /// Profils de tâches, chacun un vecteur de 6 poids (D,M,R,A,C,V).
    pub tasks: Vec<[f64; 6]>,
    /// Exigence calculatoire normalisée de chaque tâche ∈ [0, 1].
    pub demand: Vec<f64>,
    /// Poids de μ, normalisés (∑ = 1).
    pub weights: Vec<f64>,
}

impl IntelligenceSurface {
    /// Tire Ω ~ μ : profils de besoins via Dirichlet, importances uniformes.
    pub fn sample(n_tasks: usize, rng: &mut Rng) -> Self {
        let mut tasks = Vec::with_capacity(n_tasks);
        let mut demand = Vec::with_capacity(n_tasks);
        let mut weights = Vec::with_capacity(n_tasks);

        let alpha = [1.0; 6];
        let mut raw_demand = Vec::with_capacity(n_tasks);
        for _ in 0..n_tasks {
            let d = rng.dirichlet(&alpha);
            let task = [d[0], d[1], d[2], d[3], d[4], d[5]];
            tasks.push(task);
            // exigence calculatoire ∝ concentration des besoins (avant normalisation)
            let raw: f64 = rng.uniform_range(0.5, 2.0);
            raw_demand.push(raw);
            weights.push(rng.uniform_range(0.2, 1.0));
        }
        // normalise la demande dans [0,1]
        let max_d = raw_demand.iter().cloned().fold(f64::MIN, f64::max).max(1e-9);
        for r in &raw_demand {
            demand.push(r / max_d);
        }
        // normalise μ
        let sum_w: f64 = weights.iter().sum::<f64>().max(1e-12);
        for w in weights.iter_mut() {
            *w /= sum_w;
        }

        IntelligenceSurface { tasks, demand, weights }
    }

    /// C_réel(x,t) = min( Φ_x(S), g_x(P_eff) ) pour chaque tâche de Ω.
    pub fn real_capability(&self, state: &CognitiveState, substrate: &Substrate) -> Vec<f64> {
        let p_eff = substrate.effective_power();
        let caps = state.capability_array();
        self.tasks
            .iter()
            .zip(&self.demand)
            .map(|(task, &dem)| phi(task, &caps).min(g(p_eff, dem)))
            .collect()
    }

    /// SI_global(t) = ∫_Ω C_réel dμ ≈ Σ_i w_i · C_réel(x_i).
    pub fn si_global(&self, state: &CognitiveState, substrate: &Substrate) -> f64 {
        let c = self.real_capability(state, substrate);
        dot(&self.weights, &c)
    }

    /// Diagnostic : la compétence est-elle bridée par le cognitif (Φ) ou par
    /// le substrat (g) ?
    pub fn bottleneck(&self, state: &CognitiveState, substrate: &Substrate) -> Bottleneck {
        let p_eff = substrate.effective_power();
        let caps = state.capability_array();
        let mut frac_substrate = 0.0;
        let mut mean_phi = 0.0;
        let mut mean_g = 0.0;
        for ((task, &dem), &w) in self.tasks.iter().zip(&self.demand).zip(&self.weights) {
            let p = phi(task, &caps);
            let gg = g(p_eff, dem);
            if gg < p {
                frac_substrate += w;
            }
            mean_phi += w * p;
            mean_g += w * gg;
        }
        Bottleneck {
            frac_limited_by_substrate: frac_substrate,
            frac_limited_by_cognition: 1.0 - frac_substrate,
            mean_phi,
            mean_g,
        }
    }
}

/// Diagnostic de goulot d'étranglement (cognitif vs substrat).
#[derive(Clone, Copy, Debug)]
pub struct Bottleneck {
    pub frac_limited_by_substrate: f64,
    pub frac_limited_by_cognition: f64,
    pub mean_phi: f64,
    pub mean_g: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::Dims;

    #[test]
    fn si_in_unit_interval() {
        let mut rng = Rng::new(5);
        let surf = IntelligenceSurface::sample(256, &mut rng);
        let state = CognitiveState::random(Dims::uniform(4), &mut rng, 0.3);
        let sub = Substrate::default_with(4, 4, &mut rng);
        let si = surf.si_global(&state, &sub);
        assert!((0.0..=1.0).contains(&si), "SI = {si}");
    }

    #[test]
    fn higher_state_higher_si() {
        let mut rng = Rng::new(9);
        let surf = IntelligenceSurface::sample(512, &mut rng);
        let sub = Substrate::default_with(4, 4, &mut rng);
        let low = CognitiveState::from_vector(&[0.1; 24], Dims::uniform(4));
        let high = CognitiveState::from_vector(&[0.9; 24], Dims::uniform(4));
        assert!(surf.si_global(&high, &sub) >= surf.si_global(&low, &sub));
    }
}
