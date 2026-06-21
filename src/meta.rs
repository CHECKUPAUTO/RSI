//! §5 — BOUCLE DISCRÈTE ET MÉTA-FONCTION ÉVOLUTIVE
//!
//! ```text
//! S_{t+1} = S_t + ℳ(S_t, V_t, H, O) + ΔS_appr
//! ℳ_{t+1} = arg max_ℳ  SI_global( ℳ(S_t) )      (méta-révision)
//! ```
//!
//! La méta-fonction `ℳ` est une *politique d'auto-modification* : elle décide
//! où investir l'effort d'amélioration (sur les composantes de S) et comment
//! l'agent réécrit son propre logiciel `O` pour augmenter P_eff — c'est le
//! cœur récursif du RSI (l'agent améliore *la façon dont il s'améliore*).
//!
//! La méta-révision `ℳ_{t+1} = argmax_ℳ SI_global(ℳ(S_t))` explore un
//! voisinage de stratégies candidates et retient celle qui maximise
//! l'intelligence globale projetée.

use crate::rng::Rng;
use crate::state::CognitiveState;
use crate::substrate::Substrate;
use crate::surface::IntelligenceSurface;

/// Politique d'auto-modification ℳ.
///
/// - `focus` : répartition de l'effort cognitif sur (D,M,R,A,C,V), normalisée ;
/// - `software_edit` : direction de réécriture du logiciel O (auto-amélioration
///   du substrat) ;
/// - `gain` : amplitude globale de la proposition de ℳ.
#[derive(Clone, Debug)]
pub struct MetaStrategy {
    pub focus: [f64; 6],
    pub software_edit: Vec<f64>,
    pub gain: f64,
}

impl MetaStrategy {
    /// Stratégie neutre : effort uniforme, pas de réécriture logicielle.
    pub fn neutral(n_software: usize) -> Self {
        MetaStrategy {
            focus: [1.0 / 6.0; 6],
            software_edit: vec![0.0; n_software],
            gain: 0.05,
        }
    }

    /// Perturbation aléatoire de la stratégie (génère un candidat voisin).
    pub fn perturb(&self, rng: &mut Rng, scale: f64) -> MetaStrategy {
        let mut focus = self.focus;
        for f in focus.iter_mut() {
            *f = (*f + rng.normal(0.0, scale)).max(0.0);
        }
        let sum: f64 = focus.iter().sum::<f64>().max(1e-9);
        for f in focus.iter_mut() {
            *f /= sum;
        }
        let software_edit = self
            .software_edit
            .iter()
            .map(|&w| w + rng.normal(0.0, scale))
            .collect();
        let gain = (self.gain + rng.normal(0.0, scale * 0.2)).clamp(0.005, 0.25);
        MetaStrategy { focus, software_edit, gain }
    }

    /// ℳ(S_t, V_t, H, O) — proposition d'auto-modification.
    ///
    /// Retourne (ΔS_meta, substrat_modifié). L'effort est *aligné sur les
    /// valeurs* V (`focus` pondéré par le niveau de V) et la réécriture du
    /// logiciel ne s'applique que dans la limite de l'autonomie A de l'agent.
    pub fn apply(
        &self,
        state: &CognitiveState,
        substrate: &Substrate,
    ) -> (CognitiveState, Substrate) {
        // Alignement valeurs : un agent dont V est faible auto-modifie peu.
        let alignment = crate::linalg::mean(&state.v).clamp(0.0, 1.0);
        let autonomy = crate::linalg::mean(&state.a).clamp(0.0, 1.0);
        let g = self.gain * (0.25 + 0.75 * alignment);

        let mut delta = CognitiveState::zeros(state.dims());
        let push = |comp: &mut Vec<f64>, w: f64| {
            comp.iter_mut().for_each(|x| *x = g * w);
        };
        push(&mut delta.d, self.focus[0]);
        push(&mut delta.m, self.focus[1]);
        push(&mut delta.r, self.focus[2]);
        push(&mut delta.a, self.focus[3]);
        push(&mut delta.c, self.focus[4]);
        push(&mut delta.v, self.focus[5]);

        // Auto-réécriture du logiciel O (limitée par l'autonomie).
        let mut new_sub = substrate.clone();
        for (o, &e) in new_sub.o.iter_mut().zip(&self.software_edit) {
            *o = (*o + autonomy * g * e).max(0.0);
        }

        (delta, new_sub)
    }
}

/// Méta-optimiseur : réalise la méta-révision argmax_ℳ SI_global(ℳ(S_t)).
pub struct MetaOptimizer {
    /// Nombre de stratégies candidates explorées par méta-révision.
    pub candidates: usize,
    /// Amplitude de perturbation du voisinage de stratégies.
    pub explore_scale: f64,
    rng: Rng,
}

impl MetaOptimizer {
    pub fn new(candidates: usize, explore_scale: f64, seed: u64) -> Self {
        MetaOptimizer {
            candidates,
            explore_scale,
            rng: Rng::new(seed),
        }
    }

    /// ℳ_{t+1} = arg max_ℳ SI_global( ℳ(S_t) ).
    ///
    /// Explore `candidates` perturbations de la stratégie courante, évalue le
    /// SI_global projeté de chacune (après application de ℳ), et retourne la
    /// meilleure ainsi que sa valeur projetée.
    pub fn revise(
        &mut self,
        current: &MetaStrategy,
        state: &CognitiveState,
        substrate: &Substrate,
        surface: &IntelligenceSurface,
    ) -> (MetaStrategy, f64) {
        // référence : la stratégie courante
        let (best, best_si) = {
            let (ds, sub2) = current.apply(state, substrate);
            let projected = state.add(&ds).clipped(0.0, 1.0);
            (current.clone(), surface.si_global(&projected, &sub2))
        };

        let mut best = best;
        let mut best_si = best_si;

        for _ in 0..self.candidates {
            let cand = current.perturb(&mut self.rng, self.explore_scale);
            let (ds, sub2) = cand.apply(state, substrate);
            let projected = state.add(&ds).clipped(0.0, 1.0);
            let si = surface.si_global(&projected, &sub2);
            if si > best_si {
                best = cand;
                best_si = si;
            }
        }
        (best, best_si)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::Dims;

    #[test]
    fn revision_never_worse_than_current() {
        let mut rng = Rng::new(4);
        let surf = IntelligenceSurface::sample(256, &mut rng);
        let state = CognitiveState::random(Dims::uniform(4), &mut rng, 0.3);
        let sub = Substrate::default_with(4, 4, &mut rng);
        let strat = MetaStrategy::neutral(sub.o.len());

        // SI projeté avec la stratégie courante
        let (ds, sub2) = strat.apply(&state, &sub);
        let base = surf.si_global(&state.add(&ds).clipped(0.0, 1.0), &sub2);

        let mut meta = MetaOptimizer::new(32, 0.1, 123);
        let (_best, best_si) = meta.revise(&strat, &state, &sub, &surf);
        assert!(best_si >= base - 1e-9);
    }
}
