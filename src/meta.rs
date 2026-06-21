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
//! La méta-révision `ℳ_{t+1} = argmax_ℳ SI_global(ℳ(S_t))` est abstraite
//! derrière le trait [`MetaSearch`], ce qui permet d'échanger la stratégie de
//! recherche : [`MetaOptimizer`] (recherche aléatoire de voisinage) ou
//! [`CmaEsMeta`] (sep-CMA-ES, recherche guidée par covariance adaptative).

use crate::cma::SepCmaEs;
use crate::linalg::{mean, sigmoid};
use crate::rng::Rng;
use crate::state::CognitiveState;
use crate::substrate::Substrate;
use crate::surface::IntelligenceSurface;

/// Bornes du paramètre `gain` d'une [`MetaStrategy`] (utilisées par CMA-ES).
const GAIN_LO: f64 = 0.005;
const GAIN_HI: f64 = 0.25;

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
        let gain = (self.gain + rng.normal(0.0, scale * 0.2)).clamp(GAIN_LO, GAIN_HI);
        MetaStrategy { focus, software_edit, gain }
    }

    /// ℳ(S_t, V_t, H, O) — proposition d'auto-modification.
    ///
    /// Retourne (ΔS_meta, substrat_modifié). L'effort est *aligné sur les
    /// valeurs* V (`gain` pondéré par le niveau de V) et la réécriture du
    /// logiciel ne s'applique que dans la limite de l'autonomie A de l'agent.
    pub fn apply(
        &self,
        state: &CognitiveState,
        substrate: &Substrate,
    ) -> (CognitiveState, Substrate) {
        let alignment = mean(&state.v).clamp(0.0, 1.0);
        let autonomy = mean(&state.a).clamp(0.0, 1.0);
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

    /// SI_global projeté si l'on applique cette stratégie à (state, substrate).
    fn projected_si(
        &self,
        state: &CognitiveState,
        substrate: &Substrate,
        surface: &IntelligenceSurface,
    ) -> f64 {
        let (delta, sub2) = self.apply(state, substrate);
        let projected = state.add(&delta).clipped(0.0, 1.0);
        surface.si_global(&projected, &sub2)
    }

    // --- encodage pour optimiseur en espace non contraint (CMA-ES) -------- //

    /// Encode la stratégie en un vecteur ℝ^(7+n) *non contraint* :
    /// `[ ln(focus)×6, logit(gain), software_edit×n ]`.
    fn encode(&self) -> Vec<f64> {
        let mut theta = Vec::with_capacity(7 + self.software_edit.len());
        for &f in &self.focus {
            theta.push(f.max(1e-6).ln());
        }
        // logit de gain ramené dans (GAIN_LO, GAIN_HI)
        let x = self.gain.clamp(GAIN_LO + 1e-6, GAIN_HI - 1e-6);
        theta.push(((x - GAIN_LO) / (GAIN_HI - x)).ln());
        theta.extend_from_slice(&self.software_edit);
        theta
    }

    /// Décode un vecteur non contraint en stratégie valide :
    /// `focus = softmax(·)`, `gain = GAIN_LO + (GAIN_HI−GAIN_LO)·σ(·)`.
    fn decode(theta: &[f64], n_software: usize) -> MetaStrategy {
        let max = theta[0..6].iter().cloned().fold(f64::MIN, f64::max);
        let exps: [f64; 6] = std::array::from_fn(|i| (theta[i] - max).exp());
        let sum: f64 = exps.iter().sum::<f64>().max(1e-12);
        let focus: [f64; 6] = std::array::from_fn(|i| exps[i] / sum);

        let gain = GAIN_LO + (GAIN_HI - GAIN_LO) * sigmoid(theta[6]);
        let software_edit = theta[7..7 + n_software].to_vec();
        MetaStrategy { focus, software_edit, gain }
    }
}

/// Stratégie de recherche pour la méta-révision `argmax_ℳ SI_global(ℳ(S_t))`.
///
/// Toute implémentation doit garantir que la stratégie retournée n'est **pas
/// pire** que `current` (méta-révision monotone) afin de préserver les
/// garde-fous de stabilité du système.
pub trait MetaSearch {
    fn revise(
        &mut self,
        current: &MetaStrategy,
        state: &CognitiveState,
        substrate: &Substrate,
        surface: &IntelligenceSurface,
    ) -> (MetaStrategy, f64);
}

/// Méta-optimiseur par **recherche aléatoire de voisinage**.
///
/// Explore `candidates` perturbations de la stratégie courante et retient la
/// meilleure (élitisme par rapport à `current`).
pub struct MetaOptimizer {
    pub candidates: usize,
    pub explore_scale: f64,
    rng: Rng,
}

impl MetaOptimizer {
    pub fn new(candidates: usize, explore_scale: f64, seed: u64) -> Self {
        MetaOptimizer { candidates, explore_scale, rng: Rng::new(seed) }
    }
}

impl MetaSearch for MetaOptimizer {
    fn revise(
        &mut self,
        current: &MetaStrategy,
        state: &CognitiveState,
        substrate: &Substrate,
        surface: &IntelligenceSurface,
    ) -> (MetaStrategy, f64) {
        let mut best = current.clone();
        let mut best_si = current.projected_si(state, substrate, surface);

        for _ in 0..self.candidates {
            let cand = current.perturb(&mut self.rng, self.explore_scale);
            let si = cand.projected_si(state, substrate, surface);
            if si > best_si {
                best = cand;
                best_si = si;
            }
        }
        (best, best_si)
    }
}

/// Méta-optimiseur par **sep-CMA-ES**.
///
/// À chaque révision, lance une optimisation sep-CMA-ES de quelques
/// générations, initialisée autour de la stratégie courante (encodée en
/// espace non contraint), et retient la meilleure stratégie trouvée — sans
/// jamais régresser sous `current`.
pub struct CmaEsMeta {
    pub population: usize,  // 0 ⇒ défaut 4 + ⌊3 ln N⌋
    pub generations: usize,
    pub sigma0: f64,
    seed: u64,
    counter: u64,
}

impl CmaEsMeta {
    pub fn new(population: usize, generations: usize, sigma0: f64, seed: u64) -> Self {
        CmaEsMeta { population, generations, sigma0, seed, counter: 0 }
    }
}

impl MetaSearch for CmaEsMeta {
    fn revise(
        &mut self,
        current: &MetaStrategy,
        state: &CognitiveState,
        substrate: &Substrate,
        surface: &IntelligenceSurface,
    ) -> (MetaStrategy, f64) {
        let n_sw = current.software_edit.len();
        let dim = 7 + n_sw;

        // graine variée par appel pour éviter des trajectoires figées
        let seed = self.seed ^ self.counter.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        self.counter = self.counter.wrapping_add(1);

        let mut cma = SepCmaEs::new(dim, self.population, seed);
        let mean0 = current.encode();
        let objective = |theta: &[f64]| -> f64 {
            MetaStrategy::decode(theta, n_sw).projected_si(state, substrate, surface)
        };
        let (best_theta, best_si) =
            cma.optimize(&mean0, self.sigma0, self.generations, objective);

        // garde-fou : ne jamais faire pire que la stratégie courante
        let cur_si = current.projected_si(state, substrate, surface);
        if best_si >= cur_si {
            (MetaStrategy::decode(&best_theta, n_sw), best_si)
        } else {
            (current.clone(), cur_si)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::Dims;

    fn setup() -> (IntelligenceSurface, CognitiveState, Substrate, MetaStrategy) {
        let mut rng = Rng::new(4);
        let surf = IntelligenceSurface::sample(256, &mut rng);
        let state = CognitiveState::random(Dims::uniform(4), &mut rng, 0.3);
        let sub = Substrate::default_with(4, 4, &mut rng);
        let strat = MetaStrategy::neutral(sub.o.len());
        (surf, state, sub, strat)
    }

    #[test]
    fn random_revision_never_worse() {
        let (surf, state, sub, strat) = setup();
        let base = strat.projected_si(&state, &sub, &surf);
        let mut meta = MetaOptimizer::new(32, 0.1, 123);
        let (_b, best_si) = meta.revise(&strat, &state, &sub, &surf);
        assert!(best_si >= base - 1e-9);
    }

    #[test]
    fn cma_revision_never_worse() {
        let (surf, state, sub, strat) = setup();
        let base = strat.projected_si(&state, &sub, &surf);
        let mut meta = CmaEsMeta::new(0, 8, 0.3, 777);
        let (_b, best_si) = meta.revise(&strat, &state, &sub, &surf);
        assert!(best_si >= base - 1e-9);
    }

    #[test]
    fn cma_matches_or_beats_random_on_average() {
        let (surf, state, sub, strat) = setup();
        let mut rnd = MetaOptimizer::new(64, 0.1, 1);
        let mut cma = CmaEsMeta::new(0, 12, 0.3, 1);
        let (_a, si_rnd) = rnd.revise(&strat, &state, &sub, &surf);
        let (_b, si_cma) = cma.revise(&strat, &state, &sub, &surf);
        // les deux dépassent la base ; CMA-ES est compétitif
        let base = strat.projected_si(&state, &sub, &surf);
        assert!(si_rnd >= base - 1e-9 && si_cma >= base - 1e-9);
    }

    #[test]
    fn encode_decode_roundtrip() {
        let s = MetaStrategy {
            focus: [0.1, 0.2, 0.3, 0.15, 0.05, 0.2],
            software_edit: vec![0.3, -0.2, 0.1, 0.0],
            gain: 0.1,
        };
        let theta = s.encode();
        let d = MetaStrategy::decode(&theta, 4);
        for (a, b) in s.focus.iter().zip(&d.focus) {
            assert!((a - b).abs() < 1e-6, "focus {a} vs {b}");
        }
        assert!((s.gain - d.gain).abs() < 1e-6);
        assert_eq!(s.software_edit, d.software_edit);
    }
}
