//! # scirust-rsi
//!
//! Moteur d'**auto-amélioration récursive** : des boucles *bornées, élitistes,
//! reproductibles* — « propose → évalue → garde si meilleur → répète » — avec
//! garantie de **non-régression**.
//!
//! > ⚠️ **Reconstruction API-compatible** (vendorisée comme dépendance `path`).
//! > Ce crate reproduit fidèlement l'**API publique publiée** de
//! > `CHECKUPAUTO/scirust :: scirust-rsi` (cf. `INTEGRATION.md`) afin de
//! > débloquer le build hors-ligne sans accès réseau au dépôt. Il ne dépend que
//! > de `rand`. Remplacer ce dossier par le crate amont (`vendor_scirust_rsi.sh`
//! > ou la git-dependency) dès qu'il est accessible — l'API est identique.
//!
//! ```
//! use scirust_rsi::{Guard, Fitness};
//! use scirust_rsi::refine::{RefineTask, SelfRefiner};
//! use rand::rngs::StdRng;
//!
//! struct Climb { target: f64 }
//! impl RefineTask for Climb {
//!     type Solution = f64;
//!     fn initial(&self, _rng: &mut StdRng) -> f64 { 0.0 }
//!     fn score(&self, s: &f64) -> Fitness { -(self.target - *s).abs() }
//!     fn refine(&self, s: &f64, _rng: &mut StdRng) -> f64 { s + 1.0 }
//! }
//! let (_best, report) = SelfRefiner::new(7)
//!     .run(&Climb { target: 5.0 }, &Guard::new().max_iters(20).target(0.0));
//! assert!(report.is_monotone());
//! ```

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Score d'aptitude — **plus grand = meilleur**. Minimiser un coût = le nier.
pub type Fitness = f64;

/// Raison d'arrêt d'une boucle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// Borne dure `max_iters` atteinte.
    MaxIterations,
    /// Cible de fitness atteinte.
    TargetReached,
    /// Plus d'amélioration pendant `patience` itérations.
    Converged,
}

/// Garde-fou de boucle : bornes + arrêt anticipé.
#[derive(Debug, Clone, Copy)]
pub struct Guard {
    pub max_iters: usize,
    pub patience: usize,
    pub target: Option<Fitness>,
    pub min_delta: Fitness,
}

impl Default for Guard {
    fn default() -> Self {
        Guard { max_iters: 1000, patience: 0, target: None, min_delta: 0.0 }
    }
}

impl Guard {
    pub fn new() -> Self {
        Guard::default()
    }
    pub fn max_iters(mut self, n: usize) -> Self {
        self.max_iters = n;
        self
    }
    /// `0` désactive la patience.
    pub fn patience(mut self, n: usize) -> Self {
        self.patience = n;
        self
    }
    pub fn target(mut self, t: Fitness) -> Self {
        self.target = Some(t);
        self
    }
    pub fn min_delta(mut self, d: Fitness) -> Self {
        self.min_delta = d;
        self
    }
}

/// Compte rendu d'une boucle.
#[derive(Debug, Clone)]
pub struct Report {
    pub iterations: usize,
    pub accepted: usize,
    pub best_fitness: Fitness,
    /// fitness de l'incumbent après chaque itération (index 0 = initial).
    pub history: Vec<Fitness>,
    pub stop_reason: StopReason,
}

impl Report {
    /// **Non-régression** : l'historique de l'incumbent est non décroissant.
    pub fn is_monotone(&self) -> bool {
        self.history.windows(2).all(|w| w[1] >= w[0] - 1e-12)
    }
    /// Gain total (dernier − premier).
    pub fn total_gain(&self) -> Fitness {
        match (self.history.first(), self.history.last()) {
            (Some(a), Some(b)) => b - a,
            _ => 0.0,
        }
    }
}

/// Boucle d'**ascension élitiste bornée**. `propose` reçoit l'incumbent, le
/// numéro d'itération et le RNG, et renvoie un candidat + sa fitness. Le
/// candidat n'est adopté que si sa fitness dépasse **strictement** l'incumbent
/// (par `min_delta`) ⇒ aucune régression.
pub fn ascend<S, P>(
    initial: S,
    init_fit: Fitness,
    mut propose: P,
    guard: &Guard,
    rng: &mut StdRng,
) -> (S, Report)
where
    P: FnMut(&S, usize, &mut StdRng) -> (S, Fitness),
{
    let mut best = initial;
    let mut best_fit = init_fit;
    let mut history = vec![best_fit];
    let mut accepted = 0usize;
    let mut stale = 0usize;
    let mut iters = 0usize;
    let mut stop = StopReason::MaxIterations;

    for i in 0..guard.max_iters {
        iters = i + 1;
        let (cand, fit) = propose(&best, i, rng);
        if fit > best_fit + guard.min_delta {
            best = cand;
            best_fit = fit;
            accepted += 1;
            stale = 0;
        } else {
            stale += 1;
        }
        history.push(best_fit);

        if let Some(t) = guard.target {
            if best_fit >= t {
                stop = StopReason::TargetReached;
                break;
            }
        }
        if guard.patience > 0 && stale >= guard.patience {
            stop = StopReason::Converged;
            break;
        }
    }

    (
        best,
        Report { iterations: iters, accepted, best_fitness: best_fit, history, stop_reason: stop },
    )
}

/// Échantillonne une loi normale centrée réduite (Box–Muller) à partir du RNG.
fn gaussian(rng: &mut StdRng) -> f64 {
    let u1: f64 = rng.gen::<f64>().max(1e-12);
    let u2: f64 = rng.gen::<f64>();
    (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
}

/// Fonctions-test classiques (à **minimiser** ⇒ nier pour une `Fitness`).
pub mod bench {
    /// Sphère : somme des carrés (minimum 0 en 0).
    pub fn sphere(x: &[f64]) -> f64 {
        x.iter().map(|v| v * v).sum()
    }
    /// Rastrigin (multimodale).
    pub fn rastrigin(x: &[f64]) -> f64 {
        let a = 10.0;
        a * x.len() as f64
            + x.iter()
                .map(|v| v * v - a * (std::f64::consts::TAU * v).cos())
                .sum::<f64>()
    }
    /// Rosenbrock (vallée).
    pub fn rosenbrock(x: &[f64]) -> f64 {
        x.windows(2)
            .map(|w| 100.0 * (w[1] - w[0] * w[0]).powi(2) + (1.0 - w[0]).powi(2))
            .sum()
    }
}

// ---- refine (Self-Refine) -------------------------------------------------- //
pub mod refine {
    use super::{ascend, Fitness, Guard, Report, StdRng, SeedableRng};

    /// Tâche d'auto-amélioration : un évaluateur (`score`) et un générateur
    /// (`refine`), avec une solution initiale (`initial`).
    pub trait RefineTask {
        type Solution: Clone;
        fn initial(&self, rng: &mut StdRng) -> Self::Solution;
        /// ÉVALUATEUR.
        fn score(&self, sol: &Self::Solution) -> Fitness;
        /// GÉNÉRATEUR : révision critiquée du candidat.
        fn refine(&self, sol: &Self::Solution, rng: &mut StdRng) -> Self::Solution;
    }

    /// Pilote « propose → évalue → garde si meilleur ».
    pub struct SelfRefiner {
        seed: u64,
    }
    impl SelfRefiner {
        pub fn new(seed: u64) -> Self {
            SelfRefiner { seed }
        }
        pub fn run<T: RefineTask>(&self, task: &T, guard: &Guard) -> (T::Solution, Report) {
            let mut rng = StdRng::seed_from_u64(self.seed);
            let init = task.initial(&mut rng);
            let init_fit = task.score(&init);
            ascend(
                init,
                init_fit,
                |sol, _i, rng| {
                    let cand = task.refine(sol, rng);
                    let f = task.score(&cand);
                    (cand, f)
                },
                guard,
                &mut rng,
            )
        }
    }
}

// ---- star (Self-Taught Reasoner) ------------------------------------------- //
pub mod star {
    use super::{Fitness, Guard, Report, StdRng, SeedableRng, StopReason};

    pub trait BootstrapTask {
        type Problem: Clone;
        type Solution: Clone;
        type Model: Clone;
        fn problems(&self) -> Vec<Self::Problem>;
        fn base_model(&self) -> Self::Model;
        fn attempt(&self, model: &Self::Model, problem: &Self::Problem, rng: &mut StdRng) -> Self::Solution;
        fn is_correct(&self, problem: &Self::Problem, sol: &Self::Solution) -> bool;
        fn learn(&self, base: &Self::Model, data: &[(Self::Problem, Self::Solution)]) -> Self::Model;
        fn evaluate(&self, model: &Self::Model) -> Fitness;
    }

    pub struct Star {
        seed: u64,
        samples: usize,
        accumulate: bool,
    }
    impl Star {
        pub fn new(seed: u64) -> Self {
            Star { seed, samples: 1, accumulate: false }
        }
        pub fn samples(mut self, k: usize) -> Self {
            self.samples = k.max(1);
            self
        }
        pub fn accumulate(mut self, b: bool) -> Self {
            self.accumulate = b;
            self
        }
        pub fn run<T: BootstrapTask>(&self, task: &T, guard: &Guard) -> (T::Model, Report) {
            let mut rng = StdRng::seed_from_u64(self.seed);
            let problems = task.problems();
            let mut model = task.base_model();
            let mut best_fit = task.evaluate(&model);
            let mut best = model.clone();
            let mut history = vec![best_fit];
            let mut acc: Vec<(T::Problem, T::Solution)> = Vec::new();
            let mut accepted = 0usize;
            let mut stale = 0usize;
            let mut iters = 0usize;
            let mut stop = StopReason::MaxIterations;

            for i in 0..guard.max_iters {
                iters = i + 1;
                // collecte des solutions correctes (auto-apprentissage)
                let mut data: Vec<(T::Problem, T::Solution)> =
                    if self.accumulate { acc.clone() } else { Vec::new() };
                for p in &problems {
                    for _ in 0..self.samples {
                        let sol = task.attempt(&model, p, &mut rng);
                        if task.is_correct(p, &sol) {
                            data.push((p.clone(), sol));
                            break;
                        }
                    }
                }
                if self.accumulate {
                    acc = data.clone();
                }
                let cand = task.learn(&best, &data);
                let fit = task.evaluate(&cand);
                if fit > best_fit + guard.min_delta {
                    best = cand.clone();
                    best_fit = fit;
                    accepted += 1;
                    stale = 0;
                } else {
                    stale += 1;
                }
                model = cand; // le modèle continue d'évoluer même si non adopté comme best
                history.push(best_fit);

                if let Some(t) = guard.target {
                    if best_fit >= t {
                        stop = StopReason::TargetReached;
                        break;
                    }
                }
                if guard.patience > 0 && stale >= guard.patience {
                    stop = StopReason::Converged;
                    break;
                }
            }
            (
                best,
                Report { iterations: iters, accepted, best_fitness: best_fit, history, stop_reason: stop },
            )
        }
    }
}

// ---- expert_iteration ------------------------------------------------------ //
pub mod expert_iteration {
    use super::{Fitness, Guard, Report, StdRng, SeedableRng, StopReason};

    pub trait ExpertIterationTask {
        type Sample: Clone;
        type Policy: Clone;
        type Target: Clone;
        fn samples(&self, rng: &mut StdRng) -> Vec<Self::Sample>;
        fn base_policy(&self) -> Self::Policy;
        fn expert(&self, policy: &Self::Policy, sample: &Self::Sample, rng: &mut StdRng) -> Self::Target;
        fn distil(&self, base: &Self::Policy, data: &[(Self::Sample, Self::Target)]) -> Self::Policy;
        fn evaluate(&self, policy: &Self::Policy) -> Fitness;
    }

    pub struct ExpertIteration {
        seed: u64,
    }
    impl ExpertIteration {
        pub fn new(seed: u64) -> Self {
            ExpertIteration { seed }
        }
        pub fn run<T: ExpertIterationTask>(&self, task: &T, guard: &Guard) -> (T::Policy, Report) {
            let mut rng = StdRng::seed_from_u64(self.seed);
            let mut policy = task.base_policy();
            let mut best = policy.clone();
            let mut best_fit = task.evaluate(&policy);
            let mut history = vec![best_fit];
            let mut accepted = 0usize;
            let mut stale = 0usize;
            let mut iters = 0usize;
            let mut stop = StopReason::MaxIterations;

            for i in 0..guard.max_iters {
                iters = i + 1;
                let samples = task.samples(&mut rng);
                let data: Vec<(T::Sample, T::Target)> = samples
                    .iter()
                    .map(|s| (s.clone(), task.expert(&policy, s, &mut rng)))
                    .collect();
                let cand = task.distil(&best, &data);
                let fit = task.evaluate(&cand);
                if fit > best_fit + guard.min_delta {
                    best = cand.clone();
                    best_fit = fit;
                    accepted += 1;
                    stale = 0;
                } else {
                    stale += 1;
                }
                policy = cand;
                history.push(best_fit);

                if let Some(t) = guard.target {
                    if best_fit >= t {
                        stop = StopReason::TargetReached;
                        break;
                    }
                }
                if guard.patience > 0 && stale >= guard.patience {
                    stop = StopReason::Converged;
                    break;
                }
            }
            (
                best,
                Report { iterations: iters, accepted, best_fitness: best_fit, history, stop_reason: stop },
            )
        }
    }
}

// ---- pbt (Population-Based Training) --------------------------------------- //
pub mod pbt {
    use super::{Fitness, Guard, Report, StdRng, SeedableRng, StopReason, Rng};

    pub trait PbtTask {
        type Hyper: Clone;
        fn init_member(&self, rng: &mut StdRng) -> (Vec<f64>, Self::Hyper);
        fn step(&self, params: &mut Vec<f64>, hyper: &Self::Hyper, rng: &mut StdRng) -> Fitness;
        fn perturb(&self, hyper: &Self::Hyper, rng: &mut StdRng) -> Self::Hyper;
    }

    pub struct Pbt {
        seed: u64,
        pop_size: usize,
        steps_per_gen: usize,
        exploit_frac: f64,
    }
    impl Pbt {
        pub fn new(seed: u64) -> Self {
            Pbt { seed, pop_size: 8, steps_per_gen: 1, exploit_frac: 0.2 }
        }
        pub fn pop_size(mut self, n: usize) -> Self {
            self.pop_size = n.max(1);
            self
        }
        pub fn steps_per_gen(mut self, n: usize) -> Self {
            self.steps_per_gen = n.max(1);
            self
        }
        pub fn exploit_frac(mut self, f: f64) -> Self {
            self.exploit_frac = f.clamp(0.0, 0.5);
            self
        }
        pub fn run<T: PbtTask>(&self, task: &T, guard: &Guard) -> (Vec<f64>, T::Hyper, Report) {
            let mut rng = StdRng::seed_from_u64(self.seed);
            let mut pop: Vec<(Vec<f64>, T::Hyper, Fitness)> = (0..self.pop_size)
                .map(|_| {
                    let (p, h) = task.init_member(&mut rng);
                    (p, h, f64::NEG_INFINITY)
                })
                .collect();

            let mut best_fit = f64::NEG_INFINITY;
            let mut best: (Vec<f64>, T::Hyper) = (pop[0].0.clone(), pop[0].1.clone());
            let mut history: Vec<Fitness> = Vec::new();
            let mut accepted = 0usize;
            let mut stale = 0usize;
            let mut iters = 0usize;
            let mut stop = StopReason::MaxIterations;

            for i in 0..guard.max_iters {
                iters = i + 1;
                for m in pop.iter_mut() {
                    let mut f = m.2;
                    for _ in 0..self.steps_per_gen {
                        f = task.step(&mut m.0, &m.1, &mut rng);
                    }
                    m.2 = f;
                }
                pop.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));

                let gen_best = pop[0].2;
                if gen_best > best_fit + guard.min_delta {
                    best_fit = gen_best;
                    best = (pop[0].0.clone(), pop[0].1.clone());
                    accepted += 1;
                    stale = 0;
                } else {
                    if gen_best > best_fit {
                        best_fit = gen_best;
                        best = (pop[0].0.clone(), pop[0].1.clone());
                    }
                    stale += 1;
                }
                history.push(best_fit);

                // exploit/explore : les pires copient les meilleurs + perturbation
                let n_ex = ((self.pop_size as f64) * self.exploit_frac).round() as usize;
                if n_ex > 0 {
                    let top: Vec<(Vec<f64>, T::Hyper)> =
                        pop[..n_ex].iter().map(|m| (m.0.clone(), m.1.clone())).collect();
                    let len = pop.len();
                    for (k, m) in pop[len - n_ex..].iter_mut().enumerate() {
                        let src = &top[k % top.len()];
                        m.0 = src.0.clone();
                        m.1 = task.perturb(&src.1, &mut rng);
                        m.2 = f64::NEG_INFINITY;
                    }
                }
                let _ = rng.gen::<f64>(); // garde l'avancée déterministe du flux

                if let Some(t) = guard.target {
                    if best_fit >= t {
                        stop = StopReason::TargetReached;
                        break;
                    }
                }
                if guard.patience > 0 && stale >= guard.patience {
                    stop = StopReason::Converged;
                    break;
                }
            }
            (
                best.0,
                best.1,
                Report { iterations: iters, accepted, best_fitness: best_fit, history, stop_reason: stop },
            )
        }
    }
}

// ---- evo ((1+λ)-ES + règle du 1/5) ----------------------------------------- //
pub mod evo {
    use super::{gaussian, Fitness, Guard, Report, StdRng, SeedableRng, StopReason};

    /// Stratégie d'évolution (1+λ) avec adaptation du pas par la règle du 1/5.
    pub struct OnePlusLambda {
        seed: u64,
        lambda: usize,
        sigma0: f64,
        c: f64,
        window: usize,
    }
    impl OnePlusLambda {
        pub fn new(seed: u64) -> Self {
            OnePlusLambda { seed, lambda: 10, sigma0: 0.3, c: 0.817, window: 10 }
        }
        pub fn lambda(mut self, l: usize) -> Self {
            self.lambda = l.max(1);
            self
        }
        pub fn sigma0(mut self, s: f64) -> Self {
            self.sigma0 = s.max(1e-12);
            self
        }
        pub fn c(mut self, c: f64) -> Self {
            self.c = c;
            self
        }
        pub fn window(mut self, w: usize) -> Self {
            self.window = w.max(1);
            self
        }
        pub fn optimize<F: Fn(&[f64]) -> Fitness>(
            &self,
            x0: Vec<f64>,
            f: F,
            guard: &Guard,
        ) -> (Vec<f64>, Fitness, Report) {
            let mut rng = StdRng::seed_from_u64(self.seed);
            let mut parent = x0;
            let mut parent_fit = f(&parent);
            let mut sigma = self.sigma0;
            let mut history = vec![parent_fit];
            let mut accepted = 0usize;
            let mut stale = 0usize;
            let mut iters = 0usize;
            let mut stop = StopReason::MaxIterations;
            let mut succ: Vec<bool> = Vec::with_capacity(self.window);

            for i in 0..guard.max_iters {
                iters = i + 1;
                let mut best_child = parent.clone();
                let mut best_child_fit = f64::NEG_INFINITY;
                for _ in 0..self.lambda {
                    let child: Vec<f64> =
                        parent.iter().map(|v| v + sigma * gaussian(&mut rng)).collect();
                    let cf = f(&child);
                    if cf > best_child_fit {
                        best_child_fit = cf;
                        best_child = child;
                    }
                }
                let success = best_child_fit > parent_fit + guard.min_delta;
                if success {
                    parent = best_child;
                    parent_fit = best_child_fit;
                    accepted += 1;
                    stale = 0;
                } else {
                    stale += 1;
                }
                // règle du 1/5 : viser ~20 % de succès
                succ.push(success);
                if succ.len() > self.window {
                    succ.remove(0);
                }
                let rate = succ.iter().filter(|b| **b).count() as f64 / succ.len() as f64;
                if rate > 0.2 {
                    sigma /= self.c;
                } else if rate < 0.2 {
                    sigma *= self.c;
                }
                history.push(parent_fit);

                if let Some(t) = guard.target {
                    if parent_fit >= t {
                        stop = StopReason::TargetReached;
                        break;
                    }
                }
                if guard.patience > 0 && stale >= guard.patience {
                    stop = StopReason::Converged;
                    break;
                }
            }
            (
                parent,
                parent_fit,
                Report { iterations: iters, accepted, best_fitness: parent_fit, history, stop_reason: stop },
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::refine::{RefineTask, SelfRefiner};
    use super::*;

    struct Climb {
        target: f64,
    }
    impl RefineTask for Climb {
        type Solution = f64;
        fn initial(&self, _rng: &mut StdRng) -> f64 {
            0.0
        }
        fn score(&self, s: &f64) -> Fitness {
            -(self.target - *s).abs()
        }
        fn refine(&self, s: &f64, rng: &mut StdRng) -> f64 {
            // propose +1 ou -1 ; l'élitisme rejette la régression
            if rng.gen::<bool>() {
                s + 1.0
            } else {
                s - 1.0
            }
        }
    }

    #[test]
    fn refine_is_monotone_and_bounded() {
        let (best, report) =
            SelfRefiner::new(1).run(&Climb { target: 6.0 }, &Guard::new().max_iters(100).target(0.0));
        assert!(report.is_monotone());
        assert!(report.iterations <= 100);
        assert_eq!(report.stop_reason, StopReason::TargetReached);
        assert!((best - 6.0).abs() < 1e-9);
        assert!(report.total_gain() >= 0.0);
    }

    #[test]
    fn patience_converges() {
        struct OnlyWorse;
        impl RefineTask for OnlyWorse {
            type Solution = f64;
            fn initial(&self, _r: &mut StdRng) -> f64 {
                0.0
            }
            fn score(&self, s: &f64) -> Fitness {
                -s.abs()
            }
            fn refine(&self, s: &f64, _r: &mut StdRng) -> f64 {
                s + 1.0
            }
        }
        let (_b, report) = SelfRefiner::new(2).run(&OnlyWorse, &Guard::new().max_iters(50).patience(3));
        assert_eq!(report.stop_reason, StopReason::Converged);
        assert_eq!(report.accepted, 0);
        assert!(report.is_monotone());
    }

    #[test]
    fn evo_minimizes_sphere() {
        let f = |x: &[f64]| -bench::sphere(x); // maximiser le négatif = minimiser
        let (_x, fit, report) = evo::OnePlusLambda::new(3)
            .lambda(12)
            .optimize(vec![3.0, -2.5, 1.0], f, &Guard::new().max_iters(300));
        assert!(report.is_monotone());
        assert!(fit > -1.0, "fit = {fit}");
    }

    #[test]
    fn ascend_free_function() {
        let mut rng = StdRng::seed_from_u64(0);
        let (best, report) = ascend(
            0i64,
            0.0,
            |s, _i, _r| (s + 1, (s + 1) as f64),
            &Guard::new().max_iters(10),
            &mut rng,
        );
        assert_eq!(best, 10);
        assert!(report.is_monotone());
        assert_eq!(report.iterations, 10);
    }
}
