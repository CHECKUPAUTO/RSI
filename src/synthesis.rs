//! Domaine de **synthèse symbolique** : l'agent génère des *expressions*
//! candidates puis les améliore via [`crate::ascent`] (contrat scirust-rsi).
//!
//! ## Sandbox (garde-fou clé)
//! Le « code » candidat est un **AST arithmétique** que J'ÉVALUE dans mon
//! propre interpréteur ([`Expr::eval`]) — **jamais** compilé ni exécuté comme
//! du code arbitraire, aucun sous-processus. C'est le sandbox que contrôle RSI :
//! l'évaluateur lit une fitness, le moteur d'ascension ne voit que des nombres.
//!
//! - ÉVALUATEUR (`score`) : fraction de cas de test réussis (|sortie − cible| ≤
//!   tolérance) **moins** une pénalité de complexité (taille de l'AST).
//! - GÉNÉRATEUR (`refine`) : produit la meilleure de `lambda` mutations
//!   déterministes (révision « critiquée », façon 1+λ).
//!
//! Entièrement déterministe (graine) et borné.

use crate::ascent::RefineTask;
use crate::rng::Rng;

/// Expression arithmétique sur une variable `x` (AST candidat).
#[derive(Clone, Debug, PartialEq)]
pub enum Expr {
    X,
    Const(f64),
    Add(Box<Expr>, Box<Expr>),
    Sub(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Neg(Box<Expr>),
}

impl Expr {
    /// ÉVALUATION en sandbox (interpréteur maison ; aucune exécution externe).
    pub fn eval(&self, x: f64) -> f64 {
        match self {
            Expr::X => x,
            Expr::Const(c) => *c,
            Expr::Add(a, b) => a.eval(x) + b.eval(x),
            Expr::Sub(a, b) => a.eval(x) - b.eval(x),
            Expr::Mul(a, b) => a.eval(x) * b.eval(x),
            Expr::Neg(a) => -a.eval(x),
        }
    }

    /// Nombre de nœuds (complexité).
    pub fn size(&self) -> usize {
        match self {
            Expr::X | Expr::Const(_) => 1,
            Expr::Neg(a) => 1 + a.size(),
            Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) => 1 + a.size() + b.size(),
        }
    }

    /// Représentation lisible (pour le log).
    pub fn pretty(&self) -> String {
        match self {
            Expr::X => "x".into(),
            Expr::Const(c) => format!("{c:.3}"),
            Expr::Add(a, b) => format!("({} + {})", a.pretty(), b.pretty()),
            Expr::Sub(a, b) => format!("({} - {})", a.pretty(), b.pretty()),
            Expr::Mul(a, b) => format!("({} * {})", a.pretty(), b.pretty()),
            Expr::Neg(a) => format!("(-{})", a.pretty()),
        }
    }

    /// Renvoie une copie du sous-arbre d'indice `idx` (préordre).
    fn subtree_at(&self, idx: usize, cur: &mut usize) -> Option<Expr> {
        let here = *cur;
        *cur += 1;
        if here == idx {
            return Some(self.clone());
        }
        match self {
            Expr::X | Expr::Const(_) => None,
            Expr::Neg(a) => a.subtree_at(idx, cur),
            Expr::Add(a, b) | Expr::Sub(a, b) | Expr::Mul(a, b) => {
                a.subtree_at(idx, cur).or_else(|| b.subtree_at(idx, cur))
            }
        }
    }

    /// Remplace le nœud d'indice `idx` (préordre) par `repl`.
    fn replace_at(&self, idx: usize, repl: &Expr, cur: &mut usize) -> Expr {
        let here = *cur;
        *cur += 1;
        if here == idx {
            return repl.clone();
        }
        match self {
            Expr::X | Expr::Const(_) => self.clone(),
            Expr::Neg(a) => Expr::Neg(Box::new(a.replace_at(idx, repl, cur))),
            Expr::Add(a, b) => Expr::Add(
                Box::new(a.replace_at(idx, repl, cur)),
                Box::new(b.replace_at(idx, repl, cur)),
            ),
            Expr::Sub(a, b) => Expr::Sub(
                Box::new(a.replace_at(idx, repl, cur)),
                Box::new(b.replace_at(idx, repl, cur)),
            ),
            Expr::Mul(a, b) => Expr::Mul(
                Box::new(a.replace_at(idx, repl, cur)),
                Box::new(b.replace_at(idx, repl, cur)),
            ),
        }
    }
}

fn random_terminal(rng: &mut Rng) -> Expr {
    if rng.uniform() < 0.5 {
        Expr::X
    } else {
        // constante dans [-2, 2], arrondie au quart (favorise les entiers simples)
        let c = (rng.uniform_range(-2.0, 2.0) * 4.0).round() / 4.0;
        Expr::Const(c)
    }
}

fn random_expr(rng: &mut Rng, depth: usize) -> Expr {
    if depth == 0 || rng.uniform() < 0.3 {
        return random_terminal(rng);
    }
    let a = Box::new(random_expr(rng, depth - 1));
    let b = Box::new(random_expr(rng, depth - 1));
    match (rng.uniform() * 3.0) as u32 {
        0 => Expr::Add(a, b),
        1 => Expr::Sub(a, b),
        _ => Expr::Mul(a, b),
    }
}

/// Tâche de régression symbolique : ajuster une fonction cible sur des cas de
/// test, sous pénalité de complexité. Implémente [`RefineTask`].
pub struct SymbolicSynthesis {
    /// cas de test (x, cible) — la « suite de tests » du candidat.
    cases: Vec<(f64, f64)>,
    /// tolérance d'acceptation par cas.
    tol: f64,
    /// pénalité par nœud d'AST (favorise la simplicité).
    complexity_penalty: f64,
    /// nombre de mutations évaluées par `refine` (1+λ).
    lambda: usize,
    rng: Rng,
}

impl SymbolicSynthesis {
    /// Construit la tâche à partir d'une fonction cible échantillonnée sur
    /// `n` points de `[lo, hi]`.
    pub fn from_target(
        target: impl Fn(f64) -> f64,
        lo: f64,
        hi: f64,
        n: usize,
        seed: u64,
    ) -> Self {
        let n = n.max(2);
        let cases = (0..n)
            .map(|i| {
                let x = lo + (hi - lo) * i as f64 / (n - 1) as f64;
                (x, target(x))
            })
            .collect();
        SymbolicSynthesis {
            cases,
            tol: 0.25,
            complexity_penalty: 0.01,
            lambda: 16,
            rng: Rng::new(seed),
        }
    }

    pub fn with_lambda(mut self, lambda: usize) -> Self {
        self.lambda = lambda.max(1);
        self
    }
    pub fn with_tolerance(mut self, tol: f64) -> Self {
        self.tol = tol.max(0.0);
        self
    }

    /// Candidat initial trivial.
    pub fn seed_candidate(&self) -> Expr {
        Expr::Const(0.0)
    }

    /// Fraction de cas réussis (sans pénalité) — utile pour le log/diagnostic.
    pub fn pass_fraction(&self, e: &Expr) -> f64 {
        let passed = self
            .cases
            .iter()
            .filter(|(x, t)| (e.eval(*x) - t).abs() <= self.tol)
            .count();
        passed as f64 / self.cases.len() as f64
    }

    /// Une mutation déterministe du candidat. Trois opérateurs :
    /// - **remplacement** d'un sous-arbre par un petit arbre aléatoire ;
    /// - **grow** : enrober le sous-arbre choisi dans un opérateur binaire avec
    ///   un terminal (construit de la structure, ex. `x → x*x`) ;
    /// - **perturbation** de constante.
    fn mutate(&mut self, e: &Expr) -> Expr {
        let n = e.size();
        let idx = (self.rng.uniform() * n as f64) as usize % n;
        let r = self.rng.uniform();
        let repl = if r < 0.45 {
            // grow : enrober le sous-arbre existant
            let mut cur = 0;
            let sub = e.subtree_at(idx, &mut cur).unwrap_or(Expr::X);
            let term = random_terminal(&mut self.rng);
            let (a, b) = (Box::new(sub), Box::new(term));
            match (self.rng.uniform() * 3.0) as u32 {
                0 => Expr::Add(a, b),
                1 => Expr::Sub(a, b),
                _ => Expr::Mul(a, b),
            }
        } else if r < 0.8 {
            random_expr(&mut self.rng, 2)
        } else {
            let c = (self.rng.uniform_range(-3.0, 3.0) * 4.0).round() / 4.0;
            Expr::Const(c)
        };
        let mut cur = 0;
        e.replace_at(idx, &repl, &mut cur)
    }
}

impl RefineTask for SymbolicSynthesis {
    type Cand = Expr;

    /// ÉVALUATEUR : fraction de tests réussis − pénalité de complexité.
    fn score(&self, cand: &Expr) -> f64 {
        let mut passed = 0usize;
        for (x, t) in &self.cases {
            let y = cand.eval(*x);
            if y.is_finite() && (y - t).abs() <= self.tol {
                passed += 1;
            }
        }
        let frac = passed as f64 / self.cases.len() as f64;
        frac - self.complexity_penalty * cand.size() as f64
    }

    /// GÉNÉRATEUR : meilleure de `lambda` mutations (révision critiquée, 1+λ).
    fn refine(&mut self, cand: &Expr, _iter: usize) -> Expr {
        let mut best = self.mutate(cand);
        let mut best_fit = self.score(&best);
        for _ in 1..self.lambda {
            let m = self.mutate(cand);
            let f = self.score(&m);
            if f > best_fit {
                best = m;
                best_fit = f;
            }
        }
        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ascent::{ascend, Guard};

    #[test]
    fn eval_and_size() {
        // x*x + 1
        let e = Expr::Add(
            Box::new(Expr::Mul(Box::new(Expr::X), Box::new(Expr::X))),
            Box::new(Expr::Const(1.0)),
        );
        assert!((e.eval(3.0) - 10.0).abs() < 1e-12);
        assert_eq!(e.size(), 5);
    }

    #[test]
    fn synthesis_improves_monotonically_and_terminates() {
        // cible : x^2 + 1
        let mut task = SymbolicSynthesis::from_target(|x| x * x + 1.0, -2.0, 2.0, 21, 42);
        let init = task.seed_candidate();
        let init_fit = task.score(&init);
        let guard = Guard::new().max_iters(60).patience(15).target(0.99).min_delta(0.0);
        let (best, report) = ascend(&mut task, init, &guard);

        // Contrat (garanti) : non-régression + terminaison bornée + amélioration.
        assert!(report.is_monotone(), "non-régression (élitisme)");
        assert!(report.iters <= 60, "terminaison bornée");
        assert!(report.best() > init_fit, "la fitness s'améliore");
        assert!(report.accepted >= 1, "au moins une révision adoptée");
        // amélioration substantielle de la couverture de tests vs l'initial
        let init = task.seed_candidate();
        assert!(
            task.pass_fraction(&best) > task.pass_fraction(&init),
            "couverture: {} ({})",
            task.pass_fraction(&best),
            best.pretty()
        );
    }

    #[test]
    fn deterministic_given_seed() {
        let run = || {
            let mut t = SymbolicSynthesis::from_target(|x| 2.0 * x - 1.0, -3.0, 3.0, 15, 7);
            let g = Guard::new().max_iters(40).target(0.99);
            let c = t.seed_candidate();
            let (_b, r) = ascend(&mut t, c, &g);
            r.best()
        };
        assert_eq!(run(), run()); // même graine ⇒ même résultat
    }
}
