//! P1.1 — INTÉGRATION LLM : « le LLM propose, le moteur dispose ».
//!
//! Ce module branche un producteur de propositions externe (un LLM) sur la
//! boucle élitiste bornée, **sans jamais lui donner le contrôle de la boucle**.
//! Le LLM ne fait que produire des chaînes ; le moteur les parse, les valide
//! (sûreté), les évalue (fitness) et les adopte élitistement (strictement
//! meilleur) ou les rejette — sous garde-fous bornés (`LlmGuard`).
//!
//! Architecture (cf. `docs/P1_DESIGN_SPIKE.md`) :
//! - [`LlmClient`] : backend interchangeable (Ollama local par défaut, Claude
//!   sélectionnable, [`MockLlmClient`] déterministe pour les tests hors-ligne).
//!   Le cœur reste **std-only** : les backends réseau vivront derrière des
//!   features (`llm-ollama`, `llm-claude`) ; le mock n'a aucune dépendance.
//! - [`LlmRefineTask`] : le *domaine* (ce que le LLM voit, comment on parse ses
//!   propositions, l'éval held-out anti-Goodhart, les interdits de sûreté).
//! - [`ascend_llm`] : le pilote, qui réutilise l'élitisme et étend les
//!   garde-fous au **budget** (appels, temps) et à l'**intégrité d'éval**
//!   (écart train/held-out).
//!
//! Le LLM ne voit jamais `LlmGuard` : il reçoit un prompt, rend `k` propositions.

use crate::ascent::RefineTask;
use std::time::{Duration, Instant};

/// Erreur d'un backend LLM.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LlmError {
    /// Le backend a échoué (réseau, modèle indisponible…).
    Backend(String),
    /// Le backend n'a renvoyé aucune proposition.
    Empty,
}

/// Producteur de propositions interchangeable. Le moteur ne connaît que ça ;
/// il ignore quel modèle tourne derrière. **Aucun appel réseau dans le cœur.**
pub trait LlmClient {
    /// Rend `k` propositions (texte brut) pour `prompt`. À charge du domaine
    /// ([`LlmRefineTask::parse_proposals`]) de les interpréter/valider.
    fn propose(&self, prompt: &str, k: usize) -> Result<Vec<String>, LlmError>;
}

/// Violation d'une contrainte de sûreté propre à un domaine (§3.4 du spike).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SafetyViolation(pub String);

/// Domaine auto-améliorable piloté par LLM. Étend [`RefineTask`] : le moteur
/// garde la main sur `score` (évaluateur) et la boucle ; le LLM n'intervient
/// que via les propositions textuelles, jamais sur les bornes.
pub trait LlmRefineTask: RefineTask {
    /// Vue prompt-friendly de l'incumbent (ce que le LLM « voit »).
    fn describe(&self, incumbent: &Self::Cand) -> String;

    /// Transforme les propositions brutes du LLM en candidats typés. Les
    /// chaînes malformées sont simplement ignorées (filtrées).
    fn parse_proposals(&self, raw: &[String]) -> Vec<Self::Cand>;

    /// Évaluation **held-out** (anti-Goodhart, §3) : NE pilote PAS l'adoption,
    /// sert au reporting et à la détection de sur-apprentissage. Par défaut,
    /// retombe sur `score` (à surcharger par les domaines à held-out réel).
    fn score_heldout(&self, cand: &Self::Cand) -> f64 {
        self.score(cand)
    }

    /// Contraintes de sûreté du domaine (§3.4) : un candidat qui échoue est
    /// rejeté quel que soit son score. Par défaut, tout est permis.
    fn safety_check(&self, _cand: &Self::Cand) -> Result<(), SafetyViolation> {
        Ok(())
    }
}

/// Garde-fous de la boucle LLM : bornes classiques **plus** budget (§2) et
/// garde-fou anti-overfitting (§3). Autonome (ne dépend pas de `ascent::Guard`)
/// pour ne pas exposer les champs privés de ce dernier.
#[derive(Clone, Debug)]
pub struct LlmGuard {
    /// Borne dure d'itérations (terminaison garantie).
    pub max_iters: usize,
    /// Arrêt après `patience` itérations sans amélioration (0 = désactivé).
    pub patience: usize,
    /// Arrêt si la fitness atteint cette cible.
    pub target: Option<f64>,
    /// Seuil d'amélioration *stricte* pour adopter.
    pub min_delta: f64,
    /// Nombre de propositions demandées par itération (batching, §2).
    pub k: usize,
    /// Budget : nombre maximal d'appels LLM (terminaison côté coût).
    pub max_llm_calls: usize,
    /// Budget : temps mur maximal (None = illimité).
    pub max_wall_clock: Option<Duration>,
    /// Garde-fou overfitting (§3) : écart `score_train − score_heldout` maximal
    /// toléré sur l'incumbent (None = désactivé).
    pub max_overfit_gap: Option<f64>,
}

impl Default for LlmGuard {
    fn default() -> Self {
        LlmGuard {
            max_iters: 50,
            patience: 0,
            target: None,
            min_delta: 0.0,
            k: 4,
            max_llm_calls: 100,
            max_wall_clock: None,
            max_overfit_gap: None,
        }
    }
}

/// Raison d'arrêt de la boucle LLM (distincte de `ascent::StopReason` pour ne
/// pas perturber les `match` exhaustifs existants).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LlmStop {
    MaxIters,
    Patience,
    Target,
    /// Budget épuisé (appels ou temps).
    BudgetExhausted,
    /// Garde-fou anti-overfitting déclenché (écart train/held-out).
    OverfitGuard,
}

/// Compte rendu d'une ascension pilotée par LLM.
#[derive(Clone, Debug)]
pub struct LlmReport {
    /// fitness **train** de l'incumbent par itération (index 0 = initial) —
    /// monotone non décroissante (élitisme).
    pub history: Vec<f64>,
    /// score **held-out** de l'incumbent par itération (reporting, peut varier).
    pub heldout: Vec<f64>,
    pub iters: usize,
    /// candidats strictement meilleurs adoptés.
    pub accepted: usize,
    /// candidats rejetés car non strictement meilleurs.
    pub rejected_worse: usize,
    /// candidats rejetés par `safety_check`.
    pub rejected_unsafe: usize,
    /// nombre d'appels LLM effectués (budget consommé).
    pub llm_calls: usize,
    pub stop: LlmStop,
}

impl LlmReport {
    /// Meilleure fitness train atteinte.
    pub fn best(&self) -> f64 {
        self.history.last().copied().unwrap_or(f64::NEG_INFINITY)
    }

    /// Dernier score held-out de l'incumbent.
    pub fn best_heldout(&self) -> f64 {
        self.heldout.last().copied().unwrap_or(f64::NEG_INFINITY)
    }

    /// **Non-régression** : l'historique train de l'incumbent est non décroissant.
    pub fn is_monotone(&self) -> bool {
        self.history.windows(2).all(|w| w[1] >= w[0] - 1e-12)
    }
}

/// Boucle d'ascension élitiste **pilotée par un LLM**, bornée et budgétée.
///
/// À chaque itération : (1) on décrit l'incumbent, (2) le `client` propose `k`
/// candidats, (3) on parse, (4) chaque candidat passe `safety_check` puis
/// `score`, (5) on adopte s'il est **strictement** meilleur. Les garde-fous
/// (bornes, budget, overfitting) sont appliqués par le moteur ; le LLM n'y a
/// jamais accès.
pub fn ascend_llm<T, C>(
    task: &mut T,
    init: T::Cand,
    client: &C,
    guard: &LlmGuard,
) -> (T::Cand, LlmReport)
where
    T: LlmRefineTask,
    C: LlmClient,
{
    let mut best = init;
    let mut best_fit = task.score(&best);
    let mut history = vec![best_fit];
    let mut heldout = vec![task.score_heldout(&best)];
    let mut accepted = 0usize;
    let mut rejected_worse = 0usize;
    let mut rejected_unsafe = 0usize;
    let mut llm_calls = 0usize;
    let mut stale = 0usize;
    let mut iters = 0usize;
    let mut stop = LlmStop::MaxIters;
    let start = Instant::now();

    for i in 0..guard.max_iters {
        iters = i + 1;

        // --- garde-fous de BUDGET (§2), avant tout appel LLM --------------- //
        if llm_calls >= guard.max_llm_calls {
            stop = LlmStop::BudgetExhausted;
            iters = i;
            break;
        }
        if let Some(max) = guard.max_wall_clock {
            if start.elapsed() >= max {
                stop = LlmStop::BudgetExhausted;
                iters = i;
                break;
            }
        }

        // --- proposition (le LLM lit l'incumbent et propose k candidats) --- //
        let prompt = task.describe(&best);
        let raw = client.propose(&prompt, guard.k);
        llm_calls += 1;
        let raw = match raw {
            Ok(v) => v,
            // un appel infructueux compte au budget mais n'altère pas l'incumbent
            Err(_) => {
                history.push(best_fit);
                heldout.push(*heldout.last().unwrap());
                stale += 1;
                if guard.patience > 0 && stale >= guard.patience {
                    stop = LlmStop::Patience;
                    break;
                }
                continue;
            }
        };

        // --- évaluation élitiste : sûreté PUIS score ----------------------- //
        let mut improved = false;
        for cand in task.parse_proposals(&raw) {
            if task.safety_check(&cand).is_err() {
                rejected_unsafe += 1;
                continue;
            }
            let fit = task.score(&cand);
            if fit > best_fit + guard.min_delta {
                best = cand; // adoption seulement si STRICTEMENT meilleur ET sûr
                best_fit = fit;
                accepted += 1;
                improved = true;
            } else {
                rejected_worse += 1;
            }
        }

        history.push(best_fit);
        let ho = task.score_heldout(&best);
        heldout.push(ho);
        if improved {
            stale = 0;
        } else {
            stale += 1;
        }

        // --- garde-fou anti-overfitting (§3) ------------------------------- //
        if let Some(max_gap) = guard.max_overfit_gap {
            if best_fit - ho > max_gap {
                stop = LlmStop::OverfitGuard;
                break;
            }
        }
        // --- cible / patience ---------------------------------------------- //
        if let Some(t) = guard.target {
            if best_fit >= t {
                stop = LlmStop::Target;
                break;
            }
        }
        if guard.patience > 0 && stale >= guard.patience {
            stop = LlmStop::Patience;
            break;
        }
    }

    (
        best,
        LlmReport {
            history,
            heldout,
            iters,
            accepted,
            rejected_worse,
            rejected_unsafe,
            llm_calls,
            stop,
        },
    )
}

/// Client LLM **déterministe** pour les tests et le développement hors-ligne.
/// Encapsule une closure `(prompt, k) -> Vec<String>` : on y scripte le
/// comportement d'un LLM sans aucun appel réseau ni dépendance.
pub struct MockLlmClient {
    proposer: Box<Proposer>,
}

/// Closure de proposition d'un [`MockLlmClient`] (alias pour la lisibilité).
type Proposer = dyn Fn(&str, usize) -> Vec<String> + Send + Sync;

impl MockLlmClient {
    pub fn new(proposer: impl Fn(&str, usize) -> Vec<String> + Send + Sync + 'static) -> Self {
        MockLlmClient {
            proposer: Box::new(proposer),
        }
    }
}

impl LlmClient for MockLlmClient {
    fn propose(&self, prompt: &str, k: usize) -> Result<Vec<String>, LlmError> {
        let out = (self.proposer)(prompt, k);
        if out.is_empty() {
            Err(LlmError::Empty)
        } else {
            Ok(out)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ascent::RefineTask;

    /// Domaine jouet : rapprocher un entier d'une cible. Sert à valider toute la
    /// mécanique `ascend_llm` hors-ligne (le « vrai » domaine prompts vient en P1.3).
    struct NumberGame {
        target: i64,
    }

    impl RefineTask for NumberGame {
        type Cand = i64;
        fn score(&self, c: &i64) -> f64 {
            // plus proche de la cible = plus grand (max 0)
            -(((*c - self.target) as f64).powi(2))
        }
        fn refine(&mut self, c: &i64, _iter: usize) -> i64 {
            *c + 1 // générateur déterministe de repli (non utilisé par le chemin LLM)
        }
    }

    impl LlmRefineTask for NumberGame {
        fn describe(&self, c: &i64) -> String {
            format!("incumbent={c}")
        }
        fn parse_proposals(&self, raw: &[String]) -> Vec<i64> {
            raw.iter().filter_map(|s| s.trim().parse().ok()).collect()
        }
        fn safety_check(&self, c: &i64) -> Result<(), SafetyViolation> {
            if *c < 0 {
                Err(SafetyViolation("valeur négative interdite".into()))
            } else {
                Ok(())
            }
        }
    }

    /// Mock « recherche locale » : lit l'incumbent dans le prompt et propose ses
    /// voisins n±1..n±k — stand-in déterministe d'un LLM qui lit puis propose.
    fn neighbor_client() -> MockLlmClient {
        MockLlmClient::new(|prompt: &str, k: usize| {
            let n: i64 = prompt
                .strip_prefix("incumbent=")
                .and_then(|s| s.trim().parse().ok())
                .unwrap_or(0);
            (1..=k as i64)
                .flat_map(|d| [(n + d).to_string(), (n - d).to_string()])
                .collect()
        })
    }

    #[test]
    fn mock_drives_convergence_to_target() {
        let mut task = NumberGame { target: 17 };
        let client = neighbor_client();
        let guard = LlmGuard {
            target: Some(0.0), // fitness 0 = cible atteinte
            max_iters: 100,
            ..LlmGuard::default()
        };
        let (best, report) = ascend_llm(&mut task, 0, &client, &guard);
        assert_eq!(best, 17, "doit converger vers la cible");
        assert_eq!(report.stop, LlmStop::Target);
        assert!(report.is_monotone(), "incumbent train non monotone");
        assert!(report.accepted > 0);
    }

    #[test]
    fn budget_caps_llm_calls() {
        let mut task = NumberGame { target: 1_000_000 }; // hors d'atteinte rapide
        let client = neighbor_client();
        let guard = LlmGuard {
            max_llm_calls: 3,
            max_iters: 10_000,
            target: None,
            ..LlmGuard::default()
        };
        let (_best, report) = ascend_llm(&mut task, 0, &client, &guard);
        assert!(report.llm_calls <= 3, "budget d'appels dépassé: {}", report.llm_calls);
        assert_eq!(report.stop, LlmStop::BudgetExhausted);
    }

    #[test]
    fn safety_check_blocks_forbidden_candidates() {
        // cible négative : un LLM naïf proposerait des candidats < 0 (interdits).
        let mut task = NumberGame { target: -50 };
        let client = neighbor_client();
        let guard = LlmGuard {
            max_iters: 60,
            ..LlmGuard::default()
        };
        let (best, report) = ascend_llm(&mut task, 5, &client, &guard);
        // jamais adopter un candidat interdit ⇒ l'incumbent reste ≥ 0
        assert!(best >= 0, "un candidat interdit a été adopté: {best}");
        assert!(report.rejected_unsafe > 0, "aucun rejet de sûreté observé");
    }

    #[test]
    fn empty_proposals_consume_budget_without_changing_incumbent() {
        let mut task = NumberGame { target: 10 };
        // mock qui ne propose jamais rien ⇒ LlmError::Empty
        let client = MockLlmClient::new(|_p, _k| Vec::new());
        let guard = LlmGuard { max_iters: 5, ..LlmGuard::default() };
        let (best, report) = ascend_llm(&mut task, 3, &client, &guard);
        assert_eq!(best, 3, "incumbent inchangé sans proposition valide");
        assert!(report.is_monotone());
        assert_eq!(report.accepted, 0);
    }

    #[test]
    fn ascend_llm_is_deterministic() {
        let run = || {
            let mut task = NumberGame { target: 23 };
            let client = neighbor_client();
            let guard = LlmGuard { target: Some(0.0), max_iters: 100, ..LlmGuard::default() };
            ascend_llm(&mut task, 0, &client, &guard)
        };
        let (b1, r1) = run();
        let (b2, r2) = run();
        assert_eq!(b1, b2);
        assert_eq!(r1.history, r2.history);
        assert_eq!(r1.llm_calls, r2.llm_calls);
    }
}
