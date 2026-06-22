//! §2bis — SOURCE DE CONNAISSANCES (ancrage de la composante D)
//!
//! La composante `D` (connaissances) n'est plus une abstraction : le port
//! [`KnowledgeSource`] l'alimente depuis une **vraie source** (documents). À
//! chaque absorption, la source ingère du contenu réel, en extrait des
//! *concepts* distincts, et renvoie un **niveau de connaissance** ∈ [0,1]
//! (saturant) vers lequel l'agent fait tendre `D`.
//!
//! [`CorpusKnowledge`] lit des textes en mémoire ou un répertoire de fichiers.
//! Une source lourde (p. ex. PAPERS) se brancherait via un adaptateur en
//! sous-processus implémentant le même trait, sans alourdir le cœur.

use std::collections::HashSet;
use std::path::Path;

/// Source de connaissances : chaque `absorb` ingère un lot et renvoie le niveau
/// cumulé de connaissance ∈ [0,1] (monotone croissant).
pub trait KnowledgeSource {
    fn absorb(&mut self) -> f64;
    /// Niveau courant sans nouvelle ingestion.
    fn level(&self) -> f64;
}

/// Source de connaissances adossée à un corpus de documents réels.
///
/// Ingère un document par appel ; le niveau sature avec le nombre de **concepts
/// distincts** appris : `level = 1 − exp(−|concepts| / scale)`.
pub struct CorpusKnowledge {
    documents: Vec<String>,
    cursor: usize,
    concepts: HashSet<String>,
    scale: f64,
}

impl CorpusKnowledge {
    /// Construit depuis des textes en mémoire.
    pub fn from_texts(documents: Vec<String>) -> Self {
        CorpusKnowledge { documents, cursor: 0, concepts: HashSet::new(), scale: 64.0 }
    }

    /// Construit depuis tous les fichiers d'un répertoire (lecture paresseuse au fil
    /// des `absorb`). Les fichiers illisibles sont ignorés.
    pub fn from_dir(dir: impl AsRef<Path>) -> std::io::Result<Self> {
        let mut docs = Vec::new();
        for entry in std::fs::read_dir(dir)? {
            let path = entry?.path();
            if path.is_file() {
                if let Ok(text) = std::fs::read_to_string(&path) {
                    docs.push(text);
                }
            }
        }
        Ok(CorpusKnowledge::from_texts(docs))
    }

    /// Règle l'échelle de saturation (nombre de concepts pour ~63 % du niveau).
    pub fn with_scale(mut self, scale: f64) -> Self {
        self.scale = scale.max(1.0);
        self
    }

    pub fn concept_count(&self) -> usize {
        self.concepts.len()
    }

    fn extract(text: &str, into: &mut HashSet<String>) {
        for raw in text.split(|c: char| !c.is_alphanumeric()) {
            let tok = raw.to_lowercase();
            // concept = jeton alphabétique de longueur significative
            if tok.len() >= 4 && tok.chars().any(|c| c.is_alphabetic()) {
                into.insert(tok);
            }
        }
    }

    fn compute_level(&self) -> f64 {
        1.0 - (-(self.concepts.len() as f64) / self.scale).exp()
    }
}

impl KnowledgeSource for CorpusKnowledge {
    fn absorb(&mut self) -> f64 {
        if self.cursor < self.documents.len() {
            let doc = self.documents[self.cursor].clone();
            Self::extract(&doc, &mut self.concepts);
            self.cursor += 1;
        }
        self.compute_level()
    }

    fn level(&self) -> f64 {
        self.compute_level()
    }
}

/// Source triviale de niveau constant (utile pour tests / calibration).
pub struct StaticKnowledge(pub f64);

impl KnowledgeSource for StaticKnowledge {
    fn absorb(&mut self) -> f64 {
        self.0.clamp(0.0, 1.0)
    }
    fn level(&self) -> f64 {
        self.0.clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_grows_with_documents() {
        let docs = vec![
            "recursive self improvement geometry surface intelligence".to_string(),
            "substrate hardware software coupling efficiency multiplicative".to_string(),
            "criticality failure modes risk priority number wireheading".to_string(),
        ];
        let mut k = CorpusKnowledge::from_texts(docs).with_scale(16.0);
        let l0 = k.level();
        let l1 = k.absorb();
        let l2 = k.absorb();
        let l3 = k.absorb();
        assert!(l0 == 0.0);
        assert!(l1 > l0 && l2 > l1 && l3 > l2, "{l0} {l1} {l2} {l3}");
        assert!(l3 < 1.0 && l3 > 0.0);
        // au-delà du corpus, le niveau se stabilise
        let l4 = k.absorb();
        assert!((l4 - l3).abs() < 1e-12);
    }
}
