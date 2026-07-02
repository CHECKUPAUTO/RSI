//! Benchmark du parseur `rsi::json` pour la boucle DGM — première cible
//! **self** : le sujet optimisé est un vrai module de RSI, pas un kernel jouet.
//!
//! Imprime `RSI_BENCH_SCORE=<débit>` (Mo parsés/seconde, plus grand = mieux).
//! Le gate DGM exécute toute la suite (`cargo test`), qui couvre largement le
//! parseur (fuzz « ne panique jamais », profondeur bornée, échappements UTF-16,
//! rejets de déchets en fin d'entrée…) : une réécriture plus rapide mais moins
//! robuste échoue au gate.
//!
//! Lancement direct : `cargo run --release --example bench_json`
//! Via DGM :
//!   rsi-dgm . --goal "accelere Json::parse, meme comportement" \
//!       --allow src/json.rs \
//!       --bench "run --release --example bench_json" --min-gain 0.05

use rsi::json::Json;
use std::fmt::Write as _;
use std::time::Instant;

/// Document synthétique représentatif (~120 Ko) : objets imbriqués, tableaux,
/// chaînes avec échappements, nombres variés — déterministe.
fn build_document() -> String {
    let mut s = String::with_capacity(400_000);
    s.push_str("{\"sessions\":[");
    for i in 0..600 {
        if i > 0 {
            s.push(',');
        }
        let _ = write!(
            s,
            "{{\"id\":\"sess-{i:05}\",\"score\":{},\"active\":{},\"tags\":[\"a{i}\",\"b\\n{}\",\"c\\u00e9{}\"],\
             \"metrics\":{{\"si\":{},\"phi\":{},\"steps\":{}}},\"note\":\"itération {i} — \\\"guillemets\\\" et \\\\antislash\"}}",
            (i as f64) * 0.503 - 42.0,
            i % 3 == 0,
            i % 7,
            i % 11,
            (i as f64) * 1e-3,
            ((i * 37) % 1000) as f64 / 999.0,
            i * 13
        );
    }
    s.push_str("]}");
    s
}

fn main() {
    let doc = build_document();
    let mb = doc.len() as f64 / (1024.0 * 1024.0);

    // Échauffement + sanité : le document DOIT parser.
    let parsed = Json::parse(&doc).expect("le document de bench doit parser");
    std::hint::black_box(&parsed);

    // Médiane de `reps` mesures ; chaque mesure = `iters` parses complets.
    let reps = 9;
    let iters = 20u64;
    let mut times: Vec<f64> = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t0 = Instant::now();
        for _ in 0..iters {
            let v = Json::parse(std::hint::black_box(&doc)).expect("parse");
            std::hint::black_box(&v);
        }
        times.push(t0.elapsed().as_secs_f64());
    }
    times.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let median = times[times.len() / 2].max(1e-12);

    let score = iters as f64 * mb / median; // Mo/s
    println!(
        "json::parse: {:.1} Ko/doc, mediane {:.2} ms / {iters} parses",
        doc.len() as f64 / 1024.0,
        median * 1e3
    );
    println!("RSI_BENCH_SCORE={score}");
}
