//! §3 (extension) — **Sonde matérielle réelle** pour ancrer le substrat.
//!
//! Lit des métriques **réelles** de la machine hôte pour construire le vecteur
//! matériel `H` du [`Substrate`] (§3), au lieu de valeurs synthétiques :
//!   - charge CPU (`/proc/loadavg` ÷ nb cœurs) ;
//!   - mémoire utilisée (`/proc/meminfo`) ;
//!   - charge GPU (sysfs Tegra/Jetson, sinon `nvidia-smi`) — **optionnelle**.
//!
//! 100 % std-only, **dégradation propre** : sur une machine sans GPU exposé (CI,
//! conteneur), `gpu_load_frac` vaut `None` et le reste fonctionne. Sur un
//! **Jetson** (Thor/Orin), la charge GPU est lue réellement → `P_eff` reflète
//! l'état matériel courant.
//!
//! Sémantique : `H` encode la **capacité disponible** (`1 − utilisation`) — une
//! machine chargée a moins de puissance effective disponible, donc un `P_eff`
//! plus faible (`hardware_efficiency = σ(HᵀA H)`).
//!
//! > Comme toute mesure réelle, c'est **non déterministe** (dépend de la charge
//! > instantanée) — à n'utiliser que là où l'on veut ancrer le modèle sur le
//! > matériel réel, pas dans les tests bit-exacts. Cf. `docs/SAFETY.md`.

use std::io::Read;
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use crate::rng::Rng;
use crate::substrate::Substrate;

/// Instantané des métriques matérielles réelles de l'hôte.
#[derive(Clone, Debug)]
pub struct HardwareSnapshot {
    /// Nombre de cœurs logiques.
    pub cpu_count: usize,
    /// Charge CPU normalisée `loadavg(1 min) / cpu_count`, clampée `[0,1]`.
    pub cpu_load_frac: f64,
    /// Fraction de mémoire utilisée `1 − MemAvailable/MemTotal`, `[0,1]`.
    pub mem_used_frac: f64,
    /// Charge GPU `[0,1]` si lisible (Jetson sysfs ou `nvidia-smi`), sinon `None`.
    pub gpu_load_frac: Option<f64>,
    /// D'où vient la mesure GPU (diagnostic) : `"sysfs:<path>"`, `"nvidia-smi"`,
    /// ou `"absent"`.
    pub gpu_source: String,
}

impl HardwareSnapshot {
    /// Effectue une mesure réelle de l'hôte (best-effort, dégradation propre).
    pub fn probe() -> Self {
        let cpu_count = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);

        let cpu_load_frac = read_loadavg()
            .map(|l| (l / cpu_count as f64).clamp(0.0, 1.0))
            .unwrap_or(0.0);

        let mem_used_frac = read_mem_used_frac().unwrap_or(0.0);

        let (gpu_load_frac, gpu_source) = read_gpu_load();

        HardwareSnapshot { cpu_count, cpu_load_frac, mem_used_frac, gpu_load_frac, gpu_source }
    }

    /// Vecteur matériel `H` (capacité **disponible** ∈ `[0,1]`) :
    /// `[cpu_dispo, mem_dispo, gpu_dispo]`. GPU absent ⇒ capacité neutre `0.5`.
    pub fn hardware_vector(&self) -> Vec<f64> {
        vec![
            (1.0 - self.cpu_load_frac).clamp(0.0, 1.0),
            (1.0 - self.mem_used_frac).clamp(0.0, 1.0),
            self.gpu_load_frac.map(|g| (1.0 - g).clamp(0.0, 1.0)).unwrap_or(0.5),
        ]
    }
}

/// Construit un [`Substrate`] dont le vecteur matériel `H` provient d'une mesure
/// **réelle** (le reste — matrices A/B/C, vecteur logiciel O — est tiré par
/// `default_with`). `P_eff` reflète alors l'état matériel courant de l'hôte.
pub fn measured_hardware_substrate(
    snap: &HardwareSnapshot,
    n_software: usize,
    rng: &mut Rng,
) -> Substrate {
    let hv = snap.hardware_vector();
    let mut sub = Substrate::default_with(hv.len(), n_software, rng);
    sub.h = hv;
    sub
}

// --------------------------------------------------------------------------- //
// Lecteurs bas niveau (std-only, dégradation propre)
// --------------------------------------------------------------------------- //

/// Charge moyenne sur 1 minute (`/proc/loadavg`, 1ᵉʳ champ).
fn read_loadavg() -> Option<f64> {
    let s = std::fs::read_to_string("/proc/loadavg").ok()?;
    s.split_whitespace().next()?.parse().ok()
}

/// Fraction de mémoire utilisée via `/proc/meminfo`.
fn read_mem_used_frac() -> Option<f64> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    let mut total = None;
    let mut avail = None;
    for line in s.lines() {
        if let Some(v) = line.strip_prefix("MemTotal:") {
            total = v.split_whitespace().next().and_then(|x| x.parse::<f64>().ok());
        } else if let Some(v) = line.strip_prefix("MemAvailable:") {
            avail = v.split_whitespace().next().and_then(|x| x.parse::<f64>().ok());
        }
    }
    let (t, a) = (total?, avail?);
    if t <= 0.0 {
        return None;
    }
    Some((1.0 - a / t).clamp(0.0, 1.0))
}

/// Chemins sysfs candidats exposant la charge GPU en *per-mille* (0–1000),
/// couvrant les générations Tegra/Jetson (Orin, Xavier, Thor…).
const GPU_SYSFS: &[&str] = &[
    "/sys/devices/platform/gpu.0/load",
    "/sys/devices/gpu.0/load",
    "/sys/class/devfreq/17000000.gpu/device/load",
    "/sys/class/devfreq/17000000.ga10b/device/load",
    "/sys/class/devfreq/gpu.0/device/load",
];

/// Charge GPU `[0,1]` : essaie sysfs (per-mille), puis `nvidia-smi` (%).
fn read_gpu_load() -> (Option<f64>, String) {
    for path in GPU_SYSFS {
        if let Ok(s) = std::fs::read_to_string(path) {
            if let Ok(permille) = s.trim().parse::<f64>() {
                return (Some((permille / 1000.0).clamp(0.0, 1.0)), format!("sysfs:{path}"));
            }
        }
    }
    if let Some(pct) = read_gpu_via_nvidia_smi() {
        return (Some((pct / 100.0).clamp(0.0, 1.0)), "nvidia-smi".to_string());
    }
    (None, "absent".to_string())
}

/// `nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits`
/// (sous-processus **borné** : 2 s, sortie plafonnée). `None` si indisponible.
fn read_gpu_via_nvidia_smi() -> Option<f64> {
    const TIMEOUT: Duration = Duration::from_secs(2);
    const MAX_OUTPUT: u64 = 64 * 1024;

    let mut cmd = Command::new("nvidia-smi");
    cmd.args(["--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"])
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::null());
    let mut child = cmd.spawn().ok()?;

    let stdout = child.stdout.take()?;
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let mut buf = Vec::new();
        let _ = stdout.take(MAX_OUTPUT).read_to_end(&mut buf);
        let _ = tx.send(buf);
    });

    let deadline = Instant::now() + TIMEOUT;
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    return None;
                }
                std::thread::sleep(Duration::from_millis(20));
            }
            Err(_) => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
        }
    }

    let buf = rx.recv_timeout(Duration::from_secs(1)).ok()?;
    let text = String::from_utf8_lossy(&buf);
    // moyenne des GPU listés (une valeur par ligne)
    let vals: Vec<f64> = text
        .lines()
        .filter_map(|l| l.trim().parse::<f64>().ok())
        .collect();
    if vals.is_empty() {
        None
    } else {
        Some(vals.iter().sum::<f64>() / vals.len() as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn probe_returns_sane_ranges() {
        let s = HardwareSnapshot::probe();
        assert!(s.cpu_count >= 1);
        assert!((0.0..=1.0).contains(&s.cpu_load_frac));
        assert!((0.0..=1.0).contains(&s.mem_used_frac));
        if let Some(g) = s.gpu_load_frac {
            assert!((0.0..=1.0).contains(&g));
        }
    }

    #[test]
    fn hardware_vector_is_unit_bounded() {
        let s = HardwareSnapshot::probe();
        let h = s.hardware_vector();
        assert_eq!(h.len(), 3);
        assert!(h.iter().all(|&x| (0.0..=1.0).contains(&x)));
    }

    #[test]
    fn substrate_from_probe_has_valid_p_eff() {
        let mut rng = Rng::new(42);
        let snap = HardwareSnapshot::probe();
        let sub = measured_hardware_substrate(&snap, 4, &mut rng);
        let p = sub.effective_power();
        assert!(p > 0.0 && p < 1.0, "P_eff = {p}");
        assert_eq!(sub.h, snap.hardware_vector());
    }

    #[test]
    fn mem_frac_parses_proc() {
        // /proc/meminfo existe sur Linux (CI) — sinon None toléré.
        if let Some(f) = read_mem_used_frac() {
            assert!((0.0..=1.0).contains(&f));
        }
    }
}
