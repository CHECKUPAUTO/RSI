//! Petites primitives d'algèbre linéaire (std-only) : vecteurs, matrices
//! denses, formes quadratiques xᵀMx, et la sigmoïde σ utilisée partout.

/// σ(x) = 1 / (1 + e⁻ˣ), implémentation numériquement stable.
#[inline]
pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Produit scalaire de deux vecteurs de même longueur.
#[inline]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Norme euclidienne ‖v‖.
#[inline]
pub fn norm(v: &[f64]) -> f64 {
    dot(v, v).sqrt()
}

/// Moyenne des éléments (0.0 si vide).
#[inline]
pub fn mean(v: &[f64]) -> f64 {
    if v.is_empty() {
        0.0
    } else {
        v.iter().sum::<f64>() / v.len() as f64
    }
}

/// Matrice dense en ligne-major.
#[derive(Clone, Debug)]
pub struct Matrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Matrix {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Matrix {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn from_vec(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), rows * cols, "dimensions incompatibles");
        Matrix { rows, cols, data }
    }

    pub fn identity(n: usize) -> Self {
        let mut m = Matrix::zeros(n, n);
        for i in 0..n {
            m.data[i * n + i] = 1.0;
        }
        m
    }

    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.cols + j]
    }

    #[inline]
    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        self.data[i * self.cols + j] = v;
    }

    /// Produit matrice-vecteur M·v (v de taille `cols`, sortie de taille `rows`).
    pub fn matvec(&self, v: &[f64]) -> Vec<f64> {
        assert_eq!(v.len(), self.cols, "matvec: dimension du vecteur");
        let mut out = vec![0.0; self.rows];
        for i in 0..self.rows {
            let mut acc = 0.0;
            let base = i * self.cols;
            for j in 0..self.cols {
                acc += self.data[base + j] * v[j];
            }
            out[i] = acc;
        }
        out
    }

    /// Forme bilinéaire aᵀ · M · b.
    pub fn bilinear(&self, a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), self.rows);
        let mb = self.matvec(b);
        dot(a, &mb)
    }

    /// Forme quadratique xᵀ·M·x.
    pub fn quadratic(&self, x: &[f64]) -> f64 {
        self.bilinear(x, x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigmoid_bounds() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-12);
        assert!(sigmoid(40.0) > 0.999);
        assert!(sigmoid(-40.0) < 0.001);
    }

    #[test]
    fn quadratic_identity() {
        let m = Matrix::identity(3);
        let x = [1.0, 2.0, 2.0];
        assert!((m.quadratic(&x) - 9.0).abs() < 1e-12);
    }
}
