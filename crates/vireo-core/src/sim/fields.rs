use bytemuck::{Pod, Zeroable};
use half::f16;
use rand_chacha::ChaCha8Rng;
use rand::{Rng, SeedableRng};
use std::f32::consts::TAU;

/// Field data structure for GPU compute
#[repr(C)]
#[derive(Debug, Clone, Copy, Zeroable, Pod)]
pub struct FieldData {
    pub R: f32,  // Resource concentration
    pub W: f32,  // Waste concentration
    pub _pad: [f32; 2], // Padding for alignment
}

impl FieldData {
    pub fn new(resource: f32, waste: f32) -> Self {
        Self {
            R: resource,
            W: waste,
            _pad: [0.0, 0.0],
        }
    }
    
    pub fn zero() -> Self {
        Self {
            R: 0.0,
            W: 0.0,
            _pad: [0.0, 0.0],
        }
    }
}

impl Default for FieldData {
    fn default() -> Self { Self::zeroed() }
}

/// Field statistics for metrics collection
#[derive(Debug, Clone)]
pub struct FieldStats {
    pub mean_R: f32,
    pub mean_W: f32,
    pub var_R: f32,
    pub var_W: f32,
    pub mean_grad_R: f32,
    pub max_R: f32,
    pub max_W: f32,
    pub min_R: f32,
    pub min_W: f32,
}

impl Default for FieldStats {
    fn default() -> Self {
        Self {
            mean_R: 0.0,
            mean_W: 0.0,
            var_R: 0.0,
            var_W: 0.0,
            mean_grad_R: 0.0,
            max_R: 0.0,
            max_W: 0.0,
            min_R: 0.0,
            min_W: 0.0,
        }
    }
}

/// Field manager for CPU-side operations
pub struct FieldManager {
    pub size: [u32; 2],
    pub data: Vec<FieldData>,
    pub stats: FieldStats,
}

impl FieldManager {
    pub fn new(size: [u32; 2]) -> Self {
        let data = vec![FieldData::zero(); (size[0] * size[1]) as usize];
        
        Self {
            size,
            data,
            stats: FieldStats::default(),
        }
    }
    
    pub fn get_index(&self, x: u32, y: u32) -> usize {
        (y * self.size[0] + x) as usize
    }
    
    pub fn get(&self, x: u32, y: u32) -> FieldData {
        let idx = self.get_index(x, y);
        self.data[idx]
    }
    
    pub fn set(&mut self, x: u32, y: u32, data: FieldData) {
        let idx = self.get_index(x, y);
        self.data[idx] = data;
    }
    
    pub fn get_resource(&self, x: u32, y: u32) -> f32 {
        self.get(x, y).R
    }
    
    pub fn get_waste(&self, x: u32, y: u32) -> f32 {
        self.get(x, y).W
    }
    
    pub fn set_resource(&mut self, x: u32, y: u32, value: f32) {
        let idx = self.get_index(x, y);
        self.data[idx].R = value;
    }
    
    pub fn set_waste(&mut self, x: u32, y: u32, value: f32) {
        let idx = self.get_index(x, y);
        self.data[idx].W = value;
    }
    
    /// Helper: clamp sigma in pixels so we don't create needle-thin gaussians on tiny worlds
    fn sigma_px(&self, min_dim: f32, pct: f32, min_px: f32) -> f32 {
        (min_dim * pct).max(min_px)
    }

    /// Helper: safe percent span → absolute [lo, hi] in pixels (always non-empty for size>0)
    fn span_pct(&self, size: f32, lo_pct: f32, hi_pct: f32) -> (f32, f32) {
        // ensure monotonic and in [0,1]
        let lo = size * lo_pct.min(hi_pct).max(0.0);
        let hi = size * hi_pct.max(lo_pct).min(1.0);
        // if degenerate, center 0.5±small_pad
        if hi <= lo {
            let c = 0.5 * size;
            (c - 1.0, c + 1.0)
        } else {
            (lo, hi)
        }
    }

    /// Initialize field with gaussian blobs for resources
    pub fn seed_resources(&mut self, seed: u64) {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        let w = self.size[0] as f32;
        let h = self.size[1] as f32;
        let min_dim = w.min(h);

        // Optional: fast-fail on absurdly tiny worlds
        if min_dim < 32.0 {
            log::warn!("World min dimension < 32; seeding will be very coarse.");
        }

        // 0) Clear / baseline
        for data in &mut self.data {
            *data = FieldData::zero();
        }

        // 1) Primary center source
        let center_x = 0.5 * w;
        let center_y = 0.5 * h;
        let amp_center = 0.8;                         // baseline amplitude
        let sig_center = self.sigma_px(min_dim, 0.07, 2.0); // ~7% of min dimension, ≥2px
        
        for y in 0..self.size[1] {
            for x in 0..self.size[0] {
                let dx = x as f32 - center_x;
                let dy = y as f32 - center_y;
                let r2 = (dx * dx + dy * dy) / (2.0 * sig_center * sig_center);
                let resource = amp_center * (-r2).exp();
                self.set_resource(x, y, resource);
            }
        }
        
        // 2) Clusters — scale count by size
        let num_clusters: usize = if min_dim < 192.0 { 4 } else { 8 };
        let (cx_lo, cx_hi) = self.span_pct(w, 0.15, 0.85);
        let (cy_lo, cy_hi) = self.span_pct(h, 0.15, 0.85);
        
        for _ in 0..num_clusters {
            let cluster_x = rng.gen_range(cx_lo..cx_hi);
            let cluster_y = rng.gen_range(cy_lo..cy_hi);
            let amp = rng.gen_range(0.3..0.7);
            let sigma = self.sigma_px(min_dim, 0.05, 2.0);
            
            for y in 0..self.size[1] {
                for x in 0..self.size[0] {
                    let dx = x as f32 - cluster_x;
                    let dy = y as f32 - cluster_y;
                    let r2 = (dx * dx + dy * dy) / (2.0 * sigma * sigma);
                    let resource = amp * (-r2).exp();
                    let current = self.get_resource(x, y);
                    self.set_resource(x, y, current + resource);
                }
            }
        }
        
        // 3) Scattered sources — also size-aware
        let num_sources: usize = if min_dim < 192.0 { 8 } else { 15 };
        let (sx_lo, sx_hi) = self.span_pct(w, 0.05, 0.95);
        let (sy_lo, sy_hi) = self.span_pct(h, 0.05, 0.95);
        
        for _ in 0..num_sources {
            let cx = rng.gen_range(sx_lo..sx_hi);
            let cy = rng.gen_range(sy_lo..sy_hi);
            let amp = rng.gen_range(0.2..0.5);
            let sigma = self.sigma_px(min_dim, 0.02, 1.5);
            
            for y in 0..self.size[1] {
                for x in 0..self.size[0] {
                    let dx = x as f32 - cx;
                    let dy = y as f32 - cy;
                    let r2 = (dx * dx + dy * dy) / (2.0 * sigma * sigma);
                    let resource = amp * (-r2).exp();
                    let current = self.get_resource(x, y);
                    self.set_resource(x, y, current + resource);
                }
            }
        }
        
        // 4) Gentle gradient (directional ramp)
        let theta = rng.gen_range(0.0..TAU);
        let dir_x = theta.cos();
        let dir_y = theta.sin();
        let grad_amp = 0.15 * amp_center; // subtle
        
        for y in 0..self.size[1] {
            for x in 0..self.size[0] {
                let dx = x as f32 - center_x;
                let dy = y as f32 - center_y;
                let proj = dx * dir_x + dy * dir_y;
                let gradient_factor = (proj / min_dim).max(-0.5).min(0.5);
                let current = self.get_resource(x, y);
                self.set_resource(x, y, current + gradient_factor * grad_amp);
            }
        }

        // 5) Final clamp (non-negative)
        for data in &mut self.data {
            data.R = data.R.max(0.0);
        }
    }
    
    /// Calculate field statistics
    pub fn update_stats(&mut self) {
        let mut sum_R = 0.0;
        let mut sum_W = 0.0;
        let mut sum_R_sq = 0.0;
        let mut sum_W_sq = 0.0;
        let mut sum_grad_R = 0.0;
        let mut max_R = f32::NEG_INFINITY;
        let mut max_W = f32::NEG_INFINITY;
        let mut min_R = f32::INFINITY;
        let mut min_W = f32::INFINITY;
        
        let count = self.data.len() as f32;
        
        for y in 0..self.size[1] {
            for x in 0..self.size[0] {
                let data = self.get(x, y);
                let R = data.R;
                let W = data.W;
                
                sum_R += R;
                sum_W += W;
                sum_R_sq += R * R;
                sum_W_sq += W * W;
                
                max_R = max_R.max(R);
                max_W = max_W.max(W);
                min_R = min_R.min(R);
                min_W = min_W.min(W);
                
                // Calculate gradient magnitude (central differences)
                if x > 0 && x < self.size[0] - 1 && y > 0 && y < self.size[1] - 1 {
                    let dx = (self.get_resource(x + 1, y) - self.get_resource(x - 1, y)) / 2.0;
                    let dy = (self.get_resource(x, y + 1) - self.get_resource(x, y - 1)) / 2.0;
                    let grad_mag = (dx * dx + dy * dy).sqrt();
                    sum_grad_R += grad_mag;
                }
            }
        }
        
        let mean_R = sum_R / count;
        let mean_W = sum_W / count;
        let var_R = (sum_R_sq / count) - (mean_R * mean_R);
        let var_W = (sum_W_sq / count) - (mean_W * mean_W);
        let mean_grad_R = sum_grad_R / count;
        
        self.stats = FieldStats {
            mean_R,
            mean_W,
            var_R: var_R.max(0.0), // Ensure variance is non-negative
            var_W: var_W.max(0.0),
            mean_grad_R,
            max_R,
            max_W,
            min_R,
            min_W,
        };
    }
    
    /// Add noise to resource field
    pub fn add_noise(&mut self, sigma: f32, seed: u64) {
        if sigma <= 0.0 {
            return;
        }
        
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        
        for y in 0..self.size[1] {
            for x in 0..self.size[0] {
                let noise = rng.gen_range(-sigma..sigma);
                let current = self.get_resource(x, y);
                self.set_resource(x, y, (current + noise).max(0.0)); // Clamp to non-negative
            }
        }
    }
    
    /// Convert to RGBA16F format for GPU texture
    pub fn to_rgba16f(&self) -> Vec<f16> {
        let mut result = Vec::with_capacity(self.data.len() * 4);
        
        for data in &self.data {
            result.push(f16::from_f32(data.R));
            result.push(f16::from_f32(data.W));
            result.push(f16::from_f32(0.0)); // Unused channel
            result.push(f16::from_f32(0.0)); // Unused channel
        }
        
        result
    }
    
    /// Convert from RGBA16F format from GPU texture
    pub fn from_rgba16f(&mut self, data: &[f16]) {
        let expected_len = self.data.len() * 4;
        if data.len() != expected_len {
            panic!("Invalid data length: expected {}, got {}", expected_len, data.len());
        }
        
        for (i, data_slice) in data.chunks_exact(4).enumerate() {
            if i < self.data.len() {
                self.data[i] = FieldData {
                    R: data_slice[0].to_f32(),
                    W: data_slice[1].to_f32(),
                    _pad: [0.0, 0.0],
                };
            }
        }
    }
}
