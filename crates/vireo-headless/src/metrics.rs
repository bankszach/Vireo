use std::path::PathBuf;
use std::fs::File;
use std::io::Write;
use csv::Writer;
use std::time::Duration;
use vireo_core::sim::{FieldStats, AgentStats};

/// Metrics writer for CSV output and performance logging
pub struct MetricsWriter {
    csv_writer: Writer<File>,
    step_count: u32,
    herbivore_history: Vec<u32>, // Track herbivore counts for cycle detection
    energy_history: Vec<f32>,    // Track energy for cycle detection
}

impl MetricsWriter {
    /// Create a new metrics writer
    pub fn new(output_dir: &PathBuf) -> Result<Self, anyhow::Error> {
        let csv_path = output_dir.join("metrics.csv");
        let file = File::create(&csv_path)?;
        
        let mut csv_writer = Writer::from_writer(file);
        
        // Write CSV header with enhanced metrics
        csv_writer.write_record(&[
            "step",
            "mean_R", "mean_W", "var_R", "var_W", "mean_grad_R",
            "max_R", "max_W", "min_R", "min_W",
            "alive_count", "total_energy", "mean_energy", "mean_velocity", "foraging_efficiency",
            "cycle_score", "foraging_efficiency_enhanced",
            "wall_time_ms", "fps_proxy"
        ])?;
        
        Ok(Self {
            csv_writer,
            step_count: 0,
            herbivore_history: Vec::new(),
            energy_history: Vec::new(),
        })
    }
    
    /// Compute cycle score based on autocorrelation of herbivore count
    fn compute_cycle_score(&self, current_count: u32) -> f32 {
        if self.herbivore_history.len() < 50 {
            return 0.0; // Need more data for meaningful cycle detection
        }
        
        // Simple autocorrelation-based cycle score
        let window_size = 20.min(self.herbivore_history.len() / 2);
        let mut autocorr_sum = 0.0;
        let mut count = 0;
        
        for lag in 1..=window_size {
            if lag < self.herbivore_history.len() {
                let current = self.herbivore_history[self.herbivore_history.len() - 1] as f32;
                let lagged = self.herbivore_history[self.herbivore_history.len() - 1 - lag] as f32;
                autocorr_sum += (current - lagged).abs();
                count += 1;
            }
        }
        
        if count > 0 {
            let avg_variation = autocorr_sum / count as f32;
            // Normalize to [0, 1] range, higher means more cycling
            (avg_variation / 100.0).min(1.0)
        } else {
            0.0
        }
    }
    
    /// Compute enhanced foraging efficiency (energy gain per unit movement)
    fn compute_enhanced_foraging_efficiency(&self, agent_stats: &AgentStats) -> f32 {
        if agent_stats.mean_velocity > 0.0 {
            // Enhanced efficiency: energy per unit movement, normalized
            let base_efficiency = agent_stats.mean_energy / agent_stats.mean_velocity;
            // Normalize to reasonable range [0, 1]
            (base_efficiency / 10.0).min(1.0)
        } else {
            0.0
        }
    }
    
    /// Write metrics for a single simulation step
    pub fn write_step(
        &mut self,
        step: u32,
        field_stats: &FieldStats,
        agent_stats: &AgentStats,
        step_time: Duration,
    ) -> Result<(), anyhow::Error> {
        // Update history for cycle detection
        self.herbivore_history.push(agent_stats.alive_count);
        self.energy_history.push(agent_stats.mean_energy);
        
        // Keep only last 200 entries to avoid memory bloat
        if self.herbivore_history.len() > 200 {
            self.herbivore_history.remove(0);
            self.energy_history.remove(0);
        }
        
        let wall_time_ms = step_time.as_millis() as f64;
        let fps_proxy = if wall_time_ms > 0.0 { 1000.0 / wall_time_ms } else { 0.0 };
        
        let cycle_score = self.compute_cycle_score(agent_stats.alive_count);
        let foraging_efficiency_enhanced = self.compute_enhanced_foraging_efficiency(agent_stats);
        
        self.csv_writer.write_record(&[
            &step.to_string(),
            &field_stats.mean_R.to_string(),
            &field_stats.mean_W.to_string(),
            &field_stats.var_R.to_string(),
            &field_stats.var_W.to_string(),
            &field_stats.mean_grad_R.to_string(),
            &field_stats.max_R.to_string(),
            &field_stats.max_W.to_string(),
            &field_stats.min_R.to_string(),
            &field_stats.min_W.to_string(),
            &agent_stats.alive_count.to_string(),
            &agent_stats.total_energy.to_string(),
            &agent_stats.mean_energy.to_string(),
            &agent_stats.mean_velocity.to_string(),
            &agent_stats.foraging_efficiency.to_string(),
            &cycle_score.to_string(),
            &foraging_efficiency_enhanced.to_string(),
            &wall_time_ms.to_string(),
            &fps_proxy.to_string(),
        ])?;
        
        self.csv_writer.flush()?;
        self.step_count += 1;
        
        Ok(())
    }
    
    /// Get the number of steps written
    pub fn step_count(&self) -> u32 {
        self.step_count
    }
}
