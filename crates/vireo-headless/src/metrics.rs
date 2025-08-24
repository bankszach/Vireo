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
}

impl MetricsWriter {
    /// Create a new metrics writer
    pub fn new(output_dir: &PathBuf) -> Result<Self, anyhow::Error> {
        let csv_path = output_dir.join("metrics.csv");
        let file = File::create(&csv_path)?;
        
        let mut csv_writer = Writer::from_writer(file);
        
        // Write CSV header
        csv_writer.write_record(&[
            "step",
            "mean_R", "mean_W", "var_R", "var_W", "mean_grad_R",
            "max_R", "max_W", "min_R", "min_W",
            "alive_count", "total_energy", "mean_energy", "mean_velocity", "foraging_efficiency",
            "wall_time_ms", "fps_proxy"
        ])?;
        
        Ok(Self {
            csv_writer,
            step_count: 0,
        })
    }
    
    /// Write metrics for a single simulation step
    pub fn write_step(
        &mut self,
        step: u32,
        field_stats: &FieldStats,
        agent_stats: &AgentStats,
        step_time: Duration,
    ) -> Result<(), anyhow::Error> {
        let wall_time_ms = step_time.as_millis() as f64;
        let fps_proxy = if wall_time_ms > 0.0 { 1000.0 / wall_time_ms } else { 0.0 };
        
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
