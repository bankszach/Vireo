use std::path::PathBuf;
use std::fs::File;
use std::io::Write;
use csv::Writer;
use image::{ImageBuffer, Rgb, RgbImage};
use vireo_core::sim::{FieldManager, AgentManager};

/// Snapshot writer for field images and agent data
pub struct SnapshotWriter {
    output_dir: PathBuf,
}

impl SnapshotWriter {
    /// Create a new snapshot writer
    pub fn new(output_dir: &PathBuf) -> Result<Self, anyhow::Error> {
        Ok(Self {
            output_dir: output_dir.clone(),
        })
    }
    
    /// Write a field snapshot as PNG image
    pub fn write_field_snapshot(
        &self,
        step: u32,
        field_manager: &FieldManager,
    ) -> Result<(), anyhow::Error> {
        let filename = format!("R_{:04}.png", step);
        let filepath = self.output_dir.join(&filename);
        
        // Create RGB image from field data
        let mut img: RgbImage = ImageBuffer::new(field_manager.size[0], field_manager.size[1]);
        
        // Find value range for normalization
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;
        
        for y in 0..field_manager.size[1] {
            for x in 0..field_manager.size[0] {
                let data = field_manager.get(x, y);
                min_val = min_val.min(data.R);
                max_val = max_val.max(data.R);
            }
        }
        
        let range = max_val - min_val;
        let range = if range > 0.0 { range } else { 1.0 };
        
        // Convert to RGB (R channel = resource, G channel = waste)
        for y in 0..field_manager.size[1] {
            for x in 0..field_manager.size[0] {
                let data = field_manager.get(x, y);
                
                // Normalize resource value to [0, 255]
                let r_val = ((data.R - min_val) / range * 255.0) as u8;
                let g_val = ((data.W * 255.0).min(255.0)) as u8; // Waste in green channel
                let b_val = 0; // Blue channel unused
                
                img.put_pixel(x, y, Rgb([r_val, g_val, b_val]));
            }
        }
        
        // Save PNG
        img.save(&filepath)?;
        
        Ok(())
    }
    
    /// Write agent positions and states to CSV
    pub fn write_agents_snapshot(
        &self,
        step: u32,
        agent_manager: &AgentManager,
    ) -> Result<(), anyhow::Error> {
        let filename = format!("agents_{:04}.csv", step);
        let filepath = self.output_dir.join(&filename);
        
        let file = File::create(&filepath)?;
        let mut csv_writer = Writer::from_writer(file);
        
        // Write CSV header
        csv_writer.write_record(&[
            "id", "x", "y", "vx", "vy", "energy", "alive"
        ])?;
        
        // Write agent data
        for (i, agent) in agent_manager.agents.iter().enumerate() {
            csv_writer.write_record(&[
                &i.to_string(),
                &agent.pos[0].to_string(),
                &agent.pos[1].to_string(),
                &agent.vel[0].to_string(),
                &agent.vel[1].to_string(),
                &agent.energy.to_string(),
                &agent.alive.to_string(),
            ])?;
        }
        
        csv_writer.flush()?;
        
        Ok(())
    }
}
