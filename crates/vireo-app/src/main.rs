//! Vireo Interactive App
//! 
//! Interactive GUI for the ecosystem simulation with real-time visualization.

mod viewer;
mod renderer;

use clap::Parser;
use std::path::PathBuf;
use vireo_params::SimulationConfig;
use anyhow::Result;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Configuration file path
    #[arg(short, long, default_value = "lab/configs/best-demo.yaml")]
    config: PathBuf,
    
    /// Random seed for reproducible simulations
    #[arg(short, long, default_value = "1337")]
    seed: u64,
}

fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();
    
    let cli = Cli::parse();
    
    // Load configuration
    println!("Loading configuration from {}", cli.config.display());
    let mut config: SimulationConfig = serde_yaml::from_str(
        &std::fs::read_to_string(&cli.config)?
    )?;
    
    // Override seed if provided
    config.world.seed = cli.seed;
    
    println!("Starting Vireo Interactive Viewer");
    println!("World size: {}x{}", config.world.size[0], config.world.size[1]);
    println!("Agents: {}", config.agents.herbivores);
    println!("Seed: {}", config.world.seed);
    
    // Run the interactive viewer
    pollster::block_on(viewer::run_viewer(config))?;
    
    Ok(())
}
