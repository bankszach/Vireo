mod metrics;
mod snapshots;

use clap::Parser;
use std::path::PathBuf;
use std::time::Instant;
use vireo_core::sim::SimulationConfig;
use vireo_core::gpu::GpuDevice;
use vireo_core::gpu::{FieldTextures, ComputePipelines};
use vireo_core::sim::{FieldManager, AgentManager, RDParams, AgentParams};
use metrics::MetricsWriter;
use snapshots::SnapshotWriter;

#[derive(Parser)]
#[command(name = "vireo-headless")]
#[command(about = "Headless CLI runner for Vireo ecosystem experiments")]
struct Cli {
    /// Configuration file path
    #[arg(short, long, value_name = "FILE")]
    config: PathBuf,
    
    /// Output directory for results
    #[arg(short, long, value_name = "DIR")]
    out: PathBuf,
    
    /// Enable strict mode (fail on any errors)
    #[arg(long)]
    strict: bool,
}

fn main() -> Result<(), anyhow::Error> {
    let cli = Cli::parse();
    
    // Load configuration
    println!("Loading configuration from {}", cli.config.display());
    let config: SimulationConfig = serde_yaml::from_str(
        &std::fs::read_to_string(&cli.config)?
    )?;
    
    // Validate configuration
    let w = config.world.size[0] as u32;
    let h = config.world.size[1] as u32;
    if w < 32 || h < 32 {
        anyhow::bail!("World size too small ({}x{}). Minimum supported is 32x32.", w, h);
    }
    if config.world.steps == 0 {
        anyhow::bail!("Step count must be greater than 0.");
    }
    if config.world.dt <= 0.0 {
        anyhow::bail!("Time step (dt) must be positive.");
    }
    
    // Create output directory
    std::fs::create_dir_all(&cli.out)?;
    
    // Initialize GPU
    println!("Initializing GPU...");
    let gpu = pollster::block_on(GpuDevice::new());
    println!("{}", gpu.info());
    
    // Create simulation components
    let mut field_manager = FieldManager::new(config.world.size);
    let mut agent_manager = AgentManager::new(
        config.agents.herbivores,
        [config.world.size[0] as f32, config.world.size[1] as f32],
        config.agents.E0,
        config.world.seed,
    );
    
    // Seed the field
    println!("Seeding field with resources...");
    field_manager.seed_resources(config.world.seed);
    
    // Create GPU resources
    let field_textures = FieldTextures::new(&gpu.device, config.world.size);
    let compute_pipelines = ComputePipelines::new(&gpu.device);
    
    // Upload initial data
    field_textures.upload_field_data(&gpu.queue, &field_manager);
    
    // Create GPU buffers
    let rd_params = RDParams::from(&config);
    let agent_params = AgentParams::from(&config);
    
    let rd_params_buffer = gpu.create_rd_params_buffer(&rd_params);
    let agent_params_buffer = gpu.create_agent_params_buffer(&agent_params);
    let agents_buffer = gpu.create_agents_buffer(&agent_manager.agents);
    let occupancy_buffer = gpu.create_occupancy_buffer(config.world.size);
    
    // Create bind groups
    let rd_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &compute_pipelines.rd_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&field_textures.view_a_sample),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&field_textures.view_b_store),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: rd_params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: occupancy_buffer.as_entire_binding(),
            },
        ],
        label: Some("rd_bg"),
    });
    
    let agent_bg = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &compute_pipelines.agent_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: agents_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&field_textures.view_a_sample),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: agent_params_buffer.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: occupancy_buffer.as_entire_binding(),
            },
        ],
        label: Some("agent_bg"),
    });
    
    // Initialize metrics collection
    let mut metrics_writer = MetricsWriter::new(&cli.out)?;
    let mut snapshot_writer = SnapshotWriter::new(&cli.out)?;
    
    // Main simulation loop
    println!("Starting simulation for {} steps...", config.world.steps);
    let start_time = Instant::now();
    let mut use_a_as_src = true;
    
    for step in 0..=config.world.steps {
        let step_start = Instant::now();
        
        // Zero occupancy buffer
        let zero_occupancy = vec![0u32; (config.world.size[0] * config.world.size[1]) as usize];
        gpu.queue.write_buffer(&occupancy_buffer, 0, bytemuck::cast_slice(&zero_occupancy));
        
        // Agents pass -> occupancy
        {
            let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("agent_pass"),
            });
            
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("agent pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&compute_pipelines.agent_pipeline);
                cpass.set_bind_group(0, &agent_bg, &[]);
                
                let gx = (config.agents.herbivores + 127) / 128;
                cpass.dispatch_workgroups(gx, 1, 1);
            } // cpass is dropped here
            
            gpu.submit(encoder.finish());
        }
        
        // RD pass uses occupancy
        {
            let (src_view, dst_view) = field_textures.get_ping_pong_views(use_a_as_src);
            
            let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rd_pass"),
            });
            
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("rd pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&compute_pipelines.rd_pipeline);
                cpass.set_bind_group(0, &rd_bg, &[]);
                
                let gx = (config.world.size[0] + 7) / 8;
                let gy = (config.world.size[1] + 7) / 8;
                cpass.dispatch_workgroups(gx, gy, 1);
            } // cpass is dropped here
            
            gpu.submit(encoder.finish());
        }
        
        // Flip ping-pong
        use_a_as_src = !use_a_as_src;
        
        // Optional: add noise to R
        if config.noise.sigma > 0.0 {
            // For now, we'll add noise on the CPU side after downloading
            // In a production version, this could be done on GPU
        }
        
        // Metrics and logging every 50 steps
        if step % 50 == 0 {
            // Download field data for metrics
            field_textures.download_field_data(&gpu.device, &gpu.queue, &mut field_manager);
            
            // Update statistics
            field_manager.update_stats();
            agent_manager.update_stats();
            
            // Write metrics
            let step_time = step_start.elapsed();
            metrics_writer.write_step(step, &field_manager.stats, &agent_manager.stats, step_time)?;
            
            println!("Step {}: R={:.3}, W={:.3}, Agents={}, Time={:?}", 
                step, 
                field_manager.stats.mean_R, 
                field_manager.stats.mean_W,
                agent_manager.stats.alive_count,
                step_time
            );
        }
        
        // Snapshots at specific steps
        if matches!(step, 0 | 200 | 1000 | 2000) {
            // Download field data for snapshot
            field_textures.download_field_data(&gpu.device, &gpu.queue, &mut field_manager);
            
            // Write snapshots
            snapshot_writer.write_field_snapshot(step, &field_manager)?;
            snapshot_writer.write_agents_snapshot(step, &agent_manager)?;
            
            println!("Snapshot written for step {}", step);
        }
        
        // Check for extinction
        if agent_manager.get_alive_count() == 0 {
            println!("Warning: All agents died at step {}", step);
            break;
        }
    }
    
    let total_time = start_time.elapsed();
    println!("Simulation completed in {:?}", total_time);
    println!("Results written to {}", cli.out.display());
    
    Ok(())
}
