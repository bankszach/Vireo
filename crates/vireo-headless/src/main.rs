mod metrics;
mod snapshots;

use clap::Parser;
use clap::ValueEnum;
use std::path::PathBuf;
use std::time::Instant;
use vireo_core::sim::SimulationConfig;
use vireo_core::gpu::GpuDevice;
use vireo_core::gpu::{FieldTextures, ComputePipelines};
use vireo_core::sim::{FieldManager, AgentManager, RDParams, AgentParams};
use metrics::MetricsWriter;
use snapshots::SnapshotWriter;

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct Cli {
    /// Configuration file path
    #[arg(short, long, default_value = "lab/configs/best-demo.yaml")]
    config: PathBuf,
    
    /// Output directory for results
    #[arg(short, long, default_value = "results")]
    out: PathBuf,
    
    /// Enable strict validation
    #[arg(long)]
    strict: bool,
    
    /// Enable debug scenarios for testing individual components
    #[arg(long)]
    debug_scenario: bool,
    
    /// Test specific scenario: reaction-only, diffusion-only, uptake-only, damping-only
    #[arg(long, value_enum)]
    scenario: Option<Scenario>,
}

#[derive(ValueEnum, Clone)]
enum Scenario {
    ReactionOnly,
    DiffusionOnly,
    UptakeOnly,
    DampingOnly,
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
    
    // Debug scenario: Modify parameters to produce obvious changes
    let mut debug_rd_params = rd_params;
    let mut debug_agent_params = agent_params;
    
    if cli.debug_scenario || cli.scenario.is_some() {
        println!("DEBUG SCENARIO: Using modified parameters for testing");
        
        match cli.scenario.as_ref() {
            Some(Scenario::ReactionOnly) => {
                println!("SCENARIO: Reaction-only (σ>0, λ=0, D=0) → mean R ↑");
                // Test 1: Pure reaction (no diffusion, no uptake) - R should increase
                debug_rd_params.D_R = 0.0;
                debug_rd_params.D_W = 0.0;
                debug_rd_params.lambda_R = 0.0;
                debug_rd_params.lambda_W = 0.0;
                debug_rd_params.alpha_H = 0.0; // No herbivore uptake
                debug_rd_params.sigma_R = 0.02; // High replenishment
            }
            Some(Scenario::DiffusionOnly) => {
                println!("SCENARIO: Diffusion-only (D>0, σ=λ=0) → max↓, min↑, mean steady");
                // Test 2: Pure diffusion (no reaction, no uptake) - max↓, min↑, mean steady
                debug_rd_params.sigma_R = 0.0;
                debug_rd_params.lambda_R = 0.0;
                debug_rd_params.alpha_H = 0.0; // No herbivore uptake
                debug_rd_params.D_R = 1.0; // High diffusion
            }
            Some(Scenario::UptakeOnly) => {
                println!("SCENARIO: Uptake-only (σ=0, H>0) → mean R ↓");
                // Test 3: Pure uptake (no replenishment, herbivores consume) - mean R ↓
                debug_rd_params.sigma_R = 0.0; // No replenishment
                debug_rd_params.alpha_H = 0.2; // High herbivore uptake
                debug_rd_params.D_R = 0.0; // No diffusion
            }
            Some(Scenario::DampingOnly) => {
                println!("SCENARIO: Damping-only (χ=0, γ>0) → mean v ↓");
                // Test 4: Pure damping (no chemotaxis) - velocity should decay
                debug_agent_params.chi_R = 0.0;
                debug_agent_params.chi_W = 0.0;
                debug_agent_params.gamma = 0.2; // High damping
            }
            None => {
                // Default debug scenario (original logic)
                // Test 1: Pure reaction (no diffusion, no uptake) - R should increase
                debug_rd_params.D_R = 0.0;
                debug_rd_params.D_W = 0.0;
                debug_rd_params.lambda_R = 0.0;
                debug_rd_params.lambda_W = 0.0;
                debug_rd_params.alpha_H = 0.0; // No herbivore uptake
                debug_rd_params.sigma_R = 0.02; // High replenishment
                
                // Test 2: Pure damping (no chemotaxis) - velocity should decay
                debug_agent_params.chi_R = 0.0;
                debug_agent_params.chi_W = 0.0;
                debug_agent_params.gamma = 0.2; // High damping
            }
        }
        
        println!("DEBUG RD params: D_R={} sigma_R={} lambda_R={} alpha_H={}", 
            debug_rd_params.D_R, debug_rd_params.sigma_R, debug_rd_params.lambda_R, debug_rd_params.alpha_H);
        println!("DEBUG Agent params: chi_R={} gamma={}", debug_agent_params.chi_R, debug_agent_params.gamma);
    }
    
    let rd_params_buffer = gpu.create_rd_params_buffer(&debug_rd_params);
    let agent_params_buffer = gpu.create_agent_params_buffer(&debug_agent_params);
    let agents_buffer = gpu.create_agents_buffer(&agent_manager.agents);
    let occupancy_buffer = gpu.create_occupancy_buffer(config.world.size);
    
    // Log initial parameters for debugging
    println!("RD params: D_R={} D_W={} sigma_R={} alpha_H={} beta_H={} lambda_R={} lambda_W={} dt={}",
        rd_params.D_R, rd_params.D_W, rd_params.sigma_R, rd_params.alpha_H, rd_params.beta_H, rd_params.lambda_R, rd_params.lambda_W, rd_params.dt);
    println!("Agent params: chi_R={} chi_W={} gamma={} eps0={} eta_R={} dt={}",
        agent_params.chi_R, agent_params.chi_W, agent_params.gamma, agent_params.eps0, agent_params.eta_R, agent_params.dt);
    
    // Create bind groups for both ping-pong states
    let rd_bg_a = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
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
        label: Some("rd_bg_a"),
    });
    
    let rd_bg_b = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
        layout: &compute_pipelines.rd_bgl,
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::TextureView(&field_textures.view_b_sample),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: wgpu::BindingResource::TextureView(&field_textures.view_a_store),
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
        label: Some("rd_bg_b"),
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
        
        // Update uniform buffers each frame with current parameters
        gpu.queue.write_buffer(&rd_params_buffer, 0, bytemuck::bytes_of(&debug_rd_params));
        gpu.queue.write_buffer(&agent_params_buffer, 0, bytemuck::bytes_of(&debug_agent_params));
        
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
        
        // Save occupancy PNG at specific steps
        if step == 0 || step == 200 || step == 1000 || step == 2000 {
            // Read back occupancy buffer for PNG dump
            let staging_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("occupancy_png_staging"),
                size: (config.world.size[0] * config.world.size[1] * 4) as u64,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            
            let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("occupancy_png_copy"),
            });
            encoder.copy_buffer_to_buffer(&occupancy_buffer, 0, &staging_buffer, 0, (config.world.size[0] * config.world.size[1] * 4) as u64);
            gpu.submit(encoder.finish());
            
            staging_buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            gpu.device.poll(wgpu::Maintain::Wait);
            
            let data = staging_buffer.slice(..).get_mapped_range();
            let occupancy_data: Vec<u32> = data
                .chunks_exact(4)
                .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .collect();
            
            drop(data);
            staging_buffer.unmap();
            
            // Save occupancy PNG
            let png_path = cli.out.join(format!("occupancy_{:04}.png", step));
            if let Err(e) = snapshots::save_occupancy_png(&occupancy_data, config.world.size, &png_path) {
                eprintln!("Warning: Failed to save occupancy PNG: {}", e);
            } else {
                println!("Saved occupancy PNG: {}", png_path.display());
            }
        }
        
        // Debug: Check occupancy after agent pass (every 100 steps)
        if cli.debug_scenario && step % 100 == 0 {
            // Read back a small portion of the occupancy buffer to verify it's working
            let staging_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("occupancy_debug"),
                size: 1024, // Read first 256 u32s
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            
            let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("occupancy_debug_copy"),
            });
            encoder.copy_buffer_to_buffer(&occupancy_buffer, 0, &staging_buffer, 0, 1024);
            gpu.submit(encoder.finish());
            
            staging_buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            gpu.device.poll(wgpu::Maintain::Wait);
            
            let data = staging_buffer.slice(..).get_mapped_range();
            let occupancy_sample: Vec<u32> = data.chunks_exact(4).map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]])).collect();
            drop(data);
            staging_buffer.unmap();
            
            let total_occupancy: u32 = occupancy_sample.iter().sum();
            println!("Step {}: Total occupancy after agents: {}", step, total_occupancy);
        }
        
        // RD pass uses occupancy - select correct bind group based on ping-pong state
        let current_rd_bg = if use_a_as_src { &rd_bg_a } else { &rd_bg_b };
        
        if cli.debug_scenario && step % 100 == 0 {
            println!("Step {}: RD dispatch - groups=({}, {}), ping_pong={}", 
                step, 
                (config.world.size[0] + 7) / 8, 
                (config.world.size[1] + 7) / 8,
                if use_a_as_src { "A->B" } else { "B->A" }
            );
        }
        
        {
            let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("rd_pass"),
            });
            
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("rd pass"),
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&compute_pipelines.rd_pipeline);
                cpass.set_bind_group(0, current_rd_bg, &[]);
                
                let gx = (config.world.size[0] + 7) / 8;
                let gy = (config.world.size[1] + 7) / 8;
                cpass.dispatch_workgroups(gx, gy, 1);
            } // cpass is dropped here
            
            gpu.submit(encoder.finish());
        }
        
        // Flip ping-pong
        use_a_as_src = !use_a_as_src;
        
        // Debug: Check if field is actually changing (every 100 steps)
        if cli.debug_scenario && step % 100 == 0 {
            // Read back a single pixel from the current front texture to verify changes
            let current_texture = if use_a_as_src { &field_textures.texture_a } else { &field_textures.texture_b };
            
            let staging_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("pixel_debug"),
                size: 8, // Single pixel (4 channels × 2 bytes f16)
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });
            
            let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("pixel_debug_copy"),
            });
            
            encoder.copy_texture_to_buffer(
                wgpu::ImageCopyTexture {
                    texture: current_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                wgpu::ImageCopyBuffer {
                    buffer: &staging_buffer,
                    layout: wgpu::ImageDataLayout {
                        offset: 0,
                        bytes_per_row: Some(((8 + 255) / 256) * 256), // Align to 256-byte boundary
                        rows_per_image: Some(1),
                    },
                },
                wgpu::Extent3d {
                    width: 1,
                    height: 1,
                    depth_or_array_layers: 1,
                },
            );
            
            gpu.submit(encoder.finish());
            
            staging_buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
            gpu.device.poll(wgpu::Maintain::Wait);
            
            let data = staging_buffer.slice(..).get_mapped_range();
            let r_bytes = [data[0], data[1]];
            let r_value = half::f16::from_le_bytes(r_bytes).to_f32();
            drop(data);
            staging_buffer.unmap();
            
            println!("Step {}: Pixel (0,0) R value: {:.6}", step, r_value);
        }
        
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
