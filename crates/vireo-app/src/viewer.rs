//! Interactive viewer for the Vireo ecosystem simulation

use std::sync::Arc;
use std::time::Instant;
use winit::{
    event::{Event, WindowEvent, ElementState, KeyEvent},
    event_loop::EventLoop,
    window::{Window, WindowBuilder},
    dpi::LogicalSize,
};
use wgpu::{Instance, Device, Queue, Surface, SurfaceConfiguration, RequestAdapterOptions, util::DeviceExt};
use anyhow::Result;
use bytemuck;

use vireo_params::SimulationConfig;
use vireo_core::{
    gpu::{FieldPingPong, ComputePipelines},
    gpu::layouts::Layouts,
    sim::{FieldManager, AgentManager},
    RDParams, AgentParams,
};

use crate::renderer::Renderer;

/// Central GPU context that owns all GPU resources
pub struct GpuContext {
    pub device: Device,
    pub queue: Queue,
    pub surface: Surface<'static>,
    pub config: SurfaceConfiguration,
}

/// Main viewer state
pub struct Viewer {
    window: Arc<Window>,
    
    // Centralized layouts
    layouts: Layouts,
    
    // Simulation state
    field_manager: FieldManager,
    agent_manager: AgentManager,
    field_textures: FieldPingPong,
    compute_pipelines: ComputePipelines,
    
    // GPU buffers
    rd_params_buffer: wgpu::Buffer,
    agent_params_buffer: wgpu::Buffer,
    agents_buffer: wgpu::Buffer,
    occupancy_buffer: wgpu::Buffer,
    
    // Simulation parameters
    sim_config: SimulationConfig,
    current_step: u32,
    last_frame_time: Instant,
    frame_count: u32,
    
    // Overlay state
    show_r_field: bool,
    show_w_field: bool,
    show_occupancy: bool,
    show_gradients: bool,
    scenario_mode: Option<String>,
}

impl Viewer {
    /// Create a new viewer instance
    pub fn new(
        window: Arc<Window>, 
        gpu: &GpuContext,
        sim_config: SimulationConfig,
    ) -> Result<Self> {
        // Create centralized layouts first
        let layouts = Layouts::new(&gpu.device);
        
        // Create simulation components
        let field_manager = FieldManager::new(sim_config.world.size);
        let agent_manager = AgentManager::new(
            sim_config.agents.herbivores,
            [sim_config.world.size[0] as f32, sim_config.world.size[1] as f32],
            sim_config.agents.E0,
            sim_config.world.seed,
        );
        
        // Seed the field
        let mut field_manager = field_manager;
        field_manager.seed_resources(sim_config.world.seed);
        
        // Create GPU resources using centralized layouts
        let compute_pipelines = ComputePipelines::new(&gpu.device, &layouts);
        
        // Create GPU buffers first (needed for FieldPingPong)
        let rd_params = RDParams::from(&sim_config);
        let agent_params = AgentParams::from(&sim_config);
        
        let rd_params_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("rd_params"),
            contents: bytemuck::cast_slice(&[rd_params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        
        let agent_params_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("agent_params"),
            contents: bytemuck::cast_slice(&[agent_params]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        
        let agents_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("agents_buffer"),
            contents: bytemuck::cast_slice(&agent_manager.agents),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        
        // Create occupancy buffer
        let occupancy_size = (sim_config.world.size[0] * sim_config.world.size[1]) as u64 * 4; // u32 per cell
        let occupancy_buffer = gpu.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("occupancy"),
            size: occupancy_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        
        // Create FieldPingPong with centralized layouts
        let field_textures = FieldPingPong::new(
            &gpu.device,
            sim_config.world.size,
            &layouts,
            &rd_params_buffer,
            &occupancy_buffer,
            &gpu.device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some("field_sampler"),
                address_mode_u: wgpu::AddressMode::ClampToEdge,
                address_mode_v: wgpu::AddressMode::ClampToEdge,
                address_mode_w: wgpu::AddressMode::ClampToEdge,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::FilterMode::Nearest,
                compare: None,
                ..Default::default()
            }),
        );
        
        // Upload initial data
        field_textures.upload_field_data(&gpu.queue, &field_manager);
        
        Ok(Self {
            window,
            layouts,
            field_manager,
            agent_manager,
            field_textures,
            compute_pipelines,
            rd_params_buffer,
            agent_params_buffer,
            agents_buffer,
            occupancy_buffer,
            sim_config,
            current_step: 0,
            last_frame_time: Instant::now(),
            frame_count: 0,
            show_r_field: true,
            show_w_field: false,
            show_occupancy: false,
            show_gradients: false,
            scenario_mode: None,
        })
    }
    
    /// Handle window resize
    pub fn resize(&mut self, gpu: &mut GpuContext, new_size: winit::dpi::PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            gpu.config.width = new_size.width;
            gpu.config.height = new_size.height;
            gpu.surface.configure(&gpu.device, &gpu.config);
            
            // Recreate field textures and bind groups using centralized layouts
            self.field_textures.recreate(
                &gpu.device,
                &self.layouts,
                &self.rd_params_buffer,
                &self.occupancy_buffer,
                &gpu.device.create_sampler(&wgpu::SamplerDescriptor {
                    label: Some("field_sampler"),
                    address_mode_u: wgpu::AddressMode::ClampToEdge,
                    address_mode_v: wgpu::AddressMode::ClampToEdge,
                    address_mode_w: wgpu::AddressMode::ClampToEdge,
                    mag_filter: wgpu::FilterMode::Linear,
                    min_filter: wgpu::FilterMode::Linear,
                    mipmap_filter: wgpu::FilterMode::Nearest,
                    compare: None,
                    ..Default::default()
                }),
            );
        }
    }
    
    /// Update the simulation state
    pub fn update(&mut self, gpu: &GpuContext) -> Result<()> {
        // Update uniform buffers every frame
        let rd_params = RDParams::from(&self.sim_config);
        let agent_params = AgentParams::from(&self.sim_config);
        
        gpu.queue.write_buffer(&self.rd_params_buffer, 0, bytemuck::cast_slice(&[rd_params]));
        gpu.queue.write_buffer(&self.agent_params_buffer, 0, bytemuck::cast_slice(&[agent_params]));
        
        // Run agent pass
        self.run_agent_pass(gpu)?;
        
        // Run RD pass
        self.run_rd_pass(gpu)?;
        
        // Clear occupancy buffer
        self.clear_occupancy_buffer(gpu)?;
        
        // Swap ping-pong buffers (this updates the centralized state)
        self.field_textures.swap();
        
        // Debug: log the swap
        if self.current_step % 10 == 0 {
            println!("Ping-pong swapped, front_is_a: {}", self.field_textures.front_is_a());
        }
        
        self.current_step += 1;
        Ok(())
    }
    
    /// Render the current frame
    pub fn render(&mut self, gpu: &GpuContext, renderer: &Renderer) -> Result<()> {
        let output = gpu.surface.get_current_texture()?;
        let view = output.texture.create_view(&wgpu::TextureViewDescriptor::default());
        
        // Debug: Check surface dimensions
        if self.frame_count % 60 == 0 {  // Every second at 60 FPS
            println!("Rendering frame {}: surface size {}x{}, texture size {}x{}", 
                self.frame_count, 
                gpu.config.width, 
                gpu.config.height,
                output.texture.size().width,
                output.texture.size().height);
        }
        
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("render_encoder"),
        });
        
        // Create SimParams buffer for this frame
        let sim_params = [
            self.sim_config.world.size[0] as f32,  // world_size.x
            self.sim_config.world.size[1] as f32,  // world_size.y
            self.current_step as f32 * 0.016,      // time: 60 FPS
            1.0,                                   // zoom: default zoom
            0.0,                                   // camera.x: centered
            0.0,                                   // camera.y: centered
            0.0,                                   // _pad0.x
            0.0,                                   // _pad0.y
        ];
        let sim_params_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("sim_params_frame"),
            contents: bytemuck::cast_slice(&sim_params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        
        // Render the particles
        renderer.render(&gpu.device, &mut encoder, &view, &sim_params_buffer, &self.agents_buffer, self.sim_config.agents.herbivores, &self.layouts.particle_render)?;
        
        gpu.queue.submit(Some(encoder.finish()));
        output.present();
        
        self.frame_count += 1;
        
        // Display HUD info every 30 frames (about once per second at 60 FPS)
        if self.frame_count % 30 == 0 {
            let (alive_agents, mean_r, mean_gradient, foraging_efficiency) = self.get_stats();
            println!("=== HUD (Step {}) ===", self.current_step);
            println!("Alive agents: {}", alive_agents);
            println!("Mean R: {:.3}", mean_r);
            println!("Mean |∇R|: {:.3}", mean_gradient);
            println!("Foraging efficiency: {:.3}", foraging_efficiency);
            println!("Overlays: R={}, W={}, Occ={}, ∇={}", 
                self.show_r_field, self.show_w_field, self.show_occupancy, self.show_gradients);
            if let Some(scenario) = &self.scenario_mode {
                println!("Scenario: {}", scenario);
            }
            println!("==================");
        }
        
        Ok(())
    }
    
    /// Clear the occupancy buffer
    fn clear_occupancy_buffer(&self, gpu: &GpuContext) -> Result<()> {
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("clear_occupancy_encoder"),
        });
        
        // Create a dimensions buffer for the clear occupancy shader
        let dimensions = [self.sim_config.world.size[0], self.sim_config.world.size[1]];
        let dimensions_buffer = gpu.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("clear_occupancy_dims"),
            contents: bytemuck::cast_slice(&dimensions),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });
        
        // Create bind group for clear occupancy pass using centralized layouts
        let clear_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("clear_occupancy_bind_group"),
            layout: &self.layouts.clear_occupancy,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.occupancy_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(dimensions_buffer.as_entire_buffer_binding()),
                },
            ],
        });
        
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("clear_occupancy_pass"),
            timestamp_writes: None,
        });
        
        compute_pass.set_pipeline(&self.compute_pipelines.clear_occupancy_pipeline);
        compute_pass.set_bind_group(0, &clear_bind_group, &[]);
        
        // Dispatch clear occupancy compute pass
        let total_cells = self.sim_config.world.size[0] * self.sim_config.world.size[1];
        let workgroup_size = 128;
        let workgroup_count = (total_cells + workgroup_size - 1) / workgroup_size;
        
        compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        drop(compute_pass);
        
        gpu.queue.submit(Some(encoder.finish()));
        Ok(())
    }
    
    /// Run the agent simulation pass
    fn run_agent_pass(&self, gpu: &GpuContext) -> Result<()> {
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("agent_encoder"),
        });
        
        // Create bind group for agent pass using centralized layouts
        let agent_bind_group = gpu.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("agent_bind_group"),
            layout: &self.layouts.agent,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.agents_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(self.field_textures.front_sample_view()),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: self.agent_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.occupancy_buffer.as_entire_binding(),
                },
            ],
        });
        
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("agent_pass"),
            timestamp_writes: None,
        });
        
        compute_pass.set_pipeline(&self.compute_pipelines.agent_pipeline);
        compute_pass.set_bind_group(0, &agent_bind_group, &[]);
        
        // Dispatch agent compute pass
        let agent_count = self.sim_config.agents.herbivores;
        let workgroup_size = 128;
        let workgroup_count = (agent_count + workgroup_size - 1) / workgroup_size;
        
        compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        drop(compute_pass);
        
        gpu.queue.submit(Some(encoder.finish()));
        
        // Debug: log agent pass completion
        if self.current_step % 10 == 0 {
            println!("Agent pass completed, workgroups: {}", workgroup_count);
        }
        
        Ok(())
    }
    
    /// Run the reaction-diffusion pass
    fn run_rd_pass(&self, gpu: &GpuContext) -> Result<()> {
        let mut encoder = gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("rd_encoder"),
        });
        
        // Use the centralized bind group from FieldPingPong
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("rd_pass"),
            timestamp_writes: None,
        });
        
        compute_pass.set_pipeline(&self.compute_pipelines.rd_pipeline);
        compute_pass.set_bind_group(0, self.field_textures.rd_bind_group(), &[]);
        
        // Dispatch RD compute pass
        let size = self.sim_config.world.size;
        let workgroup_size = 8;
        let workgroup_count_x = (size[0] + workgroup_size - 1) / workgroup_size;
        let workgroup_count_y = (size[1] + workgroup_size - 1) / workgroup_size;
        
        compute_pass.dispatch_workgroups(workgroup_count_x, workgroup_count_y, 1);
        drop(compute_pass);
        
        gpu.queue.submit(Some(encoder.finish()));
        
        // Debug: log RD pass completion
        if self.current_step % 10 == 0 {
            println!("RD pass completed, workgroups: {}x{}", workgroup_count_x, workgroup_count_y);
        }
        
        Ok(())
    }

    /// Handle key press for overlay toggles and scenario modes
    pub fn handle_key(&mut self, key: &winit::keyboard::Key) -> Result<()> {
        match key {
            winit::keyboard::Key::Character(c) if c == "1" => {
                self.show_r_field = !self.show_r_field;
                self.show_w_field = false;
                self.show_occupancy = false;
                self.show_gradients = false;
                log::info!("R field overlay: {}", self.show_r_field);
            }
            winit::keyboard::Key::Character(c) if c == "2" => {
                self.show_r_field = false;
                self.show_w_field = !self.show_w_field;
                self.show_occupancy = false;
                self.show_gradients = false;
                log::info!("W field overlay: {}", self.show_w_field);
            }
            winit::keyboard::Key::Character(c) if c == "3" => {
                self.show_r_field = false;
                self.show_w_field = false;
                self.show_occupancy = !self.show_occupancy;
                self.show_gradients = false;
                log::info!("Occupancy overlay: {}", self.show_occupancy);
            }
            winit::keyboard::Key::Character(c) if c == "g" || c == "G" => {
                self.show_r_field = false;
                self.show_w_field = false;
                self.show_occupancy = false;
                self.show_gradients = !self.show_gradients;
                log::info!("Gradient overlay: {}", self.show_gradients);
            }
            winit::keyboard::Key::Named(winit::keyboard::NamedKey::F1) => {
                self.scenario_mode = Some("baseline".to_string());
                log::info!("Scenario: Baseline (all systems enabled)");
            }
            winit::keyboard::Key::Named(winit::keyboard::NamedKey::F2) => {
                self.scenario_mode = Some("clumpy".to_string());
                log::info!("Scenario: Clumpy (high chemotaxis, low damping)");
            }
            winit::keyboard::Key::Named(winit::keyboard::NamedKey::F3) => {
                self.scenario_mode = Some("flat".to_string());
                log::info!("Scenario: Flat (low chemotaxis, high damping)");
            }
            _ => {}
        }
        Ok(())
    }

    /// Get current simulation statistics for HUD display
    pub fn get_stats(&self) -> (u32, f32, f32, f32) {
        let alive_agents = self.agent_manager.agents.iter()
            .filter(|a| a.alive == 1)
            .count() as u32;
        
        // For now, return placeholder values - we'll implement proper metrics later
        let mean_r = 0.5; // Placeholder
        let mean_gradient = 0.1; // Placeholder
        let foraging_efficiency = 0.8; // Placeholder
        
        (alive_agents, mean_r, mean_gradient, foraging_efficiency)
    }
}

/// Run the interactive viewer
pub async fn run_viewer(sim_config: SimulationConfig) -> Result<()> {
    println!("Creating event loop...");
    let event_loop = EventLoop::new()?;
    
    // Create window and wrap in Arc for proper ownership
    println!("Creating window...");
    let window = Arc::new(WindowBuilder::new()
        .with_title("Vireo Ecosystem Simulation")
        .with_inner_size(LogicalSize::new(1024.0, 768.0))
        .build(&event_loop)?);
    
    println!("Creating viewer...");
    let instance = Instance::default();
    let surface = instance.create_surface(window.clone()).unwrap();
    let adapter = instance
        .request_adapter(&RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        })
        .await
        .expect("Failed to find an appropriate adapter");
    
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                label: None,
            },
            None,
        )
        .await
        .expect("Failed to create device");
    
    let surface_caps = surface.get_capabilities(&adapter);
    let surface_format = surface_caps.formats.iter()
        .copied()
        .find(|f| f.is_srgb())
        .unwrap_or(surface_caps.formats[0]);
    
    let config = SurfaceConfiguration {
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        format: surface_format,
        width: window.inner_size().width,
        height: window.inner_size().height,
        present_mode: surface_caps.present_modes[0],
        alpha_mode: surface_caps.alpha_modes[0],
        view_formats: vec![],
        desired_maximum_frame_latency: 2,
    };
    surface.configure(&device, &config);

    let mut gpu = GpuContext {
        device,
        queue,
        surface,
        config,
    };

    // Check RGBA16Float filtering support for runtime fallback
    let format = wgpu::TextureFormat::Rgba16Float;
    let format_features = adapter.get_texture_format_features(format);
    // For now, assume filtering is supported - we can implement proper fallback later
    let supports_filtering = true;
    println!("RGBA16Float format features: {:?}", format_features);
    println!("RGBA16Float supports filtering: {} (assumed for testing)", supports_filtering);
    
    if !supports_filtering {
        println!("WARNING: RGBA16Float does not support filtering on this GPU. Consider implementing non-filtering fallback.");
    }

    let mut viewer = Viewer::new(window.clone(), &gpu, sim_config)?;
    let renderer = Renderer::new(&gpu.device, &gpu.config, &viewer.layouts)?;
    println!("Viewer created successfully!");
    
    // Request initial redraw to start the simulation
    window.request_redraw();
    
    println!("Starting event loop...");
    
    // Create a simple timer to ensure simulation runs
    let mut last_update = std::time::Instant::now();
    let target_fps = 60.0;
    let frame_duration = std::time::Duration::from_secs_f32(1.0 / target_fps);
    
    event_loop.run(move |event, elwt| {
        match event {
            Event::WindowEvent {
                ref event,
                window_id,
            } if window_id == viewer.window.id() => {
                match event {
                    WindowEvent::CloseRequested => {
                        println!("Window close requested");
                        elwt.exit();
                    }
                    WindowEvent::Resized(physical_size) => {
                        viewer.resize(&mut gpu, *physical_size);
                        // Request a redraw after resize
                        viewer.window.request_redraw();
                    }
                    WindowEvent::KeyboardInput {
                        event: KeyEvent {
                            logical_key: winit::keyboard::Key::Named(winit::keyboard::NamedKey::Escape),
                            state: ElementState::Pressed,
                            ..
                        },
                        ..
                    } => {
                        println!("Escape key pressed");
                        elwt.exit();
                    }
                    WindowEvent::KeyboardInput {
                        event: KeyEvent {
                            logical_key,
                            state: ElementState::Pressed,
                            ..
                        },
                        ..
                    } => {
                        if let Err(e) = viewer.handle_key(logical_key) {
                            log::error!("Key handling error: {}", e);
                        }
                    }
                    _ => {}
                }
            }
            Event::DeviceEvent {
                event: winit::event::DeviceEvent::MouseWheel { delta: _, .. },
                ..
            } => {
                // Handle mouse wheel for zoom
            }
            Event::AboutToWait => {
                // Check if it's time for the next frame
                let now = std::time::Instant::now();
                if now.duration_since(last_update) >= frame_duration {
                    last_update = now;
                    
                    // Directly update and render instead of requesting redraw
                    if let Err(e) = viewer.update(&gpu) {
                        log::error!("Simulation update error: {}", e);
                    }
                    
                    if let Err(e) = viewer.render(&gpu, &renderer) {
                        log::error!("Render error: {}", e);
                    }
                }
            }
            Event::WindowEvent {
                event: WindowEvent::RedrawRequested,
                ..
            } => {
                // Update simulation
                if let Err(e) = viewer.update(&gpu) {
                    log::error!("Simulation update error: {}", e);
                }
                
                // Render frame
                if let Err(e) = viewer.render(&gpu, &renderer) {
                    log::error!("Render error: {}", e);
                }
                
                // Request next redraw to keep the loop going
                viewer.window.request_redraw();
            }
            _ => {}
        }
    })?;
    
    Ok(())
}
