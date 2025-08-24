
use std::time::Instant;
use std::cmp::min;
use std::f32;

use bytemuck::{Pod, Zeroable};
use glam::{Vec2, vec2};
use rand::Rng;
use wgpu::util::DeviceExt;
use winit::{
    dpi::PhysicalSize,
    event::{Event, WindowEvent, ElementState, KeyEvent},
    keyboard::Key,
    event_loop::{EventLoop},
    window::WindowBuilder,
};

// ------------------------ Config ------------------------

const DEFAULT_GRID_W: u32 = 1024;
const DEFAULT_GRID_H: u32 = 576;
const DEFAULT_PARTICLES: u32 = 20_000; // try 50_000 on stronger GPUs
const GROUP_SIZE: u32 = 12;            // particles per ring-spring group
const WORKGROUP_2D: (u32, u32) = (16, 16);
const WORKGROUP_1D: u32 = 256;


// ------------------------ GPU Data ------------------------

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct SimParams {
    dt: f32,
    world_w: f32,
    world_h: f32,
    _pad0: f32,
    grid_w: u32,
    grid_h: u32,
    group_size: u32,
    paused: u32,
    time: f32,
    diffusion: f32,
    decay: f32,
    _pad1: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug, Default)]
struct Particle {
    // 32 bytes aligned
    pos: [f32; 2],
    vel: [f32; 2],
    energy: f32,
    kind: u32,   // 0 = plant, 1 = herbivore, 2 = predator
    age: f32,    // Age in seconds
    reproduction_cooldown: f32, // Time until can reproduce again
    state_flags: u32,  // Visual state indicators
}

impl Particle {
    fn new(pos: Vec2, kind: u32) -> Self {
        Self {
            pos: [pos.x, pos.y],
            vel: [0.0, 0.0],
            energy: 1.0,
            kind,
            age: 0.0,
            reproduction_cooldown: 0.0,
            state_flags: 0,
        }
    }
}

// ------------------------ App ------------------------

struct Pipelines {
    diffuse_pipeline: wgpu::ComputePipeline,
    diffuse_bgl: wgpu::BindGroupLayout,

    particle_pipeline: wgpu::ComputePipeline,
    particle_bgl: wgpu::BindGroupLayout,

    render_pipeline: wgpu::RenderPipeline,
    render_bgl: wgpu::BindGroupLayout,
}

struct Gfx<'a> {
    surface: wgpu::Surface<'a>,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    size: PhysicalSize<u32>,

    // Field ping-pong
    field_a: wgpu::Texture,
    field_b: wgpu::Texture,
    field_a_view_sample: wgpu::TextureView,
    field_a_view_store: wgpu::TextureView,
    field_b_view_sample: wgpu::TextureView,
    field_b_view_store: wgpu::TextureView,

    // Particles
    particle_buf: wgpu::Buffer,
    particle_count: u32,

    // Uniforms
    params: SimParams,
    params_buf: wgpu::Buffer,
    params_bg: wgpu::BindGroup,

    // Bind groups that depend on textures/bufs
    diffuse_bg_a2b: wgpu::BindGroup,
    diffuse_bg_b2a: wgpu::BindGroup,
    particle_bg_read_a: wgpu::BindGroup,
    particle_bg_read_b: wgpu::BindGroup,
    render_bg: wgpu::BindGroup,

    pipelines: Pipelines,
    use_a_as_src: bool,
}

impl Gfx<'_> {
    async fn new(window: &winit::window::Window, grid_w: u32, grid_h: u32, particle_count: u32) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::default();
        let surface = unsafe { instance.create_surface_unsafe(wgpu::SurfaceTargetUnsafe::from_window(window).unwrap()) }.expect("surface");
        let adapter = instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface: Some(&surface),
            force_fallback_adapter: false,
        }).await.expect("No adapter");

        let (device, queue) = adapter.request_device(
            &wgpu::DeviceDescriptor {
                required_features: wgpu::Features::PUSH_CONSTANTS,
                required_limits: wgpu::Limits {
                    max_push_constant_size: 128,
                    ..wgpu::Limits::default()
                },
                label: None,
            },
            None
        ).await.expect("device");

        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps.formats.iter().copied()
            .find(|f| f.is_srgb()).unwrap_or(surface_caps.formats[0]);

        // Create sampler for texture sampling
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("field sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);

        // --- Create field ping-pong textures (RGBA16F for filtering support) ---
        let field_format = wgpu::TextureFormat::Rgba16Float;
        let usage = wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC;
        let extent = wgpu::Extent3d { width: grid_w, height: grid_h, depth_or_array_layers: 1 };
        let field_a = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("field_a"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: field_format,
            usage,
            view_formats: &[],
        });
        let field_b = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("field_b"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: field_format,
            usage,
            view_formats: &[],
        });

        let field_a_view_sample = field_a.create_view(&wgpu::TextureViewDescriptor::default());
        let field_b_view_sample = field_b.create_view(&wgpu::TextureViewDescriptor::default());
        let field_a_view_store = field_a.create_view(&wgpu::TextureViewDescriptor {
            label: Some("field_a_store"),
            format: Some(field_format),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(1),
        });
        let field_b_view_store = field_b.create_view(&wgpu::TextureViewDescriptor {
            label: Some("field_b_store"),
            format: Some(field_format),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(1),
        });

        
        // --- Particles buffer ---
        let mut rng = rand::thread_rng();
        let world_w = grid_w as f32;
        let world_h = grid_h as f32;
        let mut particles = Vec::with_capacity(particle_count as usize);
        
        // Debug counters for particle types
        let mut plant_count = 0;
        let mut herbivore_count = 0;
        let mut predator_count = 0;
        
        // Calculate target counts for each type
        let target_plants = particle_count / 15;  // ~6.7%
        let target_predators = particle_count / 15;  // ~6.7%
        let target_herbivores = particle_count - target_plants - target_predators;  // ~86.6%
        
        // Spawn plants first - distribute them more evenly
        for i in 0..target_plants {
            let x = rng.gen_range(50.0..(world_w - 50.0));
            let y = rng.gen_range(50.0..(world_h - 50.0));
            let particle = Particle::new(vec2(x, y), 0);
            particles.push(particle);
            plant_count += 1;
        }
        
        // Spawn herbivores - some clustering but not too much
        for i in 0..target_herbivores {
            let x = if i < target_herbivores / 2 {
                // First half: left side with some clustering
                rng.gen_range(100.0..(world_w * 0.4))
            } else {
                // Second half: right side with some clustering
                rng.gen_range((world_w * 0.6)..(world_w - 100.0))
            };
            let y = rng.gen_range(100.0..(world_h - 100.0));
            let mut particle = Particle::new(vec2(x, y), 1);
            particle.vel = [rng.gen_range(-0.5..0.5), rng.gen_range(-0.5..0.5)];
            particle.age = rng.gen_range(0.0..5.0);
            particle.reproduction_cooldown = rng.gen_range(0.0..10.0);
            particles.push(particle);
            herbivore_count += 1;
        }
        
        // Spawn predators - more spread out to avoid clustering
        for i in 0..target_predators {
            let x = rng.gen_range(150.0..(world_w - 150.0));
            let y = rng.gen_range(150.0..(world_h - 150.0));
            let mut particle = Particle::new(vec2(x, y), 2);
            particle.vel = [rng.gen_range(-1.0..1.0), rng.gen_range(-1.0..1.0)];
            particle.age = rng.gen_range(0.0..3.0);
            particle.reproduction_cooldown = rng.gen_range(0.0..15.0);
            particles.push(particle);
            predator_count += 1;
        }
        
        // Debug output for particle distribution
        println!("Particle distribution:");
        println!("  Plants (Green): {} ({:.1}%)", plant_count, (plant_count as f32 / particle_count as f32) * 100.0);
        println!("  Herbivores (Blue): {} ({:.1}%)", herbivore_count, (herbivore_count as f32 / particle_count as f32) * 100.0);
        println!("  Predators (Red): {} ({:.1}%)", predator_count, (predator_count as f32 / particle_count as f32) * 100.0);
        println!("  Total particles: {}", particle_count);

        let particle_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("particles"),
            contents: bytemuck::cast_slice(&particles),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });

        // --- Params ---
        let params = SimParams {
            dt: 1.0/60.0,
            world_w, world_h, _pad0: 0.0,
            grid_w, grid_h,
            group_size: GROUP_SIZE,
            paused: 0,
            time: 0.0,
            diffusion: 0.03,  // Further reduced from 0.08
            decay: 0.002,      // Further increased from 0.001
            _pad1: 0.0,
        };
        let params_buf = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("params"),
            contents: bytemuck::bytes_of(&params),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // --- Pipelines ---
        let pipelines = create_pipelines(&device, surface_format);

        // --- Bind groups ---
        // Common params bindgroup (render + compute share layout)
        let render_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &pipelines.render_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::Buffer(params_buf.as_entire_buffer_binding()) },
                wgpu::BindGroupEntry { binding: 1, resource: particle_buf.as_entire_binding() },
            ],
            label: Some("render_bg"),
        });
        
        // Debug: Verify particle buffer binding
        println!("Particle buffer created with {} bytes", particle_buf.size());
        println!("First few particles for verification:");
        for i in 0..min(5, particles.len()) {
            let p = &particles[i];
            println!("  Particle {}: pos=({:.1}, {:.1}), kind={}, energy={:.1}", 
                    i, p.pos[0], p.pos[1], p.kind, p.energy);
        }
        
        let render_bg_clone = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &pipelines.render_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::Buffer(params_buf.as_entire_buffer_binding()) },
                wgpu::BindGroupEntry { binding: 1, resource: particle_buf.as_entire_binding() },
            ],
            label: Some("render_bg_clone"),
        });

        // Diffuse A->B
        let diffuse_bg_a2b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &pipelines.diffuse_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&field_a_view_sample) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&field_b_view_store) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Buffer(params_buf.as_entire_buffer_binding()) },
            ],
            label: Some("diffuse_bg_a2b"),
        });
        // Diffuse B->A
        let diffuse_bg_b2a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &pipelines.diffuse_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&field_b_view_sample) },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&field_a_view_store) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Buffer(params_buf.as_entire_buffer_binding()) },
            ],
            label: Some("diffuse_bg_b2a"),
        });

        // Particles read field A
        let particle_bg_read_a = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &pipelines.particle_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: particle_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&field_a_view_sample) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Buffer(params_buf.as_entire_buffer_binding()) },
            ],
            label: Some("particle_bg_read_a"),
        });
        // Particles read field B
        let particle_bg_read_b = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &pipelines.particle_bgl,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: particle_buf.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(&field_b_view_sample) },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(&sampler) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Buffer(params_buf.as_entire_buffer_binding()) },
            ],
            label: Some("particle_bg_read_b"),
        });

        let mut gfx = Self {
            surface, adapter, device, queue, config, size,
            field_a, field_b,
            field_a_view_sample, field_a_view_store,
            field_b_view_sample, field_b_view_store,
            particle_buf,
            particle_count,
            params,
            params_buf,
            params_bg: render_bg_clone,
            diffuse_bg_a2b,
            diffuse_bg_b2a,
            particle_bg_read_a,
            particle_bg_read_b,
            render_bg,
            pipelines,
            use_a_as_src: true,
        };

        gfx.seed_field();
        gfx
    }

    fn seed_field(&mut self) {
        // seed channel 0 with gaussian blobs as "food/scent"
        let w = self.params.grid_w as usize;
        let h = self.params.grid_h as usize;
        let mut data = vec![0f32; w*h*4];
        let mut rng = rand::thread_rng();
        
        // Create more distributed food sources instead of heavy clustering
        let center_x = self.params.world_w * 0.5;
        let center_y = self.params.world_h * 0.5;
        
        // Primary food source at center (reduced intensity)
        let primary_amp = 0.8;  // Reduced from 1.2
        let primary_sigma = 80.0;  // Increased from 60.0 for wider distribution
        for y in 0..h {
            for x in 0..w {
                let dx = x as f32 - center_x;
                let dy = y as f32 - center_y;
                let r2 = (dx*dx + dy*dy) / (2.0*primary_sigma*primary_sigma);
                data[(y*w + x)*4 + 0] += primary_amp * (-r2).exp();
            }
        }
        
        // Create multiple food clusters across the world for better distribution
        let num_clusters = 8;  // Reduced from 25 for better distribution
        for i in 0..num_clusters {
            // Distribute clusters more evenly across the world
            let cluster_x = if i < 4 {
                // Left half of world
                rng.gen_range(100.0..(self.params.world_w * 0.4))
            } else {
                // Right half of world
                rng.gen_range((self.params.world_w * 0.6)..(self.params.world_w - 100.0))
            };
            
            let cluster_y = if i % 2 == 0 {
                // Upper half
                rng.gen_range(100.0..(self.params.world_h * 0.4))
            } else {
                // Lower half
                rng.gen_range((self.params.world_h * 0.6)..(self.params.world_h - 100.0))
            };
            
            let amp = rng.gen_range(0.3..0.7);  // Reduced amplitude
            let sigma = rng.gen_range(40.0..80.0);  // Varied sizes
            
            for y in 0..h {
                for x in 0..w {
                    let dx = x as f32 - cluster_x;
                    let dy = y as f32 - cluster_y;
                    let r2 = (dx*dx + dy*dy) / (2.0*sigma*sigma);
                    data[(y*w + x)*4 + 0] += amp * (-r2).exp();
                }
            }
        }
        
        // Add some random scattered food sources for natural variation
        for _ in 0..15 {
            let cx = rng.gen_range(50.0..(self.params.world_w - 50.0));
            let cy = rng.gen_range(50.0..(self.params.world_h - 50.0));
            let amp = rng.gen_range(0.2..0.5);
            let sigma = rng.gen_range(20.0..50.0);
            
            for y in 0..h {
                for x in 0..w {
                    let dx = x as f32 - cx;
                    let dy = y as f32 - cy;
                    let r2 = (dx*dx + dy*dy) / (2.0*sigma*sigma);
                    data[(y*w + x)*4 + 0] += amp * (-r2).exp();
                }
            }
        }
        
        // Add a very gentle gradient from center to edges (reduced intensity)
        for y in 0..h {
            for x in 0..w {
                let dx = x as f32 - center_x;
                let dy = y as f32 - center_y;
                let dist_to_center = (dx*dx + dy*dy).sqrt();
                let max_dist = (center_x*center_x + center_y*center_y).sqrt();
                let gradient_factor = 1.0 - (dist_to_center / max_dist);
                data[(y*w + x)*4 + 0] += gradient_factor * 0.05; // Reduced from 0.1
            }
        }
        
        // write into A (source)
        let layout = wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some((w*16) as u32), // 4 * f32
            rows_per_image: Some(h as u32),
        };
        let size = wgpu::Extent3d { width: self.params.grid_w, height: self.params.grid_h, depth_or_array_layers: 1 };
        self.queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.field_a,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            bytemuck::cast_slice(&data),
            layout,
            size
        );
    }

    fn resize(&mut self, new_size: PhysicalSize<u32>) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            self.surface.configure(&self.device, &self.config);
        }
    }

    fn update_params(&mut self, dt: f32) {
        if self.params.paused == 0 {
            self.params.time += dt;
        }
        self.params.dt = dt;
        self.queue.write_buffer(&self.params_buf, 0, bytemuck::bytes_of(&self.params));
        

    }
    
    fn frame(&mut self) -> Result<(), wgpu::SurfaceError> {
        let frame = self.surface.get_current_texture()?;
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder = self.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame encoder") });

        // --- Diffuse pass (ping-pong) ---
        {
            let (pipeline, bg) = if self.use_a_as_src {
                (&self.pipelines.diffuse_pipeline, &self.diffuse_bg_a2b)
            } else {
                (&self.pipelines.diffuse_pipeline, &self.diffuse_bg_b2a)
            };
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { 
                label: Some("diffuse pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(pipeline);
            cpass.set_bind_group(0, bg, &[]);
            let gx = (self.params.grid_w + WORKGROUP_2D.0 - 1) / WORKGROUP_2D.0;
            let gy = (self.params.grid_h + WORKGROUP_2D.1 - 1) / WORKGROUP_2D.1;
            cpass.dispatch_workgroups(gx, gy, 1);
        }

        // Flip source for next pass
        self.use_a_as_src = !self.use_a_as_src;

        // --- Particles pass (read the *current* source) ---
        {
            let bg = if self.use_a_as_src { &self.particle_bg_read_a } else { &self.particle_bg_read_b };
            let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor { 
                label: Some("particles pass"),
                timestamp_writes: None,
            });
            cpass.set_pipeline(&self.pipelines.particle_pipeline);
            cpass.set_bind_group(0, bg, &[]);
            let gx = (self.particle_count + WORKGROUP_1D - 1) / WORKGROUP_1D;
            cpass.dispatch_workgroups(gx, 1, 1);
        }

        // --- Render ---
        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("render pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.02, g: 0.02, b: 0.03, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            rpass.set_pipeline(&self.pipelines.render_pipeline);
            rpass.set_bind_group(0, &self.render_bg, &[]);
            rpass.draw(0..6, 0..self.particle_count);
        }

        self.queue.submit(Some(encoder.finish()));
        frame.present();
        Ok(())
    }
}

fn create_pipelines(device: &wgpu::Device, surface_format: wgpu::TextureFormat) -> Pipelines {
    // --- Diffuse pipeline ---
    let diffuse_src = include_str!("../shaders/diffuse.wgsl");
    let diffuse_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("diffuse shader"),
        source: wgpu::ShaderSource::Wgsl(diffuse_src.into()),
    });
    let diffuse_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("diffuse bgl"),
        entries: &[
            // src sampled texture
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: false },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            // dst storage texture
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: wgpu::TextureFormat::Rgba16Float,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            // params
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let diffuse_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("diffuse pl"),
        bind_group_layouts: &[&diffuse_bgl],
        push_constant_ranges: &[],
    });
    let diffuse_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("diffuse pipeline"),
        layout: Some(&diffuse_pl),
        module: &diffuse_mod,
        entry_point: "main",
    });

    // --- Particles pipeline ---
    let particles_src = include_str!("../shaders/particles.wgsl");
    let particles_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("particles shader"),
        source: wgpu::ShaderSource::Wgsl(particles_src.into()),
    });
    let particle_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("particle bgl"),
        entries: &[
            // storage buffer (read_write)
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // field texture (sampled)
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            // sampler
            wgpu::BindGroupLayoutEntry {
                binding: 2,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                count: None,
            },
            // params
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let particle_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("particle pl"),
        bind_group_layouts: &[&particle_bgl],
        push_constant_ranges: &[],
    });
    let particle_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("particle pipeline"),
        layout: Some(&particle_pl),
        module: &particles_mod,
        entry_point: "main",
    });

    // --- Render pipeline ---
    let render_src = include_str!("../shaders/render.wgsl");
    let render_mod = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("render shader"),
        source: wgpu::ShaderSource::Wgsl(render_src.into()),
    });
    let render_bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("render bgl"),
        entries: &[
            // params
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            // particles (read-only in vertex stage)
            wgpu::BindGroupLayoutEntry {
                binding: 1,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let render_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("render pl"),
        bind_group_layouts: &[&render_bgl],
        push_constant_ranges: &[],
    });
    let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("render pipeline"),
        layout: Some(&render_pl),
        vertex: wgpu::VertexState {
            module: &render_mod,
            entry_point: "vs_main",
            buffers: &[],
        },
        fragment: Some(wgpu::FragmentState {
            module: &render_mod,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format: surface_format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: None,
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: None,
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
    });

    Pipelines {
        diffuse_pipeline,
        diffuse_bgl,
        particle_pipeline,
        particle_bgl,
        render_pipeline,
        render_bgl,
    }
}

fn env_u32(key: &str, default: u32) -> u32 {
    std::env::var(key).ok().and_then(|s| s.parse::<u32>().ok()).unwrap_or(default)
}

fn main() {
    let grid_w = env_u32("VIREO_GRID_W", DEFAULT_GRID_W);
    let grid_h = env_u32("VIREO_GRID_H", DEFAULT_GRID_H);
    let particle_count = env_u32("VIREO_PARTICLES", DEFAULT_PARTICLES);

    env_logger::init();

    let event_loop = EventLoop::new().unwrap();
    let window = WindowBuilder::new()
        .with_title("Vireo — Ecosystem Sandbox")
        .with_inner_size(PhysicalSize::new(grid_w, grid_h))
        .build(&event_loop).unwrap();

    let mut state = pollster::block_on(Gfx::new(&window, grid_w, grid_h, particle_count));

    let mut last = Instant::now();
    event_loop.run(move |event, elwt| {
        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => elwt.exit(),
                WindowEvent::Resized(sz) => state.resize(sz),
                WindowEvent::KeyboardInput { event: KeyEvent { logical_key: code, state: ks, .. }, .. } => {
                    if ks == ElementState::Pressed {
                        match code {
                            Key::Character(s) if s == "Escape" => elwt.exit(),
                            Key::Character(s) if s == " " => { state.params.paused ^= 1; },
                            Key::Character(s) if s == "r" || s == "R" => { state.seed_field(); },
                            _ => {}
                        }
                    }
                }
                WindowEvent::RedrawRequested => {
                    match state.frame() {
                        Ok(()) => {}
                        Err(wgpu::SurfaceError::Lost) => state.resize(state.size),
                        Err(wgpu::SurfaceError::OutOfMemory) => elwt.exit(),
                        Err(e) => eprintln!("Surface error: {e:?}"),
                    }
                }
                _ => {}
            },
            Event::AboutToWait => {
                let now = Instant::now();
                let dt = (now - last).as_secs_f32().min(1.0/30.0);
                last = now;
                state.update_params(dt);
                // request redraw
                window.request_redraw();
            }

            _ => {}
        }
    }).unwrap();
}
