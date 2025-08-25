//! Renderer for the Vireo ecosystem simulation

use wgpu::{SurfaceConfiguration, CommandEncoder, TextureView};
use anyhow::Result;

use vireo_core::gpu::layouts::Layouts;

/// Simple renderer for displaying particles
pub struct Renderer {
    render_pipeline: wgpu::RenderPipeline,
}

impl Renderer {
    /// Create a new renderer using centralized layouts
    pub fn new(
        device: &wgpu::Device,
        config: &SurfaceConfiguration,
        layouts: &Layouts, // Use centralized layouts instead of FieldPingPong
    ) -> Result<Self> {
        // Create shader
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("particle_render_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/render.wgsl").into()),
        });

        // Use the centralized particle render layout
        let bind_group_layout = &layouts.particle_render;

        // Create pipeline layout
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("particle_pipeline_layout"),
            bind_group_layouts: &[bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create render pipeline
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("particle_render_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[], // No vertex buffers needed for instanced particles
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        Ok(Self {
            render_pipeline,
        })
    }
    
    /// Render the particles
    pub fn render(
        &self,
        device: &wgpu::Device,
        encoder: &mut CommandEncoder,
        view: &TextureView,
        sim_params_buffer: &wgpu::Buffer,
        particles_buffer: &wgpu::Buffer,
        particle_count: u32,
        render_layout: &wgpu::BindGroupLayout,
    ) -> Result<()> {
        // Debug: log render call
        println!("Renderer: Starting particle render pass");
        println!("Renderer: Particle count: {}", particle_count);
        
        // Create bind group for particle rendering BEFORE the render pass
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("particle_render_bind_group"),
            layout: render_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: sim_params_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: particles_buffer.as_entire_binding(),
                },
            ],
        });
        
        let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("particle_render_pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: 0.1,
                        g: 0.1,
                        b: 0.1,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: None,
            occlusion_query_set: None,
            timestamp_writes: None,
        });

        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &bind_group, &[]);
        
        // Draw particles: 6 vertices per quad, particle_count instances
        render_pass.draw(0..6, 0..particle_count);

        println!("Renderer: Particle render pass completed");
        Ok(())
    }
}
