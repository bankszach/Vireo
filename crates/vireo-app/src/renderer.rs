//! Renderer for the Vireo ecosystem simulation

use wgpu::{SurfaceConfiguration, CommandEncoder, TextureView};
use anyhow::Result;

use vireo_core::gpu::layouts::Layouts;

/// Simple renderer for displaying particles
pub struct Renderer {
    render_pipeline: wgpu::RenderPipeline,
    field_bg_pipeline: wgpu::RenderPipeline,
}

impl Renderer {
    /// Create a new renderer using centralized layouts
    pub fn new(
        device: &wgpu::Device,
        config: &SurfaceConfiguration,
        layouts: &Layouts, // Use centralized layouts instead of FieldPingPong
    ) -> Result<Self> {
        // Create particle shader
        let particle_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("particle_render_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/render.wgsl").into()),
        });

        // Create field background shader
        let field_bg_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("field_bg_shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("../shaders/field_bg.wgsl").into()),
        });

        // Use the centralized particle render layout
        let particle_bind_group_layout = &layouts.particle_render;
        let field_bg_bind_group_layout = &layouts.field_render;

        // Create particle pipeline layout
        let particle_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("particle_pipeline_layout"),
            bind_group_layouts: &[particle_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create field background pipeline layout
        let field_bg_pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("field_bg_pipeline_layout"),
            bind_group_layouts: &[field_bg_bind_group_layout],
            push_constant_ranges: &[],
        });

        // Create particle render pipeline
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("particle_render_pipeline"),
            layout: Some(&particle_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &particle_shader,
                entry_point: "vs_main",
                buffers: &[], // No vertex buffers needed for instanced particles
            },
            fragment: Some(wgpu::FragmentState {
                module: &particle_shader,
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

        // Create field background render pipeline
        let field_bg_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("field_bg_pipeline"),
            layout: Some(&field_bg_pipeline_layout),
            vertex: wgpu::VertexState {
                module: &field_bg_shader,
                entry_point: "vs_main",
                buffers: &[], // No vertex buffers needed for fullscreen triangle
            },
            fragment: Some(wgpu::FragmentState {
                module: &field_bg_shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: config.format,
                    blend: None, // No blending for background
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None, // No culling for fullscreen triangle
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
            field_bg_pipeline,
        })
    }
    
    /// Render the field background and particles
    pub fn render(
        &self,
        device: &wgpu::Device,
        encoder: &mut CommandEncoder,
        view: &TextureView,
        sim_params_buffer: &wgpu::Buffer,
        particles_buffer: &wgpu::Buffer,
        particle_count: u32,
        render_layout: &wgpu::BindGroupLayout,
        field_bg_layout: &wgpu::BindGroupLayout,
        field_texture: &wgpu::TextureView,
        field_sampler: &wgpu::Sampler,
    ) -> Result<()> {
        // Debug: log render call
        println!("Renderer: Starting render pass");
        println!("Renderer: Particle count: {}", particle_count);
        
        // Create bind group for field background rendering
        let field_bg_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("field_bg_bind_group"),
            layout: field_bg_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(field_texture),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(field_sampler),
                },
            ],
        });

        // Create bind group for particle rendering
        let particle_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
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
            label: Some("render_pass"),
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

        // 1. Draw field background first
        render_pass.set_pipeline(&self.field_bg_pipeline);
        render_pass.set_bind_group(0, &field_bg_bind_group, &[]);
        render_pass.draw(0..3, 0..1); // Fullscreen triangle

        // 2. Draw particles on top
        render_pass.set_pipeline(&self.render_pipeline);
        render_pass.set_bind_group(0, &particle_bind_group, &[]);
        render_pass.draw(0..6, 0..particle_count); // 6 vertices per quad, particle_count instances

        println!("Renderer: Render pass completed");
        Ok(())
    }
}
