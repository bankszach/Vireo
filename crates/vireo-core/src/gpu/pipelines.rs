use wgpu::{Device, BindGroupLayout, ComputePipeline};
use crate::shaders;

/// Compute pipelines for the simulation
pub struct ComputePipelines {
    pub rd_pipeline: ComputePipeline,
    pub rd_bgl: BindGroupLayout,
    
    pub agent_pipeline: ComputePipeline,
    pub agent_bgl: BindGroupLayout,
}

impl ComputePipelines {
    /// Create all compute pipelines
    pub fn new(device: &Device) -> Self {
        let (rd_pipeline, rd_bgl) = Self::create_rd_pipeline(device);
        let (agent_pipeline, agent_bgl) = Self::create_agent_pipeline(device);
        
        Self {
            rd_pipeline,
            rd_bgl,
            agent_pipeline,
            agent_bgl,
        }
    }
    
    /// Create the reaction-diffusion compute pipeline
    fn create_rd_pipeline(device: &Device) -> (ComputePipeline, BindGroupLayout) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rd_shader"),
            source: wgpu::ShaderSource::Wgsl(shaders::rd_step().into()),
        });
        
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rd_bgl"),
            entries: &[
                // Source texture (R, W channels) - sampled for reading
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
                // Destination texture
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
                // Parameters
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
                // Herbivore density (occupancy)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rd_pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rd_pipeline"),
            layout: Some(&pl),
            module: &shader,
            entry_point: "main",
        });
        
        (pipeline, bgl)
    }
    
    /// Create the agent chemotaxis compute pipeline
    fn create_agent_pipeline(device: &Device) -> (ComputePipeline, BindGroupLayout) {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("agent_shader"),
            source: wgpu::ShaderSource::Wgsl(shaders::agent_step().into()),
        });
        
        let bgl = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("agent_bgl"),
            entries: &[
                // Agents storage buffer
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
                // Field texture (R, W channels)
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                // Parameters
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
                // Herbivore occupancy (write)
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("agent_pl"),
            bind_group_layouts: &[&bgl],
            push_constant_ranges: &[],
        });
        
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("agent_pipeline"),
            layout: Some(&pl),
            module: &shader,
            entry_point: "main",
        });
        
        (pipeline, bgl)
    }
}
