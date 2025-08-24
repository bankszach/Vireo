use wgpu::{Device, ComputePipeline};
use crate::gpu::layouts::Layouts;

/// Compute pipelines for the simulation
pub struct ComputePipelines {
    pub rd_pipeline: ComputePipeline,
    pub agent_pipeline: ComputePipeline,
    pub clear_occupancy_pipeline: ComputePipeline,
}

impl ComputePipelines {
    /// Create all compute pipelines using centralized layouts
    pub fn new(device: &Device, layouts: &Layouts) -> Self {
        let rd_pipeline = Self::create_rd_pipeline(device, &layouts.rd);
        let agent_pipeline = Self::create_agent_pipeline(device, &layouts.agent);
        let clear_occupancy_pipeline = Self::create_clear_occupancy_pipeline(device, &layouts.clear_occupancy);
        
        Self {
            rd_pipeline,
            agent_pipeline,
            clear_occupancy_pipeline,
        }
    }
    
    /// Create the reaction-diffusion compute pipeline
    fn create_rd_pipeline(device: &Device, rd_layout: &wgpu::BindGroupLayout) -> ComputePipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("rd_shader"),
            source: wgpu::ShaderSource::Wgsl(crate::shaders::rd_step().into()),
        });
        
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("rd_pl"),
            bind_group_layouts: &[rd_layout],
            push_constant_ranges: &[],
        });
        
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("rd_pipeline"),
            layout: Some(&pl),
            module: &shader,
            entry_point: "main",
        })
    }
    
    /// Create the agent chemotaxis compute pipeline
    fn create_agent_pipeline(device: &Device, agent_layout: &wgpu::BindGroupLayout) -> ComputePipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("agent_shader"),
            source: wgpu::ShaderSource::Wgsl(crate::shaders::agent_step().into()),
        });
        
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("agent_pl"),
            bind_group_layouts: &[agent_layout],
            push_constant_ranges: &[],
        });
        
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("agent_pipeline"),
            layout: Some(&pl),
            module: &shader,
            entry_point: "main",
        })
    }
    
    /// Create the clear occupancy compute pipeline
    fn create_clear_occupancy_pipeline(device: &Device, clear_layout: &wgpu::BindGroupLayout) -> ComputePipeline {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("clear_occupancy_shader"),
            source: wgpu::ShaderSource::Wgsl(crate::shaders::clear_occupancy().into()),
        });
        
        let pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("clear_occupancy_pl"),
            bind_group_layouts: &[clear_layout],
            push_constant_ranges: &[],
        });
        
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("clear_occupancy_pipeline"),
            layout: Some(&pl),
            module: &shader,
            entry_point: "main",
        })
    }
}
