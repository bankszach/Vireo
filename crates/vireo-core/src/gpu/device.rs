use wgpu::{Adapter, Device, Instance, Queue, RequestAdapterOptions};
use crate::{RDParams, AgentParams};
use crate::sim::Agent;
use wgpu::util::DeviceExt;
use bytemuck;

/// GPU device manager for headless compute operations
pub struct GpuDevice {
    pub instance: Instance,
    pub adapter: Adapter,
    pub device: Device,
    pub queue: Queue,
}

impl GpuDevice {
    /// Create a new GPU device for headless compute
    pub async fn new() -> Self {
        let instance = Instance::default();
        
        let adapter = instance
            .request_adapter(&RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None, // Headless, no surface needed
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
        
        Self {
            instance,
            adapter,
            device,
            queue,
        }
    }
    
    /// Get device info for logging
    pub fn info(&self) -> String {
        let info = self.adapter.get_info();
        format!(
            "GPU: {} ({:?}), Features: {:?}",
            info.name,
            info.backend,
            self.device.features()
        )
    }
    
    /// Create a buffer with initial data
    pub fn create_buffer_with_data<T: bytemuck::Pod>(
        &self,
        label: &str,
        usage: wgpu::BufferUsages,
        data: &[T],
    ) -> wgpu::Buffer {
        self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(label),
            contents: bytemuck::cast_slice(data),
            usage,
        })
    }
    
    /// Create a uniform buffer for parameters
    pub fn create_rd_params_buffer(&self, params: &RDParams) -> wgpu::Buffer {
        self.create_buffer_with_data(
            "rd_params",
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            &[*params],
        )
    }
    
    /// Create a uniform buffer for agent parameters
    pub fn create_agent_params_buffer(&self, params: &AgentParams) -> wgpu::Buffer {
        self.create_buffer_with_data(
            "agent_params",
            wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            &[*params],
        )
    }
    
    /// Create a storage buffer for agents
    pub fn create_agents_buffer(&self, agents: &[Agent]) -> wgpu::Buffer {
        self.create_buffer_with_data(
            "agents",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            agents,
        )
    }
    
    /// Create a storage buffer for herbivore occupancy
    pub fn create_occupancy_buffer(&self, size: [u32; 2]) -> wgpu::Buffer {
        let occupancy_data = vec![0u32; (size[0] * size[1]) as usize];
        self.create_buffer_with_data(
            "herbivore_occupancy",
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
            &occupancy_data,
        )
    }
    
    /// Clear the occupancy buffer to zero (called each frame before agent update)
    pub fn clear_occupancy_buffer(&self, buffer: &wgpu::Buffer, size: [u32; 2]) {
        let zero_data = vec![0u32; (size[0] * size[1]) as usize];
        self.queue.write_buffer(buffer, 0, bytemuck::cast_slice(&zero_data));
    }
    
    /// Submit commands to the GPU
    pub fn submit(&self, commands: wgpu::CommandBuffer) {
        self.queue.submit(Some(commands));
    }
    
    /// Wait for GPU operations to complete
    pub fn wait(&self) {
        self.queue.on_submitted_work_done(|| {});
        self.device.poll(wgpu::Maintain::Wait);
    }
}
