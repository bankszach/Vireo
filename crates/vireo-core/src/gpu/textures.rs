use wgpu::{Device, Queue, Texture, TextureView, TextureViewDescriptor, BindGroup};
use crate::sim::FieldManager;
use crate::gpu::layouts::Layouts;

/// Centralized ping-pong struct that owns textures, views, and bind groups
pub struct FieldPingPong {
    // textures + views
    tex_a: Texture,
    tex_b: Texture,
    view_a_sample: TextureView,   // sampled for reading
    view_b_sample: TextureView,
    view_a_store:  TextureView,   // storage for writing
    view_b_store:  TextureView,

    // pre-built bind groups (compute: A→B and B→A)
    rd_a2b_bg: BindGroup,
    rd_b2a_bg: BindGroup,

    // pre-built bind groups (render front = A or front = B)
    show_a_bg: BindGroup,
    show_b_bg: BindGroup,

    // the *single* source of truth
    front_is_a: bool,
    
    // grid size
    size: [u32; 2],
    

}

impl FieldPingPong {
    /// Create a new FieldPingPong with all textures, views, and bind groups
    pub fn new(
        device: &Device, 
        size: [u32; 2],
        layouts: &Layouts,
        rd_params_buffer: &wgpu::Buffer,
        occupancy_buffer: &wgpu::Buffer,
        sampler: &wgpu::Sampler,
    ) -> Self {
        let format = wgpu::TextureFormat::Rgba16Float;
        let usage = wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC;
        
        let extent = wgpu::Extent3d {
            width: size[0],
            height: size[1],
            depth_or_array_layers: 1,
        };
        
        let tex_a = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("field_a"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        });
        
        let tex_b = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("field_b"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        });
        
        let view_a_sample = tex_a.create_view(&TextureViewDescriptor {
            label: Some("field_a_sample"),
            format: Some(format),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(1),
        });
        let view_b_sample = tex_b.create_view(&TextureViewDescriptor {
            label: Some("field_b_sample"),
            format: Some(format),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(1),
        });
        
        let view_a_store = tex_a.create_view(&TextureViewDescriptor {
            label: Some("field_a_store"),
            format: Some(format),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(1),
        });
        
        let view_b_store = tex_b.create_view(&TextureViewDescriptor {
            label: Some("field_b_store"),
            format: Some(format),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(1),
        });

        // Create bind groups for RD compute (A→B and B→A) using borrowed layouts
        let rd_a2b_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rd_a2b_bg"),
            layout: &layouts.rd, // borrow the layout
            entries: &[
                // @binding(0) src A (sampled)
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view_a_sample),
                },
                // @binding(1) dst B (storage)
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view_b_store),
                },
                // @binding(2) RDParams uniform
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(rd_params_buffer.as_entire_buffer_binding()),
                },
                // @binding(3) occupancy buffer
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(occupancy_buffer.as_entire_buffer_binding()),
                },
            ],
        });

        let rd_b2a_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rd_b2a_bg"),
            layout: &layouts.rd, // borrow the layout
            entries: &[
                // @binding(0) src B (sampled)
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view_b_sample),
                },
                // @binding(1) dst A (storage)
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&view_a_store),
                },
                // @binding(2) RDParams uniform
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(rd_params_buffer.as_entire_buffer_binding()),
                },
                // @binding(3) occupancy buffer
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(occupancy_buffer.as_entire_buffer_binding()),
                },
            ],
        });

        // Create bind groups for rendering (show A and show B) using borrowed layouts
        let show_a_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("show_a_bg"),
            layout: &layouts.field_render, // borrow the field render layout
            entries: &[
                // @binding(0) field A texture
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view_a_sample),
                },
                // @binding(1) sampler
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });

        let show_b_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("show_b_bg"),
            layout: &layouts.field_render, // borrow the field render layout
            entries: &[
                // @binding(1) field B texture
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&view_b_sample),
                },
                // @binding(1) sampler
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });

        Self {
            tex_a,
            tex_b,
            view_a_sample,
            view_b_sample,
            view_a_store,
            view_b_store,
            rd_a2b_bg,
            rd_b2a_bg,
            show_a_bg,
            show_b_bg,
            front_is_a: true,
            size,

        }
    }

    /// Recreate textures and views (for resize), then rebuild bind groups using borrowed layouts
    pub fn recreate(
        &mut self,
        device: &Device,
        layouts: &Layouts,
        rd_params_buffer: &wgpu::Buffer,
        occupancy_buffer: &wgpu::Buffer,
        sampler: &wgpu::Sampler,
    ) {
        let format = wgpu::TextureFormat::Rgba16Float;
        let usage = wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC;
        
        let extent = wgpu::Extent3d {
            width: self.size[0],
            height: self.size[1],
            depth_or_array_layers: 1,
        };
        
        // Recreate textures
        self.tex_a = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("field_a"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        });
        
        self.tex_b = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("field_b"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        });
        
        // Recreate views
        self.view_a_sample = self.tex_a.create_view(&TextureViewDescriptor {
            label: Some("field_a_sample"),
            format: Some(format),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(1),
        });
        self.view_b_sample = self.tex_b.create_view(&TextureViewDescriptor {
            label: Some("field_b_sample"),
            format: Some(format),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(1),
        });
        
        self.view_a_store = self.tex_a.create_view(&TextureViewDescriptor {
            label: Some("field_a_store"),
            format: Some(format),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(1),
        });
        
        self.view_b_store = self.tex_b.create_view(&TextureViewDescriptor {
            label: Some("field_b_store"),
            format: Some(format),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(1),
        });

        // Rebuild bind groups using borrowed layouts
        self.rd_a2b_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rd_a2b_bg"),
            layout: &layouts.rd,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.view_a_sample),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.view_b_store),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(rd_params_buffer.as_entire_buffer_binding()),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(occupancy_buffer.as_entire_buffer_binding()),
                },
            ],
        });

        self.rd_b2a_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rd_b2a_bg"),
            layout: &layouts.rd,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.view_b_sample),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(&self.view_a_store),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(rd_params_buffer.as_entire_buffer_binding()),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(occupancy_buffer.as_entire_buffer_binding()),
                },
            ],
        });

        self.show_a_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("show_a_bg"),
            layout: &layouts.field_render,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.view_a_sample),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });

        self.show_b_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("show_b_bg"),
            layout: &layouts.field_render,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&self.view_b_sample),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        });
        

    }

    /// Get the RD bind group for the current frame (read from front, write to back)
    #[inline] 
    pub fn rd_bind_group(&self) -> &BindGroup {
        if self.front_is_a { &self.rd_a2b_bg } else { &self.rd_b2a_bg }
    }
    
    /// Get the render bind group for the current frame (show front texture)
    #[inline] 
    pub fn render_bind_group(&self) -> &BindGroup {
        let bind_group = if self.front_is_a { &self.show_a_bg } else { &self.show_b_bg };
        println!("FieldPingPong: render_bind_group called, front_is_a={}, returning {} bind group", 
            self.front_is_a, if self.front_is_a { "A" } else { "B" });
        bind_group
    }
    
    /// Get the front texture view for sampling (reading)
    #[inline] 
    pub fn front_sample_view(&self) -> &TextureView {
        if self.front_is_a { &self.view_a_sample } else { &self.view_b_sample }
    }
    
    /// Get the back texture view for storage (writing)
    #[inline] 
    pub fn back_storage_view(&self) -> &TextureView {
        if self.front_is_a { &self.view_b_store } else { &self.view_a_store }
    }
    
    /// Get the front ping-pong state
    #[inline] 
    pub fn front_is_a(&self) -> bool {
        self.front_is_a
    }
    
    /// Get the A sample view (for agent pass)
    #[inline] 
    pub fn a_sample_view(&self) -> &TextureView {
        &self.view_a_sample
    }
    
    /// Swap the ping-pong state (call this after RD pass, before render)
    #[inline] 
    pub fn swap(&mut self) { 
        self.front_is_a = !self.front_is_a; 
    }
    
    /// Get the grid size
    pub fn size(&self) -> [u32; 2] {
        self.size
    }
    

    
    /// Upload field data to texture A
    pub fn upload_field_data(&self, queue: &Queue, field_manager: &FieldManager) {
        println!("FieldPingPong: Starting texture upload");
        println!("FieldPingPong: Field size: {:?}", self.size);
        
        let data = field_manager.to_rgba16f();
        println!("FieldPingPong: Converted {} RGBA16F values", data.len());
        
        // Debug: check first few values
        if data.len() >= 4 {
            println!("FieldPingPong: First RGBA values: R={:.3}, W={:.3}, A3={:.3}, A4={:.3}", 
                data[0].to_f32(), data[1].to_f32(), data[2].to_f32(), data[3].to_f32());
        }
        
        // Convert f16 to bytes manually since bytemuck doesn't support half::f16
        let mut bytes = Vec::with_capacity(data.len() * 2);
        for &f in &data {
            bytes.extend_from_slice(&f.to_le_bytes());
        }
        
        // Calculate padded bytes per row (wgpu requires 256-byte alignment)
        let bytes_per_row = self.size[0] * 8; // 4 channels × 2 bytes (f16)
        let padded_bytes_per_row = ((bytes_per_row + 255) / 256) * 256;
        
        let mut padded_bytes = Vec::with_capacity((padded_bytes_per_row * self.size[1]) as usize);
        for row in 0..self.size[1] {
            let start = (row * bytes_per_row) as usize;
            let end = start + bytes_per_row as usize;
            padded_bytes.extend_from_slice(&bytes[start..end]);
            
            // Pad to alignment boundary
            let padding = padded_bytes_per_row - bytes_per_row;
            padded_bytes.extend(std::iter::repeat(0u8).take(padding as usize));
        }
        
        println!("FieldPingPong: Uploading {} bytes to texture", padded_bytes.len());
        
        let layout = wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(padded_bytes_per_row),
            rows_per_image: Some(self.size[1]),
        };
        
        let size = wgpu::Extent3d {
            width: self.size[0],
            height: self.size[1],
            depth_or_array_layers: 1,
        };
        
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.tex_a,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &padded_bytes,
            layout,
            size,
        );
        
        println!("FieldPingPong: Texture upload completed");
    }
    
    /// Download field data from the front texture
    pub fn download_field_data(&self, device: &Device, queue: &Queue, field_manager: &mut FieldManager) {
        // Create a staging buffer to read the texture
        let buffer_size = (self.size[0] * self.size[1] * 8) as u64; // 4 channels × 2 bytes (f16)
        let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("field_download_staging"),
            size: buffer_size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        
        // Copy texture to staging buffer
        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("field_download_encoder"),
        });
        
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: if self.front_is_a {
                    &self.tex_a
                } else {
                    &self.tex_b
                },
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &staging_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(self.size[0] * 8),
                    rows_per_image: Some(self.size[1]),
                },
            },
            wgpu::Extent3d {
                width: self.size[0],
                height: self.size[1],
                depth_or_array_layers: 1,
            },
        );
        
        queue.submit(Some(encoder.finish()));
        
        // Map the buffer and read the data
        staging_buffer.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        device.poll(wgpu::Maintain::Wait);
        
        let data = staging_buffer.slice(..).get_mapped_range();
        let half_data: Vec<half::f16> = data
            .chunks_exact(8) // 4 channels × 2 bytes
            .map(|chunk| {
                let mut result = Vec::with_capacity(4);
                for i in 0..4 {
                    let bytes = [chunk[i * 2], chunk[i * 2 + 1]];
                    result.push(half::f16::from_le_bytes(bytes));
                }
                result
            })
            .flatten()
            .collect();
        
        drop(data);
        staging_buffer.unmap();
        
        // Convert to field data
        field_manager.from_rgba16f(&half_data);
    }
}
