use wgpu::{Device, Queue, Texture, TextureView, TextureViewDescriptor};
use crate::sim::FieldManager;

/// Field texture pair for ping-pong operations
pub struct FieldTextures {
    pub texture_a: Texture,
    pub texture_b: Texture,
    pub view_a_sample: TextureView,
    pub view_a_store: TextureView,
    pub view_b_sample: TextureView,
    pub view_b_store: TextureView,
    pub view_a_storage: TextureView,  // Storage texture view for RD source
    pub view_b_storage: TextureView,  // Storage texture view for RD source
    pub size: [u32; 2],
}

impl FieldTextures {
    /// Create a new pair of field textures for ping-pong operations
    pub fn new(device: &Device, size: [u32; 2]) -> Self {
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
        
        let texture_a = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("field_a"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        });
        
        let texture_b = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("field_b"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        });
        
        let view_a_sample = texture_a.create_view(&TextureViewDescriptor::default());
        let view_b_sample = texture_b.create_view(&TextureViewDescriptor::default());
        
        let view_a_store = texture_a.create_view(&TextureViewDescriptor {
            label: Some("field_a_store"),
            format: Some(format),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(1),
        });
        
        let view_b_store = texture_b.create_view(&TextureViewDescriptor {
            label: Some("field_b_store"),
            format: Some(format),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(1),
        });
        
        // Create storage texture views for RD pipeline
        let view_a_storage = texture_a.create_view(&TextureViewDescriptor {
            label: Some("field_a_storage"),
            format: Some(format),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(1),
        });
        
        let view_b_storage = texture_b.create_view(&TextureViewDescriptor {
            label: Some("field_b_storage"),
            format: Some(format),
            dimension: Some(wgpu::TextureViewDimension::D2),
            aspect: wgpu::TextureAspect::All,
            base_mip_level: 0,
            mip_level_count: Some(1),
            base_array_layer: 0,
            array_layer_count: Some(1),
        });
        
        Self {
            texture_a,
            texture_b,
            view_a_sample,
            view_a_store,
            view_b_sample,
            view_b_store,
            view_a_storage,
            view_b_storage,
            size,
        }
    }
    
    /// Upload field data to texture A
    pub fn upload_field_data(&self, queue: &Queue, field_manager: &FieldManager) {
        let half_data = field_manager.to_rgba16f();
        
        // Calculate padded bytes per row for alignment
        let bytes_per_pixel = 8; // 4 channels × 2 bytes (f16)
        let unpadded_bpr = self.size[0] as usize * bytes_per_pixel;
        let padded_bpr = ((unpadded_bpr + 255) / 256) * 256; // Round up to 256-byte boundary
        
        // Create padded buffer for texture upload
        let mut padded_bytes = Vec::with_capacity(padded_bpr * self.size[1] as usize);
        for row in 0..self.size[1] {
            let row_start = row as usize * self.size[0] as usize * 4;
            let row_end = row_start + self.size[0] as usize * 4;
            
            // Add the actual pixel data for this row
            for pixel_idx in row_start..row_end {
                padded_bytes.extend_from_slice(&half_data[pixel_idx].to_le_bytes());
            }
            
            // Pad the row to meet alignment requirement
            let row_bytes = self.size[0] as usize * bytes_per_pixel;
            let padding_needed = padded_bpr - row_bytes;
            padded_bytes.extend(std::iter::repeat(0u8).take(padding_needed));
        }
        
        let layout = wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(padded_bpr as u32),
            rows_per_image: Some(self.size[1]),
        };
        
        let size = wgpu::Extent3d {
            width: self.size[0],
            height: self.size[1],
            depth_or_array_layers: 1,
        };
        
        queue.write_texture(
            wgpu::ImageCopyTexture {
                texture: &self.texture_a,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &padded_bytes,
            layout,
            size,
        );
    }
    
    /// Download field data from texture A
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
                texture: &self.texture_a,
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
    
    /// Get the current source and destination views for ping-pong
    pub fn get_ping_pong_views(&self, use_a_as_src: bool) -> (&TextureView, &TextureView) {
        if use_a_as_src {
            (&self.view_a_sample, &self.view_b_store)
        } else {
            (&self.view_b_sample, &self.view_a_store)
        }
    }
    
    /// Swap the ping-pong buffers
    pub fn swap(&mut self) {
        std::mem::swap(&mut self.texture_a, &mut self.texture_b);
        std::mem::swap(&mut self.view_a_sample, &mut self.view_b_sample);
        std::mem::swap(&mut self.view_a_store, &mut self.view_b_store);
    }
}
