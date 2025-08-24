# WGSL Binding Layouts

This document defines the exact binding layouts that must be identical between headless and viewer simulations to ensure parity, along with best practices for organizing binding layouts in wgpu applications.

## Binding Layout Architecture Patterns

### TL;DR: Layout Ownership Strategy

* **Do not** recreate layouts every frame
* Create **all bind group layouts once** in a central place (the "app/viewer context")
* Share them by **reference** when building pipelines/bind-groups, or wrap in **`Arc`** if multiple owners must hold them
* Keep `FieldPingPong` focused on **textures + views + prebuilt bind groups**; don't make it own the layouts
* On resize, **pass the layout** back into `FieldPingPong::recreate(...)` to rebuild bind groups

This avoids lifetime tangles and matches how most wgpu apps structure things.

### Why This Pattern Matters

`wgpu::BindGroupLayout` is a handle type (RAII around an internal ref), but it:

* is **not `Clone`**; copying is not allowed
* is **owned** by the device; it must not outlive the device (enforced at runtime)
* is cheap to **borrow** (`&BindGroupLayout`) wherever needed

Trying to `clone()` a `BindGroupLayout` or store only a `&BindGroupLayout` with a `'static` expectation causes lifetime mismatches.

### Recommended Architecture

#### 1) Centralize Layouts

Make a small registry that owns layouts (and optionally samplers). Either store as plain fields (borrow when needed) or `Arc` if you want multiple long-lived holders.

```rust
use std::sync::Arc;

pub struct Layouts {
    pub rd:   wgpu::BindGroupLayout,  // compute: src sampled, dst storage, uniforms, occ
    pub show: wgpu::BindGroupLayout,  // render: sampler + sampled field
}

impl Layouts {
    pub fn new(device: &wgpu::Device) -> Self {
        let rd = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("rd_bgl"),
            entries: &[
                // @binding(0) src sampled
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
                // @binding(1) dst storage
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rg32Float,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                // @binding(2) uniforms
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
                // @binding(3) occupancy
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

        let show = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("show_bgl"),
            entries: &[
                // @binding(0) sampler
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                    count: None,
                },
                // @binding(1) sampled field
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });

        Self { rd, show }
    }
}
```

> If you prefer multiple owners: `pub struct Layouts { pub rd: Arc<BindGroupLayout>, pub show: Arc<BindGroupLayout> }`. Then pass `layouts.show.clone()` to places that need to hold onto it long-term. `Arc` is only for *sharing ownership*, not required for correctness.

#### 2) Pipelines are Created from These Layouts (by Reference)

The **renderer** builds its pipeline using the `show` layout:

```rust
let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
    label: Some("render_pl"),
    bind_group_layouts: &[&layouts.show],
    push_constant_ranges: &[],
});

let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
    label: Some("render"),
    layout: Some(&pipeline_layout),
    // ... shader, targets, etc.
});
```

The **simulation/compute** module builds its compute pipeline using the `rd` layout:

```rust
let rd_pl = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
    label: Some("rd_pl"),
    bind_group_layouts: &[&layouts.rd],
    push_constant_ranges: &[],
});
let rd_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
    label: Some("rd"),
    layout: Some(&rd_pl),
    module: &rd_shader,
    entry_point: "main",
});
```

You only pass `&BindGroupLayout` at creation time; you don't have to store the layout inside other components unless they need to *rebuild* bind groups by themselves later.

#### 3) Keep `FieldPingPong` Layout-Free (Preferred)

Let `FieldPingPong` own **textures + views** and the **flip** flag. It should **not** own the layouts. The modules that *know* their layout (renderer and sim) create and own their bind groups using the views you expose.

```rust
pub struct FieldPingPong {
    // textures + views
    view_a_sample: wgpu::TextureView,
    view_b_sample: wgpu::TextureView,
    view_a_store:  wgpu::TextureView,
    view_b_store:  wgpu::TextureView,
    front_is_a: bool,
}

impl FieldPingPong {
    pub fn front_sample(&self) -> &wgpu::TextureView {
        if self.front_is_a { &self.view_a_sample } else { &self.view_b_sample }
    }
    pub fn a_sample(&self) -> &wgpu::TextureView { &self.view_a_sample }
    pub fn b_sample(&self) -> &wgpu::TextureView { &self.view_b_sample }
    pub fn a_store(&self)  -> &wgpu::TextureView { &self.view_a_store  }
    pub fn b_store(&self)  -> &wgpu::TextureView { &self.view_b_store  }
    pub fn swap(&mut self) { self.front_is_a = !self.front_is_a; }
}
```

Then:

* **Sim** creates **two RD bind groups** (A→B and B→A) *once* using `&layouts.rd` and the views obtained from `FieldPingPong`.
* **Renderer** creates **two show bind groups** (show A, show B) *once* using `&layouts.show`.

On **resize**, call:

* `field.recreate_textures(...)`
* `sim.recreate_rd_bind_groups(&device, &layouts.rd, &field)`
* `renderer.recreate_show_bind_groups(&device, &layouts.show, &field)`

No layout is stored inside `FieldPingPong`; no circular deps.

#### 4) If You Really Want `FieldPingPong` to Build Bind Groups

Wrap the layouts in `Arc` and hand them in:

```rust
pub struct FieldPingPong {
    rd_a2b_bg: wgpu::BindGroup,
    rd_b2a_bg: wgpu::BindGroup,
    show_a_bg: wgpu::BindGroup,
    show_b_bg: wgpu::BindGroup,
    // textures/views...
    front_is_a: bool,
}

impl FieldPingPong {
    pub fn new(
        device: &wgpu::Device,
        layouts: &Layouts,        // or &Arc<Layouts>
        // ... formats, sizes, sampler, buffers
    ) -> Self {
        let rd_a2b_bg = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("rd a2b"),
            layout: &layouts.rd, // borrow here
            entries: &[
                // src A (sampled), dst B (storage), uniforms, occupancy
            ],
        });
        // build rd_b2a_bg, show_a_bg, show_b_bg similarly
        Self { /* … */ front_is_a: true }
    }

    pub fn recreate(&mut self, device: &wgpu::Device, layouts: &Layouts /*…*/) {
        // rebuild bind groups using borrowed layouts
    }

    pub fn rd_bg(&self) -> &wgpu::BindGroup {
        if self.front_is_a { &self.rd_a2b_bg } else { &self.rd_b2a_bg }
    }
    pub fn show_bg(&self) -> &wgpu::BindGroup {
        if self.front_is_a { &self.show_a_bg } else { &self.show_b_bg }
    }
}
```

Note: You **don't** store the layout; you **borrow** it when creating bind groups. If multiple places need to reconstruct bind groups asynchronously, that's when an `Arc<Layouts>` is handy so each can hold onto it safely.

### Common Pitfalls and Solutions

#### Error: `BindGroupLayout is not Clone`

**Problem**: Trying to clone a `BindGroupLayout` or store it with incorrect lifetime expectations.

**Solutions**:
* **Best**: Don't store layouts in components that don't need to rebuild bind groups
* **If you must store**: Wrap in `Arc` at definition time:
  ```rust
  pub struct Layouts { pub show: Arc<wgpu::BindGroupLayout>, /*…*/ }
  // then
  render_bgl: layouts.show.clone(), // ✅ clone the Arc
  // and when creating pipeline:
  let pl = device.create_pipeline_layout(&PipelineLayoutDescriptor {
      bind_group_layouts: &[&*layouts.show], // Arc<Deref<Target=T>> → &T
      push_constant_ranges: &[],
  });
  ```

#### Resize Handling

On window resize or texture recreation:

1. Call `field.recreate_textures(...)` to rebuild textures and views
2. Rebuild bind groups by passing `&layouts.*` again:
   ```rust
   sim.recreate_rd_bind_groups(&device, &layouts.rd, &field)
   renderer.recreate_show_bind_groups(&device, &layouts.show, &field)
   ```

### Final Recommendation

* **Adopt the "central Layouts + builder borrows" pattern**
  * Viewer/App owns `Layouts`
  * Renderer and Sim borrow `&layouts.*` to build pipelines/bind groups
  * `FieldPingPong` owns textures/views + flip + (optionally) the prebuilt bind groups, but **not** the layouts
* On **resize**, call `recreate_textures` then rebuild bind groups by passing `&layouts.*` again

This keeps responsibilities crisp, avoids lifetime/clone headaches, and mirrors established wgpu practice.

---

## Binding Group 0: Reaction-Diffusion Compute Shader

**Shader**: `rd_step.wgsl`

```wgsl
@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var dstTex: texture_storage_2d<rgba16float, write>;
@group(0) @binding(2) var<uniform> params: RDParams;
@group(0) @binding(3) var<storage, read> herbDensity: array<u32>;
```

**Bindings**:
- `@0`: Source texture (sampler2D) - current field state
- `@1`: Destination texture (storage2D write) - next field state  
- `@2`: RDParams uniform buffer - reaction-diffusion parameters
- `@3`: Herbivore occupancy buffer (storage r32uint) - agent density per cell

**RDParams Structure**:
```rust
#[repr(C)]
pub struct RDParams {
    pub D_R: f32,        // Resource diffusion coefficient
    pub D_W: f32,        // Waste diffusion coefficient
    pub sigma_R: f32,    // Resource replenishment rate
    pub alpha_H: f32,    // Herbivore resource uptake rate
    pub beta_H: f32,     // Herbivore waste emission rate
    pub lambda_R: f32,   // Resource decay rate
    pub lambda_W: f32,   // Waste decay rate
    pub dt: f32,         // Time step
    pub size: [u32; 2],  // Grid size
    pub H_SCALE: f32,    // Herbivore density scale factor (0.125)
    pub _pad: u32,       // Padding for alignment
}
```

## Binding Group 0: Agent Chemotaxis Compute Shader

**Shader**: `agent_step.wgsl`

```wgsl
@group(0) @binding(0) var<storage, read_write> agents: array<Agent>;
@group(0) @binding(1) var fieldTex: texture_2d<f32>;
@group(0) @binding(2) var<uniform> params: AgentParams;
@group(0) @binding(3) var<storage, read_write> herbOcc: array<u32>;
```

**Bindings**:
- `@0`: Agents storage buffer (read_write) - agent positions, velocities, energy
- `@1`: Field texture (sampler2D) - current resource/waste field
- `@2`: AgentParams uniform buffer - chemotaxis parameters
- `@3`: Herbivore occupancy buffer (storage r32uint) - agent count per cell

**AgentParams Structure**:
```rust
#[repr(C)]
pub struct AgentParams {
    pub chi_R: f32,      // Resource attraction strength
    pub chi_W: f32,      // Waste repulsion strength
    pub kappa: f32,      // Gradient saturation parameter
    pub gamma: f32,      // Velocity damping
    pub v_max: f32,      // Maximum velocity
    pub eps0: f32,       // Basal energy drain rate
    pub eta_R: f32,      // Energy gain from resource
    pub dt: f32,         // Time step
    pub size: [f32; 2],  // World size
    pub _pad: [f32; 2],  // Padding for alignment
}
```

**Agent Structure**:
```rust
#[repr(C)]
pub struct Agent {
    pub pos: [f32; 2],   // Position (x, y)
    pub vel: [f32; 2],   // Velocity (vx, vy)
    pub energy: f32,      // Current energy
    pub alive: u32,       // Alive flag (1 = alive, 0 = dead)
}
```

## Critical Constants

**H_SCALE**: `0.125` (1/8 per agent per cell)
- Must be identical in both simulations
- Used to normalize occupancy counts to [0,1] range
- Defined in `vireo_params::bindings::H_SCALE`

## Validation

Both headless and viewer must:
1. Use identical binding group layouts
2. Use identical parameter structures
3. Use identical H_SCALE constant
4. Clear occupancy buffer each frame before agent pass
5. Sample from correct texture after ping-pong swap

## Workgroup Sizes

- **RD Pass**: `@workgroup_size(8, 8)` - 2D grid processing
- **Agent Pass**: `@workgroup_size(128)` - 1D agent processing

## Frame Order

1. Clear occupancy buffer
2. Agent pass (updates positions, writes to occupancy)
3. RD pass (reads occupancy, updates fields)
4. Swap ping-pong buffers
5. Render (from front buffer)

## Error Prevention

- Runtime validation of binding layouts
- Assert that H_SCALE matches expected value
- Log binding group creation with layout info
- Validate texture formats match expected types
