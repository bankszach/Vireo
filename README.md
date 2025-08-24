# Vireo — Interactive Ecosystem Sandbox

An interactive ecosystem simulation built on a custom GPU engine, featuring diffusion fields, emergent behavior, and massive-scale particle systems. Built with Rust and WGPU for high-performance simulation.

> **Inspired by** [**Mote** by Peter Whidden](https://www.youtube.com/watch?v=Hju0H3NHxVI) — **not affiliated**.

## About

Vireo is a GPU-native ecosystem sandbox that explores emergent behavior through simple rules and massive scale. The simulation features a 2D diffusion field, particles with ring-spring constraints, and a behavior kernel that creates complex interactions from simple components.

This project re-implements concepts from scratch and is **not affiliated** with Mote or its author. We're grateful for the inspiration and the community enthusiasm around large-scale emergent simulations.

## Features

- **GPU diffusion field** (RGBA16F texture) with ping-pong rendering
- **Massive particle systems** (20k by default, scalable to 50k+ on stronger GPUs)
- **Ring-spring physics** for particle grouping and stability
- **Emergent behaviors** through simple rule sets:
  - Plants: stationary, stabilized by springs
  - Herbivores: follow food gradients
  - Predators: hunt herbivores
- **Real-time rendering** with instanced quads (no CPU copies)
- **Clean architecture** with separate compute and render pipelines

## Build & Run

```bash
# Requires Rust + cargo and a GPU that supports WebGPU (Vulkan/Metal/DX12)
cargo run --release
```

### Controls
- `Space` — pause/resume simulation
- `R` — re-seed the environment
- `Esc` — quit

### Environment Variables
```bash
# Particle count (default: 20,000)
VIREO_PARTICLES=50000 cargo run --release

# World dimensions (default: 1024x576)
VIREO_GRID_W=1024 VIREO_GRID_H=576 cargo run --release
```

## Project Structure

```
src/main.rs               # WGPU initialization, pipelines, main loop
shaders/diffuse.wgsl      # 2D field diffusion (laplacian + decay)
shaders/particles.wgsl    # Particle physics + behavior kernel
shaders/render.wgsl       # Instanced quad renderer
```

## Technical Details

- **Compute passes**: One for diffusion, one for particles per frame
- **Memory layout**: Optimized for GPU with minimal CPU-GPU transfers
- **Scalability**: Designed to handle 50k+ particles on consumer GPUs
- **Extensibility**: Clean separation of concerns for easy modification

## Future Directions

- **Emissions system**: Particles deposit signals back into the field
- **Advanced physics**: Replace ring-springs with sparse neighbor systems
- **Genetic algorithms**: Genome-based behavior evolution
- **ML integration**: Python bindings for reinforcement learning
- **Multi-agent systems**: Complex interaction networks

## Inspiration & Attribution

This project draws conceptual inspiration from Peter Whidden's talk on **Mote**. All code and assets in this repository are original unless otherwise noted. "Mote" is referenced for attribution only; this project is not endorsed by or associated with its author.

**References:**
- [Mote: An Interactive Ecosystem Simulation - Peter Whidden](https://www.youtube.com/watch?v=Hju0H3NHxVI)
- [Recurse Center on X](https://x.com/recursecenter/status/1958926108763529719)

## License

MIT License - see [LICENSE](LICENSE) file for details.

---

*Vireo: A world of interacting life, inspired by nature's complexity.*
