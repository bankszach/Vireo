# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2024-12-19

### Added
- Initial public release of Vireo ecosystem simulation
- GPU-native diffusion field system with ping-pong rendering
- Massive-scale particle system (20k+ particles)
- Ring-spring physics for particle grouping
- Behavior kernel with plants, herbivores, and predators
- Real-time rendering with instanced quads
- WGPU-based compute and render pipelines

### Changed
- Project renamed from "mote_lite" to "Vireo" for clarity and distinction
- Environment variables updated from `MOTE_LITE_*` to `VIREO_*`
- Window title updated to reflect new project name

### Technical
- Built with Rust and WGPU for cross-platform GPU acceleration
- Optimized memory layout for minimal CPU-GPU transfers
- Clean separation of compute and render pipelines
- Support for Vulkan, Metal, and DirectX 12 backends

### Attribution
- Project inspired by Peter Whidden's "Mote: An Interactive Ecosystem Simulation" talk
- All code and assets are original implementations
- Proper attribution and non-affiliation disclaimers added
