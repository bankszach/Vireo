//! Vireo Core Engine
//! 
//! Core engine for ecosystem simulation with reaction-diffusion fields and chemotactic agents.

pub mod gpu;
pub mod sim;
pub mod shaders;

// Re-export main types
pub use gpu::*;
pub use sim::*;
pub use shaders::*;

// Re-export params from vireo-params
pub use vireo_params::*;
