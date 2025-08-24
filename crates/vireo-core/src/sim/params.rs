use serde::{Deserialize, Serialize};
use bytemuck::{Pod, Zeroable};

/// World configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldConfig {
    pub size: [u32; 2],
    pub steps: u32,
    pub dt: f32,
    pub seed: u64,
}

/// Field reaction-diffusion parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldConfig {
    pub D_R: f32,      // Resource diffusion coefficient
    pub D_W: f32,      // Waste diffusion coefficient
    pub sigma_R: f32,  // Resource replenishment rate
    pub alpha_H: f32,  // Herbivore resource uptake rate
    pub beta_H: f32,   // Herbivore waste emission rate
    pub lambda_R: f32, // Resource decay rate
    pub lambda_W: f32, // Waste decay rate
}

/// Chemotaxis parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChemotaxisConfig {
    pub chi_R: f32,    // Resource attraction strength
    pub chi_W: f32,    // Waste repulsion strength
    pub kappa: f32,    // Gradient saturation parameter
    pub gamma: f32,    // Velocity damping
    pub v_max: f32,    // Maximum velocity
    pub eps0: f32,     // Basal energy drain rate
    pub eta_R: f32,    // Energy gain from resource
}

/// Agent configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentConfig {
    pub herbivores: u32,
    pub E0: f32,       // Initial energy
}

/// Noise configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseConfig {
    pub sigma: f32,    // Noise standard deviation
}

/// Obstacle configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObstacleConfig {
    pub enabled: bool,
}

/// Complete simulation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationConfig {
    pub world: WorldConfig,
    pub field: FieldConfig,
    pub chemotaxis: ChemotaxisConfig,
    pub agents: AgentConfig,
    pub noise: NoiseConfig,
    pub obstacles: ObstacleConfig,
}

/// GPU-compatible parameters for reaction-diffusion shader
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct RDParams {
    pub D_R: f32,
    pub D_W: f32,
    pub sigma_R: f32,
    pub alpha_H: f32,
    pub beta_H: f32,
    pub lambda_R: f32,
    pub lambda_W: f32,
    pub dt: f32,
    pub size: [u32; 2],
    pub H_SCALE: f32,  // Herbivore density scale factor
    pub _pad: u32,     // Padding for alignment
}

/// GPU-compatible parameters for agent chemotaxis shader
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct AgentParams {
    pub chi_R: f32,
    pub chi_W: f32,
    pub kappa: f32,
    pub gamma: f32,
    pub v_max: f32,
    pub eps0: f32,
    pub eta_R: f32,
    pub dt: f32,
    pub size: [f32; 2],
    pub _pad: [f32; 2], // Padding for alignment
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            world: WorldConfig {
                size: [128, 128],
                steps: 2000,
                dt: 0.1,
                seed: 1337,
            },
            field: FieldConfig {
                D_R: 0.5,
                D_W: 0.2,
                sigma_R: 0.005,
                alpha_H: 0.1,
                beta_H: 0.05,
                lambda_R: 0.005,
                lambda_W: 0.005,
            },
            chemotaxis: ChemotaxisConfig {
                chi_R: 8.0,
                chi_W: 4.0,
                kappa: 2.0,
                gamma: 0.05,
                v_max: 2.0,
                eps0: 0.02,
                eta_R: 0.2,
            },
            agents: AgentConfig {
                herbivores: 2000,
                E0: 1.0,
            },
            noise: NoiseConfig {
                sigma: 0.0,
            },
            obstacles: ObstacleConfig {
                enabled: false,
            },
        }
    }
}

impl From<&SimulationConfig> for RDParams {
    fn from(config: &SimulationConfig) -> Self {
        Self {
            D_R: config.field.D_R,
            D_W: config.field.D_W,
            sigma_R: config.field.sigma_R,
            alpha_H: config.field.alpha_H,
            beta_H: config.field.beta_H,
            lambda_R: config.field.lambda_R,
            lambda_W: config.field.lambda_W,
            dt: config.world.dt,
            size: config.world.size,
            H_SCALE: 0.125, // 1/8 per agent per cell as starting point
            _pad: 0,
        }
    }
}

impl From<&SimulationConfig> for AgentParams {
    fn from(config: &SimulationConfig) -> Self {
        Self {
            chi_R: config.chemotaxis.chi_R,
            chi_W: config.chemotaxis.chi_W,
            kappa: config.chemotaxis.kappa,
            gamma: config.chemotaxis.gamma,
            v_max: config.chemotaxis.v_max,
            eps0: config.chemotaxis.eps0,
            eta_R: config.chemotaxis.eta_R,
            dt: config.world.dt,
            size: [config.world.size[0] as f32, config.world.size[1] as f32],
            _pad: [0.0, 0.0],
        }
    }
}
