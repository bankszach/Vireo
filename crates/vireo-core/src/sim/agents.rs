use bytemuck::{Pod, Zeroable};
use glam::Vec2;
use rand_chacha::ChaCha8Rng;
use rand::{Rng, SeedableRng};

/// Agent data structure for GPU compute
#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
pub struct Agent {
    pub pos: [f32; 2],     // Position (x, y)
    pub vel: [f32; 2],     // Velocity (vx, vy)
    pub energy: f32,        // Current energy
    pub alive: u32,         // Alive flag (1 = alive, 0 = dead)
}

impl Agent {
    pub fn new(pos: Vec2, energy: f32) -> Self {
        Self {
            pos: [pos.x, pos.y],
            vel: [0.0, 0.0],
            energy,
            alive: 1,
        }
    }

    pub fn position(&self) -> Vec2 {
        Vec2::new(self.pos[0], self.pos[1])
    }

    pub fn velocity(&self) -> Vec2 {
        Vec2::new(self.vel[0], self.vel[1])
    }

    pub fn is_alive(&self) -> bool {
        self.alive == 1
    }

    pub fn kill(&mut self) {
        self.alive = 0;
    }
}

/// Agent statistics for metrics collection
#[derive(Debug, Clone)]
pub struct AgentStats {
    pub alive_count: u32,
    pub total_energy: f32,
    pub mean_energy: f32,
    pub mean_velocity: f32,
    pub foraging_efficiency: f32,
}

impl Default for AgentStats {
    fn default() -> Self {
        Self {
            alive_count: 0,
            total_energy: 0.0,
            mean_energy: 0.0,
            mean_velocity: 0.0,
            foraging_efficiency: 0.0,
        }
    }
}

/// Agent manager for CPU-side operations
pub struct AgentManager {
    pub agents: Vec<Agent>,
    pub stats: AgentStats,
}

impl AgentManager {
    pub fn new(herbivore_count: u32, world_size: [f32; 2], initial_energy: f32, seed: u64) -> Self {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        
        let mut agents = Vec::with_capacity(herbivore_count as usize);
        
        for _ in 0..herbivore_count {
            let x = rng.gen_range(10.0..(world_size[0] - 10.0));
            let y = rng.gen_range(10.0..(world_size[1] - 10.0));
            let pos = Vec2::new(x, y);
            
            // Add some initial random velocity
            let angle = rng.gen_range(0.0..std::f32::consts::TAU);
            let speed = rng.gen_range(0.1..0.5);
            let vel = Vec2::new(angle.cos() * speed, angle.sin() * speed);
            
            let mut agent = Agent::new(pos, initial_energy);
            agent.vel = [vel.x, vel.y];
            agents.push(agent);
        }
        
        Self {
            agents,
            stats: AgentStats::default(),
        }
    }
    
    pub fn update_stats(&mut self) {
        let alive_agents: Vec<_> = self.agents.iter().filter(|a| a.is_alive()).collect();
        
        if alive_agents.is_empty() {
            self.stats = AgentStats::default();
            return;
        }
        
        let alive_count = alive_agents.len() as u32;
        let total_energy: f32 = alive_agents.iter().map(|a| a.energy).sum();
        let mean_energy = total_energy / alive_count as f32;
        
        let total_velocity: f32 = alive_agents.iter()
            .map(|a| (a.vel[0] * a.vel[0] + a.vel[1] * a.vel[1]).sqrt())
            .sum();
        let mean_velocity = total_velocity / alive_count as f32;
        
        // Simple foraging efficiency: energy / (distance traveled)
        let foraging_efficiency = if mean_velocity > 0.0 {
            mean_energy / mean_velocity
        } else {
            0.0
        };
        
        self.stats = AgentStats {
            alive_count,
            total_energy,
            mean_energy,
            mean_velocity,
            foraging_efficiency,
        };
    }
    
    pub fn get_alive_count(&self) -> u32 {
        self.agents.iter().filter(|a| a.is_alive()).count() as u32
    }
    
    pub fn reset(&mut self, world_size: [f32; 2], initial_energy: f32, seed: u64) {
        *self = Self::new(self.agents.len() as u32, world_size, initial_energy, seed);
    }
}
