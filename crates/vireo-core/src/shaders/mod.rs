/// Reaction-diffusion step shader
pub fn rd_step() -> &'static str {
    include_str!("rd_step.wgsl")
}

/// Agent chemotaxis step shader
pub fn agent_step() -> &'static str {
    include_str!("agent_step.wgsl")
}

/// Clear occupancy buffer shader
pub fn clear_occupancy() -> &'static str {
    include_str!("clear_occupancy.wgsl")
}
