/// Reaction-diffusion step shader
pub fn rd_step() -> &'static str {
    include_str!("rd_step.wgsl")
}

/// Agent chemotaxis step shader
pub fn agent_step() -> &'static str {
    include_str!("agent_step.wgsl")
}
