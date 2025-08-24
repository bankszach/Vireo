# Vireo Ecosystem Simulation - Visual Legend

## Entity Types

### 🌱 Plants (Green)
- **Color**: Green `#33CC4D` (0.2, 0.8, 0.3)
- **Size**: Base size 3.0 pixels, varies with energy (0.3x to 1.5x)
- **Behavior**: Stationary with light jitter, grow slowly over time
- **Energy**: Gain energy from food-rich areas, very stable
- **Population**: ~6.7% of total particles (1,334 out of 20,000)

### 🦌 Herbivores (Blue)  
- **Color**: Blue `#3399FF` (0.2, 0.6, 1.0)
- **Size**: Base size 2.5 pixels, varies with energy (0.3x to 1.5x)
- **Behavior**: Seek food, herd together, seek mates, climb food gradients
- **Energy**: Consume energy through movement, need food to survive
- **Population**: ~86.7% of total particles (17,332 out of 20,000)

### 🐺 Predators (Red)
- **Color**: Red `#CC3333` (0.8, 0.2, 0.2)
- **Size**: Base size 4.0 pixels, varies with energy (0.3x to 1.5x)
- **Behavior**: Chase herbivores, hunt for prey, territorial
- **Energy**: Consume energy through movement and hunting
- **Population**: ~6.7% of total particles (1,334 out of 20,000)

## Visual Effects & States

### 🔄 Reproduction State
- **Indicator**: Bright white flashing effect
- **Trigger**: When `reproduction_cooldown <= 0` and `energy > 6.0`
- **Visual**: Pulsing white overlay with 20Hz frequency
- **Alpha**: 1.0 (fully opaque)

### 🍽️ Feeding State
- **Indicator**: Bright yellow/orange glow
- **Trigger**: When in food-rich areas (`local_food > 0.3`)
- **Visual**: Yellow/orange overlay with 15Hz frequency
- **Alpha**: 0.95

### ⚔️ Attack State (Predators)
- **Indicator**: Bright red flash
- **Trigger**: When predator is within 40 pixels of prey
- **Visual**: Red overlay with 25Hz frequency
- **Alpha**: 1.0

### 🐑 Herding State (Herbivores)
- **Indicator**: Cyan glow
- **Trigger**: When near 3+ other herbivores within 80 pixels
- **Visual**: Cyan overlay with 10Hz frequency
- **Alpha**: 0.9

### 💫 Energy Effects
- **High Energy Glow**: Soft yellow glow when `energy > 8.0`
- **Low Energy Stress**: Red stress indicator when `energy < 3.0`
- **Stress Flicker**: Low-energy particles flicker at 30Hz
- **Pulse Effect**: High-energy particles pulse at 8Hz

### 🌊 Movement Trails
- **Indicator**: Color-coded trails based on velocity
- **Plants**: Light green trails
- **Herbivores**: Light blue trails  
- **Predators**: Light red trails
- **Intensity**: Based on velocity magnitude (0.0 to 0.3)

## Size Variations

### Base Sizes
- **Plants**: 3.0 pixels
- **Herbivores**: 2.5 pixels
- **Predators**: 4.0 pixels

### Energy Scaling
- **Range**: 0.3x to 1.5x base size
- **Formula**: `clamp(energy / 10.0, 0.3, 1.5)`
- **Low Energy**: Particles appear smaller (0.3x)
- **High Energy**: Particles appear larger (1.5x)

### Pulse Effects
- **High Energy Pulse**: Size varies by ±20% when `energy > 7.0`
- **Frequency**: 8Hz with individual particle variation

## Environment

### 🗺️ Food Field (Diffusion Field)
- **Color**: Channel 0 (Red channel) represents food concentration
- **Distribution**: Clustered in center with Gaussian blobs
- **Primary Source**: Strong central food source (amplitude 1.2, radius 60)
- **Secondary Sources**: 25 smaller food sources around center
- **Gradient**: Gentle gradient from center to edges guides particles

### 🌍 World Boundaries
- **Size**: 1024x576 pixels (configurable via `VIREO_GRID_W` and `VIREO_GRID_H`)
- **Wrapping**: Particles wrap around edges instead of bouncing
- **Edge Forces**: Smooth repulsion prevents particles from getting stuck

## Controls

### Keyboard Shortcuts
- **Space**: Pause/Resume simulation
- **R**: Reseed food field
- **Escape**: Exit simulation

### Environment Variables
- `VIREO_GRID_W`: Grid width (default: 1024)
- `VIREO_GRID_H`: Grid height (default: 576)  
- `VIREO_PARTICLES`: Total particle count (default: 20,000)

## Simulation Parameters

### Physics
- **Time Step**: 1/60 second (60 FPS)
- **Spring Force**: 0.8 (reduced from 1.5)
- **Rest Distance**: 12.0 pixels (increased from 8.0)
- **Damping**: 0.97 (slight damping for stability)

### Ecosystem Balance
- **Plant Growth**: Very slow, energy gain from food areas
- **Herbivore Behavior**: Food-seeking, herding, mating
- **Predator Behavior**: Hunting, territorial, energy-intensive
- **Reproduction**: Type-specific cooldowns and energy costs

## Visual Rendering

### Rendering Pipeline
- **Diffuse Pass**: GPU compute shader for food field diffusion
- **Particle Pass**: GPU compute shader for particle physics and behavior
- **Render Pass**: GPU render shader for visual output

### Performance
- **Workgroup Size**: 256 particles per compute group
- **2D Workgroups**: 16x16 for diffusion field
- **Target FPS**: 60 FPS with adaptive timing

---

*This legend covers all visual elements in the Vireo ecosystem simulation. The simulation demonstrates emergent behavior through simple rules, creating complex ecological interactions between plants, herbivores, and predators.*
