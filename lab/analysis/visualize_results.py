#!/usr/bin/env python3
"""
Visualization script for Vireo simulation results
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import csv
from PIL import Image
import os

def load_metrics(csv_path):
    """Load metrics from CSV file"""
    df = pd.read_csv(csv_path)
    return df

def load_agents(csv_path):
    """Load agent data from CSV file"""
    df = pd.read_csv(csv_path)
    return df

def load_field_image(png_path):
    """Load field image and convert to numpy array"""
    img = Image.open(png_path)
    return np.array(img)

def plot_field_evolution(results_dir):
    """Plot the evolution of the resource field over time"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Resource Field Evolution (R channel)', fontsize=16)
    
    steps = [0, 200, 1000, 2000]
    for i, step in enumerate(steps):
        row, col = i // 2, i % 2
        png_path = os.path.join(results_dir, f'R_{step:04d}.png')
        
        if os.path.exists(png_path):
            field = load_field_image(png_path)
            # Extract R channel (red)
            r_channel = field[:, :, 0]
            
            im = axes[row, col].imshow(r_channel, cmap='viridis', interpolation='nearest')
            axes[row, col].set_title(f'Step {step}')
            axes[row, col].set_xlabel('X')
            axes[row, col].set_ylabel('Y')
            
            # Add colorbar
            plt.colorbar(im, ax=axes[row, col], label='Resource Value')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'field_evolution.png'), dpi=150, bbox_inches='tight')
    plt.show()

def plot_agent_distributions(results_dir):
    """Plot agent distributions at different time steps"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Agent Distributions Over Time', fontsize=16)
    
    steps = [0, 200, 1000, 2000]
    for i, step in enumerate(steps):
        row, col = i // 2, i % 2
        csv_path = os.path.join(results_dir, f'agents_{step:04d}.csv')
        
        if os.path.exists(csv_path):
            agents = load_agents(csv_path)
            
            # Plot agent positions
            axes[row, col].scatter(agents['x'], agents['y'], alpha=0.6, s=2, c='red')
            axes[row, col].set_title(f'Step {step} - {len(agents)} agents')
            axes[row, col].set_xlabel('X Position')
            axes[row, col].set_ylabel('Y Position')
            axes[row, col].set_xlim(0, 128)
            axes[row, col].set_ylim(0, 128)
            axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'agent_distributions.png'), dpi=150, bbox_inches='tight')
    plt.show()

def plot_metrics_over_time(results_dir):
    """Plot key metrics over time"""
    metrics_path = os.path.join(results_dir, 'metrics.csv')
    if not os.path.exists(metrics_path):
        print(f"Metrics file not found: {metrics_path}")
        return
    
    df = load_metrics(metrics_path)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Simulation Metrics Over Time', fontsize=16)
    
    # Resource statistics
    axes[0, 0].plot(df['step'], df['mean_R'], 'b-', label='Mean R')
    axes[0, 0].plot(df['step'], df['max_R'], 'b--', label='Max R')
    axes[0, 0].plot(df['step'], df['min_R'], 'b:', label='Min R')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Resource Value')
    axes[0, 0].set_title('Resource Field Statistics')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Agent statistics
    axes[0, 1].plot(df['step'], df['alive_count'], 'g-', label='Alive Count')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Agent Count')
    axes[0, 1].set_title('Agent Population')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Energy and velocity
    axes[1, 0].plot(df['step'], df['mean_energy'], 'r-', label='Mean Energy')
    axes[1, 0].plot(df['step'], df['mean_velocity'], 'orange', label='Mean Velocity')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('Value')
    axes[1, 0].set_title('Agent State')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Performance
    axes[1, 1].plot(df['step'], df['fps_proxy'], 'purple', label='FPS Proxy')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('FPS')
    axes[1, 1].set_title('Performance (FPS Proxy)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'metrics_over_time.png'), dpi=150, bbox_inches='tight')
    plt.show()

def create_summary_report(results_dir):
    """Create a summary report of the simulation"""
    metrics_path = os.path.join(results_dir, 'metrics.csv')
    if not os.path.exists(metrics_path):
        print(f"Metrics file not found: {metrics_path}")
        return
    
    df = load_metrics(metrics_path)
    
    print("=" * 60)
    print("Vireo Simulation Summary Report")
    print("=" * 60)
    print(f"Total Steps: {len(df)}")
    print(f"Simulation Duration: {df['wall_time_ms'].sum() / 1000:.2f} seconds")
    print(f"Average Step Time: {df['wall_time_ms'].mean():.2f} ms")
    print(f"Average FPS: {df['fps_proxy'].mean():.1f}")
    print()
    
    print("Field Statistics (Final):")
    print(f"  Mean Resource: {df.iloc[-1]['mean_R']:.6f}")
    print(f"  Max Resource: {df.iloc[-1]['max_R']:.6f}")
    print(f"  Min Resource: {df.iloc[-1]['min_R']:.6f}")
    print(f"  Resource Variance: {df.iloc[-1]['var_R']:.6f}")
    print()
    
    print("Agent Statistics (Final):")
    print(f"  Alive Count: {df.iloc[-1]['alive_count']}")
    print(f"  Mean Energy: {df.iloc[-1]['mean_energy']:.6f}")
    print(f"  Mean Velocity: {df.iloc[-1]['mean_velocity']:.6f}")
    print(f"  Foraging Efficiency: {df.iloc[-1]['foraging_efficiency']:.6f}")
    print()
    
    print("Files Generated:")
    for file in os.listdir(results_dir):
        if file.endswith(('.png', '.csv')):
            size = os.path.getsize(os.path.join(results_dir, file))
            print(f"  {file}: {size} bytes")
    
    print("=" * 60)

def main():
    """Main visualization function"""
    results_dir = "results"
    
    if not os.path.exists(results_dir):
        print(f"Results directory not found: {results_dir}")
        return
    
    print("Generating visualizations for Vireo simulation results...")
    
    # Create summary report
    create_summary_report(results_dir)
    
    # Generate plots
    try:
        plot_field_evolution(results_dir)
        plot_agent_distributions(results_dir)
        plot_metrics_over_time(results_dir)
        print("All visualizations generated successfully!")
    except Exception as e:
        print(f"Error generating visualizations: {e}")

if __name__ == "__main__":
    main()
