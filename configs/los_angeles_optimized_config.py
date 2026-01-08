"""
Optimized Los Angeles Configuration for Large Grid Simulations (868+ sites)
Minimal logging, maximum performance, reduced file sizes
"""

import os
from pathlib import Path
import logging

# Setup minimal logging for production
logger = logging.getLogger(__name__)

# Get environment variables
RSQSIM_HOME = os.environ.get('RSQSIM_HOME', str(Path.home() / 'rsqsim_mc'))
RSQSIM_WORK_DIR = os.environ.get('RSQSIM_WORK_DIR', '/scratch/olawoyiv/rsqsim_data')
RSQSIM_PROJECT_DIR = os.environ.get('RSQSIM_PROJECT_DIR', '/projects/ebelseismo/olawoyiv')

# Regional polygon (same as before, but cleaner)
REGIONAL_POLYGON = [
    [33.043, -116.303], [32.498, -117.104], [33.577, -117.971],
    [33.649, -118.429], [33.973, -118.658], [34.328, -119.974],
    [34.619, -119.974], [34.557, -117.132], [33.043, -116.303]
]

def get_region_bounds():
    """Get regional bounds from polygon"""
    lats = [point[0] for point in REGIONAL_POLYGON]
    lons = [point[1] for point in REGIONAL_POLYGON]
    
    return {
        "min_lat": min(lats),
        "max_lat": max(lats),
        "min_lon": min(lons),
        "max_lon": max(lons),
        "polygon": REGIONAL_POLYGON
    }

def generate_optimized_paths(region_name: str, mode: str = "sequential"):
    """Generate paths optimized for large grid processing"""
    
    base_output = os.path.join(RSQSIM_WORK_DIR, 'output')
    
    return {
        "project_root": Path(RSQSIM_HOME),
        "gmpe_config": str(Path(RSQSIM_HOME) / "configs" / "config.py"),
        
        # Data directories
        "windows_dir": f"{RSQSIM_WORK_DIR}/data/Catalog_4983/windows/{region_name}/{mode}",
        
        # Output directories (minimal structure for large grids)
        "output_dir": f"{base_output}/{region_name}/{mode}",
        "log_dir": f"{RSQSIM_WORK_DIR}/logs/{region_name}/{mode}",
        
        # Backup directories
        "backup_dir": f"{RSQSIM_PROJECT_DIR}/rsqsim_results/{region_name}/{mode}",
        
        # Minimal visualization directories (only essential ones)
        "viz_dir": f"{base_output}/{region_name}/{mode}/visualizations",
        "gis_dir": f"{base_output}/{region_name}/{mode}/visualizations/gis",
    }

def create_essential_directories(paths: dict):
    """Create only essential directories for large grid processing"""
    
    essential_dirs = [
        paths.get("output_dir"),
        paths.get("log_dir"), 
        paths.get("viz_dir"),
        paths.get("gis_dir"),
    ]
    
    # Filter out None values
    essential_dirs = [d for d in essential_dirs if d is not None]
    
    created_count = 0
    
    for dir_path in essential_dirs:
        try:
            Path(dir_path).mkdir(parents=True, exist_ok=True, mode=0o755)
            created_count += 1
        except Exception as e:
            logger.warning(f"Could not create directory {dir_path}: {e}")
    
    logger.info(f"Created {created_count} essential directories")
    return created_count

def load_optimized_config(mode: str = "sequential"):
    """
    Load configuration optimized for large grid simulations
    
    Key optimizations:
    - Minimal logging
    - No individual event outputs
    - Efficient memory usage
    - Fast processing settings
    """
    
    # Get regional bounds
    region_bounds = get_region_bounds()
    
    # Generate optimized paths
    paths = generate_optimized_paths('los_angeles', mode)
    
    # Create essential directories only
    create_essential_directories(paths)
    
    # Load base GMPE configuration
    try:
        from configs.config import load_gmpe_config
        base_config = load_gmpe_config()
    except Exception as e:
        logger.error(f"Failed to load base GMPE config: {e}")
        raise
    
    # OPTIMIZED CONFIGURATION FOR LARGE GRIDS
    optimized_config = {
        "region": {
            "name": "los_angeles",
            "description": "Los Angeles - Optimized for Large Grid Processing",
            "polygon": region_bounds["polygon"],
            "bounds": {
                "min_lat": region_bounds["min_lat"],
                "max_lat": region_bounds["max_lat"],
                "min_lon": region_bounds["min_lon"],
                "max_lon": region_bounds["max_lon"]
            },
            "mode": mode
        },
        
        "paths": paths,
        
        "site": {
            # CRITICAL: Grid mode for large site counts
            "grid_mode": True,
            
            # Grid configuration for 868 sites (adjust spacing as needed)
            "grid_lat_min": region_bounds["min_lat"],
            "grid_lat_max": region_bounds["max_lat"],
            "grid_lon_min": region_bounds["min_lon"],
            "grid_lon_max": region_bounds["max_lon"],
            "grid_lat_spacing": 0.1,  # ~5.5km spacing = ~868 sites
            "grid_lon_spacing": 0.1,
            
            # GMPE settings from base config
            "vs30": base_config["site_defaults"]["vs30"],
            "z1p0": base_config["site_defaults"]["z1p0"],
            "gmpe_model": base_config["site_defaults"]["gmpe_model"],
            "default_period": base_config["site_defaults"]["default_period"],
            
            # Optimized filtering for large grids
            "max_distance_km": 300.0,  # Reasonable distance limit
        },
        
        "simulation": {
            # OPTIMIZED: Parallel processing settings
            "max_workers": min(32, os.cpu_count()),  # Cap at 32 for stability
            "batch_size": 50,  # Larger batches for grid efficiency
            "memory_limit_gb": 16,  # Prevent memory explosion
            "timeout_minutes": 300,  # 5 hour timeout per batch
            
            # Filtering settings
            "filtering": {
                "max_distance_km": 300.0,
                "early_distance_filter": True,  # Filter by distance first
            }
        },
        
        "output_settings": {
            # CRITICAL: Minimal output for large grids
            "full_output": False,  # No individual event lists
            "min_ground_motion": 0.01,  # Higher threshold = fewer events
            
            # Probability settings (keep minimal)
            "probability_type": "annual",
            "probabilities": [(50, 0.02), (50, 0.1)],  # Just 2% and 10% annual
            
            # Visualization settings for large grids
            "plot_hazard_curves": False,  # Skip individual curves for 868 sites
            "max_curves": 0,  # No individual curves
            "export_gis_csv": True,  # Essential for grid visualization
            "create_hazard_map": True,  # Essential for grid visualization
            "generate_summary_plots": False,  # Skip for large grids
            
            # File settings
            "plot_format": "png",
            "plot_dpi": 150,  # Lower DPI for faster processing
            "compression": "gzip",  # Compress output files
            "precision": 4,  # Reduce precision for smaller files
        },
        
        "regional_analysis": {
            "enabled": True,
            "multi_window_comparison": True,
            "statistical_summary": True,
            "performance_metrics": True,
            "polygon_filtering": True,
        },

        # OPTIMIZED: Resource limits for large grids
        "resource_limits": {
            "max_memory_gb": 16,
            "batch_size": 50,  # Larger batches
            "max_events_per_site": 10000,  # Reasonable limit
            "max_synthetic_events": 5000,  # Reduced for memory
        },
        
        # OPTIMIZED: Performance settings
        "performance": {
            "use_memory_management": True,
            "max_distance_km": 300.0,
            "vectorized_calculations": True,  # Enable vectorization
            "early_filtering": True,  # Filter before GM calculation
            "minimal_logging": True,  # Reduce log output
        },
        
        # Synthetic catalog settings
        "synduration": 50000,  # Standard duration
        "seed": 42,  # Reproducible results
        
        # Major cities for reference (optional)
        "major_cities": {
            'Los Angeles': (-118.25, 34.05), 
            'Long Beach': (-118.19, 33.77), 
            'Anaheim': (-117.91, 33.84),
            'Santa Ana': (-117.87, 33.75),
            'Riverside': (-117.4, 33.95),
        }
    }
    
    # Merge with base configuration
    final_config = base_config.copy()
    final_config.update(optimized_config)
    
    # Final optimization check
    if final_config['site']['grid_mode']:
        # Calculate expected number of sites
        lat_range = final_config['site']['grid_lat_max'] - final_config['site']['grid_lat_min']
        lon_range = final_config['site']['grid_lon_max'] - final_config['site']['grid_lon_min']
        lat_spacing = final_config['site']['grid_lat_spacing']
        lon_spacing = final_config['site']['grid_lon_spacing']
        
        num_lat_points = int(lat_range / lat_spacing) + 1
        num_lon_points = int(lon_range / lon_spacing) + 1
        expected_sites = num_lat_points * num_lon_points
        
        logger.info(f"Optimized config loaded for {expected_sites} grid sites")
        logger.info(f"Grid: {num_lat_points} × {num_lon_points} with {lat_spacing:.3f}° spacing")
        
        # Adjust settings if too many sites
        if expected_sites > 2000:
            logger.warning(f"Large grid detected ({expected_sites} sites). Consider:")
            logger.warning(f"  - Increasing grid spacing to 0.1°")
            logger.warning(f"  - Increasing min_ground_motion threshold")
            logger.warning(f"  - Reducing max_distance_km")
    
    return final_config

def load_config(mode: str = None):
    """Main entry point - load optimized configuration"""
    if mode is None:
        mode = "sequential"
    
    return load_optimized_config(mode)

def load_config_for_mode(mode: str):
    """Load config for specific mode"""
    return load_optimized_config(mode)

def switch_to_random_mode(config: dict):
    """Switch configuration to random mode"""
    region_name = config['region']['name'] 
    random_paths = generate_optimized_paths(region_name, 'random')
    create_essential_directories(random_paths)
    
    config['paths'] = random_paths
    config['region']['mode'] = 'random'
    
    logger.info(f"Switched to random mode with optimized settings")
    return config

# USAGE EXAMPLE FOR LARGE GRIDS
def example_usage():
    """Example of how to use this optimized configuration"""
    
    print("Optimized Los Angeles Configuration for Large Grids")
    print("="*60)
    
    # Load configuration
    config = load_config("sequential")
    
    print(f"Region: {config['region']['name']}")
    print(f"Mode: {config['region']['mode']}")
    print(f"Grid mode: {config['site']['grid_mode']}")
    print(f"Grid spacing: {config['site']['grid_lat_spacing']}°")
    print(f"Output minimal: {not config['output_settings']['full_output']}")
    print(f"Max workers: {config['simulation']['max_workers']}")
    print(f"Batch size: {config['simulation']['batch_size']}")
    
    # Calculate expected sites
    lat_range = config['site']['grid_lat_max'] - config['site']['grid_lat_min']
    lon_range = config['site']['grid_lon_max'] - config['site']['grid_lon_min']
    lat_spacing = config['site']['grid_lat_spacing']
    lon_spacing = config['site']['grid_lon_spacing']
    
    num_lat = int(lat_range / lat_spacing) + 1
    num_lon = int(lon_range / lon_spacing) + 1
    total_sites = num_lat * num_lon
    
    print(f"Expected sites: {total_sites:,} ({num_lat} × {num_lon})")
    print(f"Estimated runtime: {total_sites * 0.05:.1f} minutes")
    print(f"Estimated memory: {1 + total_sites * 0.5 / 1000:.1f} GB")
    
    print("\nTo run:")
    print("python gm_simulator_optimized.py \\")
    print("  --region los_angeles \\")
    print("  --windows-dir /path/to/windows \\")
    print("  --output-dir /path/to/output \\")
    print("  --config configs/los_angeles_optimized_config.py")

if __name__ == "__main__":
    example_usage()