"""
Full California Region Configuration for Ground Motion Simulation

This configuration inherits base GMPE settings from config.py
and uses the EXACT regional polygon for Full California.
"""

import os
from pathlib import Path
import numpy as np

# Import base GMPE configuration
from configs.config import load_gmpe_config

# Get environment variables (set by job scripts or .bashrc)
RSQSIM_HOME = os.environ.get('RSQSIM_HOME', str(Path.home() / 'rsqsim_mc'))
RSQSIM_WORK_DIR = os.environ.get('RSQSIM_WORK_DIR', '/scratch/olawoyiv/rsqsim_data')
RSQSIM_PROJECT_DIR = os.environ.get('RSQSIM_PROJECT_DIR', '/projects/ebelseismo/olawoyiv')

# EXACT Full California polygon (from your los_angeles_config.py)
REGIONAL_POLYGONS = {
    'Full_california': {
        'region_polygon': [
            [43.0, -125.2], [43.0, -119.0], [39.4, -119.0], [35.7, -114.0],
            [34.3, -113.1], [32.9, -113.5], [32.2, -113.6], [31.7, -114.5],
            [31.5, -117.1], [31.9, -117.9], [32.8, -118.4], [33.7, -121.0],
            [34.2, -121.6], [37.7, -123.8], [40.2, -125.4], [40.5, -125.4],
            [43.0, -125.2]
        ]
    }
}

def get_region_bounds(region_name):
    """
    Calculate bounding box from the exact regional polygon
    
    Args:
        region_name: Name of the region ('Full_california')
        
    Returns:
        dict: Bounding box coordinates
    """
    if region_name not in REGIONAL_POLYGONS:
        raise ValueError(f"Region '{region_name}' not found. Available regions: {list(REGIONAL_POLYGONS.keys())}")
    
    polygon = REGIONAL_POLYGONS[region_name]['region_polygon']
    lats = [point[0] for point in polygon]
    lons = [point[1] for point in polygon]
    
    return {
        "min_lat": min(lats),
        "max_lat": max(lats),
        "min_lon": min(lons),
        "max_lon": max(lons),
        "polygon": polygon
    }

def load_config():
    """
    Load Full California-specific configuration
    
    Returns:
        dict: Complete configuration for Full California region
    """
    
    # Get base GMPE configuration
    base_config = load_gmpe_config()
    
    # Get the EXACT bounds used for catalog filtering
    region_bounds = get_region_bounds('Full_california')
    
    # Calculate grid parameters that fit within the actual region bounds
    lat_range = region_bounds["max_lat"] - region_bounds["min_lat"]  # ~11.5 degrees
    lon_range = region_bounds["max_lon"] - region_bounds["min_lon"]  # ~11.1 degrees
    
    # Use reasonable grid spacing for large region (about 0.1° ≈ 10km spacing)
    # For testing, we'll use larger spacing to keep site count manageable
    grid_spacing_lat = 0.2  # ~20km spacing for testing
    grid_spacing_lon = 0.2  # ~20km spacing for testing
    
    num_grid_lat = max(5, int(np.ceil(lat_range / grid_spacing_lat)))
    num_grid_lon = max(5, int(np.ceil(lon_range / grid_spacing_lon)))
    
    print(f"Full California grid will be {num_grid_lat} × {num_grid_lon} = {num_grid_lat * num_grid_lon} sites")
    
    # Full California-specific configuration
    ca_config = {
        "region": {
            "name": "Full_california",
            "description": "Full California region using exact polygon from catalog filtering",
            "polygon": region_bounds["polygon"],
            "bounds": {
                "min_lat": region_bounds["min_lat"],  # ~31.5
                "max_lat": region_bounds["max_lat"],  # ~43.0
                "min_lon": region_bounds["min_lon"],  # ~-125.2
                "max_lon": region_bounds["max_lon"]   # ~-113.1
            }
        },
        
        "paths": {
            # Code and small files stay in home
            "project_root": Path(RSQSIM_HOME),
            "gmpe_config": str(Path(RSQSIM_HOME) / "configs" / "config.py"),
            
            # Large data files use scratch for processing
            "windows_dir": f"{RSQSIM_WORK_DIR}/data/Catalog_4983/windows/Full_california/sequential",
            "output_dir": f"{RSQSIM_WORK_DIR}/output/Full_california/sequential", 
            "log_dir": f"{RSQSIM_WORK_DIR}/logs/Full_california",
            
            # Long-term storage in project directory
            "backup_dir": f"{RSQSIM_PROJECT_DIR}/rsqsim_results/Full_california",
            "archive_dir": f"{RSQSIM_PROJECT_DIR}/rsqsim_archive/Full_california",
        },
        
        "site": {
            # Site grid that fits within the EXACT regional bounds
            "grid_mode": True,           
            "num_grid_lat": num_grid_lat,        # Calculated to fit region
            "num_grid_lon": num_grid_lon,        # Calculated to fit region
            "grid_lat_min": region_bounds["min_lat"],    # EXACT region minimum
            "grid_lon_min": region_bounds["min_lon"],    # EXACT region minimum
            "grid_lat_spacing": grid_spacing_lat,        # ~20km spacing
            "grid_lon_spacing": grid_spacing_lon,        # ~20km spacing
            
            # Individual sites (used if grid_mode = False)
            "sites": [(34.05, -118.25), (37.77, -122.42), (32.72, -117.16)],  # LA, SF, San Diego
            
            # Site parameters - ADJUSTED for large catalog
            "vs30": base_config["site_defaults"]["vs30"],
            "z1p0": base_config["site_defaults"]["z1p0"],
            "gmpe_model": base_config["site_defaults"]["gmpe_model"],
            "default_period": base_config["site_defaults"]["default_period"],
            "include_scatter": base_config["site_defaults"]["include_scatter"],
            "scatter_std_dev": base_config["site_defaults"]["scatter_std_dev"],
            
            # ADJUSTED thresholds for large catalog
            "min_ground_motion": 0.001,  # Start with higher threshold since we have 5.6M events
            "max_distance_km": 300.0     # Keep 300km max distance
        }
    }
    
    # Merge base config with CA-specific config
    final_config = base_config.copy()
    final_config.update(ca_config)
    
    # Resolve paths to absolute
    project_root = final_config["paths"]["project_root"]
    for key, path in final_config["paths"].items():
        if key != "project_root" and isinstance(path, str) and not os.path.isabs(path):
            final_config["paths"][key] = str(project_root / path)
    
    # Create necessary directories
    os.makedirs(final_config["paths"]["output_dir"], exist_ok=True)
    os.makedirs(final_config["paths"]["log_dir"], exist_ok=True)
    
    return final_config

def point_in_polygon(lat, lon, polygon):
    """
    Check if a point is inside the regional polygon using ray casting algorithm
    """
    x, y = lon, lat
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0][1], polygon[0][0]  # lon, lat
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n][1], polygon[i % n][0]  # lon, lat
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside

def filter_sites_to_region(sites, region_name='Full_california'):
    """
    Filter sites to only include those within the exact regional polygon
    """
    polygon = REGIONAL_POLYGONS[region_name]['region_polygon']
    filtered_sites = []
    
    for lat, lon in sites:
        if point_in_polygon(lat, lon, polygon):
            filtered_sites.append((lat, lon))
    
    return filtered_sites

def validate_config():
    """
    Validate the Full California configuration
    """
    config = load_config()
    issues = []
    
    # Check that we're using the correct regional polygon
    if config["region"]["name"] != "Full_california":
        issues.append("Region name mismatch")
    
    if "polygon" not in config["region"]:
        issues.append("Missing regional polygon definition")
    
    # Check site parameters
    site = config["site"]
    bounds = config["region"]["bounds"]
    
    # Verify grid fits within regional bounds
    grid_lat_max = site["grid_lat_min"] + (site["num_grid_lat"]-1) * site["grid_lat_spacing"]
    grid_lon_max = site["grid_lon_min"] + (site["num_grid_lon"]-1) * site["grid_lon_spacing"]
    
    if grid_lat_max > bounds["max_lat"] or site["grid_lat_min"] < bounds["min_lat"]:
        issues.append("Grid latitude range exceeds regional bounds")
    if grid_lon_max > bounds["max_lon"] or site["grid_lon_min"] < bounds["min_lon"]:
        issues.append("Grid longitude range exceeds regional bounds")
    
    # Check GMPE files
    model = config["site"]["gmpe_model"]
    if model in config["gmpe_files"]:
        gmpe_file = config["gmpe_files"][model]
        if gmpe_file and not os.path.exists(gmpe_file):
            issues.append(f"GMPE coefficient file missing: {gmpe_file}")
    
    if issues:
        print("Full California configuration validation issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    
    print("✅ Full California configuration validation passed")
    print(f"✅ Using EXACT regional polygon with {len(config['region']['polygon'])} vertices")
    return True

# For testing/debugging
if __name__ == "__main__":
    print("Full California Configuration Test")
    print("=" * 50)
    
    # Load and validate configuration
    config = load_config()
    is_valid = validate_config()
    
    print(f"\nConfiguration valid: {is_valid}")
    print(f"Region: {config['region']['name']}")
    print(f"Regional polygon vertices: {len(config['region']['polygon'])}")
    
    # Show regional bounds
    bounds = config["region"]["bounds"]
    print(f"Regional bounds:")
    print(f"  Latitude: {bounds['min_lat']:.1f} to {bounds['max_lat']:.1f} ({bounds['max_lat'] - bounds['min_lat']:.1f}° range)")
    print(f"  Longitude: {bounds['min_lon']:.1f} to {bounds['max_lon']:.1f} ({bounds['max_lon'] - bounds['min_lon']:.1f}° range)")
    
    if config['site']['grid_mode']:
        num_sites = config['site']['num_grid_lat'] * config['site']['num_grid_lon']
        print(f"\nGrid configuration:")
        print(f"  Grid size: {config['site']['num_grid_lat']} × {config['site']['num_grid_lon']} = {num_sites} sites")
        print(f"  Grid spacing: {config['site']['grid_lat_spacing']:.2f}° (~{config['site']['grid_lat_spacing']*111:.0f} km)")
        print(f"  Threshold: {config['site']['min_ground_motion']}g")
    
    print(f"\nGMPE model: {config['site']['gmpe_model']}")
    print(f"Output directory: {config['paths']['output_dir']}")
    print(f"Expected catalog size: ~5.6M events")