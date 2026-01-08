"""
Enhanced Full California grid Site Configuration with Visualization Options
"""

import os
from pathlib import Path

# Import base GMPE configuration
from configs.config import load_gmpe_config

# Get environment variables
RSQSIM_HOME = os.environ.get('RSQSIM_HOME', str(Path.home() / 'rsqsim_mc'))
RSQSIM_WORK_DIR = os.environ.get('RSQSIM_WORK_DIR', '/scratch/olawoyiv/rsqsim_data')
RSQSIM_PROJECT_DIR = os.environ.get('RSQSIM_PROJECT_DIR', '/projects/ebelseismo/olawoyiv')

# 1. ADD THE REGIONAL_POLYGONS DICTIONARY (after imports, before load_config())
REGIONAL_POLYGONS = {
    'los_angeles': {
        'region_polygon': [
            [33.043, -116.303], [32.498, -117.104], [33.577, -117.971],
            [33.649, -118.429], [33.973, -118.658], [34.328, -119.974],
            [34.619, -119.974], [34.557, -117.132], [33.043, -116.303]
        ]
    },
    'san_francisco': {
        'region_polygon': [
            [36.890, -121.148], [36.546, -121.422], [36.535, -122.099],
            [38.489, -123.207], [38.652, -123.090], [38.692, -122.243],
            [38.987, -121.344], [38.376, -121.005], [37.503, -121.461],
            [36.859, -121.135], [36.890, -121.148]
        ]
    },
    'central_coast': {
        'region_polygon': [
            [34.406, -119.970], [34.320, -121.127], [36.519, -122.091],
            [36.547, -121.425], [36.828, -121.110], [36.814, -120.724],
            [36.463, -120.636], [35.982, -120.111], [35.212, -119.848],
            [34.378, -119.970], [34.406, -119.970]
        ]
    },
    'mid_angeles': {
        'region_polygon': [
            [35.220, -119.839], [35.988, -120.085], [36.485, -120.619],
            [37.044, -120.619], [37.241, -119.223], [37.794, -118.565],
            [37.665, -117.703], [36.452, -117.662], [36.320, -118.606],
            [35.253, -118.647], [35.220, -119.839]
        ]
    },
    'mojave': {
        'region_polygon': [
            [33.022, -114.677], [33.073, -116.287], [34.551, -117.131],
            [34.627, -119.952], [35.193, -119.860], [35.230, -118.633],
            [36.313, -118.572], [36.436, -117.652], [33.022, -114.677]
        ]
    },
    'north_cal': {
        'region_polygon': [
            [38.507, -123.222], [40.232, -124.583], [41.862, -124.598],
            [41.851, -122.276], [40.952, -122.276], [40.188, -121.536],
            [39.060, -121.196], [38.703, -122.246], [38.668, -123.104],
            [38.507, -123.222]
        ]
    },
    'north_east': {
        'region_polygon': [
            [40.640, -119.664], [40.670, -120.170], [40.779, -120.400],
            [40.870, -120.490], [41.100, -120.700], [41.346, -120.866],
            [41.716, -120.979], [41.836, -120.975], [42.131, -120.520],
            [42.400, -120.000], [42.318, -119.755], [42.084, -119.161],
            [41.918, -119.035], [40.924, -119.049], [40.776, -119.114],
            [40.640, -119.664]
        ]
    },
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

# 2. ADD THE get_region_bounds() FUNCTION (before load_config())
def get_region_bounds(region_name):
    """
    Calculate bounding box from the exact regional polygon used for catalog filtering
    
    Args:
        region_name: Name of the region (e.g., 'Full_california')
        
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
    Load Enhanced Single Site Full California configuration with visualization options
    
    Returns:
        dict: Complete configuration for single site test with visualizations
    """
    
    # Get base GMPE configuration
    base_config = load_gmpe_config()
    
    # Get the EXACT bounds used for catalog filtering
    region_bounds = get_region_bounds('Full_california')
    
    # Enhanced Full California configuration
    ca_config = {
        "region": {
            "name": "Full_california",
            "description": "Full California catalog (397K events) using exact polygon + Visualizations",
            "polygon": region_bounds["polygon"],  # EXACT polygon used for filtering
            "bounds": {
                "min_lat": region_bounds["min_lat"],
                "max_lat": region_bounds["max_lat"],
                "min_lon": region_bounds["min_lon"],
                "max_lon": region_bounds["max_lon"]
            }
        },
        
        "paths": {
            "project_root": Path(RSQSIM_HOME),
            "gmpe_config": str(Path(RSQSIM_HOME) / "configs" / "config.py"),
            
            # Full California windows directory
            "windows_dir": f"{RSQSIM_WORK_DIR}/data/Catalog_4983/windows/Full_california/sequential",
            "output_dir": f"{RSQSIM_WORK_DIR}/output/Full_california/single_site_test", 
            "log_dir": f"{RSQSIM_WORK_DIR}/logs/Full_california/single_site",
            
            # Backup directories
            "backup_dir": f"{RSQSIM_PROJECT_DIR}/rsqsim_results/Full_california/single_site",
            "archive_dir": f"{RSQSIM_PROJECT_DIR}/rsqsim_archive/Full_california/single_site",
        },
        
        "site": {
            "grid_mode": True,
            
            # California bounds with desired spacing
            "grid_lat_min": region_bounds["min_lat"],    
            "grid_lat_max": region_bounds["max_lat"],    
            "grid_lon_min": region_bounds["min_lon"],    
            "grid_lon_max": region_bounds["max_lon"], 
            
            # Choose your resolution:
            # For testing (coarse):
            "grid_lat_spacing": 0.1,   # ~55km, creates ~483 sites
            "grid_lon_spacing": 0.1,
            
            # For production (medium):  
            # "grid_lat_spacing": 0.2,   # ~22km, creates ~2,856 sites
            # "grid_lon_spacing": 0.2,
            
            # For high-res (fine):
            # "grid_lat_spacing": 0.1,   # ~11km, creates ~11,211 sites  
            # "grid_lon_spacing": 0.1,

            "sites": [(34.05, -118.25)],
            
            # Site parameters
            "vs30": base_config["site_defaults"]["vs30"],
            "z1p0": base_config["site_defaults"]["z1p0"],
            "gmpe_model": base_config["site_defaults"]["gmpe_model"],
            "default_period": base_config["site_defaults"]["default_period"],
            "include_scatter": base_config["site_defaults"]["include_scatter"],
            "scatter_std_dev": base_config["site_defaults"]["scatter_std_dev"],
            
            # Ground motion settings - optimized for large catalog
            "max_distance_km": 300.0     # 300km radius around LA
        },
        
        # ENHANCED: Visualization and Output Settings
        "output_settings": {
            "full_output": False,         # True = includes synthetic event list (large files)
            "min_ground_motion": 0.05,  # Same as site setting
            "probability_type": "annual",  # "annual" or "nonexceedence"
            "probabilities": [(50, 0.02), (50, 0.1)],  # (time_years, probability) for nonexceedence
            
            # NEW: Visualization Options (from new_Ran.py functions)
            "plot_hazard_curves": True,    # Generate hazard curve plots
            "max_curves": 10,              # Maximum number of hazard curves to plot
            "export_gis_csv": True,        # Export data for GIS mapping
            "create_hazard_map": True,     # Generate hazard maps (if multiple sites)
            "generate_summary_plots": True, # Generate summary statistics plots
            
            # Plot customization
            "plot_format": "png",          # "png", "pdf", "svg"
            "plot_dpi": 300,              # Resolution for raster plots
            "plot_style": "seaborn",      # Matplotlib style
            "color_scheme": "viridis",    # Color scheme for maps
            
            # Output file options
            "save_statistics": True,       # Save detailed statistics
            "compression": "gzip",          # "gzip" for compressed output
            "precision": 6,               # Decimal precision for output
        },
        
        # NEW: Regional Analysis Options
        "regional_analysis": {
            "enabled": True,              # Enable regional summary generation
            "multi_window_comparison": True,  # Compare across time windows
            "statistical_summary": True,  # Generate statistical summaries
            "performance_metrics": True,  # Track processing performance
            "polygon_filtering": True,
        },
        
        # Major cities for map annotation (if creating maps)
        "major_cities": {
            "Los Angeles": (-118.25, 34.05),
            "San Francisco": (-122.42, 37.77),
            "San Diego": (-117.16, 32.72),
            "Sacramento": (-121.49, 38.56),
            "Fresno": (-119.77, 36.75),
            "Oakland": (-122.27, 37.80),
            "Long Beach": (-118.19, 33.77),
            "Bakersfield": (-119.02, 35.37)
        }
    }
    
    # Merge base config with CA-specific config
    final_config = base_config.copy()
    final_config.update(ca_config)
    
    # Create necessary directories
    os.makedirs(final_config["paths"]["output_dir"], exist_ok=True)
    os.makedirs(final_config["paths"]["log_dir"], exist_ok=True)
    
    return final_config

def point_in_polygon(lat, lon, polygon):
    """
    Check if a point is inside the regional polygon using ray casting algorithm
    
    Args:
        lat, lon: Point coordinates
        polygon: List of [lat, lon] polygon vertices
        
    Returns:
        bool: True if point is inside polygon
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

# 4. ADD THE filter_sites_to_region() FUNCTION
def filter_sites_to_region(sites, region_name='Full_california'):
    """
    Filter sites to only include those within the exact regional polygon
    
    Args:
        sites: List of (lat, lon) tuples
        region_name: Name of the region
        
    Returns:
        List of sites within the regional polygon
    """
    polygon = REGIONAL_POLYGONS[region_name]['region_polygon']
    filtered_sites = []
    
    for lat, lon in sites:
        if point_in_polygon(lat, lon, polygon):
            filtered_sites.append((lat, lon))
    
    return filtered_sites


def load_multi_site_config():
    """
    Load configuration for multi-site testing (for comparison)
    """
    config = load_config()
    
    # Modify for multi-site testing
    config["site"]["grid_mode"] = True
    config["site"]["grid_lat_min"] = 33.5
    config["site"]["grid_lat_max"] = 34.5
    config["site"]["grid_lon_min"] = -119.0
    config["site"]["grid_lon_max"] = -117.5
    config["site"]["grid_lat_spacing"] = 0.25  # 4 points
    config["site"]["grid_lon_spacing"] = 0.25  # 6 points
    # This creates a 4x6 = 24 site grid
    
    return config

def validate_config():
    """Validate the enhanced configuration"""
    config = load_config()
    issues = []
    
    # Check visualization dependencies
    try:
        import matplotlib.pyplot as plt
        print("✅ Matplotlib available for plotting")
    except ImportError:
        issues.append("Matplotlib not available - plotting features will be disabled")
    
    try:
        import pygmt
        print("✅ PyGMT available for mapping")
    except ImportError:
        issues.append("PyGMT not available - advanced mapping features will be disabled")

    # Check polygon definition
    if "polygon" not in config["region"]:
        issues.append("Missing regional polygon definition")
    else:
        polygon = config["region"]["polygon"]
        if len(polygon) < 3:
            issues.append("Polygon must have at least 3 vertices")
        print(f"✅ Using EXACT regional polygon with {len(polygon)} vertices")
    
    # Check single site mode
    if config["site"]["grid_mode"]:
        issues.append("Should use grid_mode=False for single site test")
    
    if not config["site"]["sites"]:
        issues.append("No sites specified for single site test")
    
    # Check windows directory exists
    windows_dir = config["paths"]["windows_dir"]
    if not os.path.exists(windows_dir):
        issues.append(f"Windows directory not found: {windows_dir}")
    else:
        # Check for CSV files
        import glob
        csv_files = glob.glob(os.path.join(windows_dir, "*.csv"))
        if not csv_files:
            issues.append(f"No CSV files found in: {windows_dir}")
        else:
            print(f"✅ Found {len(csv_files)} window files")
    
    # Check GMPE files
    model = config["site"]["gmpe_model"]
    if model in config["gmpe_files"]:
        gmpe_file = config["gmpe_files"][model]
        if gmpe_file and not os.path.exists(gmpe_file):
            issues.append(f"GMPE coefficient file missing: {gmpe_file}")
    
    # Check output settings
    if config["output_settings"]["plot_hazard_curves"] and len(config["site"]["sites"]) == 0:
        issues.append("Cannot plot hazard curves without sites")
    
    if issues:
        print("Enhanced configuration validation issues:")
        for issue in issues:
            print(f"  ⚠️  {issue}")
        return False
    
    print("✅ Enhanced configuration validation passed")
    return True

def generate_sites_in_polygon(region_name, spacing=0.1):
    """
    Generate sites directly within polygon - memory efficient
    
    Args:
        region_name: Name of the region (e.g., 'los_angeles')
        spacing: Grid spacing in degrees
        
    Returns:
        List of (lat, lon) tuples for sites within the polygon
    """
    bounds = get_region_bounds(region_name)
    polygon = REGIONAL_POLYGONS[region_name]['region_polygon']
    
    sites = []
    total_tested = 0
    
    # Generate grid points and test each one
    lat = bounds['min_lat']
    while lat <= bounds['max_lat']:
        lon = bounds['min_lon']
        while lon <= bounds['max_lon']:
            total_tested += 1
            if point_in_polygon(lat, lon, polygon):
                sites.append((lat, lon))
            lon += spacing
        lat += spacing
    
    # Calculate efficiency
    efficiency = (len(sites) / total_tested) * 100 if total_tested > 0 else 0
    
    print(f"Site generation for {region_name}:")
    print(f"  Grid spacing: {spacing:.3f}° (~{spacing*111:.1f} km)")
    print(f"  Total grid points tested: {total_tested}")
    print(f"  Sites within polygon: {len(sites)}")
    print(f"  Grid efficiency: {efficiency:.1f}%")
    
    return sites

def get_optimized_grid_sites(config):
    """
    Get grid sites using efficient polygon-aware generation
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of (lat, lon) tuples for grid sites
    """
    if not config['site']['grid_mode']:
        return config['site']['sites']
    
    region_name = config['region']['name']
    spacing = config['site']['grid_lat_spacing']  # Assume square grid
    
    return generate_sites_in_polygon(region_name, spacing)

def estimate_grid_efficiency(region_name, spacing=0.1):
    """
    Estimate how efficient the grid generation will be
    
    Args:
        region_name: Name of the region
        spacing: Grid spacing in degrees
        
    Returns:
        Dictionary with efficiency metrics
    """
    bounds = get_region_bounds(region_name)
    
    # Calculate rectangular grid size
    lat_range = bounds['max_lat'] - bounds['min_lat']
    lon_range = bounds['max_lon'] - bounds['min_lon']
    
    num_lat_points = int(lat_range / spacing) + 1
    num_lon_points = int(lon_range / spacing) + 1
    total_rectangle_sites = num_lat_points * num_lon_points
    
    # Estimate polygon area (rough approximation using shoelace formula)
    polygon = bounds['polygon']
    area = 0.0
    n = len(polygon)
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i][0] * polygon[j][1]
        area -= polygon[j][0] * polygon[i][1]
    polygon_area = abs(area) / 2.0
    
    # Rectangle area
    rectangle_area = lat_range * lon_range
    
    # Estimated efficiency
    estimated_efficiency = (polygon_area / rectangle_area) * 100
    estimated_sites_in_polygon = int(total_rectangle_sites * estimated_efficiency / 100)
    
    return {
        'total_rectangle_sites': total_rectangle_sites,
        'estimated_sites_in_polygon': estimated_sites_in_polygon,
        'estimated_efficiency': estimated_efficiency,
        'polygon_area_deg2': polygon_area,
        'rectangle_area_deg2': rectangle_area
    }

# For testing/debugging
if __name__ == "__main__":
    print("Enhanced Los Angeles Configuration with Efficient Site Generation")
    print("=" * 80)
    
    # Show available regions
    print(f"Available regions: {list(REGIONAL_POLYGONS.keys())}")
    
    # Load and validate configuration
    config = load_config()
    is_valid = validate_config()
    
    print(f"\nConfiguration valid: {is_valid}")
    print(f"Region: {config['region']['name']}")
    print(f"Regional polygon vertices: {len(config['region']['polygon'])}")
    
    # Test efficient site generation
    if config['site']['grid_mode']:
        print("\n🚀 TESTING EFFICIENT SITE GENERATION:")
        
        # Get efficiency estimate first
        spacing = config['site']['grid_lat_spacing']
        efficiency_est = estimate_grid_efficiency(config['region']['name'], spacing)
        
        print(f"\n📊 EFFICIENCY ESTIMATE:")
        print(f"  Rectangle grid sites: {efficiency_est['total_rectangle_sites']}")
        print(f"  Estimated polygon sites: {efficiency_est['estimated_sites_in_polygon']}")
        print(f"  Estimated efficiency: {efficiency_est['estimated_efficiency']:.1f}%")
        
        # Generate actual sites efficiently
        print(f"\n⚡ GENERATING SITES:")
        sites = get_optimized_grid_sites(config)
        
        print(f"\n✅ RESULTS:")
        print(f"  Final site count: {len(sites)}")
        print(f"  Grid spacing: {spacing:.3f}° (~{spacing*111:.1f} km)")
        
    else:
        print(f"Individual sites: {len(config['site']['sites'])}")
    
    print(f"\nVisualizations enabled: {config['output_settings']['plot_hazard_curves']}")
    print(f"Output directory: {config['paths']['output_dir']}")
    
    if is_valid:
        print(f"\n🚀 READY TO RUN:")
        print(f"python scripts/run_mc.py --region Full_california --mode sequential --log-level INFO")
    else:
        print(f"\n❌ Fix validation issues first")