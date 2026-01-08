"""
Enhanced North Cal Configuration with Visualization Options

Northern California region using exact polygon + Visualizations
"""

import os
from pathlib import Path
import numpy as np
import logging

# Import base GMPE configuration
from configs.config import load_gmpe_config

# Setup logging
logger = logging.getLogger(__name__)

# Get environment variables
RSQSIM_HOME = os.environ.get('RSQSIM_HOME', str(Path.home() / 'rsqsim_mc'))
RSQSIM_WORK_DIR = os.environ.get('RSQSIM_WORK_DIR', '/scratch/olawoyiv/rsqsim_data')
RSQSIM_PROJECT_DIR = os.environ.get('RSQSIM_PROJECT_DIR', '/projects/ebelseismo/olawoyiv')

# EXACT regional polygons used for catalog filtering
REGIONAL_POLYGONS = {
    'north_cal': {
        'region_polygon': [
            [38.507, -123.222], [40.232, -124.583], [41.862, -124.598],
            [41.851, -122.276], [40.952, -122.276], [40.188, -121.536],
            [39.060, -121.196], [38.703, -122.246], [38.668, -123.104],
            [38.507, -123.222]
        ]
    }
}

def get_region_bounds(region_name):
    """Calculate bounding box from the exact regional polygon"""
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

def generate_standardized_paths(region_name: str, mode: str = "sequential"):
    """Generate standardized paths for any region configuration"""
    
    base_output = os.path.join(RSQSIM_WORK_DIR, 'output')
    
    return {
        "project_root": Path(RSQSIM_HOME),
        "gmpe_config": str(Path(RSQSIM_HOME) / "configs" / "config.py"),
        
        # STANDARDIZED DATA DIRECTORIES
        "windows_dir": f"{RSQSIM_WORK_DIR}/data/Catalog_4983/windows/{region_name}/{mode}",
        
        # FIXED: Proper output directory structure
        "output_dir": f"{base_output}/{region_name}/{mode}",
        "log_dir": f"{RSQSIM_WORK_DIR}/logs/{region_name}/{mode}",
        
        # STANDARDIZED BACKUP DIRECTORIES
        "backup_dir": f"{RSQSIM_PROJECT_DIR}/rsqsim_results/{region_name}/{mode}",
        "archive_dir": f"{RSQSIM_PROJECT_DIR}/rsqsim_archive/{region_name}/{mode}",
        
        # STANDARDIZED VISUALIZATION DIRECTORIES
        "viz_dir": f"{base_output}/{region_name}/{mode}/visualizations",
        "maps_dir": f"{base_output}/{region_name}/{mode}/visualizations/maps",
        "plots_dir": f"{base_output}/{region_name}/{mode}/visualizations/plots",
        "gis_dir": f"{base_output}/{region_name}/{mode}/visualizations/gis",
        
        # REGIONAL SUMMARY DIRECTORY
        "regional_summary_dir": f"{base_output}/{region_name}/{mode}/regional_summary"
    }

def create_standardized_directories(paths: dict, force_create: bool = True):
    """
    Create all standardized directories with comprehensive error handling
    
    Args:
        paths: Dictionary of paths to create
        force_create: If True, create directories even if they exist
    
    Returns:
        tuple: (success_count, failed_directories)
    """
    
    # Essential directories that MUST be created
    essential_dirs = [
        paths.get("output_dir"),
        paths.get("log_dir"), 
        paths.get("viz_dir"),
        paths.get("maps_dir"),
        paths.get("plots_dir"),
        paths.get("gis_dir"),
        paths.get("regional_summary_dir")
    ]
    
    # Optional directories (create if possible, but don't fail if not)
    optional_dirs = [
        paths.get("backup_dir"),
        paths.get("archive_dir")
    ]
    
    # Filter out None values
    essential_dirs = [d for d in essential_dirs if d is not None]
    optional_dirs = [d for d in optional_dirs if d is not None]
    
    created_dirs = []
    failed_dirs = []
    
    logger.info(f"🏗️  Creating {len(essential_dirs)} essential directories...")
    
    # Create essential directories
    for dir_path in essential_dirs:
        try:
            Path(dir_path).mkdir(parents=True, exist_ok=True, mode=0o755)
            created_dirs.append(dir_path)
            logger.debug(f"✅ Created essential: {dir_path}")
        except PermissionError as e:
            failed_dirs.append((dir_path, f"Permission denied: {e}"))
            logger.error(f"❌ Permission denied creating {dir_path}: {e}")
        except Exception as e:
            failed_dirs.append((dir_path, str(e)))
            logger.error(f"❌ Failed to create {dir_path}: {e}")
    
    # Create optional directories (don't fail if these can't be created)
    logger.debug(f"🔧 Creating {len(optional_dirs)} optional directories...")
    for dir_path in optional_dirs:
        try:
            Path(dir_path).mkdir(parents=True, exist_ok=True, mode=0o755)
            created_dirs.append(dir_path)
            logger.debug(f"✅ Created optional: {dir_path}")
        except Exception as e:
            logger.warning(f"⚠️  Could not create optional directory {dir_path}: {e}")
            # Don't add to failed_dirs since these are optional
    
    # Log summary
    if failed_dirs:
        logger.error(f"❌ Failed to create {len(failed_dirs)} essential directories:")
        for path, error in failed_dirs:
            logger.error(f"   {path}: {error}")
    else:
        logger.info(f"✅ All essential directories created successfully")
    
    logger.info(f"📊 Directory creation summary: {len(created_dirs)} created, {len(failed_dirs)} failed")
    
    return len(created_dirs), failed_dirs

def load_config(mode: str = None):
    """
    Load Enhanced north cal configuration with GUARANTEED directory creation
    
    Args:
        mode: Analysis mode ('sequential' or 'random')
    
    Returns:
        dict: Complete configuration with all directories created
    """
    
    # Try to detect mode from command line arguments if not provided
    if mode is None:
        import sys
        if '--mode' in sys.argv:
            mode_idx = sys.argv.index('--mode')
            if mode_idx + 1 < len(sys.argv):
                mode = sys.argv[mode_idx + 1]
            else:
                mode = "sequential"
        else:
            mode = "sequential"
    
    logger.info(f"🚀 Loading north cal config for {mode} mode")
    
    # Get base GMPE configuration
    try:
        base_config = load_gmpe_config()
        logger.debug("✅ Loaded base GMPE configuration")
    except Exception as e:
        logger.error(f"❌ Failed to load base GMPE config: {e}")
        raise
    
    # Get the EXACT bounds used for catalog filtering
    try:
        region_bounds = get_region_bounds('north_cal')
        logger.debug("✅ Loaded regional bounds")
    except Exception as e:
        logger.error(f"❌ Failed to load regional bounds: {e}")
        raise
    
    # Generate standardized paths
    paths = generate_standardized_paths('north_cal', mode)
    logger.debug(f"✅ Generated standardized paths for {mode} mode")
    
    # CRITICAL: Create ALL directories before proceeding
    try:
        created_count, failed_dirs = create_standardized_directories(paths, force_create=True)
        
        if failed_dirs:
            # If essential directories failed, this is a critical error
            error_msg = f"Failed to create {len(failed_dirs)} essential directories"
            logger.error(f"❌ {error_msg}")
            for path, error in failed_dirs:
                logger.error(f"   {path}: {error}")
            raise RuntimeError(error_msg)
        
        logger.info(f"✅ All {created_count} directories created successfully")
        
    except Exception as e:
        logger.error(f"❌ Critical error in directory creation: {e}")
        raise
    
    # north cal specific configuration
    north_cal_config = {
        "region": {
            "name": "north_cal",
            "description": "north cal Bay Area using exact polygon + Guaranteed Directories",
            "polygon": region_bounds["polygon"],
            "bounds": {
                "min_lat": region_bounds["min_lat"],
                "max_lat": region_bounds["max_lat"],
                "min_lon": region_bounds["min_lon"],
                "max_lon": region_bounds["max_lon"]
            },
            "mode": mode
        },
        
        # STANDARDIZED PATHS (all directories now guaranteed to exist)
        "paths": paths,
        
        "site": {
            "grid_mode": True,
            "grid_lat_min": region_bounds["min_lat"],
            "grid_lat_max": region_bounds["max_lat"],
            "grid_lon_min": region_bounds["min_lon"],
            "grid_lon_max": region_bounds["max_lon"],
            "grid_lat_spacing": 0.1,
            "grid_lon_spacing": 0.1,
            "sites": [(37.77, -122.42), (37.8, -122.27), (37.42, -122.08), (37.54, -122.31)],
            "vs30": base_config["site_defaults"]["vs30"],
            "z1p0": base_config["site_defaults"]["z1p0"],
            "gmpe_model": base_config["site_defaults"]["gmpe_model"],
            "default_period": base_config["site_defaults"]["default_period"],
            "include_scatter": base_config["site_defaults"]["include_scatter"],
            "scatter_std_dev": base_config["site_defaults"]["scatter_std_dev"],
            "max_distance_km": 200.0
        },
        
        "output_settings": {
            "full_output": False,
            "min_ground_motion": 0.01,
            "probability_type": "annual",
            "probabilities": [(50, 0.02), (50, 0.1)],
            "plot_hazard_curves": True,
            "max_curves": 10,
            "export_gis_csv": True,
            "create_hazard_map": True,
            "generate_summary_plots": True,
            "plot_format": "png",
            "plot_dpi": 300,
            "plot_style": "seaborn",
            "color_scheme": "viridis",
            "save_statistics": True,
            "compression": "gzip",
            "precision": 6,
        },
        
        "regional_analysis": {
            "enabled": True,
            "multi_window_comparison": True,
            "statistical_summary": True,
            "performance_metrics": True,
            "polygon_filtering": True,
        },
        
        "major_cities": {'Redding': (-122.39, 40.59), 'Chico': (-121.84, 39.73), 'Yuba City': (-121.61, 39.16), 'Eureka': (-124.08, 40.87), 'Ukiah': (-123.21, 39.15), 'Santa Rosa': (-122.71, 38.44), 'Petaluma': (-122.64, 38.23), 'Napa': (-122.29, 38.3), 'Fairfield': (-122.04, 38.25)}
    }
    
    # Merge configurations
    final_config = base_config.copy()
    final_config.update(north_cal_config)
    
    # Verify critical paths exist
    critical_paths = ['output_dir', 'viz_dir', 'maps_dir', 'plots_dir', 'gis_dir']
    for path_key in critical_paths:
        if path_key in paths:
            path_value = paths[path_key]
            if not os.path.exists(path_value):
                logger.error(f"❌ Critical path does not exist after creation: {path_value}")
                raise RuntimeError(f"Critical directory missing: {path_value}")
    
    logger.info(f"✅ north cal configuration loaded successfully for {mode} mode")
    logger.info(f"📁 Output directory: {paths['output_dir']}")
    logger.info(f"🎨 Visualizations directory: {paths['viz_dir']}")
    
    return final_config

def load_config_for_mode(mode: str):
    """Load config for specific mode"""
    return load_config(mode)

def switch_to_random_mode(config: dict):
    """Switch configuration to random mode with directory creation"""
    region_name = config['region']['name']
    random_paths = generate_standardized_paths(region_name, 'random')
    
    # Create directories for random mode
    created_count, failed_dirs = create_standardized_directories(random_paths, force_create=True)
    
    if failed_dirs:
        logger.error(f"❌ Failed to create directories for random mode: {failed_dirs}")
        raise RuntimeError("Failed to create random mode directories")
    
    config['paths'] = random_paths
    config['region']['mode'] = 'random'
    
    logger.info(f"✅ Switched to random mode with {created_count} directories created")
    return config

def point_in_polygon(lat, lon, polygon):
    """Check if point is inside polygon using ray casting"""
    x, y = lon, lat
    n = len(polygon)
    inside = False
    
    p1x, p1y = polygon[0][1], polygon[0][0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n][1], polygon[i % n][0]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p1y = p2x, p2y
    
    return inside

def generate_sites_in_polygon(region_name, spacing=0.1):
    """Generate sites directly within polygon - memory efficient"""
    bounds = get_region_bounds(region_name)
    polygon = REGIONAL_POLYGONS[region_name]['region_polygon']
    
    sites = []
    total_tested = 0
    
    lat = bounds['min_lat']
    while lat <= bounds['max_lat']:
        lon = bounds['min_lon']
        while lon <= bounds['max_lon']:
            total_tested += 1
            if point_in_polygon(lat, lon, polygon):
                sites.append((lat, lon))
            lon += spacing
        lat += spacing
    
    efficiency = (len(sites) / total_tested) * 100 if total_tested > 0 else 0
    
    print(f"Site generation for {region_name}:")
    print(f"  Grid spacing: {spacing:.3f}° (~{spacing*111:.1f} km)")
    print(f"  Total grid points tested: {total_tested}")
    print(f"  Sites within polygon: {len(sites)}")
    print(f"  Grid efficiency: {efficiency:.1f}%")
    
    return sites

def get_optimized_grid_sites(config):
    """Get grid sites using efficient polygon-aware generation"""
    if not config['site']['grid_mode']:
        return config['site']['sites']
    
    region_name = config['region']['name']
    spacing = config['site']['grid_lat_spacing']
    
    return generate_sites_in_polygon(region_name, spacing)

def validate_config():
    """Validate north cal configuration with directory checks"""
    
    logger.info("🔍 Validating north cal configuration...")
    
    config = load_config()
    issues = []
    warnings = []
    
    # Check visualization dependencies
    try:
        import matplotlib.pyplot as plt
        logger.info("✅ Matplotlib available for plotting")
    except ImportError:
        issues.append("Matplotlib not available - plotting features will be disabled")
    
    # Check polygon
    if "polygon" not in config["region"]:
        issues.append("Missing regional polygon definition")
    else:
        polygon = config["region"]["polygon"]
        if len(polygon) < 3:
            issues.append("Polygon must have at least 3 vertices")
        logger.info(f"✅ Using EXACT regional polygon with {len(polygon)} vertices")
    
    # Check input directories
    windows_dir = config["paths"]["windows_dir"]
    if not os.path.exists(windows_dir):
        warnings.append(f"Windows directory not found: {windows_dir}")
        logger.warning(f"⚠️  Windows directory not found: {windows_dir}")
    else:
        import glob
        csv_files = glob.glob(os.path.join(windows_dir, "*.csv"))
        if not csv_files:
            warnings.append(f"No CSV files found in: {windows_dir}")
            logger.warning(f"⚠️  No CSV files found in: {windows_dir}")
        else:
            logger.info(f"✅ Found {len(csv_files)} window files")
    
    # Check output directories (these should all exist now)
    essential_output_dirs = ['output_dir', 'viz_dir', 'maps_dir', 'plots_dir', 'gis_dir']
    for dir_key in essential_output_dirs:
        if dir_key in config['paths']:
            dir_path = config['paths'][dir_key]
            if not os.path.exists(dir_path):
                issues.append(f"Required output directory missing: {dir_path}")
            else:
                logger.debug(f"✅ Output directory exists: {dir_path}")
    
    # Check GMPE configuration
    gmpe_config = config["paths"]["gmpe_config"]
    if not os.path.exists(gmpe_config):
        issues.append(f"GMPE configuration file not found: {gmpe_config}")
    else:
        logger.info(f"✅ GMPE configuration found: {gmpe_config}")
    
    # Report results
    if issues:
        logger.error("❌ north cal configuration validation issues:")
        for issue in issues:
            logger.error(f"   • {issue}")
        return False
    
    if warnings:
        logger.warning("⚠️  north cal configuration warnings:")
        for warning in warnings:
            logger.warning(f"   • {warning}")
    
    logger.info("✅ north cal configuration validation passed")
    return True

def test_directory_creation():
    """Test directory creation functionality"""
    
    logger.info("🧪 Testing directory creation...")
    
    # Test with temporary directory
    import tempfile
    import shutil
    
    with tempfile.TemporaryDirectory() as temp_dir:
        test_paths = generate_standardized_paths('test_region', 'sequential')
        
        # Replace base paths with temp directory
        for key, path in test_paths.items():
            if isinstance(path, str):
                test_paths[key] = path.replace('/scratch/olawoyiv/rsqsim_data/output', temp_dir)
        
        # Test directory creation
        created_count, failed_dirs = create_standardized_directories(test_paths)
        
        if failed_dirs:
            logger.error(f"❌ Directory creation test failed: {failed_dirs}")
            return False
        
        # Verify directories exist
        for key, path in test_paths.items():
            if 'dir' in key and isinstance(path, str):
                if not os.path.exists(path):
                    logger.error(f"❌ Directory not created: {path}")
                    return False
        
        logger.info(f"✅ Directory creation test passed ({created_count} directories)")
        return True
    
if __name__ == "__main__":
    # Setup logging for testing
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print("Enhanced north cal Configuration with Directory Creation")
    print("=" * 80)
    
    # Test directory creation
    print("\n1. Testing directory creation functionality...")
    test_success = test_directory_creation()
    
    # Load and validate configuration
    print("\n2. Loading and validating configuration...")
    try:
        config = load_config()
        is_valid = validate_config()
        
        print(f"\n📊 Configuration Results:")
        print(f"   Directory creation test: {'✅ PASSED' if test_success else '❌ FAILED'}")
        print(f"   Configuration validation: {'✅ PASSED' if is_valid else '❌ FAILED'}")
        print(f"   Region: {config['region']['name']}")
        print(f"   Mode: {config['region']['mode']}")
        print(f"   Visualizations enabled: {config['output_settings']['plot_hazard_curves']}")
        print(f"   Output directory: {config['paths']['output_dir']}")
        
        if test_success and is_valid:
            print(f"\n🚀 READY TO RUN:")
            print(f"   ./scripts/run_on_cluster.sh north_cal sequential")
            print(f"   ./scripts/run_on_cluster.sh north_cal random")
            print(f"\n📁 Results will be saved to:")
            print(f"   {config['paths']['output_dir']}")
        else:
            print(f"\n❌ Fix issues before running simulation")
            
    except Exception as e:
        print(f"\n❌ Configuration loading failed: {e}")
        import traceback
        traceback.print_exc()
    