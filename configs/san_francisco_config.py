"""
San Francisco Region Configuration

Regional-specific configuration that inherits from base config.
Only contains SF-specific overrides - no redundant settings.
"""

import os
from pathlib import Path
import logging
import glob

# Import base GMPE configuration
from configs.config import load_gmpe_config, RSQSIM_WORK_DIR

# Setup logging
logger = logging.getLogger(__name__)

# ============================================================================
# REGION-SPECIFIC DATA (Only things unique to San Francisco)
# ============================================================================

# Regional polygon for San Francisco
SF_POLYGON = [
    [36.890, -121.148], [36.546, -121.422], [36.535, -122.099],
    [38.489, -123.207], [38.652, -123.090], [38.692, -122.243],
    [38.987, -121.344], [38.376, -121.005], [37.503, -121.461],
    [36.859, -121.135], [36.890, -121.148]
]

# Major cities for visualization
SF_MAJOR_CITIES = {
    'San Francisco': (-122.4194, 37.7749),
    'Oakland': (-122.2711, 37.8044),
    'San Jose': (-121.8863, 37.3382),
    'Fremont': (-121.9886, 37.5483),
    'Santa Rosa': (-122.7141, 38.4405)
}

# Example sites (if not using grid)
SF_EXAMPLE_SITES = [
    (37.7749, -122.4194),  # San Francisco
    (37.8044, -122.2711)   # Oakland
]


# ============================================================================
# Helper Functions
# ============================================================================

def get_region_bounds():
    """Calculate bounding box from regional polygon"""
    lats = [point[0] for point in SF_POLYGON]
    lons = [point[1] for point in SF_POLYGON]
    
    return {
        "min_lat": min(lats),
        "max_lat": max(lats),
        "min_lon": min(lons),
        "max_lon": max(lons)
    }


def generate_paths(mode: str = "sequential"):
    """Generate standardized paths for San Francisco"""
    base_output = os.path.join(RSQSIM_WORK_DIR, 'output')
    
    return {
        # Input
        "windows_dir": f"{RSQSIM_WORK_DIR}/data/Catalog_4983/windows/san_francisco/{mode}",
        
        # Output
        "output_dir": f"{base_output}/san_francisco/{mode}",
        "log_dir": f"{RSQSIM_WORK_DIR}/logs/san_francisco/{mode}",
        
        # Visualizations
        "viz_dir": f"{base_output}/san_francisco/{mode}/visualizations",
        "maps_dir": f"{base_output}/san_francisco/{mode}/visualizations/maps",
        "plots_dir": f"{base_output}/san_francisco/{mode}/visualizations/plots",
        "gis_dir": f"{base_output}/san_francisco/{mode}/visualizations/gis",
        
        # Summary
        "regional_summary_dir": f"{base_output}/san_francisco/{mode}/regional_summary"
    }


def create_directories(paths: dict):
    """Create all necessary directories"""
    essential_dirs = [
        "output_dir", "log_dir", "viz_dir", 
        "maps_dir", "plots_dir", "gis_dir", "regional_summary_dir"
    ]
    
    created = []
    failed = []
    
    for key in essential_dirs:
        if key in paths:
            try:
                Path(paths[key]).mkdir(parents=True, exist_ok=True)
                created.append(paths[key])
                logger.debug(f"✅ Created: {paths[key]}")
            except Exception as e:
                failed.append((paths[key], str(e)))
                logger.error(f"❌ Failed to create {paths[key]}: {e}")
    
    if failed:
        raise RuntimeError(f"Failed to create {len(failed)} directories")
    
    logger.info(f"✅ Created {len(created)} directories")
    return len(created)


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


def generate_grid_sites(spacing=0.1):
    """Generate sites within SF polygon"""
    bounds = get_region_bounds()
    sites = []
    
    lat = bounds['min_lat']
    while lat <= bounds['max_lat']:
        lon = bounds['min_lon']
        while lon <= bounds['max_lon']:
            if point_in_polygon(lat, lon, SF_POLYGON):
                sites.append((lat, lon))
            lon += spacing
        lat += spacing
    
    logger.info(f"Generated {len(sites)} sites with {spacing}° spacing")
    return sites


# ============================================================================
# Main Configuration Function
# ============================================================================

def load_config(mode: str = "sequential"):
    """
    Load San Francisco configuration
    
    Inherits from base config and only overrides SF-specific settings.
    
    Args:
        mode: Analysis mode ('sequential' or 'random')
    
    Returns:
        dict: Complete configuration for san_francisco
    """
    logger.info(f"🚀 Loading san_francisco config for {mode} mode")
    
    # 1. Load base configuration (contains all GMPE settings, defaults, etc.)
    base_config = load_gmpe_config()
    logger.debug("✅ Loaded base GMPE configuration")
    
    # 2. Get SF-specific data
    bounds = get_region_bounds()
    paths = generate_paths(mode)
    
    # 3. Create directories
    create_directories(paths)
    
    # 4. Build SF-specific configuration (ONLY overrides)
    sf_config = {
        # Regional information
        "region": {
            "name": "san_francisco",
            "pygmm_region": "california",
            "description": "San Francisco Bay Area",
            "polygon": SF_POLYGON,
            "bounds": bounds,
            "mode": mode,
            "major_cities": SF_MAJOR_CITIES
        },
        
        # Paths (SF-specific)
        "paths": paths,
        
        # Site configuration (SF-specific overrides)
        "site": {
            "grid_mode": True,
            "grid_lat_min": bounds["min_lat"],
            "grid_lat_max": bounds["max_lat"],
            "grid_lon_min": bounds["min_lon"],
            "grid_lon_max": bounds["max_lon"],
            "grid_lat_spacing": 0.1,
            "grid_lon_spacing": 0.1,
            "sites": SF_EXAMPLE_SITES,
            "max_distance_km": 300.0,
            # Inherit vs30, z1p0, scatter settings from base config
            **{k: v for k, v in base_config["site_defaults"].items() 
               if k in ["vs30", "z1p0", "z2p5", "include_scatter", "scatter_std_dev"]}
        },
        
        "gmpe": {
            # Choose one of the options below:
            
            # OPTION 1: NSHM 2023 Standard (recommended)
            "use_openquake": False,
            "use_ensemble": True,
            "models": ["ASK14", "BSSA14", "CB14", "CY14"],
            "ensemble_weights": {
                "ASK14": 0.25,
                "BSSA14": 0.25,
                "CB14": 0.25,
                "CY14": 0.25
            },
            
            # OPTION 2: Custom 2-model ensemble (uncomment to use)
            # "use_ensemble": True,
            # "models": ["CB14", "ASK14"],
            # "ensemble_weights": {
            #     "CB14": 0.6,
            #     "ASK14": 0.4
            # },
            
            # OPTION 3: Single model only (uncomment to use)
            # "use_ensemble": False,
            # "models": ["CB14"],
            # "ensemble_weights": {
            #     "CB14": 1.0
            # },
            
            # Common settings
            "default_model": "CB14",
            "mechanism": "strike-slip",
        },
        
        # Regional analysis (SF-specific)
        "regional_analysis": {
            "enabled": True,
            "multi_window_comparison": True,
            "statistical_summary": True,
            "polygon_filtering": True
        },
        
        # Performance (SF-specific if different from base)
        "resource_limits": {
            "max_memory_gb": 16,
            "batch_size": 50,
            "max_events_per_site": 20000
        }
    }
    
    # 5. Merge configurations (base + SF overrides)
    final_config = {**base_config, **sf_config}
    
     # 6. Quick validation
    if not os.path.exists(paths["windows_dir"]):
        logger.warning(f"⚠️  Windows directory not found: {paths['windows_dir']}")
    else:
        csv_files = glob.glob(os.path.join(paths["windows_dir"], "*.csv"))
        logger.info(f"✅ Found {len(csv_files)} window files")
    
    # 7. Log GMPE configuration
    gmpe_cfg = final_config.get('gmpe', {})
    if gmpe_cfg.get('use_ensemble'):
        logger.info(f"📊 Using ensemble: {gmpe_cfg.get('models')}")
        logger.info(f"   Weights: {gmpe_cfg.get('ensemble_weights')}")
    else:
        logger.info(f"📊 Using single model: {gmpe_cfg.get('default_model')}")
    
    logger.info(f"✅ San Francisco configuration loaded for {mode} mode")
    return final_config

# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    print("san_francisco Configuration")
    print("=" * 60)
    
    # Test loading
    config = load_config("sequential")
    
    print(f"\n✅ Configuration loaded successfully:")
    print(f"   Region: {config['region']['name']}")
    print(f"   Mode: {config['region']['mode']}")
    print(f"   GMPE Model: {config['default_model']}")  # Inherited from base
    print(f"   Output: {config['paths']['output_dir']}")
    print(f"   Grid sites: {len(generate_grid_sites(0.1))}")

