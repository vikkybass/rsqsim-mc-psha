"""
GMPE Configuration for RSQSim Ground Motion Simulation

This configuration file specifies Ground Motion Prediction Equation (GMPE)
parameters and coefficient file paths for the simulation.

This is the UNIVERSAL configuration that all regions inherit from.
Region-specific configs (like los_angeles_config.py) should import and extend this.
"""

import os
from pathlib import Path
import multiprocessing as mp

# Base configuration directory  
CONFIG_DIR = Path(__file__).parent
PROJECT_ROOT = CONFIG_DIR.parent
# Get environment variables
RSQSIM_HOME = os.environ.get('RSQSIM_HOME', str(Path.home() / 'rsqsim_mc'))
RSQSIM_WORK_DIR = os.environ.get('RSQSIM_WORK_DIR', '/scratch/olawoyiv/rsqsim_data')
RSQSIM_PROJECT_DIR = os.environ.get('RSQSIM_PROJECT_DIR', '/projects/ebelseismo/olawoyiv')

# GMPE Model Configuration (Universal settings)
GMPE_CONFIG = {
    # Default model to use for simulations (matching your working config)
    "default_model": "CB14",

    #geometry path
    "geom_paths": {
        "geometry_file": f"{RSQSIM_WORK_DIR}/data/Catalog_4983/geometry.flt",
        "catalog_base": f"{RSQSIM_WORK_DIR}/data/Catalog_4983",
    },
    
    # Default period for spectral acceleration (seconds)
    "default_period": 0.01,  # PGA
    
    # Available GMPE models and their coefficient files
    "gmpe_files": {
        "CB14": None,  # ✅ Uses PyGMM
        "BSSA14": None,  # ✅ Uses PyGMM
        "ASK14": None,  # ✅ Uses PyGMM
        "CY14": None,  # ✅ Uses PyGMM
        
        "Frankel1996": str(Path.home() / "rsqsim_mc" / "utility_files" / "coefficients" / "frankel1996.txt"),
        "Atkinson1995": None,  # Analytical model
        "Toro1997": None,      # Analytical model
        "BooreAtkinson1987": None,  # Analytical model
        "ToroMcGuire1987": None     # Analytical model
    },
    
    # RSQSim simulation settings (from your working Mid_config.py)
    "simulation": {
        "synduration": 50000,        # Duration in years (UPDATED to match your time windows)
        "seed": 42,                  # Random seed
        "gm_thresholds": [-3, -2, -1, 0, 1],  # Ground motion thresholds (log10 values)
        
        # Location variation (may be relevant for RSQSim)
        "location_variation": {
            "enabled": True,            # Whether to add spatial scatter
            "scatter_type": 2,           # 0=linear flat, 1=linear decrease, 2=gaussian  
            "max_scatter_distance": 10.0 # km
        },
        
        "processing": {
            "batch_size": 1000,      # Events to process at once
            "progress_interval": 1000  # How often to log progress
        },
        
        "filtering": {
            "min_magnitude": 5.0,    # Minimum magnitude to consider
            "max_distance_km": 300,  # Maximum source-site distance
            "min_depth_km": 0,       # Minimum depth
            "max_depth_km": 50       # Maximum depth
        }
    },
    
    # Default site parameters (can be overridden by region configs)
    "site_defaults": {
        "vs30": 760.0,              # Default Vs30 (m/s)
        "z1p0": 100.0,              # Default Z1.0 (m)
        "default_period": 0.01,     # Default period (PGA)
        "include_scatter": True,   # Whether to include aleatory variability
        "scatter_std_dev": 0.3      # Standard deviation for scatter
    },
    
    # Model-specific parameters
    "model_parameters": {
        "CB14": {
            "description": "Campbell & Bozorgnia (2014) - NGA-West2 [PyGMM]",
            "implementation": "pygmm",
            "applicable_periods": [0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3,
                                 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0],
            "magnitude_range": [3.3, 8.5],
            "distance_range": [0.1, 300],
            "vs30_range": [150, 1500],
            "mechanism_types": ["strike-slip", "reverse", "normal"],
            "reference": "Campbell, K. W., & Bozorgnia, Y. (2014). NGA-West2 ground motion model. Earthquake Spectra, 30(3), 1087-1115."
        },
        
        "BSSA14": {
            "description": "Boore, Stewart, Seyhan & Atkinson (2014) - NGA-West2 [PyGMM]",
            "implementation": "pygmm",
            "applicable_periods": [0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3,
                                 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0],
            "magnitude_range": [3.0, 8.5],
            "distance_range": [0.1, 400],
            "vs30_range": [150, 1500],
            "mechanism_types": ["strike-slip", "reverse", "normal"],
            "reference": "Boore, D. M., Stewart, J. P., Seyhan, E., & Atkinson, G. M. (2014). NGA-West2 equations. Earthquake Spectra, 30(3), 1057-1085."
        },
        
        "ASK14": {
            "description": "Abrahamson, Silva & Kamai (2014) - NGA-West2 [PyGMM]",
            "implementation": "pygmm",
            "applicable_periods": [0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3,
                                 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0],
            "magnitude_range": [3.0, 8.5],
            "distance_range": [0.1, 300],
            "vs30_range": [150, 1500],
            "mechanism_types": ["strike-slip", "reverse", "normal"],
            "reference": "Abrahamson, N., Silva, W., & Kamai, R. (2014). Summary of the ASK14 model. Earthquake Spectra, 30(3), 1025-1055."
        },
        
        "CY14": {
            "description": "Chiou & Youngs (2014) - NGA-West2 [PyGMM]",
            "implementation": "pygmm",
            "applicable_periods": [0.01, 0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3,
                                 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.5, 10.0],
            "magnitude_range": [3.5, 8.5],
            "distance_range": [0.1, 300],
            "vs30_range": [150, 1500],
            "mechanism_types": ["strike-slip", "reverse", "normal"],
            "reference": "Chiou, B. S. J., & Youngs, R. R. (2014). Update of the Chiou and Youngs NGA model. Earthquake Spectra, 30(3), 1117-1153."
        },
        
        "Frankel1996": {
            "description": "Frankel et al. (1996) - USGS National Seismic Hazard Maps",
            "applicable_periods": [0.01],  # PGA only for this implementation
            "magnitude_range": [4.4, 8.0],
            "distance_range": [1.0, 500],
            "vs30_range": [150, 1500],
            "mechanism_types": ["average"],
            "reference": "Frankel, A., Mueller, C., Barnhard, T., Perkins, D., Leyendecker, E. V., Dickman, N., ... & Hopper, M. (1996). National seismic-hazard maps: documentation June 1996 (No. 96-532). US Geological Survey."
        },
        
        "Atkinson1995": {
            "description": "Atkinson & Boore (1995) - Eastern North America",
            "applicable_periods": [0.01],
            "magnitude_range": [3.0, 8.0],
            "distance_range": [10.0, 500],
            "vs30_range": [150, 2000],
            "mechanism_types": ["average"],
            "reference": "Atkinson, G. M., & Boore, D. M. (1995). Ground-motion relations for eastern North America. Bulletin of the Seismological Society of America, 85(1), 17-30."
        },
        
        "Toro1997": {
            "description": "Toro, Abrahamson & Schneider (1997) - Central and Eastern US",
            "applicable_periods": [0.01],
            "magnitude_range": [4.0, 8.0],
            "distance_range": [1.0, 500],
            "vs30_range": [150, 2000],
            "mechanism_types": ["average"],
            "reference": "Toro, G. R., Abrahamson, N. A., & Schneider, J. F. (1997). Model of strong ground motions from earthquakes in central and eastern North America: best estimates and uncertainties. Seismological Research Letters, 68(1), 41-57."
        }
    },
    
    # Computational parameters
    "computation": {
        "include_aleatory_variability": True,  # Include random scatter in predictions
        "aleatory_std_dev": 0.3,               # Standard deviation for aleatory variability (log10 units)
        "minimum_distance": 0.1,               # Minimum source-site distance (km)
        "maximum_distance": 500.0,             # Maximum source-site distance (km)
        "distance_metric": "rupture",          # Distance metric: "rupture", "joyner_boore", "epicentral"
    },
    
    # Quality control parameters
    "quality_control": {
        "check_model_applicability": True,     # Check if parameters are within model ranges
        "warn_on_extrapolation": True,         # Warn when extrapolating beyond model ranges
        "maximum_pga_g": 10.0,                 # Maximum reasonable PGA value (g)
        "minimum_pga_g": 1e-6,                 # Minimum reasonable PGA value (g)
        "log_calculation_details": False       # Log detailed calculation information
    },
    
    "output_settings": {
        "full_output": False,
        "min_ground_motion": 0.01,
        "probability_type": "non_exceedance",
        "probabilities": [[50, 0.98], [50, 0.95], [50, 0.90]],
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
    } 
}

def load_gmpe_config():
    """
    Load and validate GMPE configuration
    
    Returns:
        dict: GMPE configuration dictionary
    """
    config = GMPE_CONFIG.copy()
    
    # NGA-West2 models that use PyGMM (don't need coefficient files)
    NGA_WEST2_MODELS = ['CB14', 'BSSA14', 'ASK14', 'CY14']
    
    # Validate that coefficient files exist for models that need them
    missing_files = []
    for model, filepath in config["gmpe_files"].items():
        # Skip NGA-West2 models - they use PyGMM
        if model in NGA_WEST2_MODELS:
            continue
            
        # Check other models
        if filepath is not None and not os.path.exists(filepath):
            missing_files.append(f"{model}: {filepath}")
    
    if missing_files:
        print("Warning: Missing GMPE coefficient files:")
        for missing in missing_files:
            print(f"  - {missing}")
        print("\nNote: NGA-West2 models (CB14, BSSA14, ASK14, CY14) use PyGMM and don't need coefficient files.")
    
    return config

# Legacy compatibility function
def load_config():
    """
    Load configuration in the original Mid_config.py format for backward compatibility
    
    Returns:
        dict: Configuration dictionary in original format
    """
    config = load_gmpe_config()
    
    # Return in original Mid_config.py format for backward compatibility
    return {
        "default_model": config["default_model"],
        "default_period": config["default_period"],
        "gmpe_files": config["gmpe_files"],
        "synduration": config["simulation"]["synduration"],
        "seed": config["simulation"]["seed"],
        "gm_thresholds": config["simulation"]["gm_thresholds"],
        "site_defaults": config["site_defaults"]
    }

def get_available_models():
    """
    Get list of available GMPE models
    
    Returns:
        list: Available model names
    """
    config = load_gmpe_config()
    
    # Check PyGMM availability
    try:
        import pygmm
        pygmm_available = True
    except ImportError:
        pygmm_available = False
    
    available = []
    for model, filepath in config["gmpe_files"].items():
        model_info = config["model_parameters"].get(model)
        
        # NGA-West2 models need PyGMM
        if model_info and model_info.get("implementation") == "pygmm":
            if pygmm_available:
                available.append(model)
        # Other models need either no file or existing file
        elif filepath is None or os.path.exists(filepath):
            available.append(model)
    
    return available

def get_model_info(model_name):
    """
    Get detailed information about a specific model
    
    Args:
        model_name: Name of the GMPE model
        
    Returns:
        dict: Model information or None if model not found
    """
    config = load_gmpe_config()
    return config["model_parameters"].get(model_name)

def validate_parameters(model_name, magnitude, distance, vs30=None, period=None):
    """
    Validate input parameters against model applicability ranges
    
    Args:
        model_name: GMPE model name
        magnitude: Earthquake magnitude
        distance: Source-site distance (km)
        vs30: Site velocity (m/s), optional
        period: Spectral period (s), optional
        
    Returns:
        tuple: (is_valid, warnings_list)
    """
    config = load_gmpe_config()
    model_info = config["model_parameters"].get(model_name)
    
    if not model_info:
        return False, [f"Unknown model: {model_name}"]
    
    warnings = []
    
    # Check magnitude range
    mag_range = model_info["magnitude_range"]
    if magnitude < mag_range[0]:
        warnings.append(f"Magnitude {magnitude} below model range [{mag_range[0]}, {mag_range[1]}]")
    elif magnitude > mag_range[1]:
        warnings.append(f"Magnitude {magnitude} above model range [{mag_range[0]}, {mag_range[1]}]")
    
    # Check distance range
    dist_range = model_info["distance_range"]
    if distance < dist_range[0]:
        warnings.append(f"Distance {distance} km below model range [{dist_range[0]}, {dist_range[1]}]")
    elif distance > dist_range[1]:
        warnings.append(f"Distance {distance} km above model range [{dist_range[0]}, {dist_range[1]}]")
    
    # Check Vs30 if provided
    if vs30 is not None:
        vs30_range = model_info["vs30_range"]
        if vs30 < vs30_range[0]:
            warnings.append(f"Vs30 {vs30} m/s below model range [{vs30_range[0]}, {vs30_range[1]}]")
        elif vs30 > vs30_range[1]:
            warnings.append(f"Vs30 {vs30} m/s above model range [{vs30_range[0]}, {vs30_range[1]}]")
    
    # Check period if provided
    if period is not None:
        applicable_periods = model_info["applicable_periods"]
        if period not in applicable_periods:
            # Find closest period
            closest_period = min(applicable_periods, key=lambda x: abs(x - period))
            warnings.append(f"Period {period}s not in model periods. Closest: {closest_period}s")
    
    is_valid = len(warnings) == 0
    return is_valid, warnings

# Default function for backward compatibility
def load_config():
    """Load GMPE configuration (alias for load_gmpe_config)"""
    return load_gmpe_config()

# For testing and debugging
if __name__ == "__main__":
    print("GMPE Configuration Test")
    print("=" * 50)
    
    # Load configuration
    config = load_gmpe_config()
    
    print(f"Default model: {config['default_model']}")
    print(f"Default period: {config['default_period']} seconds")
    print(f"Default Vs30: {config['default_vs30']} m/s")
    
    # Show available models
    available = get_available_models()
    print(f"\nAvailable models ({len(available)}):")
    for model in available:
        info = get_model_info(model)
        if info:
            print(f"  - {model}: {info['description']}")
        else:
            print(f"  - {model}: (no description available)")
    
    # Test parameter validation
    print(f"\nParameter validation test:")
    test_cases = [
        ("ASK14", 6.5, 50, 760, 0.01),
        ("ASK14", 9.0, 50, 760, 0.01),  # High magnitude
        ("ASK14", 6.5, 500, 760, 0.01), # High distance
        ("Frankel1996", 6.0, 100, 760, 0.01)
    ]
    
    for model, mag, dist, vs30, period in test_cases:
        is_valid, warnings = validate_parameters(model, mag, dist, vs30, period)
        status = "✅ Valid" if is_valid else "⚠️  Has warnings"
        print(f"  {model} M{mag} {dist}km: {status}")
        for warning in warnings:
            print(f"    - {warning}")