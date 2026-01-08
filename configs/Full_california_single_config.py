"""
Enhanced Full California Single Site Configuration with Visualization Options
"""

import os
from pathlib import Path

# Import base GMPE configuration
from configs.config import load_gmpe_config

# Get environment variables
RSQSIM_HOME = os.environ.get('RSQSIM_HOME', str(Path.home() / 'rsqsim_mc'))
RSQSIM_WORK_DIR = os.environ.get('RSQSIM_WORK_DIR', '/scratch/olawoyiv/rsqsim_data')
RSQSIM_PROJECT_DIR = os.environ.get('RSQSIM_PROJECT_DIR', '/projects/ebelseismo/olawoyiv')

def load_config():
    """
    Load Enhanced Single Site Full California configuration with visualization options
    
    Returns:
        dict: Complete configuration for single site test with visualizations
    """
    
    # Get base GMPE configuration
    base_config = load_gmpe_config()
    
    # Enhanced Single site Full California configuration
    ca_config = {
        "region": {
            "name": "Full_california",
            "description": "Single site test with Full California catalog (397K events) + Visualizations",
            "bounds": {
                "min_lat": 32.0,
                "max_lat": 42.0,
                "min_lon": -125.0,
                "max_lon": -114.0
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
            # SINGLE SITE MODE - much faster!
            "grid_mode": False,           # KEY: Use individual sites, not grid
            
            # Test sites (start with just LA)
            "sites": [
                (34.05, -118.25),  # Los Angeles - near San Andreas, Hollywood faults
                #37.77, -122.42),  # San Francisco - near Hayward, San Andreas
                #(32.72, -117.16),  # San Diego - near Rose Canyon fault
            ],
            
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
            "min_ground_motion": 0.01,  # Same as site setting
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
            "compression": None,          # "gzip" for compressed output
            "precision": 6,               # Decimal precision for output
        },
        
        # NEW: Regional Analysis Options
        "regional_analysis": {
            "enabled": True,              # Enable regional summary generation
            "multi_window_comparison": True,  # Compare across time windows
            "statistical_summary": True,  # Generate statistical summaries
            "performance_metrics": True,  # Track processing performance
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

def preview_expected_outputs():
    """Show what outputs to expect with visualization enabled"""
    
    print("\n📊 EXPECTED OUTPUTS WITH ENHANCED VISUALIZATION:")
    print("="*70)
    print("📍 Site: Los Angeles (34.05°N, -118.25°W)")
    print("📁 Catalog: 397K events from Full California")
    print("🎯 Processing time: 5-15 minutes")
    print()
    
    print("📂 OUTPUT STRUCTURE:")
    print("  /output/Full_california/single_site_test/Full_california_single/")
    print("  ├── hazard_results_Full_california_seq_window_0001.txt")
    print("  ├── summary_Full_california_seq_window_0001.csv")
    print("  ├── visualizations/")
    print("  │   ├── hazard_curve_34.0500_-118.2500.png")
    print("  │   ├── hazard_results_for_gis.csv")
    print("  │   └── summary_statistics.png")
    print("  └── regional_summary/")
    print("      ├── Full_california_regional_summary.txt")
    print("      └── combined_windows_summary.csv")
    print()
    
    print("📈 VISUALIZATION FILES:")
    print("  🎯 Hazard Curves: Individual site hazard curves with design levels")
    print("  🗺️  GIS Data: CSV file for importing into QGIS/ArcGIS")
    print("  📊 Summary Plots: Statistics and performance metrics")
    print("  📋 Regional Report: Comprehensive text summary")
    print()
    
    print("🎨 PLOT FEATURES:")
    print("  ✅ USGS-style hazard curves with design probabilities")
    print("  ✅ Logarithmic axes with grid lines")
    print("  ✅ Annotated design points (2% in 50 years, 10% in 50 years)")
    print("  ✅ High-resolution PNG output (300 DPI)")
    print("  ✅ Professional styling with proper labels")

def show_comparison_with_usgs():
    """Show how results should compare with USGS maps"""
    
    print("\n🗺️  COMPARISON WITH USGS HAZARD MAPS:")
    print("="*60)
    print("📍 Los Angeles (34.05°N, -118.25°W)")
    print()
    print("Expected hazard levels (PGA):")
    print("  📊 2% in 50 years:  ~0.4-0.8g  (USGS: ~0.6g)")
    print("  📊 10% in 50 years: ~0.1-0.3g  (USGS: ~0.2g)")
    print()
    print("🎯 Success criteria:")
    print("  ✅ Results within 50% of USGS values")
    print("  ✅ Hazard curve shape looks reasonable")
    print("  ✅ Ground motions in expected range (0.0001-10g)")
    print("  ✅ No processing errors or warnings")

# For testing/debugging
if __name__ == "__main__":
    print("Enhanced Full California Single Site Configuration with Visualizations")
    print("=" * 80)
    
    # Load and validate configuration
    config = load_config()
    is_valid = validate_config()
    
    print(f"\nConfiguration valid: {is_valid}")
    print(f"Region: {config['region']['name']}")
    print(f"Mode: Single site (not grid)")
    print(f"Test site: {config['site']['sites'][0]}")
    print(f"Visualizations enabled: {config['output_settings']['plot_hazard_curves']}")
    print(f"Windows directory: {config['paths']['windows_dir']}")
    print(f"Output directory: {config['paths']['output_dir']}")
    
    # Show expected results
    preview_expected_outputs()
    show_comparison_with_usgs()
    
    if is_valid:
        print(f"\n🚀 READY TO RUN WITH VISUALIZATIONS:")
        print(f"python scripts/run_mc.py --region Full_california_single --mode sequential --pattern \"*0001*.csv\" --log-level INFO")
    else:
        print(f"\n❌ Fix validation issues first")