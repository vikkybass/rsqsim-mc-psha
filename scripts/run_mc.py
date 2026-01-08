#!/usr/bin/env python3
"""
Main Monte Carlo Orchestrator for RSQSim Ground Motion Simulation

This script coordinates the execution of ground motion simulations across
multiple time windows for specified regions.

Usage:
    python run_mc.py --region los_angeles --mode sequential
    python run_mc.py --region los_angeles --mode random --config custom_config.py
"""

import os
import sys
import argparse
import logging
import time
from pathlib import Path
from typing import Dict, List
import glob
import importlib.util

# Set up basic logging immediately to prevent UnboundLocalError
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.gm_simulator_main import run_regional_batch_simulation

def setup_main_logging(log_level: str = "INFO", log_file: str = None):
    """Setup main logging configuration"""
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create log formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Setup handlers
    handlers = [logging.StreamHandler(sys.stdout)]
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    # Configure root logger
    logging.basicConfig(
        level=numeric_level,
        handlers=handlers,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    return logging.getLogger(__name__)

def setup_cluster_environment():
    """Ensure cluster environment is properly configured"""
    import os
    
    # Set default paths if not in environment
    defaults = {
        'RSQSIM_WORK_DIR': '/scratch/olawoyiv/rsqsim_data',
        'RSQSIM_PROJECT_DIR': '/projects/ebelseismo/olawoyiv',
        'TMPDIR': '/scratch/olawoyiv/tmp'
    }
    
    for var, default in defaults.items():
        if var not in os.environ:
            os.environ[var] = default
    
    # Create directories
    for path in defaults.values():
        os.makedirs(path, exist_ok=True)

def load_config_from_file(config_path: str) -> Dict:
    """
    Dynamically load configuration from a Python file
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    try:
        # Convert to absolute path
        config_path = Path(config_path).resolve()
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        # Load the module
        spec = importlib.util.spec_from_file_location("config", config_path)
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        # Get the configuration
        if hasattr(config_module, 'load_config'):
            return config_module.load_config()
        else:
            raise AttributeError("Configuration file must have a 'load_config()' function")
            
    except Exception as e:
        raise RuntimeError(f"Error loading configuration from {config_path}: {e}")

def find_windows_directory(base_dir: str, region: str, mode: str) -> str:
    """
    Find the windows directory for a given region and mode
    FIXED: Better pattern matching for random/sequential directories
    """
    
    # Try various directory naming patterns (IMPROVED ORDER)
    possible_patterns = [
        f"{region}/{mode}",           # los_angeles/random  
        f"{region}_{mode}",           # los_angeles_random
        f"{mode}/{region}",           # random/los_angeles
        f"{region}/{mode}_windows",   # los_angeles/random_windows
        f"{region}_{mode}_windows",   # los_angeles_random_windows
        region,                       # Just region name
        mode,                         # Just mode name
        ""                           # Base directory itself
    ]
    
    logging.info(f"🔍 SEARCHING for {region} {mode} windows...")
    
    for pattern in possible_patterns:
        if pattern:
            test_dir = Path(base_dir) / pattern
        else:
            test_dir = Path(base_dir)

        logging.debug(f"  Checking: {test_dir}")

        if test_dir.exists():
            # Check if it contains CSV files
            csv_files = list(test_dir.glob("*.csv"))
            logging.debug(f"    Found {len(csv_files)} CSV files")

            if csv_files:
                # ADDITIONAL CHECK: Ensure files match the expected pattern
                expected_patterns = [
                    f"*{region}*{mode}*.csv",
                    f"*{mode}*{region}*.csv", 
                    f"*{region}*.csv",
                    "*.csv"
                ]
                
                for pattern_check in expected_patterns:
                    matching_files = list(test_dir.glob(pattern_check))
                    if matching_files:
                        logging.info(f"✅ Found {region} {mode} directory: {test_dir}")
                        logging.info(f"   Contains {len(matching_files)} matching files")
                        return str(test_dir)
    
    # If we get here, nothing was found
    logging.error(f"❌ No windows directory found for region '{region}' mode '{mode}' in {base_dir}")

    # List what's actually available
    base_path = Path(base_dir)
    if base_path.exists():
        logging.info(f"📂 Available directories in {base_dir}:")
        for item in base_path.iterdir():
            if item.is_dir():
                csv_count = len(list(item.glob("*.csv")))
                logging.info(f"   {item.name}/ ({csv_count} CSV files)")

    raise FileNotFoundError(f"No windows directory found for region '{region}' mode '{mode}' in {base_dir}")
    
def get_available_regions(windows_base_dir: str) -> List[str]:
    """
    Get list of available regions from windows directory
    FIXED: Better region detection including random/sequential modes
    """
    regions = set()
    windows_path = Path(windows_base_dir)
    
    if not windows_path.exists():
        return []
    
    # Look for region directories (both random and sequential)
    for item in windows_path.iterdir():
        if item.is_dir():
            # Handle various naming patterns
            dir_name = item.name
            
            # Pattern: region_mode (e.g., los_angeles_random)
            if '_' in dir_name:
                parts = dir_name.split('_')
                if len(parts) >= 2:
                    if parts[-1] in ['random', 'sequential']:
                        region = '_'.join(parts[:-1])  # Everything except last part
                        regions.add(region)
                    else:
                        regions.add(parts[0])  # First part
            else:
                # Pattern: just region name or mode name
                if dir_name not in ['random', 'sequential']:
                    # Check if it contains subdirectories
                    subdirs = [d.name for d in item.iterdir() if d.is_dir()]
                    if any(mode in subdirs for mode in ['random', 'sequential']):
                        regions.add(dir_name)
                    else:
                        regions.add(dir_name)
    
    # Also check for CSV files with region prefixes
    for csv_file in windows_path.glob("*.csv"):
        # Extract region from filename
        parts = csv_file.stem.split('_')
        if len(parts) >= 2:
            # Handle multi-word regions like "los_angeles"
            if len(parts) >= 3 and parts[1] in ['angeles', 'francisco']:
                regions.add(f"{parts[0]}_{parts[1]}")
            else:
                regions.add(parts[0])
    
    return sorted(list(regions))

def validate_simulation_setup(config: Dict, region: str, windows_dir: str, logger: logging.Logger) -> bool:
    """
    Validate that all required components are available for simulation
    
    Args:
        config: Configuration dictionary
        region: Region name
        windows_dir: Windows directory path
        logger: Logger instance
        
    Returns:
        True if setup is valid
    """
    issues = []
    
    # Check windows directory and files
    if not os.path.exists(windows_dir):
        issues.append(f"Windows directory not found: {windows_dir}")
    else:
        csv_files = glob.glob(os.path.join(windows_dir, "*.csv"))
        if not csv_files:
            issues.append(f"No CSV window files found in: {windows_dir}")
        else:
            logger.info(f"Found {len(csv_files)} window files")
    
    # Check output directory can be created
    output_dir = config['paths']['output_dir']
    logger.info(f"Output directory: {output_dir}")
    try:
        os.makedirs(output_dir, exist_ok=True)
    except Exception as e:
        issues.append(f"Cannot create output directory {output_dir}: {e}")
    
    # Check GMPE configuration (supports both NEW PyGMM and LEGACY configs)
    if 'gmpe' in config:
        # NEW structure with PyGMM
        gmpe_cfg = config['gmpe']
        logger.info(f"✅ GMPE mode: {'Ensemble' if gmpe_cfg.get('use_ensemble') else 'Single model'}")
        if gmpe_cfg.get('use_ensemble'):
            logger.info(f"   Models: {gmpe_cfg.get('models')}")
            logger.info(f"   Weights: {gmpe_cfg.get('ensemble_weights')}")
        else:
            logger.info(f"   Model: {gmpe_cfg.get('default_model')}")
    else:
        # LEGACY structure (old configs)
        if 'gmpe_config' in config.get('paths', {}):
            gmpe_config = config['paths']['gmpe_config']
            if not os.path.exists(gmpe_config):
                issues.append(f"GMPE configuration file not found: {gmpe_config}")
        
        # Check GMPE coefficient files (legacy models only)
        if 'gmpe_model' in config.get('site', {}):
            selected_model = config['site']['gmpe_model']
            if 'gmpe_files' in config and selected_model in config['gmpe_files']:
                gmpe_file = config['gmpe_files'][selected_model]
                if not os.path.exists(gmpe_file):
                    issues.append(f"GMPE coefficient file not found: {gmpe_file}")
    
    # Report issues
    if issues:
        logger.error("Validation issues found:")
        for issue in issues:
            logger.error(f"  ❌ {issue}")
        return False
    
    logger.info("✅ Simulation setup validation passed")
    return True

def run_batch_simulation(args) -> None:
    """
    Main function to run batch simulation
    
    Args:
        args: Command line arguments
    """
    
    # ✅ FIXED: Setup cluster environment FIRST
    setup_cluster_environment()

    logger = setup_main_logging(args.log_level)
    
    # ✅ Load configuration with mode support
    if args.config:
        config = load_config_from_file(args.config)
        logger.info(f"Loaded custom config: {args.config}")
    else:
        config = load_config_for_region(args.region, args.mode)
        logger.info(f"Loaded default config for region: {args.region}")
    
    # Setup logging
    log_dir = Path(config["paths"].get("log_dir", "logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"run_mc_{args.region}_{args.mode}_{int(time.time())}.log"
    
    logger = setup_main_logging(args.log_level, str(log_file))
    logger.info("✅ Configuration loaded successfu    ")
    logger.info("=" * 60)
    logger.info("RSQSim Monte Carlo Ground Motion Simulation")
    logger.info("=" * 60)
    logger.info(f"Region: {args.region}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Pattern: {args.pattern}")  # SHOW the pattern being used
    logger.info(f"Log {log_file}")
    
    start_time = time.time()
    
    try:
        # FIXED: Use the config's windows directory directly
        if 'paths' in config and 'windows_dir' in config['paths']:
            windows_dir = config['paths']['windows_dir']
            logger.info(f"Using config windows directory: {windows_dir}")
            
            # Verify the directory exists
            if not os.path.exists(windows_dir):
                logger.error(f"❌ Windows directory does not exist: {windows_dir}")
                
                # Try to find the correct directory
                base_windows = f"{os.environ.get('RSQSIM_WORK_DIR', '/scratch/olawoyiv/rsqsim_data')}/data/Catalog_4983/windows"
                alternative_dir = f"{base_windows}/{args.region}/{args.mode}"
                logger.info(f"Trying alternative: {alternative_dir}")
                
                if os.path.exists(alternative_dir):
                    windows_dir = alternative_dir
                    logger.info(f"✅ Found alternative windows directory: {windows_dir}")
                else:
                    logger.error(f"❌ No windows directory found for {args.region} {args.mode}")
                    return
            else:
                logger.info(f"✅ Windows directory exists: {windows_dir}")
        else:
            # Fallback to the old method if paths not in config
            base_windows_dir = f"{os.environ.get('RSQSIM_WORK_DIR', '/scratch/olawoyiv/rsqsim_data')}/data/Catalog_4983/windows"
            logger.info(f"Using fallback base windows directory: {base_windows_dir}")
            
            try:
                windows_dir = find_windows_directory(base_windows_dir, args.region, args.mode)
                logger.info(f"Found windows directory: {windows_dir}")
            except FileNotFoundError as e:
                logger.error(f"❌ {e}")
                
                # Show available regions
                available_regions = get_available_regions(base_windows_dir)
                if available_regions:
                    logger.info(f"Available regions: {', '.join(available_regions)}")
                else:
                    logger.error(f"No regions found in {base_windows_dir}")
                
                return
    
        # Validate setup
        logger.info("Validating simulation setup...")
        if not validate_simulation_setup(config, args.region, windows_dir, logger):
            logger.error("❌ Validation failed. Please fix issues before running.")
            return
        
        # FIXED: Use the correct output directory from config
        output_dir = config['paths']['output_dir']
        logger.info(f"Output directory: {output_dir}")
        
        # Run the simulation
        logger.info("Starting batch simulation...")
        run_regional_batch_simulation(
            region_name=args.region,
            windows_dir=windows_dir,
            output_dir=output_dir,
            config=config,
            window_pattern=args.pattern
        )
            
        # Calculate runtime
        end_time = time.time()
        runtime_minutes = (end_time - start_time) / 60
        
        logger.info("=" * 60)
        logger.info("🎉 Batch simulation completed successfully!")
        logger.info(f"Total runtime: {runtime_minutes:.1f} minutes")
        logger.info(f"Results saved to: {config['paths']['output_dir']}")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"❌ Simulation failed: {e}")
        logger.exception("Full error traceback:")
        sys.exit(1)



def load_config_for_region(region_name: str, mode: str = "sequential") -> Dict:
    """Auto-load config based on region name and mode"""
    try:
        config_module_name = f"configs.{region_name}_config"
        config_module = importlib.import_module(config_module_name)
        
        # Check if the config module supports mode-specific loading
        if hasattr(config_module, 'load_config_for_mode'):
            return config_module.load_config_for_mode(mode)
        elif hasattr(config_module, 'load_config'):
            config = config_module.load_config()
            # If mode is random, try to switch it
            if mode == "random" and hasattr(config_module, 'switch_to_random_mode'):
                config = config_module.switch_to_random_mode(config)
            return config
        else:
            raise AttributeError(f"No load_config function found in {config_module_name}")
            
    except ImportError:
        print(f"⚠️  No config for region '{region_name}', using base config")
        from configs.config import load_config
        return load_config()

def main():
    """Main entry point"""
    
    parser = argparse.ArgumentParser(
        description="RSQSim Monte Carlo Ground Motion Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run sequential windows for Los Angeles
  python run_mc.py --region los_angeles --mode sequential
  
  # Run random windows with custom config
  python run_mc.py --region los_angeles --mode random --config configs/custom_config.py
  
  # Run with debug logging
  python run_mc.py --region los_angeles --mode sequential --log-level DEBUG
  
  # Process only specific window pattern
  python run_mc.py --region los_angeles --mode sequential --pattern "*window_00[1-5].csv"
        """
    )
    
    parser.add_argument(
        "--region", 
        required=True,
        help="Region name (e.g., los_angeles, san_francisco)"
    )
    
    parser.add_argument(
        "--mode",
        choices=["sequential", "random"],
        default="sequential", 
        help="Window mode: sequential or random (default: sequential)"
    )
    
    parser.add_argument(
        "--config",
        help="Path to configuration file (default: auto-select based on region)"
    )
    
    parser.add_argument(
        "--pattern",
        default="*.csv",
        help="File pattern to match window files (default: *.csv)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--list-regions",
        action="store_true",
        help="List available regions and exit"
    )
    
    parser.add_argument(
        "--validate-only",
        action="store_true", 
        help="Only validate setup without running simulation"
    )
    
    args = parser.parse_args()
    
    # Handle list regions
    if args.list_regions:
        # Try to find windows base directory from default config
        try:
            default_config_path = f"configs/{args.region}_config.py"
            if os.path.exists(default_config_path):
                config = load_config_from_file(default_config_path)
                base_dir = config['paths']['windows_dir']
            else:
                base_dir = "data/Catalog_4983/windows"  # Fallback
            
            regions = get_available_regions(base_dir)
            if regions:
                print("Available regions:")
                for region in regions:
                    print(f"  - {region}")
            else:
                print(f"No regions found in {base_dir}")
                
        except Exception as e:
            print(f"Error listing regions: {e}")
        
        return
    
    # Only validate if config was explicitly provided
    if args.config and not os.path.exists(args.config):
        print(f"❌ Configuration file not found: {args.config}")
        sys.exit(1)
    
    # Run simulation
    run_batch_simulation(args)

if __name__ == "__main__":
    main()