import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Optional
import glob
import argparse
import time
import cProfile
import pstats
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from src.memory_manager import MemoryManager, BatchProcessor
from src.spatial_indexing import SpatialCatalogIndex
from src.rupture_geometry import (
    RSQSimGeometryReader,
    RSQSimCatalogReader,
    EventRupture
)

# Add the project root to the path to import from src
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def split_sites_across_nodes(sites):
    """FIXED: Force single-node processing to avoid result collection issues"""
    import os
    
    # CRITICAL FIX: Always process all sites on single node
    # This prevents result fragmentation across multiple nodes
    logger.info(f"🔄 Processing all {len(sites)} sites on single node (multi-node disabled)")
    logger.info("📌 This ensures complete results without complex node coordination")
    
    # Log what we would have done in multi-node mode (for debugging)
    node_id = int(os.environ.get('SLURM_PROCID', 0))
    total_nodes = int(os.environ.get('SLURM_NPROCS', 1))
    
    if total_nodes > 1:
        logger.info(f"ℹ️  SLURM detected {total_nodes} nodes, but forcing single-node processing")
        logger.info(f"ℹ️  Current node ID: {node_id}")
        logger.info(f"ℹ️  Without this fix, this node would process ~{len(sites)//total_nodes} sites")
        logger.info(f"ℹ️  Now processing all {len(sites)} sites for complete results")
    
    return sites  # Return ALL sites to ensure complete processing

def get_window_time_range_from_results(all_site_results):
    """Extract actual time range from synthetic events across all sites"""
    all_times = []
    
    for site_result in all_site_results:
        synthetic_events = site_result.get('synthetic_events', [])
        if synthetic_events:
            times = [event.time for event in synthetic_events]
            all_times.extend(times)
    
    if all_times:
        min_time = min(all_times)
        max_time = max(all_times)
        return f"{min_time:,.0f} - {max_time:,.0f} years"
    
    return "No time data"

# Add at the top of the file
def setup_cluster_storage():
    """Setup storage for cluster execution"""
    import os
    work_dir = os.environ.get('RSQSIM_WORK_DIR', '/scratch/olawoyiv/rsqsim_data')
    project_dir = os.environ.get('RSQSIM_PROJECT_DIR', '/projects/ebelseismo/olawoyiv')
    
    # Create directories
    os.makedirs(f"{work_dir}/output", exist_ok=True)
    os.makedirs(f"{work_dir}/logs", exist_ok=True)
    os.makedirs(f"{project_dir}/rsqsim_results", exist_ok=True)
    
    return work_dir, project_dir

# Add backup function
def backup_results(output_file):
    """Backup results to project directory"""
    import shutil
    import os
    project_dir = os.environ.get('RSQSIM_PROJECT_DIR', '/projects/ebelseismo/olawoyiv')
    if os.path.exists(output_file):
        backup_path = f"{project_dir}/rsqsim_results/{os.path.basename(output_file)}"
        shutil.copy2(output_file, backup_path)
        print(f"Backed up to: {backup_path}")

# Import ALL the functions from new_Ran.py that we need
from src.new_Ran import (
    load_gmpe_coefficients,
    get_accel,
    MaxAcc,
    calculate_hazard_values,
    output_site_results,
    setup_logging,
    logger,
    calculate_distance,
    safe_open_file,
    # Import visualization functions from new_Ran.py
    plot_hazard_curves,
    visualize_hazard_results,
    create_hazard_map,
    get_top_hazard_sites,
    generate_visualizations,
    export_hazard_for_gis
)

# Optional: Can still import log_a for backward compatibility
# from src.unified_gmpe import log_a
from src.gmpe_adapter import RSQSimEvent, RSQSimGMPEAdapter

def setup_region_logging(region_name: str, log_dir: str = "logs") -> None:
    """Setup logging for a specific region"""
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)
    
    log_file = log_path / f"gm_simulation_{region_name}.log"
    
    # Configure logging with both file and console handlers
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )

def load_regional_bounds(windows_metadata_file: str) -> Dict:
    """
    Load regional bounds from the time window metadata file
    """
    try:
        metadata_df = pd.read_csv(windows_metadata_file)
        
        # Extract bounds from metadata
        bounds = {
            'min_lat': metadata_df['min_lat'].iloc[0],
            'max_lat': metadata_df['max_lat'].iloc[0], 
            'min_lon': metadata_df['min_lon'].iloc[0],
            'max_lon': metadata_df['max_lon'].iloc[0],
            'region_name': metadata_df['region'].iloc[0] if 'region' in metadata_df.columns else 'unknown'
        }
        
        logger.info(f"Loaded regional bounds: {bounds}")
        return bounds
        
    except Exception as e:
        logger.error(f"Error loading regional bounds: {e}")
        raise

def generate_regional_sites(bounds: Dict, site_config: Dict) -> List[Tuple[float, float]]:
    """
    Generate sites based on configuration - FIXED VERSION
    """
    
    if site_config.get('grid_mode', True):
        # FIXED: Always try to use site_config bounds first, then bounds parameter
        lat_min = site_config.get('grid_lat_min')
        lat_max = site_config.get('grid_lat_max')
        lon_min = site_config.get('grid_lon_min')
        lon_max = site_config.get('grid_lon_max')
        
        # If site_config doesn't have bounds, use bounds parameter
        if lat_min is None and bounds:
            lat_min = bounds['min_lat']
            lat_max = bounds['max_lat']
            lon_min = bounds['min_lon'] 
            lon_max = bounds['max_lon']
        
        # Final fallback to defaults (this should rarely happen now)
        if lat_min is None:
            lat_min = 33.5
            lat_max = 34.5
            lon_min = -119.0
            lon_max = -118.0
            logger.warning("Using fallback grid bounds - check your configuration!")
        
        # Get spacing from config
        lat_spacing = site_config.get('grid_lat_spacing', 0.1)
        lon_spacing = site_config.get('grid_lon_spacing', 0.1)
        
        # FIXED: Generate grid using bounds and spacing (no hardcoded counts)
        lats = np.arange(lat_min, lat_max + lat_spacing/2, lat_spacing)
        lons = np.arange(lon_min, lon_max + lon_spacing/2, lon_spacing)
        
        sites = [(lat, lon) for lat in lats for lon in lons]
        
        logger.info(f"FIXED Grid generation:")
        logger.info(f"  Bounds: lat [{lat_min:.2f}, {lat_max:.2f}], lon [{lon_min:.2f}, {lon_max:.2f}]")
        logger.info(f"  Spacing: {lat_spacing:.3f}° lat, {lon_spacing:.3f}° lon")
        logger.info(f"  Generated: {len(lats)} lats × {len(lons)} lons = {len(sites)} sites")
        
    else:
        # Individual sites mode
        sites = site_config.get('sites', [(34.05, -118.25)])
        logger.info(f"Using {len(sites)} individual sites")
    
    return sites

def process_single_site(args):
    """
    Process a single site - FULLY FIXED with PyGMM integration
    
    FIXES:
    - Uses fixed gmpe_adapter with proper distance calculations
    - Passes EventRupture for accurate R_rup and R_jb
    - Includes width and dip in calculations
    """
    site_data, catalog_events, site_config, default_model, geometry, rupture_cache = args
    site_idx, site_lat, site_lon = site_data
    
    # Import here to avoid issues in multiprocessing
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info(f"Processing site {site_idx + 1}: ({site_lat:.4f}, {site_lon:.4f})")
    
    # Get filtering threshold from config
    min_threshold = site_config.get('min_ground_motion')
    if min_threshold is None:
        min_threshold = site_config.get('output_settings', {}).get('min_ground_motion', 0.05)
    
    max_distance = site_config.get('max_distance_km')
    if max_distance is None:
        max_distance = site_config.get('simulation', {}).get('filtering', {}).get('max_distance_km', 300.0)
    
    logger.debug(f"Using threshold: {min_threshold}g, max distance: {max_distance}km")
    vs30 = site_config.get('vs30', 760.0)
    
    # =========================================================================
    # FIX: Import from fixed modules
    # =========================================================================
    from src.gmpe_adapter import RSQSimGMPEAdapter
    
    adapter = RSQSimGMPEAdapter(
        geometry_reader=geometry,  # Pass geometry for distance calculations
        use_ensemble=True,         # Use NSHM 2023 weighted ensemble
        vs30=vs30,
        mechanism='strike-slip'
    )
    
    # Storage for significant events
    synthetic_events = []
    
    # Counters for statistics
    total_events_processed = 0
    events_above_threshold = 0
    events_below_threshold = 0
    events_too_distant = 0
    
    # Process each event from the RSQSim synthetic catalog
    for i, event in enumerate(catalog_events):
        total_events_processed += 1
        
        try:
            # STEP 1: Calculate epicentral distance first (fast filter)
            epicentral_dist = calculate_distance(
                site_lat, site_lon, 
                event.cent_lat, event.cent_lon,
                cached=True
            )
            
            # STEP 2: Early rejection using epicentral distance
            # Add 50km buffer to account for maximum rupture extent
            if epicentral_dist > max_distance + 50.0:
                events_too_distant += 1
                continue
            
            # =========================================================================
            # FIX: Get EventRupture for proper distance calculations
            # =========================================================================
            event_rupture = None
            if rupture_cache is not None and event.event_id in rupture_cache:
                try:
                    event_rupture = rupture_cache[event.event_id]
                    
                    # Quick distance check using rupture
                    rrup, rjb, nearest_patch = event_rupture.distance_to_site_latlon(
                        site_lat, site_lon
                    )
                    
                    # Skip if beyond max distance
                    if rrup > max_distance:
                        events_too_distant += 1
                        continue
                    
                    if i < 10:  # Log first few events for debugging
                        logger.debug(
                            f"Event {event.event_id}: M{event.magnitude:.2f}, "
                            f"Epicentral={epicentral_dist:.1f}km, "
                            f"Rrup={rrup:.1f}km, Rjb={rjb:.1f}km, "
                            f"Patches={event_rupture.n_patches}"
                        )
                    
                except Exception as e:
                    logger.debug(f"Rupture distance failed for event {event.event_id}: {e}")
                    event_rupture = None
                    
                    # Fallback distance check
                    if epicentral_dist > max_distance:
                        events_too_distant += 1
                        continue
            else:
                # No rupture cache - use epicentral distance
                if epicentral_dist > max_distance:
                    events_too_distant += 1
                    continue
            
            # =========================================================================
            # FIX: Calculate ground motion using fixed adapter
            # This now properly uses:
            # - EventRupture for accurate R_rup and R_jb
            # - Correct width calculation with dip
            # - Model-specific scenarios
            # - NSHM 2023 weighted ensemble
            # =========================================================================
            gm_result = adapter.calculate_ground_motion_from_event(
                event=event,
                site_lat=site_lat,
                site_lon=site_lon,
                event_rupture=event_rupture,  # ✅ Pass EventRupture for proper distances
                vs30=vs30,
                period=0.01  # PGA
            )
            
            # Extract results
            log10_pga = gm_result['log10_pga']
            pga_g = gm_result['pga_g']
            
            # Log detailed info for first few events
            if i < 5:
                logger.debug(
                    f"Event {event.event_id} ground motion: "
                    f"PGA={pga_g:.4f}g, "
                    f"Rrup={gm_result['distance_rup']:.2f}km, "
                    f"Rjb={gm_result['distance_jb']:.2f}km, "
                    f"Width={gm_result['width']:.2f}km, "
                    f"Dip={gm_result['dip']:.1f}°"
                )
            
            # =========================================================================
            # Filter by threshold
            # =========================================================================
            if pga_g >= min_threshold:
                # Create MaxAcc object for hazard analysis
                synthetic_event = MaxAcc(
                    time=event.time,
                    lat=event.cent_lat,
                    lon=event.cent_lon,
                    mag=event.magnitude,
                    log10_a=log10_pga,
                    epic_d=gm_result['distance_rup']  # Use proper rupture distance
                )
                
                synthetic_events.append(synthetic_event)
                events_above_threshold += 1
            else:
                events_below_threshold += 1
                
        except Exception as e:
            logger.warning(f"Error processing event {event.event_id}: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            continue
    
    # Log filtering statistics
    logger.info(f"Site {site_idx + 1} filtering results:")
    logger.info(f"  Total events in catalog: {total_events_processed}")
    logger.info(f"  Events too distant (>{max_distance}km): {events_too_distant}")
    logger.info(f"  Events processed for GM: {total_events_processed - events_too_distant}")
    logger.info(f"  Events above threshold ({min_threshold}g): {events_above_threshold}")
    logger.info(f"  Events below threshold (filtered): {events_below_threshold}")
    
    if total_events_processed > 0:
        filter_rate = (events_above_threshold / total_events_processed) * 100
        logger.info(f"  Retention rate: {filter_rate:.2f}%")
    
    return {
        'site_lat': site_lat,
        'site_lon': site_lon,
        'site_index': site_idx,
        'synthetic_events': synthetic_events,
        'filtering_stats': {
            'processed': total_events_processed,
            'above_threshold': events_above_threshold,
            'below_threshold': events_below_threshold,
            'too_distant': events_too_distant
        }
    }

def simulate_ground_motion_for_window(
    window_file: str,
    site_config: Dict,
    gmpe_config_path: str,
    output_dir: str,
    regional_bounds: Optional[Dict] = None,
    config: Optional[Dict] = None
) -> Tuple[str, str, str]:
    """
    Simulate ground motion for RSQSim catalog window - UPDATED with improved summary data
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualization subdirectories
    viz_output_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_output_dir, exist_ok=True)
    os.makedirs(os.path.join(viz_output_dir, "maps"), exist_ok=True)
    os.makedirs(os.path.join(viz_output_dir, "plots"), exist_ok=True)
    os.makedirs(os.path.join(viz_output_dir, "gis"), exist_ok=True)
    
    try:
        geometry_file = config.get('geom_paths', {}).get('geometry_file', '/home/olawoyiv/rsqsim_mc/data/Catalog_4983/geometry.flt')
        
        logger.info(f"Loading RSQSim geometry from {geometry_file}")
        geometry = RSQSimGeometryReader(geometry_file)
        logger.info(f"Loaded {len(geometry.patches)} fault patches")
    except Exception as e:
        logger.warning(f"Could not load geometry: {e}. Falling back to epicentral distance")
        geometry = None

    # Load catalog readers for rupture reconstruction
    try:
        catalog_base = config.get('geom_paths', {}).get('catalog_base', '/home/olawoyiv/rsqsim_mc/data/Catalog_4983')
        
        catalog_reader = RSQSimCatalogReader(
            f"{catalog_base}/catalog.eList",
            f"{catalog_base}/catalog.pList",
            f"{catalog_base}/catalog.dList",
            f"{catalog_base}/catalog.tList"
        )
        logger.info("Loaded RSQSim catalog readers for rupture distance")
    except Exception as e:
        logger.warning(f"Could not load catalog readers: {e}")
        catalog_reader = None
        
        
    # Load RSQSim synthetic catalog window
    try:
        catalog_df = pd.read_csv(window_file)
        logger.info(f"Loaded RSQSim synthetic catalog: {len(catalog_df)} events from {window_file}")
    except Exception as e:
        logger.error(f"Error loading window file {window_file}: {e}")
        raise
    
    from gmpe_adapter import RSQSimEvent, validate_catalog_format
    
    if validate_catalog_format(window_file):
        logger.info("✅ New RSQSim catalog format detected")
        
        # Load as RSQSimEvent objects
        catalog_events = []
        for idx, row in catalog_df.iterrows():
            try:
                event = RSQSimEvent.from_csv_row(row)
                catalog_events.append(event)
            except Exception as e:
                logger.warning(f"Error loading event {idx}: {e}")
                continue
        
        logger.info(f"✅ Loaded {len(catalog_events)} RSQSimEvent objects")
        logger.info(f"   Sample event: M{catalog_events[0].magnitude:.1f}, "
                   f"depth_tor={catalog_events[0].get_ztor():.1f}km, "
                   f"width={catalog_events[0].get_rupture_width():.1f}km")
        
        # Show catalog statistics
        logger.info(f"Catalog statistics:")
        mags = [e.magnitude for e in catalog_events]
        logger.info(f"  Magnitude range: {min(mags):.1f} to {max(mags):.1f}")
        
        lats = [e.cent_lat for e in catalog_events]
        lons = [e.cent_lon for e in catalog_events]
        logger.info(f"  Spatial bounds: lat [{min(lats):.2f}, {max(lats):.2f}], "
                   f"lon [{min(lons):.2f}, {max(lons):.2f}]")
        
    else:
        logger.error("❌ Catalog does not have new RSQSim format!")
        raise ValueError("Catalog missing required columns for new format")
    
    # Validate required columns
    required_cols = ['magnitude', 'latitude', 'longitude', 'depth']
    missing_cols = [col for col in required_cols if col not in catalog_df.columns]
    if missing_cols:
        logger.error(f"Missing required columns in {window_file}: {missing_cols}")
        raise ValueError(f"Missing columns: {missing_cols}")
    
    # Show catalog statistics
    logger.info(f"Catalog statistics:")
    logger.info(f"  Magnitude range: {catalog_df['magnitude'].min():.1f} to {catalog_df['magnitude'].max():.1f}")
    logger.info(f"  Spatial bounds: lat [{catalog_df['latitude'].min():.2f}, {catalog_df['latitude'].max():.2f}], "
               f"lon [{catalog_df['longitude'].min():.2f}, {catalog_df['longitude'].max():.2f}]")
    
    # Load GMPE coefficients
    try:
        if config and 'gmpe_files' in config:
            gmpe_files = config['gmpe_files']
            default_model = site_config.get('gmpe_model', config.get('default_model', 'BSSA14'))
        else:
            logger.warning("GMPE files not found in config, using fallback")
            from configs.config import load_gmpe_config
            base_config = load_gmpe_config()
            gmpe_files = base_config['gmpe_files']
            default_model = site_config.get('gmpe_model', base_config['default_model'])
        
        import src.new_Ran as new_ran
        new_ran.dataframes = load_gmpe_coefficients(gmpe_files, default_model)
        
        gmpe_coeffs = load_gmpe_coefficients(gmpe_files, default_model)
        logger.info(f"Loaded GMPE coefficients for {default_model}")
    except Exception as e:
        logger.error(f"Error loading GMPE coefficients: {e}")
        raise
    
    # Generate sites based on configuration
    if site_config.get('grid_mode', True):
        sites = generate_regional_sites(regional_bounds, site_config)
    else:
        sites = site_config.get('sites', [(site_config.get('lat', 34.05), site_config.get('lon', -118.25))])
        logger.info(f"Using {len(sites)} individual sites from configuration")
    
    # Split sites across nodes for parallel processing
    sites = split_sites_across_nodes(sites)

        
       
    # Set filtering parameters
    min_gm = config.get('output_settings', {}).get('min_ground_motion')
    if min_gm is None:
        raise ValueError("Missing 'min_ground_motion' in config['output_settings']. Please define it in your region config.")
    
    site_config['min_ground_motion'] = min_gm
    
    if 'max_distance_km' not in site_config:
        max_dist = None
        
        if config:
            max_dist = config.get('simulation', {}).get('filtering', {}).get('max_distance_km')
            if max_dist is None:
                max_dist = config.get('computation', {}).get('maximum_distance')
        
        if max_dist is None:
            max_dist = 300.0
            
        site_config['max_distance_km'] = max_dist
    
    logger.info(f"Ground motion filtering settings:")
    logger.info(f"  Minimum threshold: {site_config['min_ground_motion']}g")
    logger.info(f"  Maximum distance: {site_config['max_distance_km']}km")
    logger.info(f"  Expected to filter out ~95-99% of event-site combinations")
    
    # Parallel processing with filtering
    all_site_results = []

    # NEW: let SLURM decide; fall back to cpu_count; never clamp by config
    slurm_cpus = int(os.getenv("SLURM_CPUS_PER_TASK", "0")) or mp.cpu_count()    
    max_workers = min(slurm_cpus, len(sites))

    logger.info(f"Core Utilization Check:")
    logger.info(f"  SLURM_CPUS_PER_TASK env var: {os.getenv('SLURM_CPUS_PER_TASK', 'NOT SET')}")
    logger.info(f"  Detected available cores: {slurm_cpus}")
    logger.info(f"  Sites to process: {len(sites)}")
    logger.info(f"  Workers spawned: {max_workers}")
    if max_workers < slurm_cpus:
        logger.warning(f"  Only using {max_workers}/{slurm_cpus} cores (not enough sites)")

    logger.info(f"Using {max_workers} parallel workers for {len(sites)} sites")
    
    # Check what columns we have for debugging
    logger.debug(f"Window file columns: {list(catalog_df.columns)}")
    
    # Find actual column names (handles variations)
    required_mappings = {
        'magnitude': ['magnitude', 'mag', 'Magnitude', 'MAG'],
        'latitude': ['latitude', 'lat', 'Latitude', 'LAT'], 
        'longitude': ['longitude', 'lon', 'long', 'Longitude', 'LON'],
        'depth': ['depth', 'Depth', 'DEPTH'],
        'event_id': ['event_id', 'Event ID', 'EventID', 'event_ID', 'ID', 'id']
    }
    
    actual_columns = {}
    for field, possible_names in required_mappings.items():
        found = False
        for name in possible_names:
            if name in catalog_df.columns:
                actual_columns[field] = name
                found = True
                break
        
        if not found and field not in ['depth', 'event_id']:  # depth is optional
            raise ValueError(f"Required column '{field}' not found. Tried: {possible_names}. Available: {list(catalog_df.columns)}")
    
    # Create memory-efficient arrays instead of dict
    catalog_arrays = {
        'latitude': catalog_df[actual_columns['latitude']].values,
        'longitude': catalog_df[actual_columns['longitude']].values,
        'magnitude': catalog_df[actual_columns['magnitude']].values,
        'depth': catalog_df[actual_columns.get('depth', actual_columns['latitude'])].values if 'depth' in actual_columns else np.full(len(catalog_df), 10.0),
        'event_id': catalog_df[actual_columns['event_id']].values if 'event_id' in actual_columns else None
    }
    
    logger.info(f"Created arrays for {len(catalog_arrays['latitude'])} events")
    
    if catalog_arrays['event_id'] is not None:
        logger.info(f"✅ Event IDs available - will use finite-rupture distances")
    else:
        logger.warning(f"⚠️  No Event IDs found - will fall back to epicentral distances")
    
    # Pre-build ruptures for all unique events in this window (CRITICAL OPTIMIZATION)
    if catalog_reader and catalog_arrays['event_id'] is not None:
        logger.info("Pre-building rupture surfaces for all events...")
        unique_event_ids = np.unique(catalog_arrays['event_id'])
        
        rupture_cache = {}
        start_time = time.time()
        
        for idx, event_id in enumerate(unique_event_ids):
            try:
                rupture_cache[int(event_id)] = catalog_reader.get_event_rupture(
                    int(event_id), geometry
                )
                
                if (idx + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    logger.info(f"  Built {idx+1}/{len(unique_event_ids)} ruptures "
                            f"({elapsed:.1f}s, {elapsed/(idx+1):.2f}s per event)")
            except Exception as e:
                logger.warning(f"Could not build rupture for event {event_id}: {e}")
        
        logger.info(f"Rupture cache built: {len(rupture_cache)} events in {time.time()-start_time:.1f}s")
        
        # Analyze cache composition
        if rupture_cache:
            patch_counts = [rup.n_patches for rup in rupture_cache.values()]
            large_events = sum(1 for n in patch_counts if n > 100)
            huge_events = sum(1 for n in patch_counts if n > 500)
            
            logger.info(f"Cache composition:")
            logger.info(f"  Total events: {len(rupture_cache)}")
            logger.info(f"  Average patches/event: {np.mean(patch_counts):.1f}")
            logger.info(f"  Max patches: {max(patch_counts)}")
            logger.info(f"  Large events (>100 patches): {large_events}")
            logger.info(f"  Huge events (>500 patches): {huge_events}")
            
            if huge_events > 0:
                logger.info(f"  Note: {huge_events} events with >500 patches may be slow")
    else:
        rupture_cache = None
        
    # Prepare data for parallel processing
    site_data_list = [(i, lat, lon) for i, (lat, lon) in enumerate(sites)]
    
    # Create arguments for each worker (using arrays instead of dict)
    worker_args = [
        (site_data, catalog_arrays, site_config, default_model, geometry, rupture_cache)
        for site_data in site_data_list
    ]
    
    # Process sites in parallel with filtering
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_site = {
            executor.submit(process_single_site, args): args[0] 
            for args in worker_args
        }
        
        completed = 0
        total_events_above_threshold = 0
        total_events_processed = 0
        total_events_too_distant = 0
        
        for future in as_completed(future_to_site):
            try:
                site_result = future.result()
                all_site_results.append(site_result)
                completed += 1

                # Memory cleanup every 200 sites
                if completed % 200 == 0:
                    import gc
                    gc.collect()
                    
                # Track filtering statistics
                stats = site_result['filtering_stats']
                total_events_processed += stats['processed']
                total_events_above_threshold += stats['above_threshold']
                total_events_too_distant += stats['too_distant']
                
                # Log progress every 100 sites
                if completed % 100 == 0:
                    if total_events_processed > 0:
                        overall_filter_rate = (total_events_above_threshold / total_events_processed * 100)
                    else:
                        overall_filter_rate = 0
                        
                    logger.info(f"Completed {completed}/{len(sites)} sites ({100*completed/len(sites):.1f}%)")
                    logger.info(f"Overall filtering: {total_events_above_threshold:,} significant / {total_events_processed:,} processed ({overall_filter_rate:.2f}%)")
                    
            except Exception as e:
                site_data = future_to_site[future]
                logger.error(f"Error processing site {site_data}: {e}")
    
    # Final filtering statistics
    logger.info(f"\n===== GROUND MOTION PROCESSING COMPLETE =====")
    logger.info(f"Total sites processed: {completed}")
    logger.info(f"Total event-site combinations: {len(catalog_df) * completed:,}")
    logger.info(f"Events too distant (skipped): {total_events_too_distant:,}")
    logger.info(f"Ground motion calculations: {total_events_processed:,}")
    logger.info(f"Events above threshold (significant): {total_events_above_threshold:,}")
    logger.info(f"Events below threshold (filtered out): {total_events_processed - total_events_above_threshold:,}")
    
    if total_events_processed > 0:
        overall_filter_rate = ((total_events_processed - total_events_above_threshold) / total_events_processed) * 100
        logger.info(f"Overall filtering efficiency: {overall_filter_rate:.1f}% filtered out")
    
    avg_events_per_site = total_events_above_threshold / completed if completed > 0 else 0
    logger.info(f"Average significant events per site: {avg_events_per_site:.1f}")
    
    # Validation check
    if avg_events_per_site > 5000:
        logger.warning(f"⚠️  Many events per site ({avg_events_per_site:.0f}). Consider raising min_ground_motion threshold.")
    elif avg_events_per_site < 10:
        logger.warning(f"⚠️  Very few events per site ({avg_events_per_site:.0f}). Consider lowering min_ground_motion threshold.")
    else:
        logger.info(f"✅ Events per site looks reasonable for hazard analysis")
    
    # Now process each site to build hazard curves
    logger.info("\n===== BUILDING HAZARD CURVES =====")
    
    # Generate output filenames
    base_name = os.path.splitext(os.path.basename(window_file))[0]
    main_output_file = os.path.join(output_dir, f"hazard_results_{base_name}.txt")
    summary_csv = os.path.join(output_dir, f"summary_{base_name}.csv")
    
    # Get synduration from config
    synduration = None
    if config:
        synduration = config.get('synduration')
        if synduration is None:
            synduration = config.get('simulation', {}).get('synduration', 50000)
    else:
        synduration = 50000
    
    # Get output settings
    output_settings = config.get('output_settings', {})
    if not output_settings:
        output_settings = {
            "full_output": False,
            "min_ground_motion": site_config['min_ground_motion'],
            "probability_type": "annual",
            "probabilities": [(50, 0.02), (50, 0.1)]
        }
    
    # Initialize main output file with header
    try:
        with safe_open_file(main_output_file, 'w') as f:
            f.write(f"RSQSim Ground Motion Hazard Analysis Results\n")
            f.write(f"=" * 80 + "\n")
            f.write(f"Synthetic catalog window: {os.path.basename(window_file)}\n")
            f.write(f"Ground motion model: {default_model}\n")
            f.write(f"Total sites analyzed: {len(all_site_results)}\n")
            f.write(f"Minimum ground motion threshold: {site_config['min_ground_motion']}g\n")
            f.write(f"Maximum distance: {site_config['max_distance_km']}km\n")
            f.write(f"Total significant events across all sites: {total_events_above_threshold:,}\n")
            f.write(f"Random seed: {config.get('seed', config.get('simulation', {}).get('seed', 42))}\n")
            f.write(f"Synthetic duration: {synduration} years\n")
            f.write(f"=" * 80 + "\n\n")
        
        logger.info(f"Initialized main output file: {main_output_file}")
    except Exception as e:
        logger.error(f"Error creating main output file: {e}")
        return None, None
    
    # Process each site for hazard analysis
    successful_sites = 0
    summary_data = []
    
    # Store all results for visualization
    all_results = {}
    
    for i, site_result in enumerate(all_site_results):
        site_lat = site_result['site_lat']
        site_lon = site_result['site_lon']
        synthetic_events = site_result['synthetic_events']
        
        logger.info(f"Building hazard curve for site {i+1}/{len(all_site_results)}: ({site_lat:.4f}, {site_lon:.4f})")
        
        if synthetic_events:
            try:
                # Calculate hazard values using new_Ran.py's function
                hazard_values = calculate_hazard_values(
                    synthetic_events=synthetic_events,
                    site_lat=site_lat,
                    site_lon=site_lon,
                    synduration=synduration,
                    output_settings=output_settings,
                    config=config
                )
                
                # Save results using new_Ran.py's output format
                output_site_results(
                    site_lat=site_lat,
                    site_lon=site_lon,
                    synthetic_events=synthetic_events,
                    hazard_values=hazard_values,
                    output_settings=output_settings,
                    output_file=main_output_file,
                    config=config,
                    mode='a'
                )
                
                # Store results for visualization
                all_results[(site_lat, site_lon)] = {
                    'hazard_values': hazard_values,
                    'synthetic_events': synthetic_events
                }
                
                # ===== UPDATED SUMMARY DATA COLLECTION =====
                summary_record = {
                    'site_lat': site_lat,
                    'site_lon': site_lon,
                    'num_significant_events': len(synthetic_events),
                }
                
                # Add REALISTIC ground motion statistics (not extreme outliers)
                if synthetic_events:
                    gm_values = [10**event.log10_a for event in synthetic_events]
                    gm_array = np.array(gm_values)
                    
                    # FIXED: Use percentiles instead of absolute max/min
                    summary_record['ground_motion_min_g'] = np.percentile(gm_array, 5)    # 5th percentile
                    summary_record['ground_motion_median_g'] = np.percentile(gm_array, 50) # Median
                    summary_record['ground_motion_95th_g'] = np.percentile(gm_array, 95)   # 95th percentile (realistic "max")
                    summary_record['ground_motion_99th_g'] = np.percentile(gm_array, 99)   # 99th percentile
                    
                    # Only include absolute max if it's reasonable (< 5g)
                    abs_max = np.max(gm_array)
                    if abs_max <= 5.0:
                        summary_record['ground_motion_max_g'] = abs_max
                    else:
                        summary_record['ground_motion_max_g'] = summary_record['ground_motion_99th_g']
                        summary_record['max_value_capped'] = True  # Flag for unrealistic max
                        summary_record['actual_max_g'] = abs_max   # Store but flag as unrealistic
                else:
                    summary_record['ground_motion_min_g'] = 0
                    summary_record['ground_motion_median_g'] = 0
                    summary_record['ground_motion_95th_g'] = 0
                    summary_record['ground_motion_99th_g'] = 0
                    summary_record['ground_motion_max_g'] = 0
                
                # Add hazard curve summary
                if hazard_values.get('ground_motions'):
                    summary_record['hazard_curve_points'] = len(hazard_values['ground_motions'])
                    # Use reasonable max from hazard curve
                    hc_gm = hazard_values['ground_motions']
                    reasonable_hc_max = min(max(hc_gm), 2.0)  # Cap at 2g for display
                    summary_record['hazard_curve_max_gm'] = reasonable_hc_max
                
                # MOST IMPORTANT: Add prob values 
                if hazard_values.get('probabilities'):
                    for prob_type, value in hazard_values['probabilities'].items():
                        if 'non_exceed_50yr_0.98' in prob_type:
                            summary_record['probs_2pct_50yr_g'] = value  # MCE
                        elif 'non_exceed_50yr_0.95' in prob_type:
                            summary_record['probs_5pct_50yr_g'] = value  
                        elif 'non_exceed_50yr_0.90' in prob_type:
                            summary_record['probs_10pct_50yr_g'] = value

                        # Store all probability values with cleaner names
                        summary_record[f'hazard_{prob_type}'] = value
                
                # Add quality flags
                if synthetic_events:
                    # Flag sites with suspiciously high values
                    if summary_record.get('ground_motion_95th_g', 0) > 3.0:
                        summary_record['high_values_flag'] = True
                    
                    # Flag sites with very few events (unreliable statistics)
                    if len(synthetic_events) < 10:
                        summary_record['low_data_flag'] = True
                    
                    # Calculate coefficient of variation (measure of variability)
                    if len(gm_values) > 1:
                        summary_record['ground_motion_cov'] = np.std(gm_values) / np.mean(gm_values)
                
                summary_data.append(summary_record)
                successful_sites += 1
                
            except Exception as e:
                logger.error(f"Error processing hazard curves for site ({site_lat}, {site_lon}): {e}")
        else:
            logger.warning(f"No significant events for site ({site_lat}, {site_lon}) - skipping hazard analysis")
    
    # ===== UPDATED SUMMARY SAVING WITH BETTER COLUMNS =====
    def save_summary_with_better_columns(summary_data, summary_csv):
        """Save summary data with engineering-focused column ordering"""
        if not summary_data:
            return None
            
        summary_df = pd.DataFrame(summary_data)
        
        # Define column order (most important first)
        preferred_columns = [
            'site_lat', 'site_lon',
            'num_significant_events',
            'probs_2pct_annual_g',      # MCE (was probs_2pct_annual_g)
            'probs_5pct_50yr_g',         # 5% in 50 years
            'probs_10pct_annual_g',     # DBE (was probs_10pct_annual_
            'ground_motion_median_g',    # Typical value
            'ground_motion_95th_g',      # Practical "maximum"
            'ground_motion_99th_g',      # Near-maximum
            'ground_motion_max_g',       # Capped maximum
            'ground_motion_min_g',       # Minimum
            'hazard_curve_points',
            'hazard_curve_max_gm',
            'ground_motion_cov',         # Variability measure
            'high_values_flag',          # Quality flags
            'low_data_flag',
            'max_value_capped',
            'actual_max_g'               # Raw max (might be unrealistic)
        ]
        
        # Reorder columns (keep any extra columns at the end)
        available_columns = [col for col in preferred_columns if col in summary_df.columns]
        extra_columns = [col for col in summary_df.columns if col not in preferred_columns]
        final_columns = available_columns + extra_columns
        
        summary_df = summary_df[final_columns]
        summary_df.to_csv(summary_csv, index=False, float_format='%.6f')
        
        logger.info(f"Saved summary with engineering-focused columns to {summary_csv}")
        
        # Log summary statistics
        if 'probs_2pct_50yr_g' in summary_df.columns:
            probs_2pct = summary_df['probs_2pct_50yr_g'].dropna()
            if len(probs_2pct) > 0:
                logger.info(f"MCE (2% in 50yr): {probs_2pct.min():.3f}g to {probs_2pct.max():.3f}g")
        
        if 'ground_motion_95th_g' in summary_df.columns:
            gm_95th = summary_df['ground_motion_95th_g'].dropna()
            if len(gm_95th) > 0:
                logger.info(f"95th Percentile Ground Motions: {gm_95th.min():.3f}g to {gm_95th.max():.3f}g")
        
        # Flag any concerning sites
        if 'high_values_flag' in summary_df.columns:
            high_value_sites = summary_df['high_values_flag'].sum() if 'high_values_flag' in summary_df.columns else 0
            if high_value_sites > 0:
                logger.warning(f"⚠️  {high_value_sites} sites have unusually high ground motion values (>3g)")
        
        if 'max_value_capped' in summary_df.columns:
            capped_sites = summary_df['max_value_capped'].sum() if 'max_value_capped' in summary_df.columns else 0
            if capped_sites > 0:
                logger.warning(f"⚠️  {capped_sites} sites had unrealistic maximum values that were capped")
        
        return summary_csv
    
    # Save summary CSV with improved format
    try:
        if summary_data:
            summary_csv = save_summary_with_better_columns(summary_data, summary_csv)
            logger.info(f"Saved summary data for {len(summary_data)} sites to {summary_csv}")
        else:
            logger.warning("No summary data to save")
            summary_csv = None
    except Exception as e:
        logger.error(f"Error saving summary data: {e}")
        summary_csv = None
    
    # Generate visualizations
    if all_results:
        logger.info(f"\n===== GENERATING VISUALIZATIONS =====")

        try:
            # Update config with output file path for visualization functions
            viz_config = config.copy() if config else {}
            viz_config['output_file'] = main_output_file

            # Generate hazard curves for top sites
            if viz_config.get('output_settings', {}).get('plot_hazard_curves', True):
                logger.info("Generating hazard curves for highest hazard sites...")
                try:
                    plot_hazard_curves(all_results, viz_output_dir, viz_config)
                    logger.info(f"✅ Hazard curves saved to {viz_output_dir}/plots/")
                except Exception as e:
                    logger.warning(f"Error generating hazard curves: {e}")

            # Generate hazard map
            if len(all_results) > 1:
                logger.info("Generating hazard map...")
                try:
                    map_file = create_hazard_map(all_results, viz_config)
                    if map_file:
                        logger.info(f"✅ Hazard map saved to {map_file}")

                except Exception as e:
                    logger.warning(f"Error generating hazard map: {e}")
            else:
                logger.info("Skipping hazard map (single site)")

            # Export GIS data
            if viz_config.get('output_settings', {}).get('export_gis_csv', True):
                logger.info("Generating CSV file for GIS...")
                try:
                    gis_dir = os.path.join(viz_output_dir, "gis")
                    os.makedirs(gis_dir, exist_ok=True)
                    csv_file = export_hazard_for_gis(all_results, viz_config, gis_dir)
                    if csv_file:
                        logger.info(f"✅ GIS CSV file saved to {csv_file}")
                except Exception as e:
                    logger.warning(f"Error generating GIS CSV: {e}")

            logger.info(f"✅ All visualizations saved to: {viz_output_dir}")

        except Exception as e:
            logger.warning(f"Error in visualization generation: {e}")
   
    # Extract actual time range from synthetic events
    window_time_range = get_window_time_range_from_results(all_site_results)
    logger.info(f"📅 Window time range: {window_time_range}")
    
    # Store time range in a metadata file for later use
    try:
        base_name = os.path.splitext(os.path.basename(window_file))[0]
        metadata_file = os.path.join(output_dir, f"window_metadata_{base_name}.txt")
        
        with open(metadata_file, 'w') as f:
            f.write(f"Window file: {os.path.basename(window_file)}\n")
            f.write(f"Time range: {window_time_range}\n")
            f.write(f"Sites processed: {successful_sites}\n")
            f.write(f"Total significant events: {total_events_above_threshold}\n")
            
        logger.info(f"📝 Saved window metadata: {metadata_file}")
        
    except Exception as e:
        logger.warning(f"⚠️  Could not save window metadata: {e}")
    
    # Final summary (existing code)
    logger.info(f"\n===== HAZARD ANALYSIS COMPLETE =====")
    logger.info(f"Successfully processed: {successful_sites} sites")
    logger.info(f"Failed: {len(all_site_results) - successful_sites} sites")
    logger.info(f"Main results saved to: {main_output_file}")
    if summary_csv:
        logger.info(f"Summary data saved to: {summary_csv}")
    
    # UPDATED RETURN: Include time range
    return main_output_file, summary_csv, window_time_range

def switch_to_random_mode(config: dict):
    """
    Switch a configuration from sequential to random mode
    Updates all relevant paths
    Args:
        config: Configuration dictionary
    Returns:
        Updated configuration with random paths
    """
    if 'paths' in config:
        for key in config['paths']:
            if 'sequential' in str(config['paths'][key]):
                config['paths'][key] = config['paths'][key].replace('sequential', 'random')
    
    return config    
    

def run_regional_batch_simulation(
    region_name: str,
    windows_dir: str,
    output_dir: str,
    config: Dict,
    window_pattern: str = "*.csv"
) -> None:
    """
    Run batch simulation for all windows in a region - COMPATIBLE with your existing configs and scripts
    """

    # Record start time for performance tracking
    start_time = time.time()

    # FIXED: Initialize memory management with proper error handling
    try:
        memory_manager = MemoryManager(
            max_memory_gb=config.get('max_memory_gb', 16),
            warning_threshold=0.8
        )
        
        batch_processor = BatchProcessor(
            batch_size=config.get('batch_size', 50),
            max_events_per_site=config.get('max_events_per_site', 10000)
        )
        
        # Connect memory manager to batch processor
        batch_processor.set_memory_manager(memory_manager)
        
        logger.info("✅ Memory management initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize memory management: {e}")
        logger.info("🔄 Continuing without advanced memory management")
        memory_manager = None
        batch_processor = None

    # ===== ADD RESOURCE MONITORING =====
    import psutil
    import shutil
    
    # Get system information
    process = psutil.Process()
    start_memory_mb = process.memory_info().rss / 1024 / 1024
    
    logger.info(f"🔧 System Status at Start:")
    logger.info(f"  Available Memory: {psutil.virtual_memory().available / (1024**3):.1f}GB")
    logger.info(f"  Process Memory: {start_memory_mb:.1f}MB")
    logger.info(f"  CPU Count: {psutil.cpu_count()}")
    
    # Check disk space
    try:
        output_disk_free = shutil.disk_usage(output_dir).free / (1024**3)
        logger.info(f"  Output Disk Space: {output_disk_free:.1f}GB free")
        
        if output_disk_free < 5.0:
            logger.warning(f"⚠️  Low disk space: {output_disk_free:.1f}GB")
    except Exception as e:
        logger.warning(f"⚠️  Could not check disk space: {e}")
    
    # Validate memory requirements
    config_memory_gb = config.get('max_memory_gb', 16)
    available_memory_gb = psutil.virtual_memory().available / (1024**3)
    
    if config_memory_gb > available_memory_gb * 0.8:
        logger.warning(f"⚠️  Config requests {config_memory_gb}GB but only {available_memory_gb:.1f}GB available")
    else:
        logger.info(f"✅ Memory requirements OK: {config_memory_gb}GB requested, {available_memory_gb:.1f}GB available")
    
    # Create batch processor
    batch_processor = BatchProcessor(
        batch_size=config.get('batch_size', 50),
        max_events_per_site=config.get('max_events_per_site', 10000)
    )
    
    # Setup logging for this region
    setup_region_logging(region_name)
    
    # Simple mode detection from paths
    detected_mode = "sequential"  # default
    if "random" in windows_dir.lower() or "random" in output_dir.lower():
        detected_mode = "random"
    elif "sequential" in windows_dir.lower() or "sequential" in output_dir.lower():
        detected_mode = "sequential"
    
    logger.info(f"Starting RSQSim batch simulation for region: {region_name}")
    logger.info(f"Mode: {detected_mode}")
    logger.info(f"Windows directory: {windows_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    # Verify directories exist
    if not os.path.exists(windows_dir):
        logger.error(f"❌ Windows directory does not exist: {windows_dir}")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"✅ Output directory ready: {output_dir}")
    
    # Load regional bounds if metadata file exists
    metadata_file = os.path.join(windows_dir, f"{region_name}_windows_metadata.csv")
    regional_bounds = None
    
    if os.path.exists(metadata_file):
        try:
            regional_bounds = load_regional_bounds(metadata_file)
            logger.info(f"Using regional bounds for site generation")
        except Exception as e:
            logger.warning(f"Could not load regional bounds, using config defaults: {e}")

    window_files = []

    # FIRST: Try the user-specified pattern
    if window_pattern != "*.csv":  # User provided a specific pattern
        logger.info(f"Using user-specified pattern: {window_pattern}")
        search_path = os.path.join(windows_dir, window_pattern)
        window_files = sorted(glob.glob(search_path))
        
        if window_files:
            logger.info(f"✅ Found {len(window_files)} files with user pattern: {window_pattern}")
        else:
            logger.warning(f"⚠️  No files found with user pattern: {window_pattern}")
            logger.info("Available files in directory:")
            all_files = [f for f in os.listdir(windows_dir) if f.endswith('.csv')]
            for file in all_files[:10]:  # Show first 10
                logger.info(f"  - {file}")
            if len(all_files) > 10:
                logger.info(f"  ... and {len(all_files) - 10} more files")

    # FALLBACK: If no user pattern or user pattern failed, try automatic detection
    if not window_files:
        logger.info("Trying automatic pattern detection...")
        search_patterns = [
            f"{region_name}*{detected_mode}*.csv",  # Full_california_single*sequential*.csv
            f"{region_name}*.csv",                  # Full_california_single*.csv  
            f"*{detected_mode}*.csv",               # *sequential*.csv
            "*.csv"                                 # fallback to all CSV files
        ]
        
        for pattern in search_patterns:
            search_path = os.path.join(windows_dir, pattern)
            found_files = sorted(glob.glob(search_path))
            if found_files:
                logger.info(f"✅ Found {len(found_files)} files with automatic pattern: {pattern}")
                window_files = found_files
                break

    if not window_files:
        logger.error(f"❌ No window files found in {windows_dir}")
        logger.info("Available files:")
        try:
            for file in os.listdir(windows_dir):
                if file.endswith('.csv'):
                    logger.info(f"  - {file}")
        except Exception as e:
            logger.error(f"Cannot list directory: {e}")
        return

    logger.info(f"Found {len(window_files)} window files to process")
    for i, file in enumerate(window_files[:5]):  # Show first 5
        logger.info(f"  {i+1}. {os.path.basename(file)}")
    if len(window_files) > 5:
        logger.info(f"  ... and {len(window_files) - 5} more files")
    
    # Create region-specific output directory
    mode_output_dir = output_dir

    os.makedirs(mode_output_dir, exist_ok=True)

    # Process each window
    successful_windows = 0
    failed_windows = 0
    
    # Store results from all windows for regional visualization
    all_windows_results = {}
    
    for i, window_file in enumerate(window_files):
        logger.info(f"\n" + "="*80)
        logger.info(f"Processing window {i+1}/{len(window_files)}: {os.path.basename(window_file)}")
        logger.info(f"="*80)

        # ===== ADD MEMORY CHECK BEFORE EACH WINDOW =====
        # Monitor memory usage before each window
        if i > 0:  # Skip for first window
            current_memory_mb = process.memory_info().rss / 1024 / 1024
            memory_increase_mb = current_memory_mb - start_memory_mb
            
            logger.info(f"📊 Memory Status:")
            logger.info(f"  Current: {current_memory_mb:.1f}MB (+{memory_increase_mb:+.1f}MB)")
            logger.info(f"  Available: {psutil.virtual_memory().available / (1024**3):.1f}GB")
            
            # Check for memory issues
            if current_memory_mb > config_memory_gb * 1024 * 0.8:
                logger.warning(f"⚠️  High memory usage: {current_memory_mb:.1f}MB")
                import gc
                gc.collect()
                after_gc_mb = process.memory_info().rss / 1024 / 1024
                logger.info(f"  After garbage collection: {after_gc_mb:.1f}MB")
        try:
            result = simulate_ground_motion_for_window(
                window_file=window_file,
                site_config=config['site'],
                gmpe_config_path=config['paths']['gmpe_config'],
                output_dir=mode_output_dir,
                regional_bounds=regional_bounds,
                config=config
            )

            # FIXED: Handle both 2-tuple and 3-tuple returns
            if isinstance(result, tuple) and len(result) == 3:
                main_output, summary_output, window_time_range = result
            elif isinstance(result, tuple) and len(result) == 2:
                main_output, summary_output = result
                window_time_range = "Unknown time range"
            else:
                main_output = result
                summary_output = None
                window_time_range = "Unknown time range"
            
            if main_output:
                successful_windows += 1
                
                # Store window results with time range for multi-window analysis
                window_name = os.path.splitext(os.path.basename(window_file))[0]
                all_windows_results[window_name] = {
                    'main_output': main_output,
                    'summary_output': summary_output,
                    'time_range': window_time_range,  # NEW: Store actual time range
                    'window_file': window_file
                }
                
                logger.info(f"✅ Successfully processed window: {os.path.basename(window_file)}")
                logger.info(f"   Results: {main_output}")
                logger.info(f"   Time range: {window_time_range}")  # NEW: Log time range
                if summary_output:
                    logger.info(f"   Summary: {summary_output}")
            else:
                failed_windows += 1
                logger.warning(f"❌ Failed to process window: {os.path.basename(window_file)}")
                
        except MemoryError as e:
            failed_windows += 1
            logger.error(f"💾 Memory error processing {os.path.basename(window_file)}: {e}")
            # Try to free memory
            import gc
            gc.collect()
            logger.info("🧹 Attempted memory cleanup")
            
        except FileNotFoundError as e:
            failed_windows += 1
            logger.error(f"📁 File not found processing {os.path.basename(window_file)}: {e}")
            
        except ValueError as e:
            failed_windows += 1
            logger.error(f"📊 Data error processing {os.path.basename(window_file)}: {e}")
            
        except Exception as e:
            failed_windows += 1
            logger.error(f"❌ Unexpected error processing {os.path.basename(window_file)}: {e}")
            logger.debug(f"Error details: {type(e).__name__}")
            
            # Log the error but continue with other windows
            logger.info(f"⏭️  Continuing with remaining windows...")
            continue

    # ========== FIXED: MULTI-WINDOW ANALYSIS ==========
    logger.info(f"\n===== CHECKING MULTI-WINDOW ANALYSIS CONDITIONS =====")
    logger.info(f"Successful windows: {successful_windows}")
    logger.info(f"Windows results collected: {len(all_windows_results)}")
    
    # NEW: Check if this is a single window job (parallel execution)
    # Skip multi-window analysis for individual parallel jobs
    if window_pattern != "*.csv" and "*window_*" in window_pattern:
        logger.info(f"🔄 Individual window job detected (pattern: {window_pattern})")
        logger.info(f"⏭️  Skipping multi-window analysis - will be run after all jobs complete")
        logger.info(f"💡 Use run_multiwindow.py to run multi-window analysis after all jobs finish")
    
    elif successful_windows > 1:
        logger.info(f"✅ Multiple windows detected, but SKIPPING multi-window analysis")
        logger.info(f"💡 This prevents memory issues and conflicts with dedicated multi-window tools")
        logger.info(f"💡 Run multi-window analysis separately using:")
        logger.info(f"   python run_multiwindow.py")
        logger.info(f"   or use the dedicated multi_window_analysis.py module")
    
    else:
        logger.info(f"⚠️  Single window processed - multi-window analysis not applicable")
        logger.info(f"   Current: {successful_windows} successful, {failed_windows} failed")
        
    # ========== NEW: GENERATE REGIONAL SUMMARY VISUALIZATIONS ==========
    if successful_windows > 0:
        logger.info(f"\n===== GENERATING REGIONAL SUMMARY =====")
        
        try:
            # Create regional summary directory
            regional_summary_dir = os.path.join(mode_output_dir, "regional_summary")
            os.makedirs(regional_summary_dir, exist_ok=True)
            
            # Generate regional summary report
            generate_regional_summary_report(
                region_name=region_name,
                mode=detected_mode,  
                windows_results=all_windows_results,
                output_dir=regional_summary_dir,
                config=config
            )

            # If multiple windows, generate comparison visualizations
            
            logger.info(f"✅ Regional summary saved to: {regional_summary_dir}")
            
        except Exception as e:
            logger.warning(f"Error generating regional summary: {e}")
    
    # Final summary
    logger.info(f"\n" + "="*80)
    logger.info(f"BATCH SIMULATION COMPLETED FOR REGION: {region_name.upper()}")
    logger.info(f"="*80)
    logger.info(f"Successfully processed: {successful_windows} windows")
    logger.info(f"Failed: {failed_windows} windows")
    logger.info(f"Success rate: {(successful_windows/(successful_windows+failed_windows))*100:.1f}%")
    logger.info(f"Results saved to: {mode_output_dir}")
    logger.info(f"="*80)

    # Add multi-window info to summary
    if successful_windows > 1:
        logger.info(f"Multi-window comparisons: {mode_output_dir}/visualizations/multi_window_comparisons")
    
    logger.info(f"="*80)

    # FIXED: Final memory summary with proper error handling
    logger.info(f"\n🧠 FINAL MEMORY SUMMARY")
    try:
        if memory_manager:
            memory_manager.log_memory_summary()
        else:
            runtime = time.time() - start_time
            logger.info(f"⏱️  Runtime: {runtime/60:.1f} minutes")
            logger.info(f"🔧 Memory management was disabled")
    except Exception as e:
        logger.warning(f"⚠️  Could not generate memory summary: {e}")
    finally:
        logger.info(f"✅ Batch simulation completed")

def generate_regional_summary_report(region_name: str, mode: str, windows_results: Dict, output_dir: str, config: Dict):
    """
    Generate a comprehensive regional summary report combining all windows
    """
    import csv
    from datetime import datetime

    logger.info("Generating regional summary report...")
    summary_file = os.path.join(output_dir, f"{region_name}_{mode}_regional_summary.txt")
    
    try:
        with open(summary_file, 'w') as f:
            f.write(f"RSQSim Regional Ground Motion Hazard Summary\n")
            f.write(f"=" * 80 + "\n")
            f.write(f"Region: {region_name}\n")
            f.write(f"Mode: {mode}\n")
            f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Ground Motion Model: {config.get('default_model', 'Unknown')}\n")
            f.write(f"Total Windows Processed: {len(windows_results)}\n")
            f.write(f"=" * 80 + "\n\n")
            
            # Process each window summary
            total_sites = 0
            total_events = 0
            
            for window_name, results in windows_results.items():
                f.write(f"Window: {window_name}\n")
                f.write(f"  Main Results: {results['main_output']}\n")
                if results['summary_output']:
                    f.write(f"  Summary CSV: {results['summary_output']}\n")
                    
                    # Try to read summary statistics
                    try:
                        summary_df = pd.read_csv(results['summary_output'])
                        sites_in_window = len(summary_df)
                        events_in_window = summary_df['num_significant_events'].sum()
                        
                        f.write(f"  Sites Analyzed: {sites_in_window}\n")
                        f.write(f"  Significant Events: {events_in_window:,}\n")
                        f.write(f"  Max Ground Motion: {summary_df['ground_motion_max_g'].max():.3f}g\n")
                        
                        total_sites += sites_in_window
                        total_events += events_in_window
                        
                    except Exception as e:
                        f.write(f"  Error reading summary: {e}\n")
                
                f.write("\n")
            
            # Regional totals
            f.write(f"REGIONAL TOTALS:\n")
            f.write(f"  Total Sites: {total_sites:,}\n")
            f.write(f"  Total Significant Events: {total_events:,}\n")
            f.write(f"  Average Events per Site: {total_events/total_sites:.1f}\n" if total_sites > 0 else "  Average Events per Site: N/A\n")
            
        logger.info(f"✅ Regional summary report saved to: {summary_file}")
        
    except Exception as e:
        logger.error(f"Error generating regional summary report: {e}")


if __name__ == "__main__":
    # Example usage when run directly - COMPATIBLE with your existing workflow
    if len(sys.argv) == 1:
        # Default example
        from configs.los_angeles_config import load_config
        
        config = load_config()
        run_regional_batch_simulation(
            region_name="los_angeles",
            windows_dir="data/Catalog_4983/windows/los_angeles/sequential",
            output_dir="data/output",
            config=config
        )
    else:
        # Command-line usage
        import argparse
        
        parser = argparse.ArgumentParser(description="Run RSQSim batch simulation for a region")
        parser.add_argument("--region", type=str, required=True, help="Region name (e.g., los_angeles)")
        parser.add_argument("--windows_dir", type=str, required=True, help="Directory containing window CSV files")
        parser.add_argument("--output_dir", type=str, required=True, help="Directory to save outputs")
        parser.add_argument("--config_file", type=str, required=True, help="Path to configuration file (YAML or JSON)")
        parser.add_argument("--window_pattern", type=str, default="*.csv", help="Pattern to match window files (default: *.csv)")
        
        args = parser.parse_args()
        
        # Load config
        try:
            config = load_config(args.config_file)
            logger.info(f"✅ Loaded configuration from {args.config_file}")
        except Exception as e:
            logger.error(f"❌ Failed to load configuration: {e}")
            sys.exit(1)
        
        run_regional_batch_simulation(
            region_name=args.region,
            windows_dir=args.windows_dir,
            output_dir=args.output_dir,
            config=config,
            window_pattern=args.window_pattern
        )