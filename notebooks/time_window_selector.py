"""
Time Window Selector for RSQSim Catalogs
Generates time windows from master earthquake catalogs for Monte Carlo hazard analysis.

This module provides functionality to:
1. Load RSQSim catalogs from various formats (CSV, binary .out files)
2. Generate sequential or random time windows
3. Save windowed catalogs for downstream analysis
"""

import os
import pandas as pd
import numpy as np
from typing import List, Union, Tuple, Optional
from pathlib import Path
import logging
from datetime import datetime

# RSQSim API imports
from rsqsim_api.catalogue.catalogue import RsqSimCatalogue
import rsqsim_api.io.rsqsim_constants as csts

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_catalog(input_path: str, reproject: Optional[List] = None) -> pd.DataFrame:
    """
    Load catalog from CSV or RSQSim binary files using the RSQSim API.
    
    Parameters:
    -----------
    input_path : str
        Path to catalog file (.csv or .out)
    reproject : List, optional
        [source_epsg, target_epsg] for coordinate reprojection
        
    Returns:
    --------
    pd.DataFrame
        Standardized catalog with columns: event_id, time, magnitude, latitude, longitude, depth
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Catalog file not found: {input_path}")
    
    logger.info(f"Loading catalog from: {input_path}")
    
    if input_path.suffix == ".csv":
        # Load CSV catalog
        df = pd.read_csv(input_path)
        
        # Standardize column names
        column_mapping = {
            "Event ID": "event_id",
            "event_id": "event_id", 
            "EventID": "event_id",
            "id": "event_id",
            "Occurrence Time (s)": "time",
            "time": "time",
            "t0": "time", 
            "origin_time": "time",
            "Magnitude": "magnitude",
            "magnitude": "magnitude",
            "mw": "magnitude",
            "mag": "magnitude",
            "Hypocenter Latitude": "latitude",
            "latitude": "latitude",
            "lat": "latitude",
            "y": "latitude",
            "Hypocenter Longitude": "longitude", 
            "longitude": "longitude",
            "lon": "longitude",
            "x": "longitude",
            "Hypocenter Depth (km)": "depth",
            "depth": "depth",
            "z": "depth"
        }
        
        available_mappings = {old: new for old, new in column_mapping.items() if old in df.columns}
        df = df.rename(columns=available_mappings)
        
        # Ensure required columns exist
        required_cols = ["event_id", "time", "magnitude", "latitude", "longitude"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in CSV: {missing_cols}. Available: {list(df.columns)}")
        
        # Clean and convert all numeric fields with improved cleaning
        numeric_fields = ["magnitude", "depth", "time", "latitude", "longitude"]
        for col in numeric_fields:
            if col in df.columns:
                # Convert to string and clean thoroughly
                df[col] = df[col].astype(str).str.strip()
                
                # Remove trailing periods and other common issues
                df[col] = df[col].str.rstrip('.')  # Remove trailing periods
                df[col] = df[col].str.rstrip(',')  # Remove trailing commas  
                df[col] = df[col].str.replace(r'^\.+$', '', regex=True)  # Remove strings of just periods
                df[col] = df[col].str.replace(r'^\s*$', '', regex=True)  # Remove whitespace-only strings
                
                # Replace empty strings and common non-numeric values with NaN
                df[col] = df[col].replace(['', 'nan', 'NaN', 'NULL', 'null', 'N/A', 'n/a', '-'], np.nan)
                
                # Convert to numeric, coercing errors to NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Replace infinite values with NaN
                df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        
        # Convert depth to km if likely in meters
        if "depth" in df.columns and df["depth"].notna().any():
            # Check if depth values suggest they're in meters (typically > 1000 for earthquake depths)
            if df["depth"].min() < -1000:
                df["depth"] = df["depth"] / 1000.0
        
        # Set default depth if missing
        if "depth" not in df.columns:
            df["depth"] = 0.0
            
        # Set default event_id if missing
        if "event_id" not in df.columns:
            df["event_id"] = range(len(df))
            
    elif input_path.suffix == ".out":
        # Load RSQSim binary catalog
        directory = input_path.parent
        possible_prefixes = [
            input_path.stem.replace("eqs.", "").replace(".out", ""),
            "catalog",
            "rundir*", 
            ""
        ]
        
        cat = None
        for prefix in possible_prefixes:
            try:
                logger.info(f"Attempting to load with prefix: '{prefix}'")
                cat = RsqSimCatalogue.from_catalogue_file_and_lists(
                    str(input_path),
                    list_file_directory=str(directory),
                    list_file_prefix=prefix,
                    reproject=reproject
                )
                logger.info(f"Successfully loaded with prefix: '{prefix}'")
                break
            except (FileNotFoundError, ValueError) as e:
                logger.debug(f"Failed with prefix '{prefix}': {e}")
                continue
                
        if cat is None:
            raise ValueError(f"Could not load RSQSim catalog. Check that list files exist in {directory}")
            
        df = cat.catalogue_df.copy()
        df = df.rename(columns={
            "t0": "time",
            "mw": "magnitude", 
            "x": "longitude",
            "y": "latitude",
            "z": "depth"
        })
        
        if "depth" in df.columns:
            df["depth"] = df["depth"] / 1000.0
            
        df["event_id"] = df.index
        logger.info(f"Loaded {len(df)} events from RSQSim catalog")
        
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}. Use .csv or .out")
    
    # Final sort and index reset
    df = df.sort_values("time").reset_index(drop=True)
    
    # Validate catalog (optional)
    validate_catalog_data(df)
    
    logger.info(f"Catalog loaded successfully: {len(df)} events")
    logger.info(f"Time range: {df['time'].min():.0f} to {df['time'].max():.0f} seconds")
    
    # Check for valid magnitudes before logging range
    valid_mags = df['magnitude'].notna() & np.isfinite(df['magnitude'])
    if valid_mags.any():
        logger.info(f"Magnitude range: {df.loc[valid_mags, 'magnitude'].min():.2f} to {df.loc[valid_mags, 'magnitude'].max():.2f}")
        
        # Report how many infinite values were discarded
        infinite_count = (df['magnitude'] == np.inf).sum() + (df['magnitude'] == -np.inf).sum()
        if infinite_count > 0:
            logger.info(f"Discarded {infinite_count} infinite magnitude values")
    else:
        logger.warning("No valid magnitude values found in catalog")
    
    return df


def validate_catalog_data(df: pd.DataFrame) -> None:
    """
    Validate catalog data quality and consistency.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Catalog dataframe to validate
    """
    required_cols = ["event_id", "time", "magnitude", "latitude", "longitude", "depth"]
    
    # Check required columns
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check for NaN values
    for col in required_cols:
        if df[col].isna().any():
            logger.warning(f"Found NaN values in column '{col}'")
    
    # Validate ranges
    if df["magnitude"].min() < 0 or df["magnitude"].max() > 12:
        logger.warning(f"Unusual magnitude range: {df['magnitude'].min():.2f} to {df['magnitude'].max():.2f}")
    
    if not (-90 <= df["latitude"].min() and df["latitude"].max() <= 90):
        logger.warning(f"Latitude values outside valid range: {df['latitude'].min():.2f} to {df['latitude'].max():.2f}")
    
    if not (-180 <= df["longitude"].min() and df["longitude"].max() <= 180):
        logger.warning(f"Longitude values outside valid range: {df['longitude'].min():.2f} to {df['longitude'].max():.2f}")


def generate_sequential_windows(
    df: pd.DataFrame, 
    window_years: float, 
    overlap: float = 0.0,
    min_events: int = 10
) -> List[pd.DataFrame]:
    """
    Generate sequential time windows from catalog.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input catalog
    window_years : float
        Window length in years
    overlap : float
        Overlap fraction between windows (0.0 to 0.99)
    min_events : int
        Minimum events required per window
        
    Returns:
    --------
    List[pd.DataFrame]
        List of windowed catalogs
    """
    if not 0.0 <= overlap < 1.0:
        raise ValueError("Overlap must be between 0.0 and 1.0 (exclusive)")
    
    window_seconds = window_years * csts.seconds_per_year
    start_time = df["time"].min()
    end_time = df["time"].max()
    step_seconds = window_seconds * (1 - overlap)
    
    logger.info(f"Generating sequential windows:")
    logger.info(f"  Window length: {window_years} years ({window_seconds:.0f} seconds)")
    logger.info(f"  Overlap: {overlap:.1%}")
    logger.info(f"  Step size: {step_seconds:.0f} seconds")
    
    windows = []
    current_time = start_time
    window_count = 0
    
    while current_time + window_seconds <= end_time:
        # Extract events in current window
        mask = (df["time"] >= current_time) & (df["time"] < current_time + window_seconds)
        window_df = df[mask].copy()
        
        # Check minimum events requirement
        if len(window_df) >= min_events:
            # Add window metadata
            window_df.attrs = {
                "window_id": window_count + 1,
                "start_time": current_time,
                "end_time": current_time + window_seconds,
                "window_years": window_years,
                "overlap": overlap
            }
            windows.append(window_df)
            window_count += 1
        else:
            logger.debug(f"Skipping window {window_count + 1}: only {len(window_df)} events (min: {min_events})")
        
        current_time += step_seconds
    
    logger.info(f"Generated {len(windows)} sequential windows")
    return windows


def generate_random_windows(
    df: pd.DataFrame, 
    window_years: float, 
    num_samples: int,
    min_events: int = 10,
    seed: Optional[int] = None
) -> List[pd.DataFrame]:
    """
    Generate random time windows from catalog.
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input catalog
    window_years : float
        Window length in years
    num_samples : int
        Number of random windows to generate
    min_events : int
        Minimum events required per window
    seed : int, optional
        Random seed for reproducibility
        
    Returns:
    --------
    List[pd.DataFrame]
        List of windowed catalogs
    """
    if seed is not None:
        np.random.seed(seed)
        
    window_seconds = window_years * csts.seconds_per_year
    start_time = df["time"].min()
    end_time = df["time"].max() - window_seconds
    
    if end_time <= start_time:
        raise ValueError(f"Catalog duration ({(df['time'].max() - df['time'].min()) / csts.seconds_per_year:.1f} years) "
                        f"is shorter than window length ({window_years} years)")
    
    logger.info(f"Generating random windows:")
    logger.info(f"  Window length: {window_years} years ({window_seconds:.0f} seconds)")
    logger.info(f"  Number of samples: {num_samples}")
    logger.info(f"  Random seed: {seed}")
    
    windows = []
    attempts = 0
    max_attempts = num_samples * 10  # Prevent infinite loop
    
    while len(windows) < num_samples and attempts < max_attempts:
        # Random start time
        t0 = np.random.uniform(start_time, end_time)
        t1 = t0 + window_seconds
        
        # Extract events in window
        mask = (df["time"] >= t0) & (df["time"] < t1)
        window_df = df[mask].copy()
        
        # Check minimum events requirement
        if len(window_df) >= min_events:
            # Add window metadata
            window_df.attrs = {
                "window_id": len(windows) + 1,
                "start_time": t0,
                "end_time": t1,
                "window_years": window_years,
                "random_sample": True
            }
            windows.append(window_df)
        
        attempts += 1
    
    if len(windows) < num_samples:
        logger.warning(f"Only generated {len(windows)} windows (requested {num_samples}) "
                      f"after {attempts} attempts. Consider reducing min_events or window_years.")
    
    logger.info(f"Generated {len(windows)} random windows")
    return windows


def save_window_catalogs(
    windows: List[pd.DataFrame], 
    output_dir: str, 
    prefix: str = "window",
    save_metadata: bool = True
) -> str:
    """
    Save windowed catalogs to CSV files.
    
    Parameters:
    -----------
    windows : List[pd.DataFrame]
        List of windowed catalogs
    output_dir : str
        Output directory path
    prefix : str
        Filename prefix for window files
    save_metadata : bool
        Whether to save metadata summary file
        
    Returns:
    --------
    str
        Path to metadata file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    metadata = []
    saved_files = []
    
    logger.info(f"Saving {len(windows)} windows to {output_path}")
    
    for i, window_df in enumerate(windows):
        # Generate filename
        filename = f"{prefix}_{i+1:04d}.csv"
        filepath = output_path / filename
        
        # Save window catalog
        window_df.to_csv(filepath, index=False)
        saved_files.append(str(filepath))
        
        # Collect metadata
        meta = {
            "window_id": i + 1,
            "filename": filename,
            "filepath": str(filepath),
            "num_events": len(window_df),
            "start_time": window_df["time"].min() if not window_df.empty else None,
            "end_time": window_df["time"].max() if not window_df.empty else None,
            "duration_years": (window_df["time"].max() - window_df["time"].min()) / csts.seconds_per_year if not window_df.empty else 0,
            "min_magnitude": window_df["magnitude"].min() if not window_df.empty else None,
            "max_magnitude": window_df["magnitude"].max() if not window_df.empty else None,
            "mean_magnitude": window_df["magnitude"].mean() if not window_df.empty else None
        }
        
        # Add window-specific metadata if available
        if hasattr(window_df, 'attrs') and window_df.attrs:
            meta.update({f"window_{k}": v for k, v in window_df.attrs.items()})
        
        metadata.append(meta)
    
    # Save metadata file
    metadata_df = pd.DataFrame(metadata)
    metadata_path = output_path / "window_metadata.csv"
    
    if save_metadata:
        metadata_df.to_csv(metadata_path, index=False)
        logger.info(f"Saved metadata to: {metadata_path}")
    
    # Save file list
    file_list_path = output_path / "window_files.txt"
    with open(file_list_path, 'w') as f:
        for filepath in saved_files:
            f.write(f"{filepath}\n")
    
    logger.info(f"Saved {len(windows)} window files and metadata")
    
    return str(metadata_path)


def generate_time_windows(
    catalog_input: Union[str, pd.DataFrame],
    window_length_years: float,
    mode: str = "sequential",
    overlap: float = 0.0,
    num_samples: int = 100,
    output_dir: str = "./windows",
    prefix: str = "window",
    min_events: int = 10,
    seed: Optional[int] = None,
    reproject: Optional[List] = None
) -> Tuple[List[pd.DataFrame], str]:
    """
    Main function to generate time windows from catalog.
    
    Parameters:
    -----------
    catalog_input : str or pd.DataFrame
        Path to catalog file or loaded DataFrame
    window_length_years : float
        Length of each window in years
    mode : str
        'sequential' or 'random'
    overlap : float
        Overlap fraction for sequential windows (0.0 to 0.99)
    num_samples : int
        Number of samples for random windows
    output_dir : str
        Output directory for saved windows
    prefix : str
        Filename prefix for window files
    min_events : int
        Minimum events required per window
    seed : int, optional
        Random seed for reproducible random sampling
    reproject : List, optional
        [source_epsg, target_epsg] for coordinate reprojection
        
    Returns:
    --------
    Tuple[List[pd.DataFrame], str]
        List of windowed catalogs and path to metadata file
    """
    # Load catalog if needed
    if isinstance(catalog_input, str):
        df = load_catalog(catalog_input, reproject=reproject)
    else:
        df = catalog_input.copy()
        validate_catalog_data(df)
    
    # Generate windows based on mode
    if mode.lower() == "sequential":
        windows = generate_sequential_windows(
            df, window_length_years, overlap=overlap, min_events=min_events
        )
    elif mode.lower() == "random":
        windows = generate_random_windows(
            df, window_length_years, num_samples=num_samples, 
            min_events=min_events, seed=seed
        )
    else:
        raise ValueError(f"Mode must be 'sequential' or 'random', got: {mode}")
    
    if not windows:
        raise ValueError("No valid windows generated. Check window_length_years and min_events parameters.")
    
    # Save windows
    metadata_path = save_window_catalogs(windows, output_dir, prefix=prefix)
    
    # Print summary
    logger.info(f"\nWindow Generation Summary:")
    logger.info(f"  Mode: {mode}")
    logger.info(f"  Window length: {window_length_years} years")
    logger.info(f"  Windows generated: {len(windows)}")
    logger.info(f"  Output directory: {output_dir}")
    logger.info(f"  Metadata file: {metadata_path}")
    
    return windows, metadata_path


def load_window_metadata(metadata_path: str) -> pd.DataFrame:
    """
    Load window metadata from CSV file.
    
    Parameters:
    -----------
    metadata_path : str
        Path to metadata CSV file
        
    Returns:
    --------
    pd.DataFrame
        Window metadata
    """
    return pd.read_csv(metadata_path)


def get_window_summary_stats(windows: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Calculate summary statistics for generated windows.
    
    Parameters:
    -----------
    windows : List[pd.DataFrame]
        List of windowed catalogs
        
    Returns:
    --------
    pd.DataFrame
        Summary statistics
    """
    stats = []
    
    for i, window_df in enumerate(windows):
        if window_df.empty:
            continue
            
        stat = {
            "window_id": i + 1,
            "num_events": len(window_df),
            "duration_years": (window_df["time"].max() - window_df["time"].min()) / csts.seconds_per_year,
            "min_mag": window_df["magnitude"].min(),
            "max_mag": window_df["magnitude"].max(),
            "mean_mag": window_df["magnitude"].mean(),
            "events_per_year": len(window_df) / ((window_df["time"].max() - window_df["time"].min()) / csts.seconds_per_year),
            "lat_range": window_df["latitude"].max() - window_df["latitude"].min(),
            "lon_range": window_df["longitude"].max() - window_df["longitude"].min()
        }
        stats.append(stat)
    
    return pd.DataFrame(stats)