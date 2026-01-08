import math
import csv
import os
import logging
import traceback
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import numpy.ma as ma
from scipy.spatial import cKDTree
import numpy as np
from datetime import datetime
from itertools import accumulate
import random
import pandas as pd
from geopy.distance import geodesic
import matplotlib.pyplot as plt
import argparse
import pygmt
from pathlib import Path
from src.unified_gmpe import log_a

# Create a module-level logger
logger = logging.getLogger(__name__)

def setup_logging(log_level=logging.INFO):
    """Set up logging with the specified level"""
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

# Constants related to random number generation and math
MBIG = 1000000000
MSEED = 161803398
MZ = 0
FAC = 1.0 / MBIG
SEED = 1

# Maximum limits for arrays and simulations
NRECMAX = 20000  # Maximum number of earthquakes allowed in the catalog
NGM = 50  # Number of ground motion values allowed
NGMVALUES = 5000  # Maximum number of ground motion values saved in analysis
SITES = 5000  # Maximum number of sites to analyze individually
# Global variable for GMPE coefficients
dataframes = {}

# Constants for optimization
PERFORMANCE_SETTINGS = {
    'batch_size': 10000,
    'chunk_size': 50000,
    'cache_size': 10000,
    'max_workers': 4,
    'distance_threshold': 300.0,  # km
    'min_magnitude': 4.0,
    'precision': 4
}

def init_random_seed(seed_value):
    """Initialize the random number generator with a seed value"""
    random.seed(seed_value)
    np.random.seed(seed_value)
    
class Catalog:
    """
    Enhanced Catalog class with management functionality for large datasets.
    Handles both individual entries and catalog-wide operations.
    """
    def __init__(self, space="", date="", orig_time="", lat=0.0, lon=0.0, mmi=0, 
                 mag=0.0, state="", desc="", catalog_file=None, chunk_size=5000):
        # Individual entry attributes
        self.space = space     # Space for future use       
        self.date = date       # Date of event       
        self.orig_time = orig_time    # Original time of event
        self.lat = lat                # Latitude of event
        self.lon = lon                # Longitude of event
        self.mmi = mmi                # Modified Mercalli Intensity/ Maximum Intensity
        self.mag = mag                # Magnitude of event
        self.state = state            # State of event if available
        self.desc = desc              # Other description of event

        # Catalog management attributes
        self.catalog_file = catalog_file        # Catalog file path
        self.chunk_size = chunk_size        # Chunk size for processing
        self.spatial_index = None           # KD-tree for spatial queries
        self.entries = []                   # List of catalog entries
        self.completeness = None            # Completeness configuration
        
        # Initialize if catalog file provided
        if catalog_file:
            self.load_catalog()
    
    def __lt__(self, other):
        """
        Define less than comparison for Catalog objects.
        This enables sorting of catalog entries.
        
        You can choose what makes one catalog "less than" another.
        Common options are comparing by date, magnitude, or distance.
        """
        # Compare by date (earlier date is "less than")
        if hasattr(self, 'date') and hasattr(other, 'date'):
            return self.date < other.date
        
        # Fallback to magnitude if date not available
        if hasattr(self, 'mag') and hasattr(other, 'mag'):
            return self.mag < other.mag
        
        # Default comparison if neither attribute is available
        return id(self) < id(other)
    
    def set_completeness(self, completeness_config):
        """Set completeness configuration using modern datetime handling"""
        if completeness_config and completeness_config.get("enabled"):
            self.completeness = CatalogCompleteness(completeness_config)
            
            # Filter entries using vectorized operations if possible
            if isinstance(self.entries, pd.DataFrame):
                mask = self.entries.apply(lambda row: 
                    self.completeness.is_complete(row['mag'], row['date']), axis=1)
                self.entries = self.entries[mask]
            else:
                # Filter list-based entries
                self.entries = [entry for entry in self.entries 
                            if self.completeness.is_complete(entry.mag, entry.date)]
            
            # Rebuild spatial index after filtering
            self._build_spatial_index()
    
    def load_catalog(self):
        """
        Load catalog with improved memory management, using existing chunk processing
        but with modern datetime handling.
        """
        try:
            total_records = 0
            chunk_count = 0
            
            logger.info(f"Starting to read catalog from {self.catalog_file}")
            
            # Process the file in chunks using pandas
            for chunk in pd.read_csv(self.catalog_file, chunksize=self.chunk_size):
                chunk_count += 1
                valid_records = 0
                
                for _, row in chunk.iterrows():
                    try:
                        # Use existing chunk entry processing
                        self._process_chunk_entry(row)
                        valid_records += 1
                        
                        # Build spatial index periodically
                        if len(self.entries) >= self.chunk_size:
                            self._build_spatial_index()
                            logger.info(f"Built spatial index for {len(self.entries)} records")
                            
                    except (ValueError, TypeError) as e:
                        logger.warning(f"Warning: Skipping invalid row due to {str(e)}")
                        continue
                    except Exception as e:
                        logger.warning(f"Warning: Unexpected error processing row: {str(e)}")
                        continue
                
                total_records += valid_records
                logger.info(f"Processed chunk {chunk_count}: {valid_records} valid records")
                
                # Optional: Clear memory periodically
                if len(self.entries) > self.chunk_size * 2:
                    self._build_spatial_index()  # Update index before clearing
                    self.entries = self.entries[-self.chunk_size:]  # Keep only most recent chunk
                    
            # Final spatial index update
            self._build_spatial_index()
            
            logger.info(f"Successfully loaded {total_records} records from catalog")
            logger.info(f"Final catalog contains {len(self.entries)} entries")
            
        except pd.errors.EmptyDataError:
            raise Exception("The catalog file is empty")
        except FileNotFoundError:
            raise Exception(f"Catalog file not found: {self.catalog_file}")
        except Exception as e:
            raise Exception(f"Error loading catalog: {str(e)}")
      
    def _process_chunk_entry(self, row):
        """Process a single row with modern datetime handling"""
        try:
            # Validate required fields
            required_fields = ['year', 'month', 'day', 'hour', 'minute', 'second', 'lat', 'long', 'mag']
            missing_fields = [field for field in required_fields if field not in row]
            if missing_fields:
                raise ValueError(f"Missing required fields: {missing_fields}")

            # Format date and time strings
            date_str = f"{int(row['year']):04d}{int(row['month']):02d}{int(row['day']):02d}"
            time_str = f"{int(row['hour']):02d}{int(row['minute']):02d}{int(row['second']):02d}"

            # Validate the date can be parsed
            try:
                datetime.strptime(date_str, "%Y%m%d")
            except ValueError as e:
                raise ValueError(f"Invalid date format: {e}")

            # Create entry
            entry = Catalog(
                date=date_str,
                orig_time=time_str,
                lat=float(row['lat']),
                lon=float(row['long']),
                mag=float(row['mag']),
                desc=row.get('comment', '')
            )
            
            # Apply completeness filter if configured
            if (self.completeness is None or 
                self.completeness.is_complete(entry.mag, entry.date)):
                self.entries.append(entry)

        except (ValueError, TypeError) as e:
            raise ValueError(f"Error processing row: {str(e)}")

    def _build_spatial_index(self):
        """Build KD-tree for spatial queries"""
        if self.entries:
            coords = [(entry.lat, entry.lon) for entry in self.entries]
            self.spatial_index = cKDTree(coords)
            
    def find_nearby_events(self, lat, lon, radius):
        """
        Find events within specified radius using optimized distance calculations
        
        Parameters:
        -----------
        lat, lon : float
            Site latitude and longitude
        radius : float
            Search radius in kilometers
        
        Returns:
        --------
        list
            List of catalog entries within the radius, sorted by distance
        """
        # Use spatial index if available for initial filtering
        if self.spatial_index is not None:
            # Convert radius to approximate degrees for initial filtering (rough approximation)
            radius_deg = radius / 111.0  # ~111 km per degree at the equator
            
            # Query spatial index for potential candidates (this is fast)
            indices = self.spatial_index.query_ball_point([lat, lon], radius_deg)
            
            if not indices:
                return []
            
            # Get candidate entries
            candidates = [self.entries[i] for i in indices]
            
            # If we have many candidates, use vectorized distance for efficient filtering
            if len(candidates) > 50:
                # Extract coordinates
                coords = np.array([(entry.lat, entry.lon) for entry in candidates])
                
                # Compute distances using vectorized calculation
                distances = calculate_distance(lat, lon, coords=coords, vectorized=True)
                
                # Filter by exact distance
                valid_indices = np.where(distances <= radius)[0]
                
                # Create distance-entry pairs for sorting
                distance_entries = [(distances[i], candidates[i]) for i in valid_indices]
                
            else:
                # For fewer candidates, direct calculation with caching is efficient
                distance_entries = []
                for entry in candidates:
                    dist = calculate_distance(lat, lon, entry.lat, entry.lon, cached=True)
                    if dist <= radius:
                        distance_entries.append((dist, entry))
        else:
            # Fall back to iterating through all entries if no spatial index
            distance_entries = []
            for entry in self.entries:
                dist = calculate_distance(lat, lon, entry.lat, entry.lon, cached=True)
                if dist <= radius:
                    distance_entries.append((dist, entry))
        
        # Sort by distance
        distance_entries.sort()
        
        # Return sorted entries
        return [entry for _, entry in distance_entries]

    def filter_by_magnitude(self, min_magnitude):
        """Filter catalog entries by minimum magnitude"""
        return [entry for entry in self.entries if entry.mag >= min_magnitude]

    def filter_by_date(self, start_date=None, end_date=None):
        """Filter catalog entries by date range"""
        filtered = self.entries
        if start_date:
            filtered = [e for e in filtered if e.date >= start_date]
        if end_date:
            filtered = [e for e in filtered if e.date <= end_date]
        return filtered

    def get_magnitude_range(self):
        """Get minimum and maximum magnitudes in catalog"""
        if not self.entries:
            return None, None
        magnitudes = [entry.mag for entry in self.entries]
        return min(magnitudes), max(magnitudes)

    def get_spatial_bounds(self):
        """Get spatial bounds of catalog entries"""
        if not self.entries:
            return None
        lats = [entry.lat for entry in self.entries]
        lons = [entry.lon for entry in self.entries]
        return {
            'min_lat': min(lats),
            'max_lat': max(lats),
            'min_lon': min(lons),
            'max_lon': max(lons)
        }

    def get_statistics(self):
        """Calculate basic statistics for the catalog"""
        if not self.entries:
            return None
        
        magnitudes = [entry.mag for entry in self.entries]
        return {
            'count': len(self.entries),
            'min_magnitude': min(magnitudes),
            'max_magnitude': max(magnitudes),
            'mean_magnitude': sum(magnitudes) / len(magnitudes),
            'spatial_bounds': self.get_spatial_bounds()
        }

    def create_histogram(self, bin_width=0.1):
        """Create magnitude histogram"""
        if not self.entries:
            return []
        
        magnitudes = [entry.mag for entry in self.entries]
        min_mag, max_mag = min(magnitudes), max(magnitudes)
        
        # Create histogram bins
        bin_edges = np.arange(min_mag, max_mag + bin_width, bin_width)
        bin_counts, _ = np.histogram(magnitudes, bins=bin_edges)
        
        # Create CatalogHist objects
        histogram = []
        cumulative_count = 0
        for i, count in enumerate(bin_counts):
            cumulative_count += count
            hist = CatalogHist(
                mag=bin_edges[i],
                numevents=int(count),
                cumnumevents=cumulative_count
            )
            histogram.append(hist)
            
        return histogram

    def prepare_full_histogram(self, catalog_duration, config):
        """
        Prepare a complete histogram following new.py's exact algorithm.
        
        Args:
            catalog_duration: Duration of the catalog in years
            config: Configuration dictionary
            
        Returns:
            Complete histogram with all event sources
        """
        # Generate base histogram
        base_histogram = self.create_histogram()
        if not base_histogram:
            return []
            
        # Get start and end dates for normalization
        start_date = min(entry.date for entry in self.entries)
        end_date = max(entry.date for entry in self.entries)
        
        # Convert dates to float format exactly as in new.py
        start_float = date_to_decimal(start_date)
        end_float = date_to_decimal(end_date)
        
        # Create a working copy of the histogram
        full_histogram = list(base_histogram)
        
        # Get catalog minimum and maximum magnitude
        catalog_min_mag = min(hist.mag for hist in full_histogram)
        catalog_max_mag = max(hist.mag for hist in full_histogram)
        
        # Get completeness thresholds
        thresholds = []
        comp_min_mag = catalog_min_mag
        if config.get("completeness", {}).get("enabled", False):
            thresholds = config["completeness"]["thresholds"]
            # Sort by magnitude
            thresholds.sort(key=lambda x: x[0])
            # If we have completeness thresholds, determine minimum complete magnitude
            if thresholds:
                comp_min_mag = max(catalog_min_mag, thresholds[0][0])
        
        # Number of bins in synthetic catalog histogram
        num_hist_syn = int(10 * catalog_max_mag - 10 * comp_min_mag) + 1
        
        # Normalize histogram - WITH INTEGER CONVERSION
        if thresholds:
            k = 0  # Index for thresholds
            
            # Process all bins except the last one
            for i in range(len(full_histogram) - 1):
                if k + 1 < len(thresholds) and full_histogram[i].mag >= thresholds[k+1][0]:
                    k += 1
                    
                # Get start date for this magnitude bin
                threshold_date = thresholds[k][1]
                threshold_float = date_to_decimal(threshold_date)
                
                # Apply scaling
                full_histogram[i].numevents = int(full_histogram[i].numevents * catalog_duration / (end_float - threshold_float))
            
            # Handle the last bin separately
            if len(thresholds) > 0:
                last_threshold_idx = len(thresholds) - 1
                last_threshold_date = thresholds[last_threshold_idx][1]
                last_threshold_float = date_to_decimal(last_threshold_date)
                full_histogram[-1].numevents = int(full_histogram[-1].numevents * catalog_duration / (end_float - last_threshold_float))
        else:
            # No completeness - scale all bins by catalog duration WITH INTEGER CONVERSION
            scaling_factor = catalog_duration / (end_float - start_float)
            for hist in full_histogram:
                hist.numevents = hist.numevents * scaling_factor
        
        
        # Add G-R events if enabled
        if config.get("additional_earthquakes", {}).get("enabled", False):
            a_value = config["additional_earthquakes"]["a_value"]
            b_value = config["additional_earthquakes"]["b_value"]
            max_magnitude = config["additional_earthquakes"]["max_magnitude"]
            
            # Calculate additional bins needed
            add_mags = int(max_magnitude * 10) - int(catalog_max_mag * 10)
            
            # Add new bins with G-R relationship
            for j in range(add_mags):
                m1 = catalog_max_mag + 0.1 * (j + 1)
                # Use exact formula from new.py with INTEGER CONVERSION
                n_events = int(catalog_duration * pow(10, a_value) * pow(10, -b_value * m1) * (1.0 - pow(10, -b_value * 0.1)))
                
                # Add bin to histogram
                new_hist = CatalogHist(
                    mag=m1,
                    numevents=n_events,
                    cumnumevents=0
                )
                full_histogram.append(new_hist)
        
        # Add characteristic earthquake if enabled
        if config.get("characteristic_earthquake", {}).get("enabled", False):
            char_mag = config["characteristic_earthquake"]["magnitude"]
            char_repeat = config["characteristic_earthquake"]["repeat_time"]
            
            # Calculate events with INTEGER CONVERSION
            n_events = int(catalog_duration / char_repeat)
            
            # Add to histogram
            new_hist = CatalogHist(
                mag=char_mag,
                numevents=n_events,
                cumnumevents=0
            )
            full_histogram.append(new_hist)
            
        # Sort by magnitude
        full_histogram.sort(key=lambda x: x.mag)

        # Calculate cumulative counts with INTEGER values
        cumulative = 0
        for hist in full_histogram:
            cumulative += hist.numevents
            hist.cumnumevents = cumulative
                
        return full_histogram
            
    def process_in_batches(self, batch_size=None):
        """Generator to process catalog in batches"""
        batch_size = batch_size or self.chunk_size
        for i in range(0, len(self.entries), batch_size):
            yield self.entries[i:i + batch_size]

    @classmethod
    def from_file(cls, file_path, chunk_size=5000, completeness=None):
        """Create Catalog instance from file"""
        catalog = cls(catalog_file=file_path, chunk_size=chunk_size)
        if completeness:
            catalog.set_completeness(completeness)
        return catalog

class CatalogHist:
    """
    Represents a histogram of events in different magnitude intervals.
    """
    def __init__(self, mag=0.0, numevents=0, cumnumevents=0):
        self.mag = mag  # Magnitude bin
        self.numevents = numevents  # Number of events in the bin
        self.cumnumevents = cumnumevents  # Cumulative number of events

class MagRange:
    """
    Contains information about events at or above a magnitude threshold.
    """
    def __init__(self, mag=0.0, date="", date_float=0.0):
        self.mag = mag  # Magnitude threshold
        self.date = date  # Date as string (YYYYMMDD)
        self.date_float = date_float  # Decimal year equivalent of the date

class MaxAcc:
    """
    Represents the largest ground motion values.
    """
    def __init__(self, time=0.0, lat=0.0, lon=0.0, mag=0.0, log10_a=0.0, epic_d=0.0):
        self.time = time  # Time of event
        self.lat = lat  # Latitude
        self.lon = lon  # Longitude
        self.mag = mag  # Magnitude
        self.log10_a = log10_a  # Logarithmic acceleration
        self.epic_d = epic_d  # Epicentral distance

class FrankAcc:
    """
    Represents ground motion values for a specific attenuation model (Frankel 1996).
    """
    def __init__(self):
        self.mag = [0.0] * 20  # Magnitude array
        self.log10_a = [[0.0] * 20 for _ in range(21)]  # Log acceleration array
        self.log10_epic_d = [0.0] * 21  # Log epicentral distance array

class CatalogCompleteness:
    """
    Modern implementation of catalog completeness using proper datetime handling
    """
    def __init__(self, config=None):
        self.thresholds = []  # List of (magnitude, datetime) tuples
        
        if config and config.get("enabled"):
            for magnitude, date_str in sorted(config["thresholds"]):
                self.add_threshold(magnitude, date_str)
                
    def add_threshold(self, magnitude: float, date_str: str):
        """Add a magnitude completeness threshold with associated date"""
        date = datetime.strptime(date_str, "%Y%m%d")
        self.thresholds.append((magnitude, date))
        # Sort by magnitude for efficient lookup
        self.thresholds.sort(key=lambda x: x[0])
        
    def is_complete(self, magnitude: float, date_str: str) -> bool:
        """
        Check if event meets completeness criteria using proper datetime comparison
        """
        if not self.thresholds:
            return True
            
        event_date = datetime.strptime(date_str, "%Y%m%d")
        
        # Find applicable threshold using binary search
        idx = np.searchsorted([t[0] for t in self.thresholds], magnitude)
        if idx == 0:
            return False  # Below minimum threshold
        
        # Check against relevant threshold
        return event_date >= self.thresholds[idx-1][1]

    def normalize_counts(self, histogram, start_date: str, end_date: str, 
                        target_duration: float):
        """
        Normalize event counts using proper date arithmetic
        """
        start = datetime.strptime(start_date, "%Y%m%d")
        end = datetime.strptime(end_date, "%Y%m%d")
        
        for hist in histogram:
            magnitude = hist.mag
            complete_start = start
            
            # Find applicable completeness period
            for mag, date in self.thresholds:
                if magnitude >= mag:
                    complete_start = max(complete_start, date)
            
            # Calculate actual duration in years
            actual_duration = (end - complete_start).days / 365.25
            if actual_duration > 0:
                norm_factor = target_duration / actual_duration
                hist.numevents = int(hist.numevents * norm_factor)
                
def validate_completeness_config(config):
    """
    Validate completeness configuration settings
    """
    if not isinstance(config.get("completeness", {}).get("enabled"), bool):
        raise ValueError("completeness.enabled must be a boolean")
        
    thresholds = config.get("completeness", {}).get("thresholds", [])
    if not isinstance(thresholds, list):
        raise ValueError("completeness.thresholds must be a list")
        
    for i, threshold in enumerate(thresholds):
        if not isinstance(threshold, tuple) or len(threshold) != 2:
            raise ValueError(f"Invalid threshold format at index {i}")
            
        magnitude, date = threshold
        if not isinstance(magnitude, (int, float)):
            raise ValueError(f"Invalid magnitude at index {i}")
            
        if not isinstance(date, str) or len(date) != 8 or not date.isdigit():
            raise ValueError(f"Invalid date format at index {i}")
            
    # Verify magnitudes are in ascending order
    mags = [t[0] for t in thresholds]
    if mags != sorted(mags):
        raise ValueError("Magnitude thresholds must be in ascending order")
        
    return True

def validate_inputs(data_dict):
    """Validate input data structure and values."""
    required_fields = ['lat', 'long', 'mag', 'year', 'month', 'day']
    
    # Check required fields
    missing_fields = [field for field in required_fields if field not in data_dict]
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")
    
    # Validate numeric ranges
    try:
        lat = float(data_dict['lat'])
        lon = float(data_dict['long'])
        mag = float(data_dict['mag'])
        
        if not (-90 <= lat <= 90):
            raise ValueError(f"Invalid latitude: {lat}")
        if not (-180 <= lon <= 180):
            raise ValueError(f"Invalid longitude: {lon}")
        if not (0 <= mag <= 10):
            raise ValueError(f"Suspicious magnitude: {mag}")
            
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid numeric data: {e}")
        
    return True

def load_gmpe_coefficients(gmpe_files, default_model):
    """
    Loads GMPE model coefficients for models that need them.
    
    NGA-West2 models (CB14, BSSA14, ASK14, CY14) use PyGMM and don't need 
    coefficient files - returns empty DataFrame for these.
    
    Legacy models (Frankel1996, etc.) load their coefficient files as before.
    
    Parameters
    ----------
    gmpe_files : dict
        Dictionary mapping model names to coefficient file paths
    default_model : str
        The GMPE model being used
        
    Returns
    -------
    dict
        Dictionary with model name as key and DataFrame/empty as value
    """
    try:
        # NGA-West2 models use PyGMM - no coefficient loading
        NGA_WEST2_MODELS = ['CB14', 'BSSA14', 'ASK14', 'CY14']
        
        if default_model in NGA_WEST2_MODELS:
            logger.info(f"✓ Using PyGMM for {default_model} - no coefficient loading needed")
            import pandas as pd
            return {default_model: pd.DataFrame()}  # Empty DataFrame
        
        # For legacy models, load coefficients as before
        if default_model not in gmpe_files:
            raise ValueError(f"Model {default_model} not found in coefficient files")
            
        filepath = gmpe_files.get(default_model)
        
        # Handle analytical models that don't need coefficient files
        if filepath is None:
            logger.info(f"{default_model} is an analytical model - no coefficient file needed")
            import pandas as pd
            return {default_model: pd.DataFrame()}
        
        logger.info(f"Loading coefficients for {default_model} from {filepath}")
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Coefficient file not found: {filepath}")
        
        # Special handling for Frankel1996 (uses .txt format)
        if default_model == "Frankel1996":
            logger.info(f"Frankel1996 uses table lookup - coefficient file will be loaded in log_a()")
            import pandas as pd
            return {default_model: pd.DataFrame()}
        
        # For other models using CSV format (if any remain)
        import pandas as pd
        df = pd.read_csv(filepath)
        
        logger.info(f"✓ Loaded {len(df)} coefficient rows for {default_model}")
        
        return {default_model: df}
        
    except Exception as e:
        logger.error(f"Error loading coefficients: {str(e)}")
        raise
    
def process_site_batch(sites, catalog, histogram, config):
    """Process multiple sites with proper resource management."""
    results = {}
    max_workers = min(PERFORMANCE_SETTINGS['max_workers'], len(sites))
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        try:
            # Submit jobs with timeout
            for site_lat, site_lon in sites:
                future = executor.submit(
                    generate_synthetic_events,
                    catalog.find_nearby_events(site_lat, site_lon, 
                                            radius=config.get('distance_threshold', 300.0)),
                    histogram,
                    site_lat,
                    site_lon,
                    config
                )
                futures[(site_lat, site_lon)] = future
            
            # Collect results with timeout
            for (site_lat, site_lon), future in futures.items():
                try:
                    results[(site_lat, site_lon)] = future.result(timeout=300)  # 5 min timeout
                except TimeoutError:
                    logger.warning(f"Warning: Timeout processing site ({site_lat}, {site_lon})")
                except Exception as e:
                    logger.warning(f"Warning: Error processing site ({site_lat}, {site_lon}): {e}")
                    
        except Exception as e:
            logger.error(f"Error in parallel processing: {e}")
        finally:
            # Cancel any remaining futures
            for future in futures.values():
                future.cancel()
                
    return results

def optimize_hazard_calculation(events, site_lat, site_lon):
    """
    Optimized vectorized calculations for hazard computation.
    
    Parameters:
    -----------
    events : list
        List of events
    site_lat, site_lon : float
        Site coordinates
    
    Returns:
    --------
    tuple
        (filtered distances, filtered magnitudes)
    """
    if not events:
        return np.array([]), np.array([])
        
    # Convert to numpy arrays
    coords = np.array([(e.lat, e.lon) for e in events])
    mags = np.array([e.mag for e in events])
    
    # Use vectorized distance calculation
    distances = calculate_distance(site_lat, site_lon, coords=coords, vectorized=True)
    
    # Apply distance threshold
    mask = distances <= PERFORMANCE_SETTINGS['distance_threshold']
    return distances[mask], mags[mask]

def compare_float(a, b, rel_tol=1e-09, abs_tol=0.0):
    """Safe floating point comparison."""
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def calculate_hazard_values(synthetic_events, site_lat, site_lon, synduration, output_settings, config):
    """
    Simple combined hazard calculation - switches between C-style and high-resolution
    """
    
    # Check which method to use
    calculation_method = config.get('hazard_calculation', {}).get('method', 'c_style')
    
    if calculation_method == 'c_style':
        return calculate_hazard_values_c_style(synthetic_events, site_lat, site_lon, synduration, output_settings, config)
    else:
        return calculate_hazard_values_high_res(synthetic_events, site_lat, site_lon, synduration, output_settings, config)

def calculate_hazard_values_c_style(synthetic_events, site_lat, site_lon, synduration, output_settings, config):
    """C-style calculation: 10 points, direct GM interpolation"""
    
    try:
        if not synthetic_events:
            return {'ground_motions': [], 'annual_rates': [], 'hazard_values': [], 'probabilities': {}}

        # Extract ground motion values
        gmvalues = np.array([event.log10_a for event in synthetic_events])
        gmvalues = np.power(10, gmvalues)
        
        gmmin = output_settings["min_ground_motion"]
        gmmax = np.max(gmvalues) if len(gmvalues) > 0 else gmmin * 100
        
        if gmmax <= gmmin:
            gmmax = gmmin * 100
        
        # Create exactly 10 points (C-style)
        log_gmmin = math.log10(gmmin)
        log_gmmax = math.log10(gmmax)
        gminc = (log_gmmax - log_gmmin) / 9
        
        hazgm = []
        lambda_values = []
        hazval = []
        
        for i in range(10):
            # Proper log-space distribution
            log_gm = log_gmmin + (gminc * i)
            gm_threshold = math.pow(10, log_gm)
            hazgm.append(gm_threshold)
            
            # Count exceedances
            exceedances = np.sum(gmvalues >= gm_threshold)
            rate = exceedances / synduration
            lambda_values.append(rate)
            
            # Calculate hazard value (probability of exceedance)
            hazard_value = 1 - math.exp(-rate) if rate > 0 else 0
            hazval.append(hazard_value)
        
        # CORRECTED PROBABILITY CALCULATION
        probabilities = calculate_probabilities(lambda_values, hazgm, output_settings)

        return {
            'ground_motions': hazgm,
            'annual_rates': lambda_values,
            'hazard_values': hazval,
            'probabilities': probabilities
        }
        
    except Exception as e:
        logger.error(f"Error in corrected C-style hazard calculation: {e}")
        return {'ground_motions': [], 'annual_rates': [], 'hazard_values': [], 'probabilities': {}}

def interpolate_c_style_simple(hazgm, hazval, target_prob):
    """Simple C-style interpolation"""
    
    # Check bounds
    if target_prob < hazval[9] or target_prob > hazval[0]:
        return None
    
    # Find bracket
    j = 0
    while j < 9 and target_prob < hazval[j]:
        j += 1
    
    if j == 0 or j >= 10:
        return None
    
    try:
        # C-style interpolation in log-log space
        log_hazval_j = math.log10(max(hazval[j], 1e-10))
        log_hazval_j_minus_1 = math.log10(max(hazval[j-1], 1e-10))
        log_target_prob = math.log10(max(target_prob, 1e-10))
        
        delthaz = (hazgm[j] - hazgm[j-1]) / (log_hazval_j - log_hazval_j_minus_1)
        gm_interp = (log_target_prob - log_hazval_j_minus_1) * delthaz + hazgm[j-1]
        
        return float(gm_interp)
        
    except Exception:
        return None

def calculate_hazard_values_high_res(synthetic_events, site_lat, site_lon, synduration, output_settings, config):
    """High-resolution calculation: 40 points"""
    
    try:
        if not synthetic_events:
            return {'ground_motions': [], 'annual_rates': [], 'hazard_values': [], 'probabilities': {}}

        # Extract ground motion values
        gmvalues = np.array([event.log10_a for event in synthetic_events])
        gmvalues = np.power(10, gmvalues)
        
        gmmin = output_settings["min_ground_motion"]
        gmmax = np.max(gmvalues) if len(gmvalues) > 0 else gmmin * 10
        
        # Ensure reasonable range
        if gmmax <= gmmin:
            gmmax = gmmin * 10
            
        # Create 40-point curve
        num_points = 40
        gm_thresholds = np.logspace(np.log10(gmmin), np.log10(gmmax), num_points)
        
        # Calculate exceedance rates
        lambda_values = []
        for threshold in gm_thresholds:
            exceedances = np.sum(gmvalues >= threshold)
            rate = exceedances / synduration
            lambda_values.append(rate)
            
        lambda_values = np.array(lambda_values)
        hazard_values = 1 - np.exp(-lambda_values)
        
        # CORRECTED PROBABILITY CALCULATION  
        probabilities = calculate_probabilities(lambda_values, gm_thresholds, output_settings)

        return {
            'ground_motions': gm_thresholds.tolist(),
            'annual_rates': lambda_values.tolist(),
            'hazard_values': hazard_values.tolist(),
            'probabilities': probabilities
        }
        
    except Exception as e:
        logger.error(f"Error in corrected high-res hazard calculation: {e}")
        return {'ground_motions': [], 'annual_rates': [], 'hazard_values': [], 'probabilities': {}}

def del_rad(slat, slon, elat, elon):
    """
    Computes geodetic distance between two latitude/longitude points using the
    more accurate Geodesic method.
    """
    site = (slat, slon)
    event = (elat, elon)
    
    # Use Geopy's geodesic method to get the correct distance
    distance = geodesic(site, event).kilometers

    return distance  # Distance in km

def calculate_distance(lat1, lon1, lat2=None, lon2=None, coords=None, cached=True, vectorized=False):
    """
    Unified distance calculation that can handle both individual points and arrays.
    
    Parameters:
    -----------
    lat1, lon1 : float or numpy.ndarray
        Site latitude and longitude in degrees. If computing a single distance, these are scalars.
        If using vectorized mode with a single site, these are still scalar values.
    lat2, lon2 : float or None, optional
        Second point's latitude and longitude for single distance calculation.
        Not used if coords is provided.
    coords : numpy.ndarray or None, optional
        Array of shape (n, 2) where each row is [lat, lon] for vectorized calculation.
        If provided, lat2 and lon2 are ignored.
    cached : bool, optional
        Whether to use cached results for individual calculations, default True.
        Ignored in vectorized mode.
    vectorized : bool, optional
        Whether to use vectorized calculation, default False.
        If True, will use the faster approximation method suitable for filtering.
    
    Returns:
    --------
    float or numpy.ndarray
        Distance(s) in kilometers
    """
    
    # Define cached version of individual calculation
    @lru_cache(maxsize=PERFORMANCE_SETTINGS['cache_size'])
    def _cached_distance(lat1, lon1, lat2, lon2):
        """Individual distance calculation with caching"""
        site = (lat1, lon1)
        event = (lat2, lon2)
        return geodesic(site, event).kilometers
    
    # Handle vectorized calculation
    if vectorized:
        if coords is None:
            raise ValueError("For vectorized calculation, coords must be provided")
        
        # Convert to radians
        lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
        lat2_rad, lon2_rad = np.radians(coords[:, 0]), np.radians(coords[:, 1])
        
        # Earth radius in kilometers
        R = 6371.0
        
        # Calculate great circle distance using haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        # Haversine formula
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        distances = 2 * R * np.arcsin(np.sqrt(a))
        
        return distances
    
    # Handle individual calculation
    elif coords is None:
        if lat2 is None or lon2 is None:
            raise ValueError("For individual calculation, lat2 and lon2 must be provided")
        
        if cached:
            return _cached_distance(lat1, lon1, lat2, lon2)
        else:
            return _cached_distance.__wrapped__(lat1, lon1, lat2, lon2)
    
    # Handle multiple individual calculations
    else:
        if cached:
            return np.array([_cached_distance(lat1, lon1, coord[0], coord[1]) for coord in coords])
        else:
            return np.array([_cached_distance.__wrapped__(lat1, lon1, coord[0], coord[1]) for coord in coords])
                                
def calculate_distance_vectorized(site_lat, site_lon, coords):
    """
    Vectorized distance calculation that produces results more similar to del_rad.
    Use this for preprocessing or filtering large datasets, but use del_rad for
    the final ground motion calculations.
    
    This function is retained for backward compatibility.
    Consider using calculate_distance(lat1, lon1, coords=coords, vectorized=True) instead.
    """
    return calculate_distance(site_lat, site_lon, coords=coords, vectorized=True)

def get_gmvalue_func(config):
    """
    Returns a function for ground motion calculations with full model support
    
    Uses unified_gmpe.py router which handles:
    - NGA-West2 models via PyGMM (ASK14, BSSA14, CB14, CY14)
    - Legacy models (Frankel1996, Atkinson1995, etc.)
    - NSHM 2023 ensemble mode
    
    Args:
        config: Configuration dictionary containing GMPE settings
        
    Returns:
        Function that calculates log10(PGA in g) for given magnitude and distance
    """
    
    # Extract configuration with safe defaults
    gmpe_config = config.get('gmpe', {})
    site_defaults = config.get('site_defaults', {})
    
    use_ensemble = gmpe_config.get('use_ensemble', False)
    default_model = config.get('default_model', gmpe_config.get('default_model', 'CB14'))
    vs30 = site_defaults.get('vs30', 760.0)
    period = config.get('default_period', 0.01)
    mechanism = gmpe_config.get('mechanism', 'strike-slip')
    
    # Optional basin depths (auto-calculated if None)
    z1p0 = site_defaults.get('z1p0', None)
    z2p5 = site_defaults.get('z2p5', None)
    region = config.get('region', {}).get('pygmm_region', 'california')  
    
    # Log configuration
    logger.info("=" * 70)
    if use_ensemble:
        models = gmpe_config.get('models', ['ASK14', 'BSSA14', 'CB14', 'CY14'])
        logger.info(f"📊 GMPE: NSHM 2023 Ensemble ({'+'.join(models)})")
        weights = gmpe_config.get('ensemble_weights', {})
        if weights:
            logger.info(f"   Weights: {weights}")
    else:
        logger.info(f"📊 GMPE: {default_model} (single model)")
    
    logger.info(f"   Vs30={vs30} m/s, period={period}s, mechanism={mechanism}")
    logger.info("=" * 70)
    
    # Create the calculation function
    def gmvalue_func(mag, distance):
        """
        Calculate log10(PGA in g) using unified GMPE router
        
        The unified_gmpe.log_a() function automatically:
        - Routes NGA-West2 → PyGMM (preferred) or legacy (fallback)
        - Routes legacy models → legacy implementation
        - Handles ensemble with NSHM 2023 weights
        - Calculates basin depths from Vs30
        """
        try:
            # Build optional parameters
            kwargs = {}
            if z1p0 is not None:
                kwargs['depth_1_0'] = z1p0 / 1000.0  # m to km
            if z2p5 is not None:
                kwargs['depth_2_5'] = z2p5
            kwargs['region'] = region
            
            # Call unified GMPE router
            # This single function handles EVERYTHING:
            # - Ensemble mode if use_ensemble=True
            # - Single model if use_ensemble=False
            # - NGA-West2 via PyGMM
            # - Legacy models via legacy code
            log10_pga = log_a(
                magnitude=mag,
                distance=distance,
                vs30=vs30,
                model=default_model,
                period=period,
                use_ensemble=use_ensemble,  # ⭐ KEY: Enable ensemble
                mechanism=mechanism,
                **kwargs
            )
            
            return log10_pga
            
        except Exception as e:
            logger.error(f"GMPE calculation failed for M{mag} at {distance}km: {e}")
            raise
    
    # Add metadata for debugging
    if use_ensemble:
        models = gmpe_config.get('models', ['ASK14', 'BSSA14', 'CB14', 'CY14'])
        gmvalue_func.model_name = f"Ensemble({'+'.join(sorted(models))})"
    else:
        gmvalue_func.model_name = default_model
    
    gmvalue_func.use_ensemble = use_ensemble
    gmvalue_func.vs30 = vs30
    
    # Quick validation
    try:
        test_result = gmvalue_func(6.0, 20.0)
        if -5.0 < test_result < 2.0:  # Reasonable range
            logger.info(f"✅ GMPE function ready: {gmvalue_func.model_name}")
        else:
            logger.warning(f"⚠️  Test result unusual: {test_result:.4f}")
    except Exception as e:
        logger.error(f"❌ GMPE validation failed: {e}")
        raise ValueError(f"GMPE function initialization failed: {e}")
    
    return gmvalue_func
    
def convert_date(date_str: str) -> datetime:
    """Convert YYYYMMDD date string to datetime object"""
    try:
        return datetime.strptime(date_str, "%Y%m%d")
    except ValueError as e:
        raise ValueError(f"Invalid date format: {date_str}. Must be YYYYMMDD. Error: {e}")

def date_to_decimal(date_str: str) -> float:
    """
    Convert date to decimal year using proper datetime arithmetic
    
    Args:
        date_str: Date string in YYYYMMDD format
    Returns:
        float: Decimal year (e.g., 2023.45)
    """
    date = convert_date(date_str)
    year = date.year
    year_start = datetime(year, 1, 1)
    year_end = datetime(year + 1, 1, 1)
    
    # Calculate fraction of year using actual number of days in the year
    days_in_year = (year_end - year_start).days
    days_elapsed = (date - year_start).days
    
    return year + days_elapsed / days_in_year
    
def randcatf(events, numeqs, histogram, nummags, totalevents, seed=None):
    """
    Find randomized mag, lat and lon by random sampling.
    Based on the original implementation in new.py with consistent seed handling.
    
    Args:
        events: List of catalog events
        numeqs: Number of events in the catalog
        histogram: Magnitude histogram
        nummags: Number of magnitude bins in histogram
        totalevents: Total number of events (for sampling)
        seed: Random seed (optional)
        
    Returns:
        tuple: (magnitude, latitude, longitude)
    """
    # Set seed if provided (ensures reproducibility)
    if seed is not None:
        random.seed(seed)
    
    # Generate magnitude index
    r1 = random.random()
    magindex = int(r1 * totalevents)
    
    # Find magnitude bin 
    i = 0
    while i < nummags and magindex > histogram[i].cumnumevents:
        i += 1
    mag = histogram[i].mag
    
    # Generate location index
    r2 = random.random()
    latlonindex = int(r2 * numeqs)
    if latlonindex >= numeqs:  # Safety check
        latlonindex = numeqs - 1
        
    elat = events[latlonindex].lat
    elon = events[latlonindex].lon
    
    return mag, elat, elon

def get_accel(slat, slon, elat, elon, mag, gmvalue_func, model=None):
    """
    Calculates ground motion acceleration at a site with model-specific handling
    
    Parameters:
    -----------
    slat, slon : float
        Site latitude and longitude
    elat, elon : float
        Earthquake latitude and longitude
    mag : float
        Earthquake magnitude
    gmvalue_func : function
        Function to calculate ground motion values
    model : str, optional
        Name of the ground motion model being used
        
    Returns:
    --------
    tuple
        (log10 of ground motion in g units, distance in km)
    """
    try:
        # Calculate distance with minimum threshold
        # Use cached individual calculation for accuracy
        d = max(calculate_distance(slat, slon, elat, elon, cached=True), 0.1)
        
        # Calculate ground motion
        log10_accel = gmvalue_func(mag, d)

        log10_g = log10_accel
        
        # For older models, the value from log_a() is already in cm/s² with log10(980) added
        older_models = ["Frankel1996", "Toro1997", 
               "BooreAtkinson1987", "ToroMcGuire1987"]
        working_models = ["Atkinson1995"]

        # Group 2: NGA-West2 models that return log10(g) directly  
        nga_models = ["BSSA14", "ASK14", "CB14", "CY14", "I14"]
        
        # Validate result
        if isinstance(log10_g, float) and (-10.0 < log10_g < 5.0):  # Reasonable range
            return log10_g, d
        else:
            logger.debug(f"Invalid ground motion value: {log10_g} for M{mag} at {d}km")
            return -np.inf, d
        
    except Exception as e:
        logger.debug(f"Error in acceleration calculation: {str(e)}")
        return -np.inf, d
     
def get_coefficient_row(model, period):
    """Get coefficient row for the specified model and period"""
    if model not in dataframes:
        raise ValueError(f"Model {model} is not available.")
    coefficients = dataframes[model]
    coeff_row = coefficients[coefficients["T (s)"] == period]
    if coeff_row.empty:
        raise ValueError(f"No coefficients for period {period}s in model {model}")
    return coeff_row.iloc[0]

def validate_ground_motion_values(magnitude, distance, model_name):
    """
    Helper function to validate ground motion values across models
    
    Args:
        magnitude: Earthquake magnitude
        distance: Distance in km
        model_name: Name of the model to test
    """
    # Get the raw value from the model
    raw_value = log_a(magnitude, distance, model=model_name)
    
    # Apply conversion if needed
    models_needing_conversion = ["Frankel1996", "Atkinson1995", "Toro1997", 
                               "BooreAtkinson1987", "ToroMcGuire1987"]
    
    if model_name in models_needing_conversion:
        converted_value = raw_value - math.log10(980)
        logger.info(f"{model_name}: raw = {raw_value}, converted to g = {converted_value}")
    else:
        logger.info(f"{model_name}: value = {raw_value} (already in g)")
    
    # Convert to actual ground motion values
    raw_acceleration = 10**raw_value
    g_acceleration = raw_acceleration / 980 if model_name in models_needing_conversion else raw_acceleration
    
    logger.info(f"  Linear values: {raw_acceleration:.5f} cm/s² = {g_acceleration:.5f}g")
    
    return raw_value

def a_pick(d, bin_flag, observed_data=None, seed=None):
    """
    Returns log10(a) based on distance d, using random or averaged values from bins.
    
    Args:
        d: Distance in kilometers.
        bin_flag: Flag indicating whether to use average (1) or random pick (0).
        observed_data: Pre-loaded observed ground motion data. If None, will load from file.
        seed: Random seed (optional)
        
    Returns:
        float: Logarithmic ground motion value.
    """
    # Use static variables to store data if not provided
    if not hasattr(a_pick, "initialized"):
        a_pick.initialized = False
        a_pick.bin_data = []
    
    # Load data from file if necessary and not provided
    if observed_data is None and not a_pick.initialized:
        # Find file to load
        try:
            filename = input("Enter observed ground motion file: ").strip()
            logger.info(f"Loading observed ground motions from: {filename}")
            
            a_pick.bin_data = load_observed_data(filename)
            a_pick.initialized = True
            observed_data = a_pick.bin_data
        except Exception as e:
            logger.info(f"Error loading observed data: {e}")
            return 0
    elif observed_data is None:
        observed_data = a_pick.bin_data
    
    # Beyond valid distance range
    if d > 525:
        return 0
    
    # Find the appropriate bin
    for bin_data in observed_data:
        bin_start = bin_data["start"]
        bin_end = bin_data["end"]
        values = bin_data["values"]
        average = bin_data["average"]

        if bin_start <= d < bin_end:
            if bin_flag == 1:
                return average  # Use the average value
            elif bin_flag == 0 and values:
                # Set seed if provided
                if seed is not None:
                    random.seed(seed)
                    
                r = random.random()
                idx = int(r * len(values))
                if idx >= len(values):  # Safety check
                    idx = len(values) - 1
                return values[idx]
    
    # Try looking for nearest bin if exact match not found
    nearest_bin = None
    nearest_distance = float('inf')
    
    for bin_data in observed_data:
        bin_middle = (bin_data["start"] + bin_data["end"]) / 2
        distance = abs(bin_middle - d)
        
        if distance < nearest_distance:
            nearest_distance = distance
            nearest_bin = bin_data
    
    if nearest_bin:
        if bin_flag == 1:
            return nearest_bin["average"]
        elif bin_flag == 0 and nearest_bin["values"]:
            if seed is not None:
                random.seed(seed)
                
            r = random.random()
            idx = int(r * len(nearest_bin["values"]))
            if idx >= len(nearest_bin["values"]):
                idx = len(nearest_bin["values"]) - 1
            return nearest_bin["values"][idx]
            
    # Default fallback
    return 0

def max_acc(events):
    """
    Finds the maximum ground motion value in a list of events.
    Returns the maximum value, or a small default if no events.
    """
    if not events:
        return 0.01  # Return small non-zero value if no events
    return max(event.log10_a for event in events)

def calculate_duration(start_date: str, end_date: str) -> float:
    """
    Calculate duration in years between two dates
    
    Args:
        start_date: Start date in YYYYMMDD format
        end_date: End date in YYYYMMDD format
    Returns:
        float: Duration in years
    """
    start = convert_date(start_date)
    end = convert_date(end_date)
    
    # Calculate duration using actual calendar days
    duration_days = (end - start).days
    return duration_days / 365.25  # Using average year length for consistency

def calculate_catalog_duration(catalog, additional_eqs, char_eq):
    """
    Calculate the duration of the synthetic catalog based on settings
    
    Args:
        catalog: Catalog instance
        additional_eqs: Dictionary containing additional earthquakes settings
        char_eq: Dictionary containing characteristic earthquake settings
    
    Returns:
        float: Duration in years
    """
    if not catalog.entries:
        raise ValueError("Catalog is empty")
        
    # Get start and end dates from the entries
    start_date = min(entry.date for entry in catalog.entries)
    end_date = max(entry.date for entry in catalog.entries)
    
    # Calculate base duration using utility function
    base_duration = calculate_duration(start_date, end_date)
    
    # Adjust duration based on additional earthquakes
    if additional_eqs["enabled"]:
        # Use 10 times the repeat time of largest event
        a_value, b_value = additional_eqs["a_value"], additional_eqs["b_value"]
        max_mag = additional_eqs["max_magnitude"]
        duration_from_ab = 10 * (pow(10, -a_value) * pow(10, b_value * max_mag))
        base_duration = max(base_duration, duration_from_ab)
    
    # Consider characteristic earthquake repeat time
    if char_eq["enabled"]:
        base_duration = max(base_duration, char_eq["repeat_time"])
    
    return base_duration

def generate_site_coordinates(config):
    """Generate site coordinates based on configuration - FIXED"""
    if config["grid_mode"]:
        sites = []
        
        # FIXED: Use spacing instead of hardcoded counts
        lat_spacing = config.get("grid_lat_spacing", 0.1)
        lon_spacing = config.get("grid_lon_spacing", 0.1)
        
        lat_range = np.arange(
            config["grid_lat_min"],
            config["grid_lat_max"] + lat_spacing/2,
            lat_spacing
        )
        lon_range = np.arange(
            config["grid_lon_min"],
            config["grid_lon_max"] + lon_spacing/2, 
            lon_spacing
        )
        
        for lat in lat_range:
            for lon in lon_range:
                sites.append((lat, lon))
        return sites
    else:
        return config["sites"]

def linear_rho(nscat, ro, seed=None):
    """
    Calculates a linear rho value based on scatter type and scale parameter.
    Direct implementation of the original logic from new.py.
    
    Parameters:
    nscat (int): Type of distance scatter allowed
      0=linear flat
      1=linear decrease
      2=gaussian
    ro (float): The scale or maximum scatter distance in km
    seed (int): Random seed (optional)
    
    Returns:
    float: The calculated linear rho value
    """
    # Set seed if provided
    if seed is not None:
        random.seed(seed)
        
    # Initialize static variables on first call
    if not hasattr(linear_rho, "initialized"):
        linear_rho.initialized = False
        linear_rho.ltable = [0.0] * 10
    
    # Initialize the probability table on first call
    if not linear_rho.initialized:
        sum_val = 0.0
        for n in range(10):
            y = (n + 1) * (ro / 10)
            sum_val = (ro / 10) * dev_func(ro, y, nscat) + sum_val
            linear_rho.ltable[n] = sum_val
        
        # Normalize the table
        for n in range(10):
            linear_rho.ltable[n] = linear_rho.ltable[n] / sum_val
            
        linear_rho.initialized = True
    
    # Calculate the scatter value
    x = random.random()
    newnum = 0.0
    
    # Find the appropriate bin
    n = 0
    while n < 10 and x > linear_rho.ltable[n]:
        n += 1
        
    # Scale the value
    newnum = float(n) * 0.1 * ro
    
    return newnum

def dev_func(ro, y, nscat):
    """
    Calculates a deviation function based on scale and distance.
    Direct implementation from new.py.
    
    Parameters:
    ro (float): The scale or maximum scatter
    y (float): Distance variable
    nscat (int): Type of distance scatter allowed
    
    Returns:
    float: The calculated deviation value
    """
    if nscat == 0:
        return 1/ro
    elif nscat == 1:
        return 2 * (1/ro - y/(ro*ro))
    elif nscat == 2:
        return (1/(math.sqrt(6.28)*ro)) * math.exp(-1 * (y*y)/(2*ro*ro))
    
    # Default return in case none of the cases match
    return 0.0

def generate_synthetic_events(events, histogram, site_lat, site_lon, config):
    """
    Generate synthetic events with proper handling of Gutenberg-Richter extrapolation.
    
    Args:
        events: List of input events from the catalog
        histogram: Magnitude histogram with numevents and cumnumevents
        site_lat, site_lon: Site coordinates
        config: Configuration dictionary
        catalog_duration: Duration of the catalog in years (pre-calculated)
        
    Returns:
        List of synthetic events with ground motion values
    """
    # Validate required parameters
    required_params = ['seed', 'synduration', 'default_model', 'output_settings']
    missing_params = [param for param in required_params if param not in config]
    if missing_params:
        raise ValueError(f"Missing required parameters in config: {missing_params}")
    
    if 'min_ground_motion' not in config['output_settings']:
        raise ValueError("Missing 'min_ground_motion' in output_settings")
    
    # Calculate total events from histogram
    if histogram:
        total_events = sum(hist.numevents for hist in histogram)
    else:
        total_events = 0
    
    logger.debug(f"\n============= EVENT GENERATION DIAGNOSTICS =============")
    logger.debug(f"Debug: Starting synthetic event generation")
    logger.debug(f"Debug: Number of Events in Analysis: {int(total_events)}")
    logger.debug(f"Debug: Number of input events: {len(events)}")
    logger.debug(f"Debug: Number of histogram bins: {len(histogram)}")
    
    if not events:
        logger.error("No input events available for synthetic event generation.")
        raise ValueError("No input events available for synthetic event generation.")
    
    # Initialize the random seed
    seed = config.get('seed', 1)
    random.seed(seed)
    np.random.seed(seed)
    save_seed = seed  # Save the initial seed for resetting
    
    logger.debug(f"Debug: Using seed: {seed}")
    
    # Get the total time duration for synthetic catalog
    synduration = config["synduration"]
    
    # Get catalog duration from config
    catalog_duration = config.get('synduration', 100)  # Default to 100 years if not specified
    
    # Use the provided catalog_duration instead of recalculating it
    logger.debug(f"Debug: Catalog duration: {catalog_duration} years")
    logger.debug(f"Debug: Synthetic duration: {synduration} years")
    
    # Ensure total_events is an integer not a float
    total_events = int(total_events)
    
    # Calculate time increment (exactly as in new.py)
    time_inc = synduration / total_events
    logger.debug(f"Debug: Time increment: {time_inc}")
    
    # Initialize GMPE function
    if 'default_model' in config:
        model_name = config["default_model"]
        gmvalue_func = lambda mag, dist: log_a(
            mag, dist, 
            model=model_name, 
            model_file=config.get('gmpe_files', {}).get(model_name)
        )
    else:
        model_name = "Frankel1996"
        gmvalue_func = lambda mag, dist: log_a(mag, dist)
    
    min_threshold = config["output_settings"]["min_ground_motion"]
    logger.debug(f"Debug: Minimum ground motion threshold: {min_threshold}")
    
    # Create a list to store synthetic events
    synthetic_events = []
    
    # Counters for diagnostics
    total_events_attempted = 0
    total_events_below_threshold = 0
    total_events_saved = 0
    events_by_magnitude_bin = {}
    
    # Generate events using the approach consistent with new.py
    itime = 0
    event_time = 0
    numeqs = len(events)
    nummags = len(histogram)
    
    logger.debug(f"\n===== Starting event generation loop =====")
    
    event_limit = min(int(total_events * 1.5), 1000000)  # Reasonable limit to avoid infinite loops
    logger.debug(f"Debug: Will attempt to generate up to {event_limit} events")
    
    # Estimate progress reporting intervals
    report_interval = max(1, min(1000, event_limit // 20))
    
    while event_time < synduration and total_events_attempted < event_limit:
        # Reset seed for each site if this is the first event
        if event_time == 0:
            random.seed(save_seed)
            np.random.seed(save_seed)
            
        # Calculate event time using uniform distribution
        event_time = itime * time_inc
        
        # Get mag, lat, lon - equivalent to randcatf logic in new.py
        # First, get a random magnitude from the histogram
        mag, elat, elon = randcatf(
            events, 
            numeqs, 
            histogram, 
            nummags, 
            total_events
        )
        
        total_events_attempted += 1
        
        # Track events by magnitude bin
        mag_bin = round(mag * 10) / 10  # Round to nearest 0.1
        events_by_magnitude_bin[mag_bin] = events_by_magnitude_bin.get(mag_bin, 0) + 1
        
        if total_events_attempted % 1000 == 0:
            logger.debug(f"Debug: Generated event #{total_events_attempted} at time {event_time:.2f}:")
            logger.debug(f"Debug: Magnitude: {mag}")
            logger.debug(f"Debug: Location: ({elat}, {elon})")
        
        # Apply location scatter if enabled
        if config.get("location_variation", {}).get("enabled", False):
            scatter_type = config["location_variation"]["scatter_type"]
            max_scatter = config["location_variation"]["max_scatter_distance"]
            
            radjust = linear_rho(scatter_type, max_scatter)
                
            # Get random azimuth (0-360 degrees)
            azadjust = random.random() * 360 * 0.0175  # Convert to radians
            
            # Calculate new coordinates using same approach as new.py
            elat = elat + (radjust * math.sin(azadjust) * 0.009)
            elon = elon + (radjust * math.cos(azadjust) * 0.009 / math.cos(elat * 0.0175))
            
            if total_events_attempted % report_interval == 0 or total_events_attempted <= 10:
                logger.debug(f"Debug: After scatter: ({elat}, {elon})")
        
        # Check for characteristic earthquake
        if (config.get("characteristic_earthquake", {}).get("enabled", False) and 
            abs(mag - config["characteristic_earthquake"]["magnitude"]) < 0.001):
            # Use the coordinates for the characteristic earthquake
            elat = config["characteristic_earthquake"]["latitude"]
            elon = config["characteristic_earthquake"]["longitude"]
            if total_events_attempted % report_interval == 0 or total_events_attempted <= 10:
                logger.debug(f"Debug: Using characteristic earthquake location: ({elat}, {elon})")
        
        # Calculate ground motion
        try:
            # Use get_accel function for consistency with new.py
            log10_accel, distance = get_accel(
                site_lat, 
                site_lon, 
                elat, 
                elon, 
                mag, 
                gmvalue_func, 
                model=model_name
            )
            
            # Convert from log10 to linear scale (g units)
            gmvalue = 10**log10_accel
            
            if total_events_attempted % report_interval == 0 or total_events_attempted <= 10:
                logger.debug(f"Ground motion: {gmvalue:.6f}g at distance {distance:.2f} km")
            
            # Compare with threshold and save if above
            if gmvalue >= min_threshold:
                # Create MaxAcc object
                event = MaxAcc(
                    time=event_time,
                    lat=elat,
                    lon=elon,
                    mag=mag,
                    log10_a=log10_accel,  # Store as log10
                    epic_d=distance
                )
                synthetic_events.append(event)
                total_events_saved += 1
                
                if total_events_attempted % report_interval == 0 or total_events_attempted <= 10:
                    logger.debug(f"Event saved (above threshold)")
            else:
                total_events_below_threshold += 1
                if total_events_attempted % report_interval == 0 or total_events_attempted <= 10:
                    logger.debug(f"Event discarded (below threshold: {min_threshold}g)")
                
        except Exception as e:
            logger.warning(f"Warning: Error calculating ground motion: {e}")
        
        # Increment time index
        itime += 1
        
        # Print progress occasionally
        if itime % (report_interval * 10) == 0:
            percent_complete = min(100, (event_time / synduration) * 100)
            logger.info(f"Progress: {percent_complete:.1f}% - Generated {total_events_attempted} events, "
                  f"time: {event_time:.2f}/{synduration} years")
            logger.info(f"Saved events so far: {total_events_saved}")
            logger.info(f"Events below threshold: {total_events_below_threshold}")
    
    # Print final statistics
    logger.info(f"\n===== Event Generation Summary =====")
    logger.info(f"Total potential events attempted: {total_events_attempted}")
    logger.info(f"Events below threshold: {total_events_below_threshold}")
    logger.info(f"Events saved (above threshold): {total_events_saved}")
    
    # Print magnitude distribution
    logger.debug(f"\n===== Magnitude Distribution =====")
    for mag in sorted(events_by_magnitude_bin.keys()):
        logger.debug(f"Magnitude {mag:.1f}: {events_by_magnitude_bin[mag]} events")
    
    logger.debug(f"\nDebug: Final synthetic event count: {len(synthetic_events)}")
    return synthetic_events

def load_observed_data(file_path, delimiter=','):
    """
    Loads observed ground motion data from a file.
    :param file_path: Path to the data file.
    :param delimiter: Delimiter used in the file.
    :return: List of observed data bins.
    """
    observed_data = []

    try:
        with open(file_path, 'r') as file:
            reader = csv.reader(file, delimiter=delimiter)
            for row in reader:
                if len(row) < 4:
                    continue  # Skip invalid rows

                observed_data.append({
                    "start": float(row[0]),
                    "end": float(row[1]),
                    "values": list(map(float, row[2:-1])),
                    "average": float(row[-1])
                })

    except IOError as e:
        raise IOError(f"Error reading file: {e}")

    return observed_data

def safe_open_file(filepath, mode='r'):
    """
    Safely open files with proper error handling and path resolution.
    
    Args:
        filepath: Path to the file
        mode: File open mode ('r', 'w', 'a', etc.)
    Returns:
        File object
    """
    try:
        # Convert to Path object for better path handling
        file_path = Path(filepath).resolve()
        
        # Create parent directories if writing
        if 'w' in mode or 'a' in mode:
            file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if file exists for reading
        if 'r' in mode and not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
            
        # Open the file
        return open(file_path, mode)
        
    except (ValueError, FileNotFoundError) as e:
        raise
    except Exception as e:
        raise IOError(f"Failed to open file {filepath}: {e}")

def output_site_results(site_lat, site_lon, synthetic_events, hazard_values, output_settings, output_file, config, mode='a'):
    """Write comprehensive results with proper hazard curve and probability outputs."""
    try:
        with safe_open_file(output_file, mode) as f:
            # Site information
            f.write(f"\nSite Location: {site_lat:.4f}N, {site_lon:.4f}E\n\n")
            f.write(f"Random seed: {config.get('seed', 1)}\n")
            # ✅ FIX: Check for ensemble mode
            gmpe_cfg = config.get('gmpe', {})
            if gmpe_cfg.get('use_ensemble', False):
                models = gmpe_cfg.get('models', [])
                model_str = '+'.join(sorted(models))
                f.write(f"Ground motion model: Ensemble({model_str})\n")
            else:
                f.write(f"Ground motion model: {config.get('default_model', 'CB14')}\n")
            f.write(f"Minimum ground motion threshold: {output_settings['min_ground_motion']:.3f}g\n\n")

            # Optional: Synthetic events
            if output_settings.get("full_output") and synthetic_events:
                f.write("Time(yrs)  Lat      Lon       Mag    Distance(km)  Ground_Motion(g)\n")
                for event in synthetic_events:
                    f.write(f"{event.time:8.2f}  {event.lat:7.3f}  {event.lon:8.3f}  "
                            f"{event.mag:5.2f}  {event.epic_d:11.2f}  {pow(10, event.log10_a):14.6f}\n")
                f.write("\n")

            # Hazard curve values
            if hazard_values.get('ground_motions'):
                f.write("Ground_motion_values ")
                for gm in hazard_values['ground_motions']:
                    f.write(f"{gm:8.2e} ")
                f.write("\n")

                f.write("Annualized_rates     ")
                for rate in hazard_values['annual_rates']:
                    f.write(f"{rate:8.2e} ")
                f.write("\n")

                f.write("Annual_hazard_values ")
                for hazard in hazard_values['hazard_values']:
                    f.write(f"{hazard:8.2e} ")
                f.write("\n")

            # Specified hazards (probability-based)
            probabilities = hazard_values.get('probabilities', {})
            if probabilities:
                f.write("\nSpecified hazards are computed as:\n")
                prob_type = output_settings.get("probability_type", "non_exceedance")

                if prob_type == "non_exceedance":
                    # Example key: non_exceed_50yr_0.98
                    for key, gm_val in probabilities.items():
                        try:
                            parts = key.split('_')
                            time_period = parts[2].replace("yr", "")
                            non_exceed = float(parts[3])
                            exceed = (1 - non_exceed) * 100  # Convert to %
                            f.write(f"  {exceed:.1f}% in {time_period} yr → {gm_val:.3f} g\n")
                        except Exception:
                            f.write(f"  {key}: {gm_val:.3f} g\n")

                elif prob_type == "exceedance":
                    # Example key: exceed_50yr_0.02
                    for key, gm_val in probabilities.items():
                        try:
                            parts = key.split('_')
                            time_period = parts[1].replace("yr", "")
                            exceed_prob = float(parts[2]) * 100
                            f.write(f"  {exceed_prob:.1f}% in {time_period} yr → {gm_val:.3f} g\n")
                        except Exception:
                            f.write(f"  {key}: {gm_val:.3f} g\n")

                elif prob_type == "annual":
                    # Example key: annual_0.02
                    for key, gm_val in probabilities.items():
                        try:
                            annual_prob = float(key.split('_')[1]) * 100
                            f.write(f"  {annual_prob:.2f}% annual → {gm_val:.3f} g\n")
                        except Exception:
                            f.write(f"  {key}: {gm_val:.3f} g\n")

                else:
                    # Fallback for unknown probability types
                    for key, gm_val in probabilities.items():
                        f.write(f"  {key}: {gm_val:.3f} g\n")
            else:
                f.write("\n(No probability-based hazards computed for this site)\n")

            f.write("\n" + "="*80 + "\n")

    except Exception as e:
        logger.error(f"Error writing results: {e}")
        logger.debug(f"Full traceback: {traceback.format_exc()}")

def extract_window_time_range(window_id, config):
    """Extract time range information for window - IMPROVED ERROR HANDLING"""
    try:
        # Try to get from metadata TXT file
        output_dir = Path(config.get('output_file', '')).parent
        metadata_file = output_dir / f"window_metadata_los_angeles_seq_window_{window_id}.txt"
        
        if metadata_file.exists():
            try:
                with open(metadata_file, 'r') as f:
                    content = f.read()
                    # Look for time range line like "Window time range: 8 - 44,082 years"
                    for line in content.split('\n'):
                        if 'time range:' in line.lower():
                            # Extract the time range part after the colon
                            time_range = line.split(':')[-1].strip()
                            return time_range
            except Exception as e:
                logger.debug(f"Error reading metadata file: {e}")
        
        # Fallback: estimate from window ID (assuming 50,000 year windows)
        try:
            window_num = int(window_id.lstrip('0')) if window_id.isdigit() else 1
            start_year = (window_num - 1) * 50000
            end_year = window_num * 50000
            return f"{start_year:,} - {end_year:,} years"
        except (ValueError, TypeError):
            return f"Window {window_id}"
        
    except Exception as e:
        logger.debug(f"Could not extract time range for window {window_id}: {e}")
        return f"Window {window_id}"

def setup_return_period_axis(ax=None):
    """Setup Y-axis to show return periods instead of probabilities"""
    import matplotlib.pyplot as plt
    
    if ax is None:
        ax = plt.gca()
    
    # Standard engineering return periods
    return_periods = [10, 25, 50, 100, 475, 975, 2475, 4975, 10000]
    annual_probs = [1/rp for rp in return_periods]
    
    # Set custom ticks
    ax.set_yticks(annual_probs)
    ax.set_yticklabels([f'{rp}yr' for rp in return_periods])
    ax.set_ylabel('Return Period (years)')
    ax.set_yscale('log')
    
    return ax

def probability_to_return_period(prob_key):
    """Convert probability key to return period for labeling"""
    
    if 'non_exceed_50yr_' in prob_key:
        # Extract non-exceedance probability
        prob_str = prob_key.split('non_exceed_50yr_')[1]
        non_exceed_prob = float(prob_str)
        exceed_prob = 1 - non_exceed_prob
        annual_prob = 1 - (1 - exceed_prob)**(1/50)
        return_period = 1 / annual_prob
        return int(round(return_period))
    
    elif 'exceed_50yr_' in prob_key:
        # Extract exceedance probability  
        prob_str = prob_key.split('exceed_50yr_')[1]
        exceed_prob = float(prob_str)
        annual_prob = 1 - (1 - exceed_prob)**(1/50)
        return_period = 1 / annual_prob
        return int(round(return_period))
        
    elif 'annual_' in prob_key:
        # Direct annual probability
        prob_str = prob_key.split('annual_')[1]
        annual_prob = float(prob_str)
        return_period = 1 / annual_prob
        return int(round(return_period))
    
    return None

def calculate_probabilities(lambda_values, gm_thresholds, output_settings):
    """
    Calculate probability values based on lambda rates and output settings
    """
    probabilities = {}
    
    if output_settings.get("probability_type") == "annual":
        prob_list = output_settings.get("probabilities", [])
        
        for prob_entry in prob_list:
            if isinstance(prob_entry, (int, float)):
                target_prob = prob_entry
                key_suffix = f"{target_prob}"
            elif isinstance(prob_entry, tuple) and len(prob_entry) >= 2:
                target_prob = prob_entry[1]
                key_suffix = f"{target_prob}"
            else:
                continue
            
            # Convert probability to rate using correct Poisson relationship
            if target_prob >= 0.999:
                logger.warning(f"Probability {target_prob} too close to 1.0, skipping")
                continue
                
            target_rate = -math.log(1 - target_prob)
            
            # Convert to numpy arrays
            lambda_array = np.array(lambda_values)
            gm_array = np.array(gm_thresholds)
            
            # Filter valid data points
            valid_mask = (lambda_array > 1e-12) & (lambda_array < 100) & np.isfinite(lambda_array)
            if not np.any(valid_mask):
                logger.warning(f"No valid rates for probability {target_prob}")
                continue
                
            gm_valid = gm_array[valid_mask]
            rate_valid = lambda_array[valid_mask]
            
            # Check if target rate is within achievable bounds
            rate_min, rate_max = np.min(rate_valid), np.max(rate_valid)
            
            if target_rate < rate_min:
                logger.debug(f"Target rate {target_rate:.6f} below minimum achievable {rate_min:.6f} for P={target_prob}")
                continue
            elif target_rate > rate_max:
                logger.debug(f"Target rate {target_rate:.6f} above maximum achievable {rate_max:.6f} for P={target_prob}")
                continue
            
            # CRITICAL FIX: Proper interpolation setup
            # Sort by GM ascending (which gives rates descending)
            sort_idx = np.argsort(gm_valid)  # Sort by GM ascending
            gm_sorted = gm_valid[sort_idx]   # GM: [low ... high]
            rate_sorted = rate_valid[sort_idx]  # Rate: [high ... low]
            
            # For np.interp to work, the x-values (rates) must be ascending
            # So we need to reverse both arrays
            rate_ascending = rate_sorted[::-1]  # Rate: [low ... high] 
            gm_ascending = gm_sorted[::-1]      # GM: [high ... low]
            
            # Now interpolate with ascending rates
            gm_interp = np.interp(target_rate, rate_ascending, gm_ascending)
            probabilities[f"annual_{key_suffix}"] = float(gm_interp)
            
            # Optional verification for debugging
            if logger.isEnabledFor(logging.DEBUG):
                # Verify the result
                rate_check = np.interp(gm_interp, gm_sorted, rate_sorted)
                actual_prob = 1 - math.exp(-rate_check)
                error = abs(actual_prob - target_prob)
                logger.debug(f"P={target_prob}: GM={gm_interp:.6f}g, actual_P={actual_prob:.6f}, error={error:.6f}")
    
    # CHANGE TO (CORRECT CONVERSION):
    elif output_settings.get("probability_type") == "non_exceedance":
        prob_list = output_settings.get("probabilities", [])
        
        for prob_entry in prob_list:
            if isinstance(prob_entry, (list, tuple)) and len(prob_entry) >= 2:
                time_period, non_exceed_prob = prob_entry[0], prob_entry[1]
                
                # Convert non-exceedance to exceedance probability
                exceed_prob = 1 - non_exceed_prob  # ✅ CORRECT: 0.98 non-exceed → 0.02 exceed
                
                if exceed_prob <= 0.001:  # Skip very small probabilities
                    continue
                    
                # Convert exceedance probability to annual rate
                annual_prob = 1 - (1 - exceed_prob)**(1/time_period)
                target_rate = -math.log(1 - annual_prob)

                # Same interpolation logic as above
                lambda_array = np.array(lambda_values)
                gm_array = np.array(gm_thresholds)
                
                valid_mask = (lambda_array > 1e-12) & (lambda_array < 100) & np.isfinite(lambda_array)
                if not np.any(valid_mask):
                    continue
                    
                gm_valid = gm_array[valid_mask]
                rate_valid = lambda_array[valid_mask]
                
                rate_min, rate_max = np.min(rate_valid), np.max(rate_valid)
                
                if rate_min <= target_rate <= rate_max:
                    # Same fix as above
                    sort_idx = np.argsort(gm_valid)
                    gm_sorted = gm_valid[sort_idx]
                    rate_sorted = rate_valid[sort_idx]
                    
                    rate_ascending = rate_sorted[::-1]
                    gm_ascending = gm_sorted[::-1]
                    
                    gm_interp = np.interp(target_rate, rate_ascending, gm_ascending)
                    probabilities[f"non_exceed_{time_period}yr_{non_exceed_prob}"] = float(gm_interp)

    elif output_settings.get("probability_type") == "exceedance":
        prob_list = output_settings.get("probabilities", [])
        
        for prob_entry in prob_list:
            if isinstance(prob_entry, tuple) and len(prob_entry) >= 2:
                time_period, exceed_prob = prob_entry[0], prob_entry[1]
                
                if exceed_prob <= 0.001:  # Skip very small probabilities
                    continue
                    
                # Convert exceedance probability directly to annual rate
                annual_prob = 1 - math.pow(1 - exceed_prob, 1/time_period)
                target_rate = -math.log(1 - annual_prob)
                
                # Same interpolation logic as above
                lambda_array = np.array(lambda_values)
                gm_array = np.array(gm_thresholds)
                
                valid_mask = (lambda_array > 1e-12) & (lambda_array < 100) & np.isfinite(lambda_array)
                if not np.any(valid_mask):
                    continue
                    
                gm_valid = gm_array[valid_mask]
                rate_valid = lambda_array[valid_mask]
                
                rate_min, rate_max = np.min(rate_valid), np.max(rate_valid)
                
                if rate_min <= target_rate <= rate_max:
                    sort_idx = np.argsort(gm_valid)
                    gm_sorted = gm_valid[sort_idx]
                    rate_sorted = rate_valid[sort_idx]
                    
                    rate_ascending = rate_sorted[::-1]
                    gm_ascending = gm_sorted[::-1]
                    
                    gm_interp = np.interp(target_rate, rate_ascending, gm_ascending)
                    probabilities[f"exceed_{time_period}yr_{exceed_prob}"] = float(gm_interp)
                    
    return probabilities
  
def plot_hazard_curves(site_results, output_dir, config):
    """
    Enhanced hazard curve plotting with proper window organization
    FIXED: Better error handling, data validation, and plotting logic
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path
    
    # Check if plotting is enabled
    if not config.get('output_settings', {}).get('plot_hazard_curves', False):
        logger.info("Hazard curve plotting is disabled in configuration")
        return
    
    if not site_results:
        logger.error("No site results available for plotting")
        return
        
    # Extract window information
    output_file = Path(config.get('output_file', 'hazard_results.txt'))
    base_name = output_file.stem
    
    # Extract window ID and create window-specific directory
    window_id = "unknown"
    if "window_" in base_name:
        window_id = base_name.split("window_")[-1]
    
    # Create window-specific subdirectory
    plots_dir = Path(output_dir) / "plots"
    window_plots_dir = plots_dir / f"window_{window_id}"
    window_plots_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"📁 Saving window {window_id} plots to: {window_plots_dir}")
    
    # Get top hazard sites
    max_curves = config.get('output_settings', {}).get('max_curves', 10)
    top_sites = get_top_hazard_sites(site_results, max_curves)
    
    if not top_sites:
        logger.error("No valid sites found for plotting hazard curves")
        return
    
    logger.info(f"🎯 Generating curves for top {len(top_sites)} sites in window {window_id}")
    
    # Extract time range for this window
    time_range = extract_window_time_range(window_id, config)
    
    curves_created = 0
    
    for lat, lon in top_sites:
        try:
            result = site_results.get((lat, lon))
            if not result:
                logger.warning(f"⚠️  No result found for site ({lat:.4f}, {lon:.4f})")
                continue
                
            hazard_data = result.get('hazard_values', {})
            if not hazard_data:
                logger.warning(f"⚠️  No hazard_values for site ({lat:.4f}, {lon:.4f})")
                continue
            
            # Extract and validate data
            ground_motions = hazard_data.get('ground_motions', [])
            annual_rates = hazard_data.get('annual_rates', [])
            
            if not ground_motions or not annual_rates:
                logger.warning(f"⚠️  Missing ground motion or rate data for site ({lat:.4f}, {lon:.4f})")
                continue
                
            # Convert to numpy arrays and validate
            ground_motions = np.array(ground_motions)
            annual_rates = np.array(annual_rates)
            
            if len(ground_motions) != len(annual_rates):
                logger.warning(f"⚠️  Data length mismatch for site ({lat:.4f}, {lon:.4f}): "
                             f"GM={len(ground_motions)}, rates={len(annual_rates)}")
                continue
                
            if len(ground_motions) == 0:
                logger.warning(f"⚠️  Empty data arrays for site ({lat:.4f}, {lon:.4f})")
                continue
            
            # Filter out invalid values
            valid_mask = (
                np.isfinite(ground_motions) & 
                np.isfinite(annual_rates) & 
                (ground_motions > 0) & 
                (annual_rates > 0)
            )
            
            if not np.any(valid_mask):
                logger.warning(f"⚠️  No valid data points for site ({lat:.4f}, {lon:.4f})")
                continue
            
            plot_gm = ground_motions[valid_mask]
            plot_rates = annual_rates[valid_mask]
            
            # Filter to reasonable plotting range
            max_plot_gm = 2.0
            plot_mask = plot_gm <= max_plot_gm
            
            if not np.any(plot_mask):
                logger.warning(f"⚠️  No data in plotting range for site ({lat:.4f}, {lon:.4f})")
                continue
                
            plot_gm = plot_gm[plot_mask]
            plot_rates = plot_rates[plot_mask]
            
            # Sort data for proper plotting
            sort_idx = np.argsort(plot_gm)
            plot_gm = plot_gm[sort_idx]
            plot_rates = plot_rates[sort_idx]
            
            logger.debug(f"Plotting site ({lat:.4f}, {lon:.4f}): {len(plot_gm)} points, "
                        f"GM range: {plot_gm.min():.4f}-{plot_gm.max():.4f}g, "
                        f"Rate range: {plot_rates.min():.2e}-{plot_rates.max():.2e}")
            
            # Create individual site plot
            plt.figure(figsize=(12, 8))
            
            # Main hazard curve
            plt.loglog(plot_gm, plot_rates, 'b-', 
                      label=f'Window {window_id} Hazard Curve', linewidth=2.5, alpha=0.8)

            annotation_positions = [] 
            
            # Add design points
            probabilities = hazard_data.get('probabilities', {})
            design_levels = [
                    ('non_exceed_50yr_0.98', 'GM at 2.0% in 50yr', 'red', 'o'),
                    ('non_exceed_50yr_0.95', 'GM at 5.0% in 50yr', 'orange', 's'),
                    ('non_exceed_50yr_0.9', 'GM at 10.0% in 50yr', 'green', '^'),
                ]
            
            points_added = 0
            for prob_key, label, color, marker in design_levels:
                if prob_key in probabilities:
                    gm_value = probabilities[prob_key]
                    if 0.001 <= gm_value <= max_plot_gm:
                        # Calculate corresponding rate
                        if 'annual_' in prob_key:
                            rate = float(prob_key.split('_')[1])
                        elif 'non_exceed_50yr_' in prob_key:
                            non_exceed = float(prob_key.split('_')[-1])
                            exceed_prob = 1 - non_exceed
                            annual_prob = 1 - (1 - exceed_prob)**(1/50)
                            rate = annual_prob
                        elif 'exceed_50yr_' in prob_key:
                            exceed_prob = float(prob_key.split('_')[-1])
                            annual_prob = 1 - (1 - exceed_prob)**(1/50)
                            rate = annual_prob
                        else:
                            continue
                            
                        # Plot the marker
                        plt.loglog([gm_value], [rate], marker, color=color, markersize=10, 
                                  markeredgecolor='black', markeredgewidth=1.5, 
                                  label=label, zorder=10)
                        
                        # Smart annotation placement to avoid overlaps
                        log_gm = np.log10(gm_value)
                        log_rate = np.log10(rate)
                        
                        # Check for conflicts with previous annotations
                        best_offset = None
                        min_overlap = float('inf')
                        
                        # Try different offset positions
                        offset_candidates = [
                            (20, 20),    # Upper right
                            (20, -25),   # Lower right
                            (-50, 20),   # Upper left
                            (-50, -25),  # Lower left
                            (35, 0),     # Right
                            (-65, 0),    # Left
                        ]
                        
                        for offset_x, offset_y in offset_candidates:
                            # Calculate approximate annotation position in log space
                            # (rough estimate, doesn't account for exact figure coordinates)
                            overlap_score = 0
                            for prev_log_gm, prev_log_rate in annotation_positions:
                                distance = np.sqrt((log_gm - prev_log_gm)**2 + 
                                                 (log_rate - prev_log_rate)**2)
                                if distance < 0.3:  # Threshold in log space
                                    overlap_score += (0.3 - distance)
                            
                            if overlap_score < min_overlap:
                                min_overlap = overlap_score
                                best_offset = (offset_x, offset_y)
                        
                        offset_x, offset_y = best_offset
                        
                        # Add annotation with smart placement
                        plt.annotate(f'{gm_value:.3f}g', 
                                   (gm_value, rate), 
                                   xytext=(offset_x, offset_y), 
                                   textcoords='offset points',
                                   bbox=dict(boxstyle='round,pad=0.3', facecolor=color, 
                                           alpha=0.7, edgecolor='black', linewidth=1),
                                   fontsize=10, fontweight='bold',
                                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2',
                                                 color='black', lw=1.5))
                        
                        # Track this annotation position
                        annotation_positions.append((log_gm, log_rate))
                        points_added += 1
            
            # Enhanced formatting
            plt.xlim(max(0.01, plot_gm.min() * 0.5), max_plot_gm)
            plt.ylim(max(1e-6, plot_rates.min() * 0.1), max(10, plot_rates.max() * 2))
            
            plt.grid(True, which="major", ls="-", alpha=0.4, color='gray')
            plt.grid(True, which="minor", ls=":", alpha=0.2, color='gray')
            
            plt.xlabel('Peak Ground Acceleration (g)', fontsize=14, fontweight='bold')
            plt.ylabel('Annual Rate of Exceedance', fontsize=14, fontweight='bold')
            
            # Enhanced title with window information  
            title_lines = [
                f'Seismic Hazard Curve - Window {window_id}',
                f'Site: {lat:.4f}°N, {lon:.4f}°E'
            ]
            
            if time_range:
                title_lines.append(f'Time Range: {time_range}')
                
            plt.title('\n'.join(title_lines), fontsize=14, fontweight='bold', pad=20)
            
            plt.legend(loc='upper right', fontsize=12, framealpha=0.9)
            
            # Add return period labels on right axis
            ax = plt.gca()
            ax2 = ax.twinx()
            
            return_periods = [10, 50, 100, 500, 1000, 2500, 10000]
            return_rates = [1/rp for rp in return_periods]
            
            # Filter return periods to current y-axis range
            y_min, y_max = ax.get_ylim()
            valid_rp_mask = [(rate >= y_min and rate <= y_max) for rate in return_rates]
            valid_return_periods = [rp for rp, valid in zip(return_periods, valid_rp_mask) if valid]
            valid_return_rates = [rate for rate, valid in zip(return_rates, valid_rp_mask) if valid]
            
            if valid_return_rates:
                ax2.set_yscale('log')
                ax2.set_ylim(ax.get_ylim())
                ax2.set_yticks(valid_return_rates)
                ax2.set_yticklabels([f'{rp}' for rp in valid_return_periods])
                ax2.set_ylabel('Return Period (years)', fontsize=14, fontweight='bold')
            
            plt.tight_layout()
            
            # Save to window-specific directory
            plot_file = window_plots_dir / f"hazard_curve_{lat:.4f}_{lon:.4f}.png"
            plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
            plt.close()
            
            curves_created += 1
            logger.info(f"✅ Saved curve for site ({lat:.4f}, {lon:.4f}) to window {window_id} folder")
            
        except Exception as e:
            logger.error(f"❌ Error plotting curve for site ({lat:.4f}, {lon:.4f}): {e}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            plt.close('all')  # Clean up any open figures
            continue
    
    if curves_created == 0:
        logger.error("❌ No hazard curves were successfully created")
    else:
        logger.info(f"✅ Successfully created {curves_created} hazard curves in window {window_id}") 
              
def visualize_hazard_results(results_data, region_bounds, output_file, major_cities=None):
    """
    Create detailed hazard map using PyGMT.
    
    :param results_data: Dictionary with site coordinates and hazard values
    :param region_bounds: List [min_lon, max_lon, min_lat, max_lat]
    :param output_file: Path to save the output map
    :param major_cities: Dictionary of city names and coordinates (optional)
    """
    cpt_file = "hazard_levels.cpt"
    try: 
        # Convert results to pandas DataFrame
        data = []
        for (lat, lon), result in results_data.items():
            # Assuming we want the 2% in 50 years probability
            hazard_value = result['hazard_values'].get('annual_0.02', 0)  # Adjust based on your actual data
            data.append({
                'Latitude': lat,
                'Longitude': lon,
                'ground_motion': hazard_value
            })
        
        df = pd.DataFrame(data)


        for prob_type in df['probability_type'].unique():
            prob_data = df[df['probability_type'] == prob_type]
            
            # Create figure
            fig = pygmt.Figure()

            # Create grid from scattered data points
            grid = pygmt.surface(
                x=df['Longitude'],
                y=df['Latitude'],
                z=df['ground_motion'],
                region=region_bounds,
                spacing="0.05d",
                tension=0.5
            )

            # Create custom color palette for hazard levels
            cpt_file = "hazard_levels.cpt"
            with open(cpt_file, "w") as cpt:
                cpt.write(
                    "0   lightblue   0.1   yellow\n"
                    "0.1 yellow   0.2   green\n"
                    "0.2 green   0.3   orange\n"
                    "0.3 orange   0.5   red\n"
                    "0.5 red   1.0   darkred\n"
                )

            pygmt.makecpt(
                cmap=cpt_file,
                series=[0, 1.0],
                continuous=False
            )

            # Set up base map
            fig.basemap(
                region=region_bounds,
                projection="M6i",
                frame=["a", '+t"Seismic Hazard Map (Ground Motion Probability)"']
            )

            # Plot interpolated hazard values
            fig.grdimage(
                grid=grid,
                cmap=True
            )

            # Add contour lines
            fig.grdcontour(
                grid=grid,
                levels=[0.1, 0.2, 0.3, 0.5],
                annotation=[0.1, 0.2, 0.3, 0.5],
                pen="1p,black"
            )

            # Add coastlines and borders
            fig.coast(
                shorelines="1/0.5p,black",
                borders=["1/0.5p,black"],
                water="white",
                resolution="h"
            )

            # Add cities if provided
            if major_cities:
                for city, coords in major_cities.items():
                    fig.plot(
                        x=[coords[0]],
                        y=[coords[1]],
                        style="c0.2c",
                        fill="black",
                        pen="0.5p,black"
                    )
                    fig.text(
                        x=[coords[0]],
                        y=[coords[1]],
                        text=city,
                        offset="0.2c/0.2c",
                        font="8p,Helvetica,black"
                    )

            # Add colorbar
            fig.colorbar(
                frame=["x+l'Ground Motion (g)',y+l'Probability of Exceedance'"]
            )

            title = f"Seismic Hazard Map ({prob_type})"
            
            
            output_file_name = os.path.splitext(output_file)[0]
            prob_output_file = f"{output_file_name}_{prob_type}.png"
            fig.savefig(prob_output_file)
        pass
    finally:
        if os.path.exists(cpt_file):
            os.remove(cpt_file)

def create_hazard_map(all_results, config):
    """
    Create a hazard map for key design probability based on user settings.
    """
    try:
        import pygmt
    except ImportError:
        return create_simple_hazard_map(all_results, config)

    prob_type = config['output_settings'].get('probability_type', 'non_exceedance')
    probabilities = config['output_settings'].get('probabilities', [])
    # Pick first probability as primary map basis
    if probabilities:
        if prob_type == "non_exceedance":
            label = f"GM (g) at {(1 - probabilities[0][1]) * 100:.1f}% in {probabilities[0][0]}yr"
        elif prob_type == "exceedance":
            label = f"GM (g) at {probabilities[0][1] * 100:.1f}% in {probabilities[0][0]}yr"
        else:
            label = f"GM (g) at {probabilities[0] * 100:.2f}% annual"
    else:
        label = "Ground Motion (g)"

    data = []
    for (lat, lon), result in all_results.items():
        if result and 'hazard_values' in result:
            probs = result['hazard_values'].get('probabilities', {})
            val = list(probs.values())[0] if probs else 0
            data.append((lon, lat, val))

    if not data:
        logger.error("❌ No data for hazard map")
        return None

    df = pd.DataFrame(data, columns=["lon", "lat", "gm"])
    region = [df["lon"].min() - 0.1, df["lon"].max() + 0.1, df["lat"].min() - 0.1, df["lat"].max() + 0.1]

    fig = pygmt.Figure()
    grid = pygmt.surface(x=df["lon"], y=df["lat"], z=df["gm"], region=region, spacing="0.05")
    pygmt.makecpt(cmap="hot", series=[df["gm"].min(), df["gm"].max()])

    fig.basemap(region=region, projection="M6i", frame=["af", f"+tSeismic Hazard Map"])
    fig.grdimage(grid=grid, cmap=True)
    fig.colorbar(frame=[f"x+l{label}"])
    fig.plot(x=df["lon"], y=df["lat"], style="c0.1c", fill="black", pen="0.5p,white")

    output_file = Path(config['output_file']).parent / "visualizations" / "maps" / "hazard_map.png"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_file), dpi=300)
    logger.info(f"✅ Hazard map saved: {output_file}")
    return output_file


def create_simple_hazard_map(df, user_config):
    """Create a simple hazard map using matplotlib as fallback - FIXED"""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.tri as tri
        from matplotlib.colors import LinearSegmentedColormap

        logger.info("Creating simple matplotlib hazard map...")
        fig, ax = plt.subplots(figsize=(12, 10))

        # Create triangulation for interpolation
        triang = tri.Triangulation(df['Longitude'], df['Latitude'])        

        # Create contour plot
        levels = np.linspace(df['ground_motion'].min(), df['ground_motion'].max(), 20)
        contourf = ax.tricontourf(triang, df['ground_motion'], levels=levels, cmap='hot')
        contour = ax.tricontour(triang, df['ground_motion'], levels=levels[::2], colors='black', linewidths=0.5)

        # Add colorbar
        cbar = plt.colorbar(contourf, ax=ax)
        cbar.set_label('Ground Motion (g)', rotation=270, labelpad=15)

        # Add data points
        scatter = ax.scatter(df['Longitude'], df['Latitude'], 
                           c='white', s=20, edgecolor='black', linewidth=0.5, zorder=5)

        # Set labels and title
        ax.set_xlabel('Longitude (°)')
        ax.set_ylabel('Latitude (°)')
        ax.set_title('Seismic Hazard Map', fontsize=14, fontweight='bold')

        # Add grid
        ax.grid(True, alpha=0.3)

        # Set aspect ratio
        ax.set_aspect('equal')

        # FIXED: Create filename with window identifier
        output_file = Path(user_config.get('output_file', 'hazard_results.txt'))
        maps_dir = output_file.parent / "visualizations" / "maps"
        maps_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract window identifier
        base_name = output_file.stem
        if "window_" in base_name:
            window_id = base_name.split("window_")[-1]
            map_file = maps_dir / f"hazard_map_simple_window_{window_id}.png"
        else:
            map_file = maps_dir / "hazard_map_simple.png"

        plt.savefig(str(map_file), dpi=300, bbox_inches='tight')
        plt.close()

        logger.info(f"✅ Simple hazard map saved to: {map_file}")
        return map_file

    except Exception as e:
        logger.error(f"❌ Error creating simple hazard map: {e}")
        return None
       
def get_top_hazard_sites(all_results, n=10):
    """
    Get the top N sites with highest DESIGN hazard values (not maximum).
    FIXED: Better error handling and multiple ranking options
    
    Args:
        all_results: Dictionary of results keyed by (lat, lon)
        n: Number of top sites to return
    Returns:
        List of (lat, lon) tuples for top N sites
    """
    if not all_results:
        logger.warning("No results available for site ranking")
        return []
    
    # Extract hazard values for ranking with multiple fallback options
    site_hazards = []
    ranking_methods = [
        ('annual_0.02', '2% annual probability'),
        ('annual_0.1', '10% annual probability'), 
        ('non_exceed_50yr_0.98', '2% in 50 years'),
        ('exceed_50yr_0.02', '2% in 50 years'),
        ('non_exceed_50yr_0.90', '10% in 50 years'),
        ('exceed_50yr_0.1', '10% in 50 years')
    ]
    
    for (lat, lon), result in all_results.items():
        if not result or not result.get('hazard_values'):
            logger.debug(f"Site ({lat:.4f}, {lon:.4f}) has no hazard_values")
            continue
            
        probabilities = result['hazard_values'].get('probabilities', {})
        if not probabilities:
            logger.debug(f"Site ({lat:.4f}, {lon:.4f}) has no probabilities")
            continue
        
        # Try each ranking method until we find values
        design_value = 0
        method_used = "none"
        
        for prob_key, method_name in ranking_methods:
            if prob_key in probabilities:
                design_value = probabilities[prob_key]
                method_used = method_name
                break
        
        # Fallback: use any available probability value
        if design_value == 0 and probabilities:
            first_key = list(probabilities.keys())[0]
            design_value = probabilities[first_key]
            method_used = f"fallback ({first_key})"
            
        if design_value > 0:
            site_hazards.append(((lat, lon), design_value, method_used))
            logger.debug(f"Site ({lat:.4f}, {lon:.4f}): {design_value:.6f}g using {method_used}")
    
    if not site_hazards:
        logger.warning("No sites have valid probability data for ranking")
        return []
    
    # Sort by design hazard value and get top N
    site_hazards.sort(key=lambda x: x[1], reverse=True)
    
    logger.info(f"Found {len(site_hazards)} sites with hazard data")
    logger.info(f"Top site: ({site_hazards[0][0][0]:.4f}, {site_hazards[0][0][1]:.4f}) "
               f"= {site_hazards[0][1]:.6f}g using {site_hazards[0][2]}")
    
    return [site[0] for site in site_hazards[:n]]


def generate_visualizations(all_results, user_config):
    """
    Generate all visualizations for the hazard analysis
    
    Args:
        all_results: Dictionary of results for all sites
        user_config: Configuration dictionary
    """
    from pathlib import Path
    import os
    
    # Create output directory for visualizations
    if 'output_file' not in user_config:
        print("Error: output_file not specified in configuration")
        return
        
    output_base = Path(user_config['output_file']).parent
    viz_dir = output_base / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate hazard curves if enabled
    if user_config.get('output_settings', {}).get('plot_hazard_curves', False):
        print("\nGenerating hazard curves for highest hazard sites...")
        plot_hazard_curves(all_results, viz_dir, user_config)
    
    # Generate hazard map
    print("\nGenerating hazard map...")
    map_file = create_hazard_map(all_results, user_config)
    if map_file:
        logger.info(f"Hazard map saved to: {map_file}")
    
    # Generate GIS CSV file if enabled
    if user_config.get('output_settings', {}).get('export_gis_csv', False):
        print("\nGenerating CSV file for GIS...")
        csv_file = export_hazard_for_gis(all_results, user_config, viz_dir)
        if csv_file:
            logger.info(f"GIS CSV file saved to: {csv_file}")
    
    logger.info(f"\nAll visualizations saved to: {viz_dir}")

def export_hazard_for_gis(all_results, config, output_dir=None):
    """
    Export GIS-ready CSV with headers matching the friendly TXT output format.
    Adds proper validation and logging.
    """
    import csv
    from pathlib import Path
    
    if not config.get('output_settings', {}).get('export_gis_csv', False):
        logger.info("GIS CSV export is disabled in configuration")
        return None

    # Determine output directory
    if output_dir is None:
        output_dir = Path(config['output_file']).parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Build filename using window ID if available
    output_file = Path(config.get('output_file', 'hazard_results.txt'))
    base_name = output_file.stem
    if "window_" in base_name:
        window_id = base_name.split("window_")[-1]
        csv_filename = output_dir / f"hazard_results_for_gis_window_{window_id}.csv"
    else:
        csv_filename = output_dir / "hazard_results_for_gis.csv"

    # Probability configuration
    prob_type = config['output_settings'].get('probability_type', 'non_exceedance')
    probabilities = config['output_settings'].get('probabilities', [])

    logger.info(f"Exporting GIS data with {prob_type} probabilities: {probabilities}")

    # Friendly headers
    headers = ["Longitude", "Latitude"]
    if prob_type == "non_exceedance":
        for p in probabilities:
            if isinstance(p, (list, tuple)) and len(p) == 2:
                time_period, non_exceed = p
                exceed_percent = (1 - non_exceed) * 100
                headers.append(f"GM at {exceed_percent:.1f}% in {time_period}yr (g)")
    elif prob_type == "exceedance":
        for p in probabilities:
            if isinstance(p, (list, tuple)) and len(p) == 2:
                time_period, exceed_prob = p
                exceed_percent = exceed_prob * 100
                headers.append(f"GM at {exceed_percent:.1f}% in {time_period}yr (g)")
    elif prob_type == "annual":
        for p in probabilities:
            if isinstance(p, (int, float)):
                headers.append(f"GM at {p*100:.2f}% annual (g)")

    rows = []
    total_sites = len(all_results)
    sites_with_data = 0
    debug_first_site = True

    for (lat, lon), result in all_results.items():
        if not result or 'hazard_values' not in result:
            continue
        row = [lon, lat]

        probs_dict = result['hazard_values'].get('probabilities', {})
        if probs_dict:
            if debug_first_site:
                logger.info(f"DEBUG - First site probability keys: {list(probs_dict.keys())}")
                debug_first_site = False

            if prob_type == "non_exceedance":
                for p in probabilities:
                    time_period, non_exceed = p
                    key = f"non_exceed_{time_period}yr_{non_exceed}"
                    val = probs_dict.get(key)
                    row.append(f"{val:.6f}" if val is not None else "")
            elif prob_type == "exceedance":
                for p in probabilities:
                    time_period, exceed_prob = p
                    key = f"exceed_{time_period}yr_{exceed_prob}"
                    val = probs_dict.get(key)
                    row.append(f"{val:.6f}" if val is not None else "")
            elif prob_type == "annual":
                for p in probabilities:
                    key = f"annual_{p}"
                    val = probs_dict.get(key)
                    row.append(f"{val:.6f}" if val is not None else "")
        else:
            row += [""] * (len(headers) - 2)

        rows.append(row)
        sites_with_data += 1

    if sites_with_data == 0:
        logger.error("❌ No sites have probability data for GIS export!")
        return None

    try:
        with open(csv_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
            writer.writerows(rows)

        logger.info(f"✅ GIS CSV generated: {csv_filename}")
        logger.info(f"Headers: {headers}")
        if rows:
            logger.info(f"Sample row: {rows[0]}")

        return csv_filename

    except Exception as e:
        logger.error(f"Error creating GIS CSV file: {e}")
        return None

               
def interactive_config(defaults):
    """
    Allow users to provide all inputs interactively.
    :param defaults: Dictionary of default configuration values.
    :return: Updated configuration dictionary.
    """
    print("Interactive Configuration (Press Enter to use default values):")
    config_values = defaults.copy()
    
    # Initialize completeness settings
    if "completeness" not in config_values:
        config_values["completeness"] = {
            "enabled": False,
            "thresholds": []
        }
    
    # Completeness configuration
    print("\nCatalog Completeness Configuration:")
    if input("\nDo you want to specify completeness thresholds? (y/n) [n]: ").lower().startswith('y'):
        config_values["completeness"]["enabled"] = True
        config_values["completeness"]["thresholds"] = []
        
        print("\nEnter magnitude thresholds from SMALLEST to LARGEST magnitude.")
        print("Press Enter without input when done.")
        
        while True:
            mag_input = input("\nMagnitude threshold (<ret> when done): ")
            if not mag_input:
                break
                
            try:
                magnitude = float(mag_input)
                date = input("Date (YYYYMMDD) from which catalog is complete: ")
                if not (len(date) == 8 and date.isdigit()):
                    print("Invalid date format. Must be YYYYMMDD. Skipping this threshold.")
                    continue
                
                config_values["completeness"]["thresholds"].append((magnitude, date))
                
            except ValueError:
                print("Invalid magnitude value. Must be a number.")
                continue
        
        # Sort thresholds by magnitude
        config_values["completeness"]["thresholds"].sort(key=lambda x: x[0])
    
    
    # Characteristic earthquake configuration
    print("\nCharacteristic Earthquake Configuration:")
    if input("Add a characteristic earthquake? (y/n) [n]: ").lower().startswith('y'):
        config_values["characteristic_earthquake"]["enabled"] = True
        config_values["characteristic_earthquake"]["magnitude"] = float(
            input(f"Magnitude [{defaults['characteristic_earthquake']['magnitude']}]: ") 
            or defaults['characteristic_earthquake']['magnitude']
        )
        config_values["characteristic_earthquake"]["repeat_time"] = float(
            input(f"Repeat time (years) [{defaults['characteristic_earthquake']['repeat_time']}]: ")
            or defaults['characteristic_earthquake']['repeat_time']
        )
        config_values["characteristic_earthquake"]["latitude"] = float(
            input(f"Latitude [{defaults['characteristic_earthquake']['latitude']}]: ")
            or defaults['characteristic_earthquake']['latitude']
        )
        config_values["characteristic_earthquake"]["longitude"] = float(
            input(f"Longitude [{defaults['characteristic_earthquake']['longitude']}]: ")
            or defaults['characteristic_earthquake']['longitude']
        )

    # Additional earthquakes configuration
    print("\nAdditional Earthquakes Configuration:")
    if input("Add earthquakes above catalog maximum? (y/n) [n]: ").lower().startswith('y'):
        config_values["additional_earthquakes"]["enabled"] = True
        config_values["additional_earthquakes"]["max_magnitude"] = float(
            input(f"Maximum magnitude [{defaults['additional_earthquakes']['max_magnitude']}]: ")
            or defaults['additional_earthquakes']['max_magnitude']
        )
        config_values["additional_earthquakes"]["a_value"] = float(
            input(f"A-value [{defaults['additional_earthquakes']['a_value']}]: ")
            or defaults['additional_earthquakes']['a_value']
        )
        config_values["additional_earthquakes"]["b_value"] = float(
            input(f"B-value [{defaults['additional_earthquakes']['b_value']}]: ")
            or defaults['additional_earthquakes']['b_value']
        )

    # Location variation configuration
    print("\nLocation Variation Configuration:")
    if input("Vary locations in synthetic catalog? (y/n) [n]: ").lower().startswith('y'):
        config_values["location_variation"]["enabled"] = True
        print("Scatter types:\n0=linear flat\n1=linear decrease\n2=gaussian")
        config_values["location_variation"]["scatter_type"] = int(
            input(f"Scatter type [{defaults['location_variation']['scatter_type']}]: ")
            or defaults['location_variation']['scatter_type']
        )
        config_values["location_variation"]["max_scatter_distance"] = float(
            input(f"Maximum scatter distance (km) [{defaults['location_variation']['max_scatter_distance']}]: ")
            or defaults['location_variation']['max_scatter_distance']
        )

    # Site configuration
    print("\nSite Configuration:")
    config_values["grid_mode"] = input("Use grid mode? (y/n) [y]: ").lower().startswith('y')
    
    if config_values["grid_mode"]:
        print("\nGrid Configuration:")
        config_values["num_grid_lat"] = int(
            input(f"Number of latitude points [{defaults['num_grid_lat']}]: ")
            or defaults['num_grid_lat']
        )
        config_values["grid_lat_min"] = float(
            input(f"Lowest latitude [{defaults['grid_lat_min']}]: ")
            or defaults['grid_lat_min']
        )
        config_values["grid_lat_spacing"] = float(
            input(f"Latitude increment [{defaults['grid_lat_spacing']}]: ")
            or defaults['grid_lat_spacing']
        )
        # Similar for longitude...
    else:
        num_sites = int(input("Number of sites: "))
        sites = []
        for i in range(num_sites):
            logger.info(f"\nSite {i+1}:")
            lat = float(input("  Latitude: "))
            lon = float(input("  Longitude: "))
            sites.append((lat, lon))
        config_values["sites"] = sites

    # Output preferences
    print("\nOutput Configuration:")
    config_values["output_settings"]["full_output"] = input(
        "Include synthetic earthquake list? (y/n) [y]: "
    ).lower().startswith('y')
    
    config_values["output_settings"]["min_ground_motion"] = float(
        input(f"Minimum ground motion value [{defaults['output_settings']['min_ground_motion']}]: ")
        or defaults['output_settings']['min_ground_motion']
    )
    
    print("\nProbability type:")
    print("1: Annual probabilities")
    print("2: non_exceedance probabilities")
    print("3: Neither")
    prob_choice = input("Choice [3]: ") or "3"
    
    if prob_choice == "1":
        config_values["output_settings"]["probability_type"] = "annual"
        num_probs = int(input("Number of probability values: "))
        probs = []
        for i in range(num_probs):
            prob = float(input(f"Probability {i+1}: "))
            probs.append(prob)
        config_values["output_settings"]["probabilities"] = probs
    elif prob_choice == "2":
        config_values["output_settings"]["probability_type"] = "non_exceedance"
        num_probs = int(input("Number of probability values: "))
        probs = []
        for i in range(num_probs):
            time = float(input(f"Time period {i+1} (years): "))
            prob = float(input(f"Probability {i+1}: "))
            probs.append((time, prob))
        config_values["output_settings"]["probabilities"] = probs

    return config_values

def validate_performance_settings(settings):
    required_keys = ['batch_size', 'chunk_size', 'cache_size', 'max_workers']
    for key in required_keys:
        if key not in settings:
            raise ValueError(f"Missing required setting: {key}")
        if settings[key] <= 0:
            raise ValueError(f"Invalid value for {key}: {settings[key]}")
    return settings

def main():
    """
    Enhanced main function for seismic hazard analysis with 
    improved handling of large catalogs and parallel processing.
    """
    
    global dataframes
    
    parser = argparse.ArgumentParser(description="Seismic Hazard Analysis Tool")
    parser.add_argument("-i", "--interactive", action="store_true", help="Enable interactive configuration")
    parser.add_argument("-c", "--config", help="Path to custom configuration file")
    parser.add_argument("--log", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], 
                        default="INFO", help="Set the logging level")
    args = parser.parse_args()

    # Configure the module-level logger
    log_level = getattr(logging, args.log.upper(), logging.INFO)
    setup_logging(log_level)
    
    # Log the start of the program
    logger.info(f"Starting seismic hazard analysis")
    
    # Load configuration
    try:
        if args.config:
            config_path = Path(args.config).resolve()
            logger.info(f"Loading configuration from {config_path}...")
            
            if not config_path.exists():
                logger.error(f"Error: Configuration file {config_path} not found.")
                return
                
            # Import the config file dynamically
            import importlib.util
            try:
                spec = importlib.util.spec_from_file_location("custom_config", config_path)
                custom_config = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(custom_config)
                user_config = custom_config.__dict__.copy()
                
                # Update paths to be relative to the config file's directory
                config_dir = config_path.parent
                if 'catalog_file' in user_config and not Path(user_config['catalog_file']).is_absolute():
                    user_config['catalog_file'] = config_dir / user_config['catalog_file']
                if 'output_file' in user_config and not Path(user_config['output_file']).is_absolute():
                    user_config['output_file'] = config_dir / user_config['output_file']
                    
                # Update GMPE file paths if they exist
                if 'gmpe_files' in user_config:
                    for model, filepath in user_config['gmpe_files'].items():
                        if not Path(filepath).is_absolute():
                            user_config['gmpe_files'][model] = config_dir / filepath
                
            except Exception as e:
                logger.error(f"Error loading configuration file: {e}")
                logger.info(f"Falling back to default configuration.")
                return
        else:
            logger.info(f"Using default configuration...")
            return
    except Exception as e:
        logger.info(f"Error in configuration loading: {e}")
        return
    
    # Ensure required fields exist in config
    required_fields = ['catalog_file', 'output_file', 'default_model', 'synduration']
    missing_fields = [field for field in required_fields if field not in user_config]
    if missing_fields:
        logger.warning(f"Warning: Missing required fields in configuration: {missing_fields}")
        logger.info(f"Please check your configuration file or use interactive mode to provide these values.")
        if not args.interactive:
            return
    
    # Initialize completeness if not present
    if "completeness" not in user_config:
        user_config["completeness"] = {
            "enabled": False,
            "thresholds": []
        }
    
    # Override with interactive input if specified
    if args.interactive:
        print("Entering interactive mode...")
        try:
            user_config = interactive_config(user_config)
        except Exception as e:
            logger.error(f"Error in interactive configuration: {e}")
            return
    
    # Initialize coefficient loading with the appropriate configuration
    try:
        if 'gmpe_files' in user_config:
            dataframes = load_gmpe_coefficients(user_config['gmpe_files'], user_config['default_model'])
        else:
            logger.error(f"Error: 'gmpe_files' not found in configuration.")
            return
    except Exception as e:
        logger.error(f"Error loading GMPE coefficients: {e}")
        dataframes = {}
        
    try:
        # Validate completeness configuration
        validate_completeness_config(user_config)
        
        # Initialize catalog with completeness settings
        catalog = Catalog.from_file(
            user_config['catalog_file'],
            completeness=user_config.get('completeness')  
        )
        
        # Get initial catalog statistics
        stats = catalog.get_statistics()
        print("\nInitial Catalog Statistics:")
        # Add these lines showing catalog date range
        start_date = min(entry.date for entry in catalog.entries)
        end_date = max(entry.date for entry in catalog.entries)
        logger.info(f"Catalog begins on {start_date} and ends on {end_date}")
        logger.info(f"Magnitude range: {stats['min_magnitude']:.1f} to {stats['max_magnitude']:.1f}")
        logger.info(f"Spatial bounds: {stats['spatial_bounds']}")

        
        # Print completeness settings if enabled
        if user_config["completeness"]["enabled"]:
            print("\nCompleteness thresholds applied:")
            for mag, date in user_config["completeness"]["thresholds"]:
                logger.info(f"M{mag:.1f} from {date}")

        # Handle additional earthquakes above maximum magnitude
        if user_config["additional_earthquakes"]["enabled"]:
            print("\nAdding earthquakes above maximum catalog magnitude...")
            max_mag = user_config["additional_earthquakes"]["max_magnitude"]
            a_value = user_config["additional_earthquakes"]["a_value"]
            b_value = user_config["additional_earthquakes"]["b_value"]
            # Calculate additional events based on G-R relationship
            logger.info(f"Using G-R relationship with a={a_value}, b={b_value}, max_mag={max_mag}")
        
        # Generate histogram from catalog
        histogram = catalog.create_histogram()
        if not histogram:
            raise ValueError("Histogram is empty. Check catalog loading.")
        
        # Add detailed histogram output like in new.py
        print("\nIn the catalog")
        for hist in histogram:
            logger.info(f"  Magnitude={hist.mag:.1f}  Number of events={hist.numevents}")
        
        # Handle characteristic earthquake if enabled
        if user_config["characteristic_earthquake"]["enabled"]:
            print("\nIncluding characteristic earthquake in analysis...")
            char_eq = user_config["characteristic_earthquake"]
            logger.info(f"Magnitude: {char_eq['magnitude']}")
            logger.info(f"Location: ({char_eq['latitude']}, {char_eq['longitude']})")
            logger.info(f"Repeat time: {char_eq['repeat_time']} years")

        # Calculate catalog duration
        catalog_duration = calculate_catalog_duration(
            catalog,
            user_config["additional_earthquakes"],
            user_config["characteristic_earthquake"]
        )
        
        # Generate the full histogram including G-R and characteristic earthquakes
        histogram = catalog.prepare_full_histogram(catalog_duration, user_config)
        if not histogram:
            raise ValueError("Histogram is empty. Check catalog loading.")

        # Calculate total events from complete histogram
        total_events = int(histogram[-1].cumnumevents)
        
        # Setup for ground motion calculations
        if user_config["location_variation"]["enabled"]:
            print("\nLocation variation enabled:")
            logger.info(f"Scatter type: {user_config['location_variation']['scatter_type']}")
            logger.info(f"Maximum scatter distance: {user_config['location_variation']['max_scatter_distance']} km")

        # Generate site list
        sites = generate_site_coordinates(user_config)
        total_sites = len(sites)
        logger.info(f"\nProcessing {total_sites} sites...")

        # Initialize results storage
        all_results = {}

        # Ensure catalog is correctly loaded
        if not isinstance(catalog, Catalog):
            raise ValueError("Catalog is not initialized correctly. Check catalog loading.")

        # Process sites in batches
        batch_size = PERFORMANCE_SETTINGS['batch_size']
        for batch_start in range(0, total_sites, batch_size):
            batch_end = min(batch_start + batch_size, total_sites)
            batch_sites = sites[batch_start:batch_end]
            logger.info(f"\nProcessing batch {batch_start//batch_size + 1} of {(total_sites + batch_size - 1)//batch_size}")
            
            # Process each site in the batch
            for site_lat, site_lon in batch_sites:
                logger.info(f"\nAnalyzing site: ({site_lat:.4f}, {site_lon:.4f})")
                
                # Find nearby events for efficiency
                nearby_events = catalog.find_nearby_events(
                    site_lat, 
                    site_lon, 
                    radius=user_config.get('distance_threshold', 300.0)
                )
                
                # Ensure at least some events exist
                if len(nearby_events) < 10:
                    print("Warning: Very few nearby events found. Expanding search radius.")
                    nearby_events = catalog.find_nearby_events(site_lat, site_lon, radius=500.0)
                    
                # Generate synthetic events - pass nearby_events directly
                synthetic_events = generate_synthetic_events(
                    nearby_events,  # Pass the events list directly
                    histogram,
                    site_lat,
                    site_lon,
                    user_config
                )


                # Calculate hazard values
                hazard_values = calculate_hazard_values(
                    synthetic_events,
                    site_lat,
                    site_lon,
                    user_config["synduration"],
                    user_config["output_settings"],
                    user_config
                )
                
                # Store results
                all_results[(site_lat, site_lon)] = {
                    'hazard_values': hazard_values,
                    'synthetic_events': synthetic_events
                }
                
                # Output results for this site
                output_site_results(
                    site_lat, site_lon,
                    synthetic_events,
                    hazard_values,
                    user_config["output_settings"],
                    user_config["output_file"],
                    user_config
                )

                # Optional: Clear memory for large catalogs
                if len(all_results) > PERFORMANCE_SETTINGS['cache_size']:
                    all_results.clear()

        # Generate all visualizations
        generate_visualizations(all_results, user_config)

        print("\nSeismic hazard analysis completed successfully.")

    except Exception as e:
        logger.error(f"Error: {e}")
        raise  # Re-raise the exception for debugging

if __name__ == "__main__":
    main()