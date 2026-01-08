"""
GMPE Adapter
============

FIXES APPLIED:
1. ✅ Proper distance calculations (R_rup vs R_jb) using rupture_geometry
2. ✅ Correct width calculation (down-dip width from dip angle)
3. ✅ Integration with RSQSimEvent dataclass
4. ✅ PyGMM-compatible parameter passing

Author: Victor Olawoyin
Date: January 2026
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# FIXED: Correct import path
from src.gmpe_calculator import GMPECalculator, calculate_ground_motion
from src.rupture_geometry import EventRupture, RSQSimGeometryReader

logger = logging.getLogger(__name__)


@dataclass
class RSQSimEvent:
    """
    Data class for RSQSim events with new catalog format
    
    ENHANCED: Added methods for proper width and depth calculations
    """
    event_id: int
    time: float  # seconds
    magnitude: float
    moment: float  # N-m
    area: float  # m^2
    num_elements: int
    avg_slip: float  # m
    slip_rate: float  # m/yr
    hypo_lat: float
    hypo_lon: float
    hypo_depth: float  # km
    cent_lat: float
    cent_lon: float
    cent_depth: float  # km
    upper_depth: float  # km
    lower_depth: float  # km
    
    @classmethod
    def from_csv_row(cls, row: pd.Series) -> 'RSQSimEvent':
        """Create RSQSimEvent from pandas Series (CSV row)"""
        return cls(
            event_id=int(row['Event ID']),
            time=float(row['Occurrence Time (s)']),
            magnitude=float(row['Magnitude']),
            moment=float(row['Moment (N-m)']),
            area=float(row['Area (m^2)']),
            num_elements=int(row['Number of Participating Elements']),
            avg_slip=float(row['Average Slip (m)']),
            slip_rate=float(row['Average Element Slip Rate (m/yr)']),
            hypo_lat=float(row['Hypocenter Latitude']),
            hypo_lon=float(row['Hypocenter Longitude']),
            hypo_depth=float(row['Hypocenter Depth (km)']),
            cent_lat=float(row['Centroid Latitude']),
            cent_lon=float(row['Centroid Longitude']),
            cent_depth=float(row['Centroid Depth (km)']),
            upper_depth=float(row['Upper Depth (km)']),
            lower_depth=float(row['Lower Depth (km)'])
        )
    
    def get_vertical_extent(self) -> float:
        """
        Get vertical extent of rupture
        
        Returns:
            Vertical extent in km
        """
        return self.lower_depth - self.upper_depth
    
    def get_rupture_width(self, dip: float = 90.0) -> float:
        """
        Calculate down-dip rupture width
        
        FIX #7: Properly accounts for dip angle
        
        Args:
            dip: Fault dip angle in degrees (90 = vertical)
            
        Returns:
            Down-dip width in km
        """
        vertical_extent = self.get_vertical_extent()
        
        # For vertical faults, width = vertical extent
        if abs(dip - 90.0) < 0.1:
            return vertical_extent
        
        # For dipping faults: width = vertical_extent / sin(dip)
        dip_rad = np.radians(dip)
        width = vertical_extent / np.sin(dip_rad)
        
        return width
    
    def estimate_dip_from_geometry(self) -> float:
        """
        Estimate dip from area and vertical extent
        
        Uses: Area = length × width
              width = vertical_extent / sin(dip)
        
        Returns:
            Estimated dip angle in degrees (defaults to 90 if can't estimate)
        """
        vertical_extent_m = self.get_vertical_extent() * 1000.0  # Convert to meters
        
        if self.area > 0 and vertical_extent_m > 0:
            # Estimate width from area assuming square-ish rupture
            length_m = np.sqrt(self.area)
            width_m = self.area / length_m
            
            # Calculate dip: sin(dip) = vertical_extent / width
            sin_dip = vertical_extent_m / width_m
            
            # Clamp to valid range
            sin_dip = np.clip(sin_dip, 0.0, 1.0)
            dip = np.degrees(np.arcsin(sin_dip))
            
            # Sanity check: dip should be reasonable (30-90 degrees)
            if 30.0 <= dip <= 90.0:
                return dip
        
        # Default to vertical if can't estimate
        return 90.0
    
    def get_ztor(self) -> float:
        """Get depth to top of rupture (km)"""
        return self.upper_depth
    
    def get_depth_hyp(self) -> float:
        """Get hypocenter depth (km)"""
        return self.hypo_depth


class RSQSimGMPEAdapter:
    """
    Adapter class to use PyGMM GMPE calculator with RSQSim events
    
    FULLY FIXED:
    - Proper distance calculations using EventRupture
    - Correct width calculations accounting for dip
    - PyGMM-compatible parameter passing
    """
    
    def __init__(self,
                 geometry_reader: Optional[RSQSimGeometryReader] = None,
                 calculator: Optional[GMPECalculator] = None,
                 use_ensemble: bool = True,
                 vs30: float = 760.0,
                 mechanism: str = 'strike-slip'):
        """
        Initialize adapter
        
        Args:
            geometry_reader: RSQSim geometry reader for distance calculations
            models: List of GMPE models to use (default: all NGA-West2)
            use_ensemble: Use NSHM 2023 weighted ensemble
            vs30: Default site velocity
            mechanism: Default fault mechanism
            calculator: Optional custom GMPECalculator instance
        """
        
        # Use provided calculator or create default
        if calculator is not None:
            self.calculator = calculator  # ✅ Use custom
        else:
            self.calculator = GMPECalculator()  # Default
        
        self.geometry_reader = geometry_reader
        self.use_ensemble = use_ensemble
        self.default_vs30 = vs30
        self.default_mechanism = mechanism
        
    
    def calculate_ground_motion_from_event(self,
                                          event: RSQSimEvent,
                                          site_lat: float,
                                          site_lon: float,
                                          event_rupture: Optional[EventRupture] = None,
                                          vs30: Optional[float] = None,
                                          period: float = 0.01,
                                          mechanism: Optional[str] = None) -> Dict:
        """
        Calculate ground motion for an RSQSim event at a site
        
        FIX #5 & #7: Proper distance and width calculations
        
        Args:
            event: RSQSimEvent object
            site_lat: Site latitude
            site_lon: Site longitude  
            event_rupture: EventRupture object (for proper distances)
            vs30: Site velocity (uses default if None)
            period: Spectral period (0.01 for PGA)
            mechanism: Fault mechanism (uses default if None)
            
        Returns:
            Dictionary with ground motion results
        """
        if vs30 is None:
            vs30 = self.default_vs30
        
        if mechanism is None:
            mechanism = self.default_mechanism
        
        # ========================================================
        # FIX #5: Proper distance calculations
        # ========================================================
        if event_rupture is not None:
            # Use EventRupture for accurate R_rup and R_jb
            dist_rup, dist_jb, nearest_patch = event_rupture.distance_to_site_latlon(
                site_lat, site_lon
            )
            
            # Estimate dist_x (site coordinate) - simplified
            # For vertical SS faults, dist_x ≈ 0 near fault
            # For more accurate calculation, need strike information
            dist_x = 0.0  # Conservative assumption
            
            logger.debug(
                f"Event {event.event_id}: R_rup={dist_rup:.2f}km, "
                f"R_jb={dist_jb:.2f}km (from geometry)"
            )
            
        else:
            # Fallback: Use simple distance calculation
            # This is less accurate but still better than nothing
            logger.warning(
                f"Event {event.event_id}: No EventRupture provided, "
                f"using simplified distance calculation"
            )
            
            dist_rup = self._calculate_simple_distance(
                event.hypo_lat, event.hypo_lon, event.hypo_depth,
                site_lat, site_lon
            )
            
            # For vertical faults, R_jb ≈ R_rup
            # For dipping faults, this is an approximation
            dip = event.estimate_dip_from_geometry()
            if abs(dip - 90.0) < 5.0:
                dist_jb = dist_rup  # Nearly vertical
            else:
                # Rough approximation for dipping faults
                dist_jb = dist_rup * 0.9  # Typically smaller
            
            dist_x = 0.0  # Default
        
        # ========================================================
        # FIX #7: Proper width calculation
        # ========================================================
        dip = event.estimate_dip_from_geometry()
        width = event.get_rupture_width(dip=dip)
        depth_tor = event.get_ztor()
        depth_hyp = event.get_depth_hyp()
        
        logger.debug(
            f"Event {event.event_id}: M{event.magnitude:.2f}, "
            f"width={width:.2f}km, dip={dip:.1f}°, Ztor={depth_tor:.2f}km"
        )
        
        # Calculate ground motion
        if self.use_ensemble:
            result = self.calculator.calculate_weighted_ensemble(
                magnitude=event.magnitude,
                distance=dist_rup,
                vs30=vs30,
                mechanism=mechanism,
                period=period,
                width=width,
                depth_tor=depth_tor,
                depth_hyp=depth_hyp,
                dist_jb=dist_jb,
                dist_x=dist_x,
                dip=dip,
                include_individual=False
            )
            
            return {
                'log10_pga': result['median_log10'],
                'pga_g': result['median_g'],
                'ln_pga': result['median_ln'],
                'total_stddev': result['total_stddev_ln'],
                'aleatory_stddev': result['aleatory_stddev_ln'],
                'epistemic_stddev': result['epistemic_stddev_ln'],
                'event_id': event.event_id,
                'magnitude': event.magnitude,
                'distance_rup': dist_rup,
                'distance_jb': dist_jb,
                'width': width,
                'dip': dip,
                'vs30': vs30
            }
        else:
            # Use first model only
            result = self.calculator.calculate_single_model(
                model=self.calculator.models[0],
                magnitude=event.magnitude,
                distance=dist_rup,
                vs30=vs30,
                mechanism=mechanism,
                period=period,
                width=width,
                depth_tor=depth_tor,
                depth_hyp=depth_hyp,
                dist_jb=dist_jb,
                dist_x=dist_x,
                dip=dip
            )
            
            return {
                'log10_pga': result['median_log10'],
                'pga_g': result['median_g'],
                'ln_pga': result['median_ln'],
                'stddev': result['stddev_ln'],
                'event_id': event.event_id,
                'magnitude': event.magnitude,
                'distance_rup': dist_rup,
                'distance_jb': dist_jb,
                'width': width,
                'dip': dip,
                'vs30': vs30
            }
    
    def _calculate_simple_distance(self,
                                   event_lat: float,
                                   event_lon: float,
                                   event_depth: float,
                                   site_lat: float,
                                   site_lon: float) -> float:
        """
        Simple distance calculation (fallback when no geometry available)
        
        Uses great circle distance + depth correction
        
        Returns:
            Distance in km
        """
        # Haversine formula for surface distance
        lat1, lon1 = np.radians(event_lat), np.radians(event_lon)
        lat2, lon2 = np.radians(site_lat), np.radians(site_lon)
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Earth radius in km
        R = 6371.0
        horiz_dist = R * c
        
        # Add depth component (Pythagorean theorem)
        dist_3d = np.sqrt(horiz_dist**2 + event_depth**2)
        
        return dist_3d
    
    def process_catalog_for_site(self,
                                 catalog_file: str,
                                 site_lat: float,
                                 site_lon: float,
                                 max_distance: float = 300.0,
                                 min_magnitude: float = 5.0) -> List[Dict]:
        """
        Process entire RSQSim catalog for a single site
        
        Args:
            catalog_file: Path to RSQSim catalog CSV
            site_lat: Site latitude
            site_lon: Site longitude
            max_distance: Maximum distance to consider (km)
            min_magnitude: Minimum magnitude to consider
            
        Returns:
            List of ground motion results
        """
        logger.info(f"📖 Reading catalog: {catalog_file}")
        
        # Read catalog
        df = pd.read_csv(catalog_file)
        logger.info(f"   Total events in catalog: {len(df):,}")
        
        # Filter by magnitude
        df = df[df['Magnitude'] >= min_magnitude]
        logger.info(f"   Events with M≥{min_magnitude}: {len(df):,}")
        
        results = []
        events_processed = 0
        events_within_distance = 0
        
        for idx, row in df.iterrows():
            try:
                # Create event object
                event = RSQSimEvent.from_csv_row(row)
                
                # Quick distance check using hypocenter
                approx_dist = self._calculate_simple_distance(
                    event.hypo_lat, event.hypo_lon, event.hypo_depth,
                    site_lat, site_lon
                )
                
                if approx_dist > max_distance:
                    continue
                
                events_within_distance += 1
                
                # Calculate ground motion
                # NOTE: For best results, provide EventRupture object
                gm_result = self.calculate_ground_motion_from_event(
                    event=event,
                    site_lat=site_lat,
                    site_lon=site_lon,
                    event_rupture=None  # Will use simplified distances
                )
                
                results.append(gm_result)
                events_processed += 1
                
                if events_processed % 1000 == 0:
                    logger.info(f"   Processed {events_processed:,} events...")
                
            except Exception as e:
                logger.error(f"Error processing event {row.get('Event ID', 'unknown')}: {e}")
                continue
        
        logger.info(
            f"✅ Processed {events_processed:,} events "
            f"({events_within_distance:,} within {max_distance}km)"
        )
        
        return results


# ============================================================================
# Backward-compatible log_a functions
# ============================================================================

_global_calculator = None


def log_a(magnitude: float,
          distance: float,
          vs30: float = 760.0,
          model: str = 'CB14',
          period: float = 0.01,
          mechanism: str = 'strike-slip',
          **kwargs) -> float:
    """
    ENHANCED log_a() function - Drop-in replacement for legacy version
    
    NOTE: Uses single model, not ensemble. For ensemble, use log_a_ensemble()
    """
    global _global_calculator
    
    if _global_calculator is None:
        _global_calculator = GMPECalculator()
    
    result = _global_calculator.calculate_single_model(
        model=model,
        magnitude=magnitude,
        distance=distance,
        vs30=vs30,
        mechanism=mechanism,
        period=period,
        **kwargs
    )
    
    return result['median_log10']


def log_a_ensemble(magnitude: float,
                   distance: float,
                   vs30: float = 760.0,
                   period: float = 0.01,
                   mechanism: str = 'strike-slip',
                   **kwargs) -> float:
    """
    Calculate log10(PGA) using NSHM 2023 weighted ensemble
    
    RECOMMENDED for PSHA calculations
    """
    global _global_calculator
    
    if _global_calculator is None:
        _global_calculator = GMPECalculator()
    
    result = _global_calculator.calculate_weighted_ensemble(
        magnitude=magnitude,
        distance=distance,
        vs30=vs30,
        mechanism=mechanism,
        period=period,
        **kwargs
    )
    
    return result['median_log10']


if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("\n" + "="*70)
    print("RSQSim-GMPE Adapter - FIXED Test Suite")
    print("="*70)
    
    # Test 1: RSQSimEvent width calculation
    print("\n📋 Test 1: Width Calculation (FIX #7)")
    print("-" * 70)
    
    mock_event_data = pd.Series({
        'Event ID': 1109242,
        'Occurrence Time (s)': 64972.818,
        'Magnitude': 6.5,
        'Moment (N-m)': 9.1047e18,
        'Area (m^2)': 200000000.0,  # 200 km²
        'Number of Participating Elements': 300,
        'Average Slip (m)': 2.5,
        'Average Element Slip Rate (m/yr)': 0.0002,
        'Hypocenter Latitude': 34.070,
        'Hypocenter Longitude': -118.596,
        'Hypocenter Depth (km)': 8.0,
        'Centroid Latitude': 34.071,
        'Centroid Longitude': -118.600,
        'Centroid Depth (km)': 8.5,
        'Upper Depth (km)': 2.0,
        'Lower Depth (km)': 15.0
    })
    
    event = RSQSimEvent.from_csv_row(mock_event_data)
    
    print(f"Event M{event.magnitude}:")
    print(f"  Vertical extent: {event.get_vertical_extent():.2f} km")
    print(f"  Estimated dip: {event.estimate_dip_from_geometry():.1f}°")
    print(f"  Width (dip=90°): {event.get_rupture_width(dip=90.0):.2f} km")
    print(f"  Width (dip=60°): {event.get_rupture_width(dip=60.0):.2f} km")
    print(f"  Width (dip=45°): {event.get_rupture_width(dip=45.0):.2f} km")
    
    # Test 2: Adapter with event
    print("\n📋 Test 2: Ground Motion Calculation")
    print("-" * 70)
    
    adapter = RSQSimGMPEAdapter(use_ensemble=True)
    
    result = adapter.calculate_ground_motion_from_event(
        event=event,
        site_lat=34.05,
        site_lon=-118.25,
        event_rupture=None  # Using simplified distances
    )
    
    print(f"Ground motion at site:")
    print(f"  PGA: {result['pga_g']:.4f} g")
    print(f"  R_rup: {result['distance_rup']:.2f} km")
    print(f"  R_jb: {result['distance_jb']:.2f} km")
    print(f"  Width: {result['width']:.2f} km")
    print(f"  Dip: {result['dip']:.1f}°")
    
    print("\n" + "="*70)
    print("✅ All adapter tests completed!")
    print("="*70 + "\n")