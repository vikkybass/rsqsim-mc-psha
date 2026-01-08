"""
Model-Specific Scenario Builder for PyGMM - CORRECTED VERSION
==============================================================

FIXED: Based on actual PyGMM API (not documentation assumptions)

Key Discovery:
- PyGMM Scenario() only accepts a LIMITED set of parameters
- Model-specific flags (like is_reverse) are NOT passed to Scenario
- They are handled internally by each model

Supported Scenario Parameters:
- mag, dist_rup, dist_jb, dist_x (but dist_x only for some models)
- v_s30, dip, width, depth_tor
- depth_1_0, depth_2_5
- mechanism (format varies by model)
- region (for basin calculations)

Author: Victor Olawoyin
Date: January 2026
"""

import numpy as np
import logging
from typing import Dict, Optional

try:
    import pygmm
    PYGMM_AVAILABLE = True
except ImportError:
    PYGMM_AVAILABLE = False

logger = logging.getLogger(__name__)


class ScenarioBuilder:
    """
    Builds model-specific scenarios for PyGMM calculations
    
    CORRECTED: Only uses parameters that pygmm.Scenario() actually accepts
    """
    
    def __init__(self):
        """Initialize scenario builder"""
        if not PYGMM_AVAILABLE:
            raise ImportError("PyGMM is required")
    
    def create_scenario(self,
                       model: str,
                       magnitude: float,
                       distance: float,
                       vs30: float = 760.0,
                       mechanism: str = 'strike-slip',
                       depth_1_0: Optional[float] = None,
                       depth_2_5: Optional[float] = None,
                       dip: float = 90.0,
                       width: Optional[float] = None,
                       depth_tor: float = 0.0,
                       depth_hyp: Optional[float] = None,
                       dist_jb: Optional[float] = None,
                       dist_x: Optional[float] = None,
                       region: str = 'california',
                       **kwargs) -> 'pygmm.Scenario':
        """
        Create a scenario appropriate for the specified model
        
        CORRECTED: Only passes parameters that Scenario() accepts
        """
        # Dispatch to model-specific builder
        if model == 'ASK14':
            return self._create_ask14_scenario(
                magnitude, distance, vs30, mechanism, depth_1_0, dip, 
                width, depth_tor, dist_jb, dist_x, region, **kwargs
            )
        elif model == 'BSSA14':
            return self._create_bssa14_scenario(
                magnitude, distance, vs30, mechanism, depth_1_0, 
                dist_jb, region, **kwargs
            )
        elif model == 'CB14':
            return self._create_cb14_scenario(
                magnitude, distance, vs30, mechanism, depth_2_5, dip,
                width, depth_tor, depth_hyp, dist_jb, dist_x, region, **kwargs
            )
        elif model == 'CY14':
            return self._create_cy14_scenario(
                magnitude, distance, vs30, mechanism, depth_1_0, dip,
                width, depth_tor, dist_jb, dist_x, region, **kwargs
            )
        else:
            raise ValueError(f"Unknown model: {model}")
    
    # ========================================================================
    # ASK14 (Abrahamson, Silva & Kamai 2014)
    # ========================================================================
    
    def _create_ask14_scenario(self,
                               magnitude: float,
                               distance: float,
                               vs30: float,
                               mechanism: str,
                               depth_1_0: Optional[float],
                               dip: float,
                               width: Optional[float],
                               depth_tor: float,
                               dist_jb: Optional[float],
                               dist_x: Optional[float],
                               region: str,
                               **kwargs) -> 'pygmm.Scenario':
        """Create scenario for ASK14"""
        params = {
            'mag': magnitude,
            'dist_rup': distance,
            'v_s30': vs30,
            'dip': dip,
        }
        
        # Mechanism as STRING
        params['mechanism'] = self._mechanism_to_ask14_string(mechanism)
        
        # Distance parameters
        if dist_jb is None:
            dist_jb = self._estimate_dist_jb(distance, dip)
        params['dist_jb'] = dist_jb
        
        # NOTE: dist_x may not be accepted by Scenario for ASK14
        # Only add if explicitly provided and non-None
        if dist_x is not None:
            try:
                params['dist_x'] = dist_x
            except:
                logger.debug("dist_x not accepted for ASK14, skipping")
        
        # Rupture geometry
        if width is not None and width > 0:
            params['width'] = width
        elif magnitude > 5.0:
            params['width'] = self._estimate_width_from_magnitude(magnitude, dip)
        
        params['depth_tor'] = depth_tor
        
        # Basin depth Z1.0
        if depth_1_0 is None:
            depth_1_0 = self._calculate_depth_1_0(vs30, region)

        params['depth_1_0'] = depth_1_0
        
        # Region
        params['region'] = region
        
        logger.debug(f"ASK14 scenario: mech={params['mechanism']}, Z1.0={depth_1_0:.3f}km")
        
        return pygmm.Scenario(**params)
    
    def _mechanism_to_ask14_string(self, mechanism: str) -> str:
        """Convert mechanism to ASK14 STRING format"""
        mapping = {
            'strike-slip': 'SS',
            'strike_slip': 'SS',
            'ss': 'SS',
            'normal': 'NS',
            'n': 'NS',
            'reverse': 'RS',
            'r': 'RS',
            'thrust': 'RS',
            'unspecified': 'U',
            'unknown': 'U',
            'u': 'U'
        }
        
        mech_lower = mechanism.lower()
        return mapping.get(mech_lower, 'SS')
    
    # ========================================================================
    # BSSA14 (Boore, Stewart, Seyhan & Atkinson 2014)
    # ========================================================================
    
    def _create_bssa14_scenario(self,
                                magnitude: float,
                                distance: float,
                                vs30: float,
                                mechanism: str,
                                depth_1_0: Optional[float],
                                dist_jb: Optional[float],
                                region: str,
                                **kwargs) -> 'pygmm.Scenario':
        """Create scenario for BSSA14"""
        params = {
            'mag': magnitude,
            'dist_jb': dist_jb if dist_jb is not None else distance,
            'v_s30': vs30,
        }
        
        # BSSA14 accepts string mechanism
        params['mechanism'] = self._mechanism_to_ask14_string(mechanism)
        
        # Basin depth (optional)
        if depth_1_0 is None:
            depth_1_0 = self._calculate_depth_1_0(vs30, region)

        params['depth_1_0'] = depth_1_0
        
        # Region
        params['region'] = region if region in ['california', 'china', 'italy', 
                                                 'japan', 'new_zealand', 'taiwan', 
                                                 'turkey'] else 'global'
        
        logger.debug(f"BSSA14 scenario: mech={params['mechanism']}, Z1.0={depth_1_0:.3f}km")
        
        return pygmm.Scenario(**params)
    
    # ========================================================================
    # CB14 (Campbell & Bozorgnia 2014)
    # ========================================================================
    
    def _create_cb14_scenario(self,
                             magnitude: float,
                             distance: float,
                             vs30: float,
                             mechanism: str,
                             depth_2_5: Optional[float],
                             dip: float,
                             width: Optional[float],
                             depth_tor: float,
                             depth_hyp: Optional[float],
                             dist_jb: Optional[float],
                             dist_x: Optional[float],
                             region: str,
                             **kwargs) -> 'pygmm.Scenario':
        """
        Create scenario for CB14
        
        CORRECTED: CB14 boolean flags (is_reverse, on_hanging_wall) 
        are NOT passed to Scenario - the model calculates them internally
        """
        params = {
            'mag': magnitude,
            'dist_rup': distance,
            'v_s30': vs30,
            'dip': dip,
        }
        
        # CB14 mechanism: Pass as STRING (CB14 interprets it)
        params['mechanism'] = self._mechanism_to_cb14_string(mechanism)
        
        # Distance parameters
        if dist_jb is None:
            dist_jb = self._estimate_dist_jb(distance, dip)
        params['dist_jb'] = dist_jb
        
        # dist_x if provided (CB14 uses it to determine hanging wall)
        if dist_x is None:
            dist_x = 0.0  # Default to on-fault
            logger.debug("CB14 requires dist_x, using default=0.0")
        params['dist_x'] = dist_x
        
        # Rupture geometry
        if width is not None and width > 0:
            params['width'] = width
        elif magnitude > 5.0:
            params['width'] = self._estimate_width_from_magnitude(magnitude, dip)
        
        params['depth_tor'] = depth_tor
        
        # Depth to bottom of rupture
        if 'width' in params and params['width'] > 0:
            depth_bor = depth_tor + params['width'] * np.sin(np.radians(dip))
            params['depth_bor'] = depth_bor
        
        # Hypocenter depth
        if depth_hyp is None:
            depth_hyp = self._estimate_depth_hyp(magnitude, depth_tor, 
                                                  params.get('depth_bor', depth_tor + 10))
        params['depth_hyp'] = depth_hyp
        
        # Basin depth Z2.5 (CB14 uses Z2.5)
        if depth_2_5 is None:
            depth_2_5 = self._calculate_depth_2_5(vs30, region)  # Returns km
        else: 
            logger.debug(f"Using provided Z2.5: {depth_2_5:.3f} km")
        params['depth_2_5'] = depth_2_5
        
        logger.debug(f"CB14 Z2_5: {depth_2_5:.3f} km = {depth_2_5:.0f} m")
        
        # Region
        params['region'] = region
        
        logger.debug(f"CB14 scenario: mech={params['mechanism']}, Z2.5={depth_2_5:.3f}km")
        
        return pygmm.Scenario(**params)
    
    def _mechanism_to_cb14_string(self, mechanism: str) -> str:
        """CB14 wants STRING like ASK14"""
        return self._mechanism_to_ask14_string(mechanism)

    # ========================================================================
    # CY14 (Chiou & Youngs 2014)
    # ========================================================================
    
    def _create_cy14_scenario(self,
                             magnitude: float,
                             distance: float,
                             vs30: float,
                             mechanism: str,
                             depth_1_0: Optional[float],
                             dip: float,
                             width: Optional[float],
                             depth_tor: float,
                             dist_jb: Optional[float],
                             dist_x: Optional[float],
                             region: str,
                             **kwargs) -> 'pygmm.Scenario':
        """
        Create scenario for CY14
        
        CORRECTED: dist_x is REQUIRED by CY14 (not optional)
        """
        params = {
            'mag': magnitude,
            'dist_rup': distance,
            'v_s30': vs30,
            'dip': dip,
        }
        
        # Mechanism as STRING
        params['mechanism'] = self._mechanism_to_ask14_string(mechanism)
        
        # Distance parameters
        if dist_jb is None:
            dist_jb = self._estimate_dist_jb(distance, dip)
        params['dist_jb'] = dist_jb
        
        # CY14 REQUIRES dist_x (not optional!)
        if dist_x is None:
            # Estimate: for vertical SS faults, dist_x ≈ 0
            # For dipping faults, more complex
            dist_x = 0.0  # Default to zero (on-fault)
            logger.debug("CY14 requires dist_x, using default=0.0")
        params['dist_x'] = dist_x
        
        # Rupture geometry
        if width is not None and width > 0:
            params['width'] = width
        elif magnitude > 5.0:
            params['width'] = self._estimate_width_from_magnitude(magnitude, dip)
        
        params['depth_tor'] = depth_tor
        
        # Basin depth Z1.0
        if depth_1_0 is None:
            depth_1_0 = self._calculate_depth_1_0(vs30, region)

        params['depth_1_0'] = depth_1_0
        
        # Region
        params['region'] = region
        
        logger.debug(f"CY14 scenario: mech={params['mechanism']}, Z1.0={depth_1_0:.3f}km, dist_x={dist_x}")
        
        return pygmm.Scenario(**params)
    
    # ========================================================================
    # Helper Functions
    # ========================================================================
    
    def _estimate_dist_jb(self, dist_rup: float, dip: float) -> float:
        """Estimate Joyner-Boore distance from rupture distance"""
        if dip >= 85:
            return dist_rup
        else:
            return dist_rup * np.cos(np.radians(90 - dip))
    
    def _estimate_width_from_magnitude(self, magnitude: float, dip: float) -> float:
        """Estimate rupture width from magnitude (Wells & Coppersmith 1994)"""
        if magnitude < 5.5:
            width_km = 5.0
        elif magnitude < 6.5:
            width_km = 10.0 ** (-0.76 + 0.27 * magnitude)
        else:
            width_km = 10.0 ** (-1.01 + 0.32 * magnitude)
        
        return max(width_km, 1.0)
    
    def _estimate_depth_hyp(self, magnitude: float, depth_tor: float, 
                           depth_bor: float) -> float:
        """Estimate hypocenter depth"""
        rupture_thickness = depth_bor - depth_tor
        
        if magnitude < 6.0:
            frac = 0.3
        elif magnitude < 7.0:
            frac = 0.4
        else:
            frac = 0.5
        
        depth_hyp = depth_tor + frac * rupture_thickness
        return depth_hyp
    
    def _calculate_depth_1_0(self, vs30: float, region: str = 'california') -> float:
        """
        Calculate Z1.0 using empirical relationship
        
        Based on Chiou & Youngs (2014) equations
        """
        if vs30 < 180:
            z1_m = np.exp(6.745)
        elif vs30 > 500:
            z1_m = np.exp(5.394 - 4.48 * np.log(vs30 / 500.0))
        else:
            z1_m = np.exp(6.745 - 1.35 * np.log(vs30 / 180.0))

        z1_km = z1_m / 1000.0
        return z1_km  # returns kilometers
    
    def _calculate_depth_2_5(self, vs30: float, region: str = 'global') -> float:
        """
        Calculate Z2.5 (depth to Vs=2.5 km/s) using empirical relationship
        
        Based on Campbell & Bozorgnia (2014) Equation (33)
        
        Args:
            vs30: Time-averaged shear-wave velocity in top 30m (m/s)
            region: 'global', 'california', or 'japan'
        
        Returns:
            Z2.5 in KILOMETERS
            
        Reference:
            Campbell, K.W. and Bozorgnia, Y. (2014). NGA-West2 Ground Motion
            Model for the Average Horizontal Components of PGA, PGV, and 5%
            Damped Linear Acceleration Response Spectra. Earthquake Spectra,
            30(3), 1087-1115.
        """
        if region.lower() == 'japan':
            # Japan model: ln(Z2.5) = 5.359 - 1.102*ln(Vs30)
            ln_z2p5 = 5.359 - 1.102 * np.log(vs30)
        else:
            # Global/California model: ln(Z2.5) = 7.089 - 1.144*ln(Vs30)
            ln_z2p5 = 7.089 - 1.144 * np.log(vs30)
        
        # ✅ Apply exponential to get Z2.5 from ln(Z2.5)
        z2p5_km = np.exp(ln_z2p5)
        
        # Ensure non-negative
        z2p5_km = max(z2p5_km, 0.0)
        
        logger.debug(f"Calculated Z2.5: {z2p5_km:.3f} km for Vs30={vs30:.0f} m/s ({region})")
        
        return z2p5_km


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    
    print("\n" + "="*70)
    print("Model-Specific Scenario Builder - CORRECTED Test Suite")
    print("="*70)
    
    builder = ScenarioBuilder()
    
    test_params = {
        'magnitude': 6.5,
        'distance': 20.0,
        'vs30': 760.0,
        'mechanism': 'strike-slip',
        'dip': 90.0,
        'dist_x': 0.0  # Add explicit dist_x for CY14
    }
    
    models = ['ASK14', 'BSSA14', 'CB14', 'CY14']
    
    print("\nTest: Creating scenarios for each model")
    print("-" * 70)
    
    for model in models:
        print(f"\n{model}:")
        try:
            scenario = builder.create_scenario(model=model, **test_params)
            
            # Create model instance
            model_class = getattr(pygmm, {
                'ASK14': 'AbrahamsonSilvaKamai2014',
                'BSSA14': 'BooreStewartSeyhanAtkinson2014',
                'CB14': 'CampbellBozorgnia2014',
                'CY14': 'ChiouYoungs2014'
            }[model])
            
            model_instance = model_class(scenario)
            
            # Get PGA
            pga_g = model_instance.pga
            ln_pga = np.log(pga_g)
            ln_std = model_instance.ln_std_pga
            
            print(f"  ✅ Scenario created successfully")
            print(f"  PGA: {pga_g:.4f} g")
            print(f"  ln(PGA): {ln_pga:.4f}")
            print(f"  σ: {ln_std:.4f}")
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*70)
    print("✅ Corrected scenario builder test complete!")
    print("="*70 + "\n")