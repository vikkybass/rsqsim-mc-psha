"""
Unified GMPE Interface - REFACTORED AS SIMPLE ROUTER
=====================================================

This module provides a unified interface that routes GMPE calculations to
the appropriate implementation:

1. **NGA-West2 models** (ASK14, BSSA14, CB14, CY14) → PyGMM via gmpe_calculator
2. **Legacy models** (Frankel1996, etc.) → Legacy implementation from new_Ran
3. **Ensemble mode** → NSHM 2023 weighted ensemble (NGA-West2 only)

The router is thin and delegates all actual work to specialized modules.

Author: Victor Olawoyin
Date: January 2026
"""

import numpy as np
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

# ============================================================================
# Import implementations
# ============================================================================

# Try to import PyGMM calculator (for NGA-West2 models)
try:
    from src.gmpe_calculator import GMPECalculator
    PYGMM_AVAILABLE = True
except ImportError:
    try:
        # Try without src prefix
        from src.gmpe_calculator import GMPECalculator
        PYGMM_AVAILABLE = True
    except ImportError:
        PYGMM_AVAILABLE = False
        logger.warning("⚠️  PyGMM not available - NGA-West2 models will use legacy implementation")

# Try to import legacy GMPE functions (for non-NGA models)
try:
    from src.new_Ran import log_a as legacy_log_a
    LEGACY_AVAILABLE = True
except ImportError:
    try:
        # Try without src prefix
        from src.new_Ran import log_a as legacy_log_a
        LEGACY_AVAILABLE = True
    except ImportError:
        LEGACY_AVAILABLE = False
        logger.warning("⚠️  Legacy GMPE functions not available")


# ============================================================================
# Model definitions
# ============================================================================

NGA_WEST2_MODELS = ['ASK14', 'BSSA14', 'CB14', 'CY14']

LEGACY_MODELS = [
    'Frankel1996',
    'Atkinson1995', 
    'Toro1997',
    'BooreAtkinson1987',
    'ToroMcGuire1987',
    # Add other legacy models as needed
]


# ============================================================================
# Simple Router Class
# ============================================================================

class UnifiedGMPERouter:
    """
    Simple router that delegates GMPE calculations
    
    This class does NOT implement any GMPE logic itself.
    It only routes to the appropriate implementation.
    """
    
    def __init__(self):
        """Initialize router with available implementations"""
        self.pygmm_calculator = None
        
        # Initialize PyGMM calculator if available
        if PYGMM_AVAILABLE:
            try:
                self.pygmm_calculator = GMPECalculator()
                logger.info("✅ PyGMM calculator available for NGA-West2 models")
            except Exception as e:
                logger.warning(f"⚠️  PyGMM initialization failed: {e}")
                self.pygmm_calculator = None
        
        # Check legacy availability
        if not LEGACY_AVAILABLE:
            logger.warning("⚠️  Legacy GMPE functions not available")
    
    def calculate(self,
                  magnitude: float,
                  distance: float,
                  vs30: float = 760.0,
                  model: str = 'CB14',
                  period: float = 0.01,
                  use_ensemble: bool = False,
                  mechanism: str = 'strike-slip',
                  **kwargs) -> float:
        """
        Route GMPE calculation to appropriate implementation
        
        Args:
            magnitude: Moment magnitude
            distance: Rupture distance (km)
            vs30: Site shear wave velocity (m/s)
            model: GMPE model name
            period: Spectral period (s, 0.01 for PGA)
            use_ensemble: Use NSHM 2023 ensemble (NGA-West2 only)
            mechanism: Fault mechanism (for PyGMM)
            **kwargs: Additional parameters (width, ztor, dist_jb, etc.)
            
        Returns:
            log10(acceleration in g)
        """
        # ====================================================================
        # Route 1: NGA-West2 Ensemble Mode (RECOMMENDED for PSHA)
        # ====================================================================
        if use_ensemble:
            if self.pygmm_calculator is not None:
                logger.debug("Routing to PyGMM ensemble (NSHM 2023)")
                result = self.pygmm_calculator.calculate_weighted_ensemble(
                    magnitude=magnitude,
                    distance=distance,
                    vs30=vs30,
                    mechanism=mechanism,
                    period=period,
                    **kwargs
                )
                return result['median_log10']
            else:
                logger.warning("⚠️  PyGMM not available for ensemble mode")
                logger.warning("    Falling back to single model: CB14")
                model = 'CB14'  # Fall back to single model
        
        # ====================================================================
        # Route 2: NGA-West2 Single Model
        # ====================================================================
        if model in NGA_WEST2_MODELS:
            # Prefer PyGMM
            if self.pygmm_calculator is not None:
                logger.debug(f"Routing {model} to PyGMM")
                result = self.pygmm_calculator.calculate_single_model(
                    model=model,
                    magnitude=magnitude,
                    distance=distance,
                    vs30=vs30,
                    mechanism=mechanism,
                    period=period,
                    **kwargs
                )
                return result['median_log10']
            
            # Fall back to legacy if PyGMM unavailable
            elif LEGACY_AVAILABLE:
                logger.warning(f"⚠️  PyGMM unavailable, using legacy for {model}")
                return legacy_log_a(
                    magnitude=magnitude,
                    distance=distance,
                    vs30=vs30,
                    model=model,
                    period=period
                )
            
            else:
                raise RuntimeError(
                    f"No implementation available for {model}. "
                    f"Install PyGMM or enable legacy implementation."
                )
        
        # ====================================================================
        # Route 3: Legacy Models
        # ====================================================================
        elif model in LEGACY_MODELS:
            if not LEGACY_AVAILABLE:
                raise RuntimeError(
                    f"Legacy implementation not available for {model}. "
                    f"Check that new_Ran.py is accessible."
                )
            
            logger.debug(f"Routing {model} to legacy implementation")
            return legacy_log_a(
                magnitude=magnitude,
                distance=distance,
                vs30=vs30,
                model=model,
                period=period
            )
        
        # ====================================================================
        # Unknown model
        # ====================================================================
        else:
            available_models = NGA_WEST2_MODELS + LEGACY_MODELS
            raise ValueError(
                f"Unknown model: '{model}'. "
                f"Available models: {available_models}"
            )
    
    def get_available_models(self) -> Dict[str, List[str]]:
        """
        Get available models by implementation type
        
        Returns:
            Dictionary with keys:
            - 'nga_west2_pygmm': NGA-West2 via PyGMM (recommended)
            - 'nga_west2_legacy': NGA-West2 via legacy (fallback)
            - 'legacy_only': Legacy-only models
        """
        available = {
            'nga_west2_pygmm': [],
            'nga_west2_legacy': [],
            'legacy_only': []
        }
        
        # NGA-West2 models
        if self.pygmm_calculator is not None:
            available['nga_west2_pygmm'] = NGA_WEST2_MODELS.copy()
        elif LEGACY_AVAILABLE:
            available['nga_west2_legacy'] = NGA_WEST2_MODELS.copy()
        
        # Legacy-only models
        if LEGACY_AVAILABLE:
            available['legacy_only'] = LEGACY_MODELS.copy()
        
        return available
    
    def supports_ensemble(self, model: str = None) -> bool:
        """
        Check if ensemble mode is available
        
        Args:
            model: Specific model to check (if None, checks general availability)
            
        Returns:
            True if ensemble mode is available
        """
        if model is not None:
            # Check specific model
            return model in NGA_WEST2_MODELS and self.pygmm_calculator is not None
        else:
            # Check general availability
            return self.pygmm_calculator is not None


# ============================================================================
# Global router instance (singleton pattern)
# ============================================================================

_global_router = None


def _get_router() -> UnifiedGMPERouter:
    """Get or create global router instance"""
    global _global_router
    if _global_router is None:
        _global_router = UnifiedGMPERouter()
    return _global_router


# ============================================================================
# Main interface function (backward compatible)
# ============================================================================

def log_a(magnitude: float,
          distance: float,
          vs30: float = 760.0,
          model: str = 'CB14',
          period: float = 0.01,
          use_ensemble: bool = False,
          mechanism: str = 'strike-slip',
          **kwargs) -> float:
    """
    Calculate log10(PGA) using unified GMPE interface
    
    This is the main entry point that routes to appropriate implementation:
    - NGA-West2 models → PyGMM (preferred) or legacy (fallback)
    - Legacy models → Legacy implementation
    - Ensemble mode → NSHM 2023 weighted ensemble
    
    Args:
        magnitude: Moment magnitude
        distance: Rupture distance (km)
        vs30: Site shear wave velocity (m/s)
        model: GMPE model name (default: 'CB14')
        period: Spectral period (s, default: 0.01 for PGA)
        use_ensemble: Use NSHM 2023 ensemble (NGA-West2 only, recommended)
        mechanism: Fault mechanism ('strike-slip', 'reverse', 'normal')
        **kwargs: Additional parameters:
            - width: Rupture width (km)
            - depth_tor: Depth to top of rupture (km)
            - depth_hyp: Hypocenter depth (km)
            - dist_jb: Joyner-Boore distance (km)
            - dist_x: Site coordinate (km)
            - dip: Fault dip angle (degrees)
            - depth_1_0: Z1.0 basin depth (km)
            - depth_2_5: Z2.5 basin depth (km)
            - region: Basin region ('california', 'japan', etc.)
        
    Returns:
        log10(acceleration in g)
        
    Examples:
        >>> # Single NGA-West2 model (uses PyGMM)
        >>> pga = log_a(6.5, 20.0, vs30=760.0, model='CB14')
        
        >>> # NSHM 2023 ensemble (RECOMMENDED for PSHA)
        >>> pga = log_a(6.5, 20.0, vs30=760.0, use_ensemble=True)
        
        >>> # With rupture parameters
        >>> pga = log_a(6.5, 20.0, vs30=760.0, model='ASK14',
        ...             width=15.0, depth_tor=2.0, dip=90.0)
        
        >>> # Legacy model
        >>> pga = log_a(6.0, 50.0, vs30=760.0, model='Frankel1996')
        
    Notes:
        - For PSHA: use_ensemble=True is recommended (NSHM 2023)
        - For deterministic: use single model
        - PyGMM handles all parameter calculations automatically
    """
    router = _get_router()
    
    return router.calculate(
        magnitude=magnitude,
        distance=distance,
        vs30=vs30,
        model=model,
        period=period,
        use_ensemble=use_ensemble,
        mechanism=mechanism,
        **kwargs
    )


# ============================================================================
# Convenience functions
# ============================================================================

def get_available_models() -> Dict[str, List[str]]:
    """
    Get list of available GMPE models
    
    Returns:
        Dictionary with available models by implementation type
    """
    router = _get_router()
    return router.get_available_models()


def supports_ensemble() -> bool:
    """
    Check if ensemble mode is available
    
    Returns:
        True if NSHM 2023 ensemble mode is available
    """
    router = _get_router()
    return router.supports_ensemble()


def print_available_models():
    """Print available models (useful for debugging)"""
    available = get_available_models()
    
    print("\n" + "="*70)
    print("AVAILABLE GMPE MODELS")
    print("="*70)
    
    if available['nga_west2_pygmm']:
        print("\n✅ NGA-West2 (PyGMM - Recommended):")
        for model in available['nga_west2_pygmm']:
            print(f"   - {model}")
        print("   Supports: Ensemble mode (NSHM 2023)")
    
    if available['nga_west2_legacy']:
        print("\n⚠️  NGA-West2 (Legacy Fallback):")
        for model in available['nga_west2_legacy']:
            print(f"   - {model}")
    
    if available['legacy_only']:
        print("\n📚 Legacy Models:")
        for model in available['legacy_only']:
            print(f"   - {model}")
    
    if not any(available.values()):
        print("\n❌ No GMPE models available!")
        print("   Install PyGMM or check legacy implementation")
    
    print("="*70 + "\n")


# ============================================================================
# Testing
# ============================================================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    
    print("\n" + "="*70)
    print("UNIFIED GMPE ROUTER - Test Suite")
    print("="*70)
    
    # Show available models
    print_available_models()
    
    # Test parameters
    test_params = {
        'magnitude': 6.5,
        'distance': 20.0,
        'vs30': 760.0
    }
    
    print("\nTest scenario: M6.5 at 20 km, Vs30=760 m/s")
    print("-" * 70)
    
    # Test 1: Single NGA-West2 model
    if PYGMM_AVAILABLE:
        print("\n📋 Test 1: Single Model (CB14)")
        pga = log_a(**test_params, model='CB14')
        print(f"   CB14: log10(PGA) = {pga:.4f}, PGA = {10**pga:.4f} g")
    
    # Test 2: Ensemble mode
    if supports_ensemble():
        print("\n📋 Test 2: NSHM 2023 Ensemble")
        pga_ensemble = log_a(**test_params, use_ensemble=True)
        print(f"   Ensemble: log10(PGA) = {pga_ensemble:.4f}, PGA = {10**pga_ensemble:.4f} g")
    
    # Test 3: Legacy model
    if LEGACY_AVAILABLE:
        print("\n📋 Test 3: Legacy Model (Frankel1996)")
        try:
            pga_legacy = log_a(**test_params, model='Frankel1996')
            print(f"   Frankel1996: log10(PGA) = {pga_legacy:.4f}, PGA = {10**pga_legacy:.4f} g")
        except Exception as e:
            print(f"   ⚠️  Legacy test failed: {e}")
    
    print("\n" + "="*70)
    print("✅ Router tests complete!")
    print("="*70 + "\n")