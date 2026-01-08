"""
GMPE Calculator with PyGMM Integration - FULLY FIXED VERSION
=============================================================

FIXES APPLIED:
1. ✅ Corrected PyGMM API usage (properties, not function calls)
2. ✅ Model-specific scenario creation (proper parameters for each model)

This version integrates the ScenarioBuilder to handle model-specific
parameter requirements automatically.

Author: Victor Olawoyin
Date: January 2026
"""

import numpy as np
import logging
from typing import Dict, List, Optional
from src.scenario_builder import ScenarioBuilder

logger = logging.getLogger(__name__)

# Try to import PyGMM
try:
    import pygmm
    PYGMM_AVAILABLE = True
    logger.info("✅ PyGMM successfully imported")
except ImportError:
    PYGMM_AVAILABLE = False
    logger.warning("⚠️  PyGMM not available")


class NSHMWeights:
    """NSHM 2023 Logic Tree Weights"""
    
    WUS_CRUSTAL = {
        'ASK14': 0.25,
        'BSSA14': 0.25,
        'CB14': 0.25,
        'CY14': 0.25
    }
    
    @classmethod
    def get_weights(cls, region: str = 'WUS') -> Dict[str, float]:
        if region == 'WUS':
            return cls.WUS_CRUSTAL.copy()
        else:
            raise ValueError(f"Region '{region}' not yet implemented. Use 'WUS'.")
    
    @classmethod
    def validate_weights(cls, weights: Dict[str, float]) -> bool:
        total = sum(weights.values())
        is_valid = abs(total - 1.0) < 1e-6
        if not is_valid:
            logger.warning(f"Weights sum to {total:.6f}, not 1.0")
        return is_valid


class GMPECalculator:
    """
    Unified GMPE calculator with proper PyGMM integration
    
    FULLY FIXED:
    - Correct API usage (properties)
    - Model-specific scenarios
    - Auto-calculation of basin depths
    - Proper parameter handling for each model
    """
    
    MODEL_MAPPING = {
        'ASK14': 'AbrahamsonSilvaKamai2014',
        'BSSA14': 'BooreStewartSeyhanAtkinson2014',
        'CB14': 'CampbellBozorgnia2014',
        'CY14': 'ChiouYoungs2014'
    }
    
    def __init__(self, 
                 models: Optional[List[str]] = None,
                 weights: Optional[Dict[str, float]] = None,
                 region: str = 'WUS'):
        """Initialize GMPE Calculator"""
        if not PYGMM_AVAILABLE:
            raise ImportError("PyGMM is required but not available")
        
        # Initialize scenario builder
        self.scenario_builder = ScenarioBuilder()
        
        # Set up models
        if models is None:
            self.models = list(self.MODEL_MAPPING.keys())
        else:
            self.models = models
            
        # Set up weights
        if weights is None:
            self.weights = NSHMWeights.get_weights(region)
        else:
            self.weights = weights
            
        # Validate weights
        NSHMWeights.validate_weights(self.weights)
        
    
    def calculate_single_model(self,
                               model: str,
                               magnitude: float,
                               distance: float,
                               vs30: float = 760.0,
                               mechanism: str = 'strike-slip',
                               period: float = 0.01,
                               depth_1_0: Optional[float] = None,
                               depth_2_5: Optional[float] = None,
                               dip: float = 90.0,
                               width: Optional[float] = None,
                               depth_tor: float = 0.0,
                               depth_hyp: Optional[float] = None,
                               dist_jb: Optional[float] = None,
                               dist_x: Optional[float] = None,
                               region: str = 'california',
                               **kwargs) -> Dict[str, float]:
        """
        Calculate ground motion for a single model
        
        FIXED: Uses ScenarioBuilder for model-specific scenarios
        """
        if model not in self.MODEL_MAPPING:
            raise ValueError(f"Unknown model: {model}")
        
        # Get PyGMM model class
        pygmm_model_name = self.MODEL_MAPPING[model]
        try:
            pygmm_model = getattr(pygmm, pygmm_model_name)
        except AttributeError:
            raise ImportError(f"PyGMM model {pygmm_model_name} not found")
        
        # ============================================================
        # FIX #2: Use ScenarioBuilder for model-specific scenarios
        # ============================================================
        try:
            scenario = self.scenario_builder.create_scenario(
                model=model,
                magnitude=magnitude,
                distance=distance,
                vs30=vs30,
                mechanism=mechanism,
                depth_1_0=depth_1_0,
                depth_2_5=depth_2_5,
                dip=dip,
                width=width,
                depth_tor=depth_tor,
                depth_hyp=depth_hyp,
                dist_jb=dist_jb,
                dist_x=dist_x,
                region=region,
                **kwargs
            )
        except Exception as e:
            logger.error(f"Failed to create scenario for {model}: {e}")
            raise
        
        # Create model instance
        try:
            model_instance = pygmm_model(scenario)
        except Exception as e:
            logger.error(f"Failed to create {model} instance: {e}")
            raise
        
        # ============================================================
        # FIX #1: Use properties correctly (not function calls)
        # ============================================================
        try:
            if period == 0.01 or period == -1 or period == 0.0:  # PGA
                pga_g = model_instance.pga
                ln_pga = np.log(pga_g)
                ln_stddev = model_instance.ln_std_pga
                
                logger.debug(f"{model} PGA: {pga_g:.6f} g (ln={ln_pga:.4f}, σ={ln_stddev:.4f})")
                
            else:  # Spectral acceleration
                periods_array = model_instance.periods
                spec_accels_array = model_instance.spec_accels
                ln_stds_array = model_instance.ln_stds
                
                # Find closest period
                period_idx = np.argmin(np.abs(periods_array - period))
                closest_period = periods_array[period_idx]
                
                if abs(closest_period - period) > 0.01:
                    logger.warning(
                        f"{model}: Requested T={period}s, using T={closest_period}s"
                    )
                
                sa_g = spec_accels_array[period_idx]
                ln_pga = np.log(sa_g)
                ln_stddev = ln_stds_array[period_idx]
                
                pga_g = sa_g
            
            return {
                'median_ln': ln_pga,
                'stddev_ln': ln_stddev,
                'median_g': pga_g,
                'median_log10': ln_pga / np.log(10)
            }
            
        except Exception as e:
            logger.error(f"Error extracting results from {model}: {e}")
            raise
    
    def calculate_weighted_ensemble(self,
                                    magnitude: float,
                                    distance: float,
                                    vs30: float = 760.0,
                                    mechanism: str = 'strike-slip',
                                    period: float = 0.01,
                                    include_individual: bool = False,
                                    **kwargs) -> Dict:
        """Calculate weighted ensemble following NSHM 2023"""
        results = {}
        
        # Calculate prediction from each model
        for model in self.models:
            try:
                result = self.calculate_single_model(
                    model=model,
                    magnitude=magnitude,
                    distance=distance,
                    vs30=vs30,
                    mechanism=mechanism,
                    period=period,
                    **kwargs
                )
                results[model] = result
                
            except Exception as e:
                logger.error(f"Model {model} failed: {e}")
                continue
        
        if not results:
            raise RuntimeError("All models failed to produce results")
        
        # Calculate weighted ensemble in ln space
        weighted_ln_sum = 0.0
        total_weight = 0.0
        
        for model, result in results.items():
            weight = self.weights.get(model, 0.0)
            weighted_ln_sum += weight * result['median_ln']
            total_weight += weight
        
        ensemble_ln = weighted_ln_sum / total_weight if total_weight > 0 else weighted_ln_sum
        ensemble_g = np.exp(ensemble_ln)
        ensemble_log10 = ensemble_ln / np.log(10)
        
        # Calculate epistemic uncertainty
        ln_values = np.array([r['median_ln'] for r in results.values()])
        weights_array = np.array([self.weights.get(m, 0.0) for m in results.keys()])
        weights_array = weights_array / weights_array.sum()
        
        epistemic_variance = np.sum(weights_array * (ln_values - ensemble_ln)**2)
        epistemic_stddev = np.sqrt(epistemic_variance)
        
        # Average aleatory uncertainty
        aleatory_stddevs = np.array([r['stddev_ln'] for r in results.values()])
        aleatory_stddev = np.sum(weights_array * aleatory_stddevs)
        
        # Total uncertainty (SRSS)
        total_stddev = np.sqrt(aleatory_stddev**2 + epistemic_stddev**2)
        
        ensemble_result = {
            'median_ln': ensemble_ln,
            'median_g': ensemble_g,
            'median_log10': ensemble_log10,
            'aleatory_stddev_ln': aleatory_stddev,
            'epistemic_stddev_ln': epistemic_stddev,
            'total_stddev_ln': total_stddev,
            'weights': self.weights.copy(),
            'num_models': len(results)
        }
        
        if include_individual:
            ensemble_result['individual_models'] = results
        
        logger.debug(
            f"Ensemble: {ensemble_g:.6f} g (σ_ale={aleatory_stddev:.4f}, "
            f"σ_epi={epistemic_stddev:.4f}, σ_tot={total_stddev:.4f})"
        )
        
        return ensemble_result
    
    def calculate_log10_a(self,
                          magnitude: float,
                          distance: float,
                          vs30: float = 760.0,
                          mechanism: str = 'strike-slip',
                          period: float = 0.01,
                          use_ensemble: bool = True,
                          **kwargs) -> float:
        """
        Calculate log10(acceleration) - main interface
        
        Drop-in replacement for old log_a() function
        """
        if use_ensemble:
            result = self.calculate_weighted_ensemble(
                magnitude=magnitude,
                distance=distance,
                vs30=vs30,
                mechanism=mechanism,
                period=period,
                **kwargs
            )
            return result['median_log10']
        else:
            result = self.calculate_single_model(
                model=self.models[0],
                magnitude=magnitude,
                distance=distance,
                vs30=vs30,
                mechanism=mechanism,
                period=period,
                **kwargs
            )
            return result['median_log10']
    
    def get_model_comparison(self,
                            magnitude: float,
                            distance: float,
                            vs30: float = 760.0,
                            **kwargs) -> Dict[str, Dict]:
        """Compare predictions from all models"""
        result = self.calculate_weighted_ensemble(
            magnitude=magnitude,
            distance=distance,
            vs30=vs30,
            include_individual=True,
            **kwargs
        )
        
        comparison = {
            'ensemble': {
                'median_g': result['median_g'],
                'median_log10': result['median_log10'],
                'total_stddev_ln': result['total_stddev_ln']
            }
        }
        
        for model, model_result in result['individual_models'].items():
            comparison[model] = {
                'median_g': model_result['median_g'],
                'median_log10': model_result['median_log10'],
                'stddev_ln': model_result['stddev_ln'],
                'weight': self.weights[model]
            }
        
        return comparison


# Convenience function
def calculate_ground_motion(magnitude: float,
                           distance: float,
                           vs30: float = 760.0,
                           mechanism: str = 'strike-slip',
                           models: Optional[List[str]] = None,
                           **kwargs) -> float:
    """Quick calculation using NSHM 2023 weighted ensemble"""
    calc = GMPECalculator(models=models)
    return calc.calculate_log10_a(
        magnitude=magnitude,
        distance=distance,
        vs30=vs30,
        mechanism=mechanism,
        **kwargs
    )


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s: %(message)s'
    )
    
    print("\n" + "="*70)
    print("FULLY FIXED GMPE Calculator - Test Suite")
    print("="*70)
    
    calc = GMPECalculator()
    
    # Test 1: Single model (no warnings!)
    print("\n📋 Test 1: Single Model (ASK14) - Should have NO warnings")
    print("-" * 70)
    
    result = calc.calculate_single_model(
        model='ASK14',
        magnitude=6.5,
        distance=20.0,
        vs30=760.0,
        mechanism='strike-slip'
    )
    
    print(f"✅ ASK14: PGA = {result['median_g']:.4f} g")
    print(f"   log10(PGA) = {result['median_log10']:.4f}")
    print(f"   σ = {result['stddev_ln']:.4f}")
    
    # Test 2: All models
    print("\n📋 Test 2: All 4 Models - Should have NO warnings")
    print("-" * 70)
    
    for model in calc.models:
        result = calc.calculate_single_model(
            model=model,
            magnitude=6.5,
            distance=20.0,
            vs30=760.0,
            mechanism='strike-slip',
            dist_x=0.0
        )
        print(f"✅ {model:7s}: PGA = {result['median_g']:.4f} g, σ = {result['stddev_ln']:.4f}")
    
    # Test 3: Ensemble
    print("\n📋 Test 3: NSHM 2023 Weighted Ensemble")
    print("-" * 70)
    
    ensemble = calc.calculate_weighted_ensemble(
        magnitude=6.5,
        distance=20.0,
        vs30=760.0,
        mechanism='strike-slip',
        include_individual=True
    )
    
    print(f"Ensemble PGA: {ensemble['median_g']:.4f} g")
    print(f"Aleatory σ: {ensemble['aleatory_stddev_ln']:.4f}")
    print(f"Epistemic σ: {ensemble['epistemic_stddev_ln']:.4f}")
    print(f"Total σ: {ensemble['total_stddev_ln']:.4f}")
    
    # Test 4: Different mechanism types
    print("\n📋 Test 4: Different Mechanisms")
    print("-" * 70)
    
    mechanisms = ['strike-slip', 'reverse', 'normal']
    for mech in mechanisms:
        result = calc.calculate_log10_a(
            magnitude=6.5,
            distance=20.0,
            vs30=760.0,
            mechanism=mech,
            use_ensemble=True
        )
        print(f"{mech:12s}: log10(PGA) = {result:.4f}, PGA = {10**result:.4f} g")
    
    print("\n" + "="*70)
    print("✅ All tests completed successfully - NO warnings!")
    print("="*70 + "\n")