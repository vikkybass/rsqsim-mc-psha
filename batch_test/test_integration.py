"""
Enhanced Integration Test - With Proper EventRupture Loading
=============================================================

This version properly loads EventRuptures from the catalog to demonstrate
accurate finite-rupture distance calculations.

Uses pathlib for cross-platform compatibility.
"""

import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

from gmpe_adapter import RSQSimEvent, RSQSimGMPEAdapter
from rupture_geometry import RSQSimGeometryReader, RSQSimCatalogReader
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================================
# Setup Paths
# ============================================================================
# Path to THIS script
SCRIPT_DIR = Path(__file__).resolve().parent
# Go UP from batch_test → rsqsim_mc
PROJECT_ROOT = SCRIPT_DIR.parent

# Catalog directory (for geometry and catalog files)
CATALOG_DIR = PROJECT_ROOT / "data" / "Catalog_4983"

# File paths
# Window file is in SAME directory as this script
WINDOW_FILE = SCRIPT_DIR / "los_angeles_seq_window_0001_sample.csv"

# Geometry and catalog files are in data directory
GEOMETRY_FILE = CATALOG_DIR / "geometry.flt"
ELIST_FILE = CATALOG_DIR / "catalog.eList"
PLIST_FILE = CATALOG_DIR / "catalog.pList"
DLIST_FILE = CATALOG_DIR / "catalog.dList"
TLIST_FILE = CATALOG_DIR / "catalog.tList"

# Test configuration
SITE_LAT, SITE_LON = 34.05, -118.25  # Downtown LA
N_TEST_EVENTS = 10

# Skip full catalog loading for quick test (set to False for full geometry test)
QUICK_TEST = False  # Set to False to load EventRuptures (takes several minutes)

print("="*70)
print("ENHANCED INTEGRATION TEST - PyGMM with RSQSim + Geometry")
print("="*70)
print(f"\nProject root: {PROJECT_ROOT}")
print(f"Catalog dir: {CATALOG_DIR}")
print(f"Catalog exists: {CATALOG_DIR.exists()}")

# ============================================================================
# Step 1: Load catalog
# ============================================================================
print(f"\n1. Loading catalog from {WINDOW_FILE.name}")
print(f"   Full path: {WINDOW_FILE}")
print(f"   Exists: {WINDOW_FILE.exists()}")

if not WINDOW_FILE.exists():
    print(f"   ❌ ERROR: Window file not found!")
    print(f"   Please check the path or create window files first.")
    exit(1)

catalog_df = pd.read_csv(WINDOW_FILE, nrows=N_TEST_EVENTS)
print(f"   ✅ Loaded {len(catalog_df)} events")

# ============================================================================
# Step 2: Load geometry
# ============================================================================
print(f"\n2. Loading geometry from {GEOMETRY_FILE.name}")
print(f"   Full path: {GEOMETRY_FILE}")
print(f"   Exists: {GEOMETRY_FILE.exists()}")

if not GEOMETRY_FILE.exists():
    print(f"   ❌ ERROR: Geometry file not found!")
    exit(1)

geometry = RSQSimGeometryReader(str(GEOMETRY_FILE))
print(f"   ✅ Loaded {len(geometry.patches):,} fault patches")

# ============================================================================
# Step 3: Load full catalog reader for EventRuptures
# ============================================================================
print(f"\n3. Loading full catalog reader...")

if QUICK_TEST:
    print(f"   ⚡ QUICK TEST MODE - Skipping full catalog load")
    print(f"   Will use simplified distances (much faster)")
    print(f"   Set QUICK_TEST=False to load EventRuptures (takes ~5-10 min)")
    catalog_reader = None
    has_ruptures = False
else:
    # Check if catalog files exist
    catalog_files_exist = all([
        ELIST_FILE.exists(),
        PLIST_FILE.exists(),
        DLIST_FILE.exists(),
        TLIST_FILE.exists()
    ])
    
    if catalog_files_exist:
        print(f"   All catalog files found:")
        print(f"      {ELIST_FILE.name}: {ELIST_FILE.exists()}")
        print(f"      {PLIST_FILE.name}: {PLIST_FILE.exists()}")
        print(f"      {DLIST_FILE.name}: {DLIST_FILE.exists()}")
        print(f"      {TLIST_FILE.name}: {TLIST_FILE.exists()}")
        print(f"   ⏳ Loading full catalog (this may take 5-10 minutes)...")
        
        try:
            catalog_reader = RSQSimCatalogReader(
                elist_file=str(ELIST_FILE),
                plist_file=str(PLIST_FILE),
                dlist_file=str(DLIST_FILE),
                tlist_file=str(TLIST_FILE)
            )
            print(f"   ✅ Catalog reader initialized")
            has_ruptures = True
        except Exception as e:
            print(f"   ⚠️  Could not load catalog reader: {e}")
            print(f"   Will use simplified distances")
            catalog_reader = None
            has_ruptures = False
    else:
        print(f"   ⚠️  Some catalog files missing:")
        print(f"      {ELIST_FILE.name}: {ELIST_FILE.exists()}")
        print(f"      {PLIST_FILE.name}: {PLIST_FILE.exists()}")
        print(f"      {DLIST_FILE.name}: {DLIST_FILE.exists()}")
        print(f"      {TLIST_FILE.name}: {TLIST_FILE.exists()}")
        print(f"   Will use simplified distances")
        catalog_reader = None
        has_ruptures = False

# ============================================================================
# Step 4: Initialize GMPE adapter
# ============================================================================
print(f"\n4. Initializing GMPE adapter...")
adapter = RSQSimGMPEAdapter(
    geometry_reader=geometry,
    use_ensemble=True,
    vs30=760.0,
    mechanism='strike-slip'
)
print(f"   ✅ Adapter ready with NSHM 2023 ensemble")

# ============================================================================
# Step 5: Test ground motions WITH and WITHOUT EventRuptures
# ============================================================================
print(f"\n5. Calculating ground motions for site ({SITE_LAT}, {SITE_LON})")
print("="*70)

results_comparison = []

for i, row in catalog_df.iterrows():
    event = RSQSimEvent.from_csv_row(row)
    
    print(f"\n📍 Event {event.event_id}: M{event.magnitude:.2f}")
    
    try:
        # ====================================================================
        # Method 1: WITH EventRupture (accurate finite-rupture distances)
        # ====================================================================
        event_rupture = None
        if has_ruptures:
            try:
                event_rupture = catalog_reader.get_event_rupture(
                    event.event_id, 
                    geometry
                )
                
                result_with_rupture = adapter.calculate_ground_motion_from_event(
                    event=event,
                    site_lat=SITE_LAT,
                    site_lon=SITE_LON,
                    event_rupture=event_rupture,
                    vs30=760.0
                )
                
                print(f"   WITH Geometry:")
                print(f"      PGA: {result_with_rupture['pga_g']:.4f} g")
                print(f"      R_rup: {result_with_rupture['distance_rup']:.2f} km")
                print(f"      R_jb: {result_with_rupture['distance_jb']:.2f} km")
                print(f"      Width: {result_with_rupture['width']:.2f} km")
                print(f"      Dip: {result_with_rupture['dip']:.1f}°")
                print(f"      Patches: {event_rupture.n_patches}")
                
                # Check if distances are properly different
                dist_diff = abs(result_with_rupture['distance_rup'] - 
                               result_with_rupture['distance_jb'])
                if dist_diff > 0.1:
                    print(f"      ✅ R_rup ≠ R_jb (difference: {dist_diff:.2f} km)")
                else:
                    print(f"      ⚠️  R_rup ≈ R_jb (vertical fault)")
                
            except Exception as e:
                print(f"   ⚠️  EventRupture failed: {e}")
                result_with_rupture = None
                event_rupture = None
        else:
            result_with_rupture = None
        
        # ====================================================================
        # Method 2: WITHOUT EventRupture (simplified distances)
        # ====================================================================
        result_without_rupture = adapter.calculate_ground_motion_from_event(
            event=event,
            site_lat=SITE_LAT,
            site_lon=SITE_LON,
            event_rupture=None,  # Force simplified calculation
            vs30=760.0
        )
        
        print(f"   WITHOUT Geometry (simplified):")
        print(f"      PGA: {result_without_rupture['pga_g']:.4f} g")
        print(f"      R_rup: {result_without_rupture['distance_rup']:.2f} km")
        print(f"      R_jb: {result_without_rupture['distance_jb']:.2f} km")
        
        # ====================================================================
        # Compare results
        # ====================================================================
        if result_with_rupture:
            pga_diff_pct = (
                (result_with_rupture['pga_g'] - result_without_rupture['pga_g']) / 
                result_without_rupture['pga_g'] * 100
            )
            dist_diff_km = (
                result_with_rupture['distance_rup'] - 
                result_without_rupture['distance_rup']
            )
            
            print(f"   COMPARISON:")
            print(f"      PGA difference: {pga_diff_pct:+.1f}%")
            print(f"      Distance difference: {dist_diff_km:+.2f} km")
            
            if abs(dist_diff_km) > 5.0:
                print(f"      ✅ Significant distance improvement!")
            
            results_comparison.append({
                'event_id': event.event_id,
                'magnitude': event.magnitude,
                'pga_with_geom': result_with_rupture['pga_g'],
                'pga_without_geom': result_without_rupture['pga_g'],
                'pga_diff_pct': pga_diff_pct,
                'rrup_with_geom': result_with_rupture['distance_rup'],
                'rrup_without_geom': result_without_rupture['distance_rup'],
                'dist_diff_km': dist_diff_km,
                'n_patches': event_rupture.n_patches if event_rupture else 0
            })
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
        import traceback
        traceback.print_exc()

# ============================================================================
# Step 6: Summary Statistics
# ============================================================================
if results_comparison:
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    comp_df = pd.DataFrame(results_comparison)
    
    print(f"\nTotal events compared: {len(comp_df)}")
    print(f"\nPGA Differences (with vs without geometry):")
    print(f"   Mean difference: {comp_df['pga_diff_pct'].mean():+.1f}%")
    print(f"   Median difference: {comp_df['pga_diff_pct'].median():+.1f}%")
    print(f"   Range: {comp_df['pga_diff_pct'].min():+.1f}% to {comp_df['pga_diff_pct'].max():+.1f}%")
    
    print(f"\nDistance Differences (R_rup):")
    print(f"   Mean difference: {comp_df['dist_diff_km'].mean():+.2f} km")
    print(f"   Max difference: {comp_df['dist_diff_km'].abs().max():.2f} km")
    
    # Find most significant improvement
    max_improvement_idx = comp_df['dist_diff_km'].abs().idxmax()
    best = comp_df.loc[max_improvement_idx]
    print(f"\nLargest distance improvement:")
    print(f"   Event {best['event_id']}: M{best['magnitude']:.2f}")
    print(f"   Distance changed by {best['dist_diff_km']:+.2f} km")
    print(f"   PGA changed by {best['pga_diff_pct']:+.1f}%")
    
    print(f"\nAverage fault patches per event: {comp_df['n_patches'].mean():.0f}")

print("\n" + "="*70)
print("✅ ENHANCED INTEGRATION TEST COMPLETE!")
print("="*70)

# ============================================================================
# Recommendations
# ============================================================================
print("\n📋 RECOMMENDATIONS:")
print("-"*70)

if has_ruptures and results_comparison:
    comp_df = pd.DataFrame(results_comparison)
    avg_diff = abs(comp_df['dist_diff_km'].mean())
    
    if avg_diff > 10.0:
        print("✅ STRONG: Use EventRuptures for best accuracy")
        print(f"   Average distance improvement: {avg_diff:.1f} km")
        print("   This significantly affects ground motion predictions!")
    elif avg_diff > 2.0:
        print("✅ MODERATE: EventRuptures provide noticeable improvement")
        print(f"   Average distance improvement: {avg_diff:.1f} km")
    else:
        print("⚠️  MINIMAL: EventRuptures show small improvements")
        print(f"   Average distance improvement: {avg_diff:.1f} km")
        print("   Simplified distances may be sufficient for this site")
    
    print(f"\n💡 For production runs:")
    print(f"   - Pre-load EventRuptures into rupture_cache")
    print(f"   - Enable in gm_simulator_main.py (already implemented)")
    print(f"   - Expect significant speedup from caching")
else:
    print("⚠️  Could not compare with/without geometry")
    print("   Make sure catalog files (.eList, .pList, etc.) are available")
    print("   For production: use rupture_cache for efficiency")

print("-"*70)
print(f"\nFiles used:")
print(f"   Window: {WINDOW_FILE}")
print(f"   Geometry: {GEOMETRY_FILE}")
if has_ruptures:
    print(f"   eList: {ELIST_FILE}")
    print(f"   pList: {PLIST_FILE}")
    print(f"   dList: {DLIST_FILE}")
    print(f"   tList: {TLIST_FILE}")