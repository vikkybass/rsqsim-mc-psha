import logging
import numpy as np
import os
import matplotlib.pyplot as plt
from pathlib import Path
import pandas as pd
import re

logger = logging.getLogger(__name__)

def generate_multi_window_site_comparisons(region_name, mode, output_base_dir):
    """
    UPDATED: Generate improved multi-window comparison plots
    """
    
    logger.info(f"🔄 Generating IMPROVED multi-window comparisons for {region_name} ({mode} mode)")
    
    # Collect all window results
    output_dir = Path(output_base_dir)
    all_window_data = {}
    
    # Find all window result files
    summary_files = list(output_dir.glob(f"summary_*_window_*.csv"))
    
    if not summary_files:
        summary_files = list(output_dir.glob(f"summary_*.csv"))
    
    logger.info(f"📊 Found {len(summary_files)} window summary files")
    
    if len(summary_files) < 2:
        logger.warning("⚠️  Need at least 2 windows for comparison plots")
        return
    
    # Load data from each window
    for summary_file in summary_files:
        filename = summary_file.stem
        if "window_" in filename:
            window_id = filename.split("window_")[-1]
        else:
            # Extract window ID from filename
            window_id = filename.split("_")[-1]
        
        try:
            df = pd.read_csv(summary_file)
            
            # Extract time range
            time_range = "Unknown time range"
            base_name = os.path.splitext(summary_file.name)[0]
            window_base = base_name.replace('summary_', '').replace(f'_{mode}', '')
            metadata_file = summary_file.parent / f"window_metadata_{window_base}.txt"
            
            if metadata_file.exists():
                try:
                    with open(metadata_file, 'r') as f:
                        for line in f:
                            if line.startswith('Time range:'):
                                time_range = line.split('Time range:')[1].strip()
                                break
                except Exception:
                    pass
            
            all_window_data[window_id] = {
                'data': df,
                'time_range': time_range,
                'summary_file': summary_file
            }
            
            logger.info(f"✅ Loaded window {window_id}: {len(df)} sites")
            
        except Exception as e:
            logger.warning(f"⚠️  Could not load window {window_id}: {e}")
            continue
    
    if len(all_window_data) < 2:
        logger.warning("⚠️  Insufficient valid window data for comparisons")
        return
    
    # Identify top hazard sites
    top_sites = identify_consistent_top_sites(all_window_data, n_sites=10)
    logger.info(f"🎯 Identified {len(top_sites)} consistent top hazard sites")
    
    # Create comparison plots directory
    comparison_dir = output_dir / "visualizations" / "multi_window_comparisons"
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate IMPROVED comparison plots for each top site
    for site_lat, site_lon in top_sites:
        create_site_multi_window_plot(site_lat, site_lon, all_window_data, comparison_dir, region_name)
    
    # Generate summary comparison
    create_window_summary_comparison(all_window_data, comparison_dir, region_name, mode)
    
    logger.info(f"✅ IMPROVED multi-window comparisons saved to: {comparison_dir}")


def identify_consistent_top_sites(all_window_data, n_sites=10):
    """
    Identify sites that consistently appear in top hazard rankings across windows
    """
    
    # Collect top sites from each window
    site_rankings = {}
    
    for window_id, window_info in all_window_data.items():
        df = window_info['data']
        
        # Rank sites by 2% annual hazard
        if 'design_2pct_annual_g' in df.columns:
            df_sorted = df.sort_values('design_2pct_annual_g', ascending=False)
            
            # Get top sites for this window
            top_sites = [(row['site_lat'], row['site_lon']) 
                        for _, row in df_sorted.head(n_sites * 2).iterrows()]  # Get more than needed
            
            # Score sites based on ranking
            for rank, (lat, lon) in enumerate(top_sites):
                site_key = (round(lat, 4), round(lon, 4))  # Round for consistency
                
                if site_key not in site_rankings:
                    site_rankings[site_key] = []
                
                # Higher score for better ranking
                score = max(0, n_sites * 2 - rank)
                site_rankings[site_key].append(score)
    
    # Calculate average ranking score for each site
    site_scores = {}
    for site, scores in site_rankings.items():
        if len(scores) >= len(all_window_data) * 0.7:  # Site must appear in most windows
            site_scores[site] = np.mean(scores)
    
    # Return top sites by average score
    sorted_sites = sorted(site_scores.items(), key=lambda x: x[1], reverse=True)
    return [site for site, score in sorted_sites[:n_sites]]

def create_site_multi_window_plot(site_lat, site_lon, all_window_data, output_dir, region_name):
    """
    IMPROVED: Adaptive multi-window plotting based on number of windows
    
    - ≤ 20 windows: Individual curves with legend
    - 21-50 windows: Sample curves + statistical summary
    - 50+ windows: Statistical summary only
    """
    
    n_windows = len(all_window_data)
    logger.info(f"Creating multi-window plot for {n_windows} windows")
    
    if n_windows <= 20:
        return create_detailed_multi_window_plot(site_lat, site_lon, all_window_data, output_dir, region_name)
    elif n_windows <= 50:
        return create_hybrid_multi_window_plot(site_lat, site_lon, all_window_data, output_dir, region_name)
    else:
        return create_statistical_multi_window_plot(site_lat, site_lon, all_window_data, output_dir, region_name)

def create_hazard_curve_from_summary(summary_file, target_lat, target_lon):
    """
    Create hazard curve data from summary CSV file - SIMPLE & RELIABLE
    
    Args:
        summary_file: Path to summary CSV file  
        target_lat, target_lon: Site coordinates to find
        
    Returns:
        dict: {'ground_motions': [...], 'annual_rates': [...]} or None if not found
    """
    
    try:
        import pandas as pd
        import numpy as np
        
        if not summary_file.exists():
            logger.debug(f"Summary file not found: {summary_file}")
            return None
            
        df = pd.read_csv(summary_file)
        
        # Find the target site
        site_row = df[
            (np.abs(df['site_lat'] - target_lat) < 0.001) & 
            (np.abs(df['site_lon'] - target_lon) < 0.001)
        ]
        
        if site_row.empty:
            logger.debug(f"Target site ({target_lat:.4f}, {target_lon:.4f}) not found in summary")
            return None
        
        row = site_row.iloc[0]
        
        # Extract design values (these are the most important points)
        design_2pct = row.get('design_2pct_annual_g', 0)
        design_10pct = row.get('design_10pct_annual_g', 0)
        
        if design_2pct <= 0 or design_10pct <= 0:
            logger.debug(f"Invalid design values: 2%={design_2pct}, 10%={design_10pct}")
            return None
        
        # Create a simple but accurate 5-point hazard curve
        ground_motions = [
            design_10pct * 0.5,    # Lower point
            design_10pct,          # 10% annual point
            (design_10pct + design_2pct) / 2,  # Mid point
            design_2pct,           # 2% annual point  
            design_2pct * 1.5      # Upper point
        ]
        
        annual_rates = [
            0.2,    # Higher rate
            0.1,    # 10% annual
            0.05,   # 5% annual
            0.02,   # 2% annual
            0.01    # 1% annual
        ]
        
        logger.debug(f"✅ Created 5-point hazard curve from CSV: {design_10pct:.3f}g to {design_2pct:.3f}g")
        return {
            'ground_motions': ground_motions,
            'annual_rates': annual_rates
        }
        
    except Exception as e:
        logger.debug(f"Error creating hazard curve from summary: {e}")
        return None

def extract_hazard_curve_from_results(results_file, target_lat, target_lon):
    """
    FIXED: Extract real hazard curve data from the results file for a specific site
    
    Args:
        results_file: Path to hazard results file
        target_lat, target_lon: Site coordinates to find
        
    Returns:
        dict: {'ground_motions': [...], 'annual_rates': [...]} or None if not found
    """
    
    try:
        if not results_file.exists():
            logger.debug(f"Results file not found: {results_file}")
            return None
        
        with open(results_file, 'r') as f:
            content = f.read()
        
        # Split into lines for processing
        lines = content.split('\n')
        
        # Find the target site
        site_found = False
        site_block_start = -1
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Look for site location line
            if line.startswith("Site Location:"):
                try:
                    # Parse coordinates: "Site Location: 34.0500N, -118.2500E"
                    coord_part = line.split("Site Location:")[1].strip()
                    
                    # Handle format like "34.0500N, -118.2500E"
                    import re
                    coords = re.findall(r"(-?\d+\.?\d*)([NS]),?\s+(-?\d+\.?\d*)([EW])", coord_part)
                    
                    if coords:
                        lat_val, lat_dir, lon_val, lon_dir = coords[0]
                        site_lat = float(lat_val) * (1 if lat_dir == 'N' else -1)
                        site_lon = float(lon_val) * (-1 if lon_dir == 'W' else 1)
                        
                        # Check if this matches our target site (using more lenient tolerance)
                        if (abs(site_lat - target_lat) < 0.01 and 
                            abs(site_lon - target_lon) < 0.01):
                            site_found = True
                            site_block_start = i
                            logger.debug(f"✅ Found target site at line {i+1}: ({site_lat}, {site_lon})")
                            break
                        else:
                            logger.debug(f"Site coordinates don't match: ({site_lat}, {site_lon}) vs ({target_lat}, {target_lon})")
                    else:
                        logger.debug(f"Could not parse coordinates from: {coord_part}")
                        
                except Exception as e:
                    logger.debug(f"Error parsing site coordinates: {e}")
        
        if not site_found:
            logger.debug(f"Target site ({target_lat:.4f}, {target_lon:.4f}) not found in {results_file}")
            return None
        
        # Extract hazard curve data from the site block
        ground_motions = None
        annual_rates = None
        
        # Process lines after the site location
        for i in range(site_block_start + 1, len(lines)):
            line = lines[i].strip()
            
            # Stop if we hit another site or end marker
            if (line.startswith("Site Location:") or 
                line.startswith("================") or
                line.startswith("RSQSim Ground Motion")):
                break
            
            # Extract ground motion values
            if line.startswith("Ground_motion_values"):
                try:
                    # Split the line and skip the first element (header)
                    parts = line.split()[1:]  # Skip "Ground_motion_values"
                    ground_motions = []
                    
                    for part in parts:
                        part = part.strip()
                        if part:  # Skip empty strings
                            try:
                                # Handle scientific notation
                                value = float(part)
                                ground_motions.append(value)
                            except ValueError:
                                logger.debug(f"Could not parse ground motion value: '{part}'")
                    
                    logger.debug(f"✅ Extracted {len(ground_motions)} ground motion values")
                    
                except Exception as e:
                    logger.debug(f"Error parsing ground motion values: {e}")
            
            # Extract annual rates
            elif line.startswith("Annualized_rates"):
                try:
                    # Split the line and skip the first element (header)
                    parts = line.split()[1:]  # Skip "Annualized_rates"
                    annual_rates = []
                    
                    for part in parts:
                        part = part.strip()
                        if part:  # Skip empty strings
                            try:
                                # Handle scientific notation
                                value = float(part)
                                annual_rates.append(value)
                            except ValueError:
                                logger.debug(f"Could not parse annual rate value: '{part}'")
                    
                    logger.debug(f"✅ Extracted {len(annual_rates)} annual rate values")
                    
                except Exception as e:
                    logger.debug(f"Error parsing annual rates: {e}")
        
        # Validate the extracted data
        if ground_motions is not None and annual_rates is not None:
            if len(ground_motions) == len(annual_rates) and len(ground_motions) > 0:
                logger.debug(f"✅ Successfully extracted hazard curve with {len(ground_motions)} points")
                
                # Filter out any invalid values (negative, zero, or infinite)
                valid_points = []
                for gm, rate in zip(ground_motions, annual_rates):
                    if gm > 0 and rate > 0 and np.isfinite(gm) and np.isfinite(rate):
                        valid_points.append((gm, rate))
                
                if valid_points:
                    gm_clean, rates_clean = zip(*valid_points)
                    logger.debug(f"✅ After filtering: {len(gm_clean)} valid points")
                    return {
                        'ground_motions': list(gm_clean),
                        'annual_rates': list(rates_clean)
                    }
                else:
                    logger.debug("❌ No valid data points after filtering")
                    return None
            else:
                logger.debug(f"❌ Data length mismatch: {len(ground_motions) if ground_motions else 0} ground motions vs {len(annual_rates) if annual_rates else 0} rates")
                return None
        else:
            logger.debug(f"❌ Missing data - Ground motions: {ground_motions is not None}, Rates: {annual_rates is not None}")
            return None
        
    except Exception as e:
        logger.error(f"Error extracting hazard curve data: {e}")
        import traceback
        logger.debug(f"Full traceback: {traceback.format_exc()}")
        return None

def create_window_summary_comparison(all_window_data, output_dir, region_name, mode):
    """
    Create summary comparison showing statistics across all windows
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'{region_name.title()} - Multi-Window Analysis Summary ({mode.title()} Mode)', 
                 fontsize=16, fontweight='bold')
    
    # Collect statistics from all windows
    window_stats = {}
    
    for window_id, window_info in all_window_data.items():
        df = window_info['data']
        time_range = window_info['time_range']
        
        # FIXED: Use safer variable names without numbers at start
        stats = {
            'window_id': window_id,
            'time_range': time_range,
            'num_sites': len(df),
            'avg_design_annual_2pct': df['design_2pct_annual_g'].mean() if 'design_2pct_annual_g' in df.columns else 0,
            'max_design_annual_2pct': df['design_2pct_annual_g'].max() if 'design_2pct_annual_g' in df.columns else 0,
            'avg_events': df['num_significant_events'].mean() if 'num_significant_events' in df.columns else 0,
            'total_events': df['num_significant_events'].sum() if 'num_significant_events' in df.columns else 0
        }
        
        window_stats[window_id] = stats
    
    # Sort windows by ID for consistent plotting
    sorted_windows = sorted(window_stats.items())
    window_ids = [w[0] for w in sorted_windows]
    
    # Plot 1: Average Design Values Across Windows
    avg_annual_2pct = [w[1]['avg_design_annual_2pct'] for w in sorted_windows]
    max_annual_2pct = [w[1]['max_design_annual_2pct'] for w in sorted_windows]
    
    x_pos = np.arange(len(window_ids))
    
    axes[0, 0].bar(x_pos, avg_annual_2pct, alpha=0.7, color='skyblue', label='Average')
    axes[0, 0].bar(x_pos, max_annual_2pct, alpha=0.7, color='red', label='Maximum')
    axes[0, 0].set_title('2% Annual Design Values by Window', fontweight='bold')
    axes[0, 0].set_xlabel('Window ID')
    axes[0, 0].set_ylabel('Ground Motion (g)')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(window_ids, rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Number of Sites and Events
    num_sites = [w[1]['num_sites'] for w in sorted_windows]
    total_events = [w[1]['total_events'] for w in sorted_windows]
    
    ax2_twin = axes[0, 1].twinx()
    
    bars1 = axes[0, 1].bar(x_pos - 0.2, num_sites, 0.4, alpha=0.7, color='green', label='Sites')
    bars2 = ax2_twin.bar(x_pos + 0.2, total_events, 0.4, alpha=0.7, color='orange', label='Total Events')
    
    axes[0, 1].set_title('Sites and Events by Window', fontweight='bold')
    axes[0, 1].set_xlabel('Window ID')
    axes[0, 1].set_ylabel('Number of Sites', color='green')
    ax2_twin.set_ylabel('Total Events', color='orange')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(window_ids, rotation=45)
    
    # Combine legends
    lines1, labels1 = axes[0, 1].get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    axes[0, 1].legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Plot 3: Design Value Distribution
    all_design_values = []
    window_labels = []
    
    for window_id, window_info in all_window_data.items():
        df = window_info['data']
        if 'design_2pct_annual_g' in df.columns:
            values = df['design_2pct_annual_g'].dropna()
            all_design_values.append(values)
            window_labels.append(f"W{window_id}")
    
    if all_design_values:
        axes[1, 0].boxplot(all_design_values, labels=window_labels)
        axes[1, 0].set_title('Design Value Distribution by Window', fontweight='bold')
        axes[1, 0].set_ylabel('2% Annual Design Value (g)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Window Statistics Summary Table
    axes[1, 1].axis('off')
    
    # Create summary table
    table_data = []
    headers = ['Window', 'Time Range', 'Sites', 'Avg Design (g)', 'Max Design (g)']
    
    for window_id, stats in sorted_windows:
        time_range = stats['time_range'] or f"Window {window_id}"
        if len(time_range) > 20:  # Truncate long ranges
            time_range = time_range[:17] + "..."
            
        row = [
            f"W{window_id}",
            time_range,
            f"{stats['num_sites']}",
            f"{stats['avg_design_annual_2pct']:.3f}",
            f"{stats['max_design_annual_2pct']:.3f}"
        ]
        table_data.append(row)
    
    # Create table
    table = axes[1, 1].table(cellText=table_data, 
                           colLabels=headers,
                           cellLoc='center',
                           loc='center',
                           bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    axes[1, 1].set_title('Window Summary Statistics', fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save comparison plot
    comparison_file = output_dir / f"{region_name}_multi_window_summary.png"
    plt.savefig(comparison_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"✅ Window summary comparison saved to: {comparison_file}")

def process_multi_window_analysis(region_name, mode, output_base_dir, processed_windows=None):
    """
    UPDATED: Main function to process multi-window analysis with improved plotting
    """
    
    logger.info(f"🔄 Starting IMPROVED multi-window analysis for {region_name}")
    
    try:
        # Generate improved multi-window site comparisons
        generate_multi_window_site_comparisons(region_name, mode, output_base_dir)
        
        logger.info(f"✅ IMPROVED multi-window analysis complete for {region_name}")
        
        # Log what was created
        comparison_dir = Path(output_base_dir) / "visualizations" / "multi_window_comparisons"
        if comparison_dir.exists():
            plot_files = list(comparison_dir.glob("*.png"))
            logger.info(f"📊 Created {len(plot_files)} comparison plots:")
            for plot_file in plot_files:
                logger.info(f"  - {plot_file.name}")
        
    except Exception as e:
        logger.error(f"❌ Error in IMPROVED multi-window analysis: {e}")
        import traceback
        logger.debug(f"Full traceback: {traceback.format_exc()}")

def extract_window_time_range_simple(window_id, mode):
    """
    Simple fallback function to generate a basic time range description
    when metadata files are not available.
    
    Args:
        window_id: Window identifier (e.g., '0001', '0002')
        mode: Analysis mode ('sequential' or 'random')
        
    Returns:
        str: Simple time range description
    """
    
    try:
        # Convert window_id to integer for calculations
        window_num = int(window_id)
        
        if mode == 'sequential':
            # For sequential mode, assume windows represent consecutive time periods
            # This is a placeholder - adjust based on your actual windowing logic
            start_year = 1980 + (window_num - 1) * 5  # Example: 5-year windows
            end_year = start_year + 4
            return f"{start_year}-{end_year}"
            
        elif mode == 'random':
            # For random mode, just indicate it's a random sample
            return f"Random sample {window_id}"
            
        else:
            # Unknown mode
            return f"Window {window_id} ({mode})"
            
    except (ValueError, TypeError):
        # If window_id can't be converted to int or other error
        logger.debug(f"Could not parse window_id '{window_id}' as integer")
        return f"Window {window_id}"
    
def test_extraction_with_sample():
    """Test the extraction with your sample file"""
    sample_file = Path("hazard_results_los_angeles_seq_window_0001.txt")
    result = extract_hazard_curve_from_results(sample_file, 34.0500, -118.2500)
    
    if result:
        print(f"✅ SUCCESS! Extracted {len(result['ground_motions'])} points")
        print(f"GM range: {min(result['ground_motions']):.3e} to {max(result['ground_motions']):.3e}")
        print(f"Rate range: {min(result['annual_rates']):.3e} to {max(result['annual_rates']):.3e}")
    else:
        print("❌ FAILED")
    
    return result

def create_detailed_summary_plots(axes, sorted_windows, region_name, mode):
    """Create detailed summary plots for small numbers of windows"""
    
    window_ids = [w[0] for w in sorted_windows]
    x_pos = np.arange(len(window_ids))
    
    # Plot 1: Average Design Values
    avg_annual_2pct = [w[1]['avg_design_annual_2pct'] for w in sorted_windows]
    max_annual_2pct = [w[1]['max_design_annual_2pct'] for w in sorted_windows]
    
    axes[0, 0].bar(x_pos, avg_annual_2pct, alpha=0.7, color='skyblue', label='Average')
    axes[0, 0].bar(x_pos, max_annual_2pct, alpha=0.7, color='red', label='Maximum')
    axes[0, 0].set_title('2% Annual Design Values by Window', fontweight='bold')
    axes[0, 0].set_xlabel('Window ID')
    axes[0, 0].set_ylabel('Ground Motion (g)')
    axes[0, 0].set_xticks(x_pos)
    axes[0, 0].set_xticklabels(window_ids, rotation=45)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Number of Sites and Events
    num_sites = [w[1]['num_sites'] for w in sorted_windows]
    total_events = [w[1]['total_events'] for w in sorted_windows]
    
    ax2_twin = axes[0, 1].twinx()
    axes[0, 1].bar(x_pos - 0.2, num_sites, 0.4, alpha=0.7, color='green', label='Sites')
    ax2_twin.bar(x_pos + 0.2, total_events, 0.4, alpha=0.7, color='orange', label='Total Events')
    
    axes[0, 1].set_title('Sites and Events by Window', fontweight='bold')
    axes[0, 1].set_xlabel('Window ID')
    axes[0, 1].set_ylabel('Number of Sites', color='green')
    ax2_twin.set_ylabel('Total Events', color='orange')
    axes[0, 1].set_xticks(x_pos)
    axes[0, 1].set_xticklabels(window_ids, rotation=45)
    
    # Plot 3: Design Value Distribution
    all_design_values = []
    window_labels = []
    
    for window_id, window_info in [(w[0], w[1]) for w in sorted_windows]:
        # This would need access to the actual data, simplified here
        all_design_values.append([window_info['avg_design_annual_2pct']])
        window_labels.append(f"W{window_id}")
    
    if all_design_values:
        axes[1, 0].boxplot(all_design_values, labels=window_labels)
        axes[1, 0].set_title('Design Value Distribution by Window', fontweight='bold')
        axes[1, 0].set_ylabel('2% Annual Design Value (g)')
        axes[1, 0].tick_params(axis='x', rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Summary table
    axes[1, 1].axis('off')
    table_data = []
    headers = ['Window', 'Sites', 'Avg Design (g)', 'Max Design (g)']
    
    for window_id, stats in sorted_windows:
        row = [f"W{window_id}", f"{stats['num_sites']}", 
               f"{stats['avg_design_annual_2pct']:.3f}", 
               f"{stats['max_design_annual_2pct']:.3f}"]
        table_data.append(row)
    
    table = axes[1, 1].table(cellText=table_data, colLabels=headers, 
                           cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    axes[1, 1].set_title('Window Summary Statistics', fontweight='bold', pad=20)

def create_statistical_summary_plots(axes, sorted_windows, region_name, mode):
    """Create statistical summary plots for large numbers of windows"""
    
    # Extract data for statistical analysis
    avg_values = [w[1]['avg_design_annual_2pct'] for w in sorted_windows]
    max_values = [w[1]['max_design_annual_2pct'] for w in sorted_windows]
    num_sites = [w[1]['num_sites'] for w in sorted_windows]
    total_events = [w[1]['total_events'] for w in sorted_windows]
    
    # Plot 1: Statistical distribution of design values
    axes[0, 0].hist(avg_values, bins=20, alpha=0.7, color='skyblue', label='Average')
    axes[0, 0].hist(max_values, bins=20, alpha=0.7, color='red', label='Maximum')
    axes[0, 0].set_title('Distribution of Design Values', fontweight='bold')
    axes[0, 0].set_xlabel('2% Annual Design Value (g)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Box plot of key metrics
    data_to_plot = [avg_values, max_values, 
                    [x/1000 for x in total_events]]  # Scale events for visibility
    labels = ['Avg Design', 'Max Design', 'Total Events\n(×1000)']
    
    bp = axes[0, 1].boxplot(data_to_plot, labels=labels, patch_artist=True)
    colors = ['skyblue', 'red', 'orange']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    axes[0, 1].set_title('Statistical Summary of All Windows', fontweight='bold')
    axes[0, 1].set_ylabel('Value')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Time series trends
    window_numbers = [int(w[0]) for w in sorted_windows]
    axes[1, 0].plot(window_numbers, avg_values, 'bo-', alpha=0.7, label='Average')
    axes[1, 0].plot(window_numbers, max_values, 'ro-', alpha=0.7, label='Maximum')
    
    # Add trend lines
    if len(window_numbers) > 2:
        z_avg = np.polyfit(window_numbers, avg_values, 1)
        p_avg = np.poly1d(z_avg)
        axes[1, 0].plot(window_numbers, p_avg(window_numbers), 'b--', alpha=0.5)
        
        z_max = np.polyfit(window_numbers, max_values, 1)
        p_max = np.poly1d(z_max)
        axes[1, 0].plot(window_numbers, p_max(window_numbers), 'r--', alpha=0.5)
    
    axes[1, 0].set_title('Trends Across Windows', fontweight='bold')
    axes[1, 0].set_xlabel('Window Number')
    axes[1, 0].set_ylabel('2% Annual Design Value (g)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Variability analysis
    cv_avg = np.std(avg_values) / np.mean(avg_values) if avg_values else 0
    cv_max = np.std(max_values) / np.mean(max_values) if max_values else 0
    
    metrics = ['Coefficient of Variation']
    x = np.arange(len(metrics))
    width = 0.35
    
    axes[1, 1].bar(x - width/2, [cv_avg], width, label='Average', color='skyblue', alpha=0.7)
    axes[1, 1].bar(x + width/2, [cv_max], width, label='Maximum', color='red', alpha=0.7)
    
    axes[1, 1].set_ylabel('Coefficient of Variation')
    axes[1, 1].set_title('Variability Analysis', fontweight='bold')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metrics)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add interpretation text
    variability_text = f"Analysis Summary:\n"
    variability_text += f"Windows: {len(sorted_windows)}\n"
    variability_text += f"Mean avg design: {np.mean(avg_values):.3f}g\n"
    variability_text += f"Mean max design: {np.mean(max_values):.3f}g\n"
    variability_text += f"Variability (CV): {cv_avg:.3f}\n"
    
    if cv_avg > 0.2:
        variability_text += f"High variability detected"
    elif cv_avg > 0.1:
        variability_text += f"Moderate variability"
    else:
        variability_text += f"Low variability"
    
    axes[1, 1].text(0.02, 0.98, variability_text, transform=axes[1, 1].transAxes, 
                   fontsize=9, verticalalignment='top', horizontalalignment='left',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.8))
    
    # Plot 5: Correlation analysis (if 6 subplots)
    if len(axes.shape) == 2 and axes.shape[1] == 3:
        # Scatter plot of average vs maximum
        axes[0, 2].scatter(avg_values, max_values, alpha=0.6, color='purple')
        
        # Add correlation line
        if len(avg_values) > 1:
            correlation = np.corrcoef(avg_values, max_values)[0, 1]
            z = np.polyfit(avg_values, max_values, 1)
            p = np.poly1d(z)
            axes[0, 2].plot(avg_values, p(avg_values), 'r--', alpha=0.8)
            
            axes[0, 2].text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                          transform=axes[0, 2].transAxes, fontsize=10,
                          bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
        
        axes[0, 2].set_xlabel('Average Design Value (g)')
        axes[0, 2].set_ylabel('Maximum Design Value (g)')
        axes[0, 2].set_title('Average vs Maximum Correlation', fontweight='bold')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 6: Summary statistics table
        axes[1, 2].axis('off')
        
        summary_stats = [
            ['Metric', 'Value'],
            ['Total Windows', f'{len(sorted_windows)}'],
            ['Mean Avg Design', f'{np.mean(avg_values):.3f}g'],
            ['Std Avg Design', f'{np.std(avg_values):.3f}g'],
            ['Mean Max Design', f'{np.mean(max_values):.3f}g'],
            ['Std Max Design', f'{np.std(max_values):.3f}g'],
            ['Coefficient of Variation', f'{cv_avg:.3f}'],
            ['Min Window Size', f'{min(num_sites)}'],
            ['Max Window Size', f'{max(num_sites)}'],
            ['Total Events', f'{sum(total_events):,}']
        ]
        
        table = axes[1, 2].table(cellText=summary_stats[1:], colLabels=summary_stats[0], 
                               cellLoc='center', loc='center', bbox=[0, 0, 1, 1])
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.5)
        
        # Style the table
        table[(0, 0)].set_facecolor('#4CAF50')
        table[(0, 0)].set_text_props(weight='bold', color='white')
        table[(0, 1)].set_facecolor('#4CAF50')
        table[(0, 1)].set_text_props(weight='bold', color='white')
        
        axes[1, 2].set_title('Summary Statistics', fontweight='bold', pad=20)

def create_statistical_multi_window_plot(site_lat, site_lon, all_window_data, output_dir, region_name):
    """
    Statistical multi-window plot for large numbers of windows (50+)
    Shows percentiles, means, and confidence bands instead of individual curves
    """
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Collect all hazard data for statistical analysis
    all_hazard_data = []
    design_values_2pct = []
    design_values_10pct = []
    time_ranges = []
    
    for window_id, window_info in all_window_data.items():
        df = window_info['data']
        time_range = window_info['time_range']
        
        # Find this site in the window data
        site_row = df[
            (np.abs(df['site_lat'] - site_lat) < 0.001) & 
            (np.abs(df['site_lon'] - site_lon) < 0.001)
        ]
        
        if site_row.empty:
            continue
        
        # Extract design values
        design_2pct = site_row['design_2pct_annual_g'].iloc[0] if 'design_2pct_annual_g' in site_row.columns else 0
        design_10pct = site_row['design_10pct_annual_g'].iloc[0] if 'design_10pct_annual_g' in site_row.columns else 0
        
        if design_2pct > 0:
            design_values_2pct.append(design_2pct)
        if design_10pct > 0:
            design_values_10pct.append(design_10pct)
        
        time_ranges.append(time_range)
        
        # Get hazard curve data
        summary_file = window_info['summary_file']
        results_file = summary_file.parent / summary_file.name.replace('summary_', 'hazard_results_').replace('.csv', '.txt')
        
        hazard_data = extract_hazard_curve_from_results(results_file, site_lat, site_lon)
        if hazard_data:
            all_hazard_data.append({
                'window_id': window_id,
                'ground_motions': np.array(hazard_data['ground_motions']),
                'annual_rates': np.array(hazard_data['annual_rates']),
                'design_2pct': design_2pct,
                'design_10pct': design_10pct
            })
    
    if len(all_hazard_data) < 2:
        logger.warning(f"Not enough hazard data for statistical plotting")
        plt.close()
        return
    
    logger.info(f"Creating statistical plot with {len(all_hazard_data)} valid windows")
    
    # PLOT 1: Statistical Hazard Curves
    create_statistical_hazard_plot(ax1, all_hazard_data, design_values_2pct, design_values_10pct)
    
    # PLOT 2: Design Value Distributions
    create_design_value_distribution_plot(ax2, design_values_2pct, design_values_10pct)
    
    # PLOT 3: Time Series of Design Values
    create_time_series_plot(ax3, all_hazard_data, time_ranges)
    
    # PLOT 4: Variability Analysis
    create_variability_analysis_plot(ax4, design_values_2pct, design_values_10pct, len(all_hazard_data))
    
    # Overall title
    fig.suptitle(f'Multi-Window Statistical Analysis - {region_name.title()}\n'
                 f'Site: {site_lat:.4f}°N, {site_lon:.4f}°E - {len(all_hazard_data)} Windows', 
                 fontsize=16, fontweight='bold')
    
    # Adjust layout to accommodate text outside plot
    plt.tight_layout()
    plt.subplots_adjust(right=0.75)  # Make room for text on the right
    
    # Save plot
    plot_file = output_dir / f"statistical_multi_window_site_{site_lat:.4f}_{site_lon:.4f}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"✅ Statistical multi-window plot saved: {plot_file}")

def create_hybrid_multi_window_plot(site_lat, site_lon, all_window_data, output_dir, region_name):
    """
    Hybrid plot for medium numbers of windows (21-50)
    Shows sample of individual curves + statistical summary
    """
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
    
    # Collect all data
    all_hazard_data = []
    design_values_2pct = []
    design_values_10pct = []
    
    for window_id, window_info in all_window_data.items():
        df = window_info['data']
        
        site_row = df[
            (np.abs(df['site_lat'] - site_lat) < 0.001) & 
            (np.abs(df['site_lon'] - site_lon) < 0.001)
        ]
        
        if site_row.empty:
            continue
        
        design_2pct = site_row['design_2pct_annual_g'].iloc[0] if 'design_2pct_annual_g' in site_row.columns else 0
        design_10pct = site_row['design_10pct_annual_g'].iloc[0] if 'design_10pct_annual_g' in site_row.columns else 0
        
        if design_2pct > 0:
            design_values_2pct.append(design_2pct)
        if design_10pct > 0:
            design_values_10pct.append(design_10pct)
        
        summary_file = window_info['summary_file']
        results_file = summary_file.parent / summary_file.name.replace('summary_', 'hazard_results_').replace('.csv', '.txt')
        
        hazard_data = extract_hazard_curve_from_results(results_file, site_lat, site_lon)
        if hazard_data:
            all_hazard_data.append({
                'window_id': window_id,
                'ground_motions': np.array(hazard_data['ground_motions']),
                'annual_rates': np.array(hazard_data['annual_rates']),
                'design_2pct': design_2pct,
                'design_10pct': design_10pct
            })
    
    # PLOT 1: Sample of individual curves + statistical bands
    create_sample_curves_plot(ax1, all_hazard_data, design_values_2pct, design_values_10pct)
    
    # PLOT 2: Design value distributions
    create_design_value_distribution_plot(ax2, design_values_2pct, design_values_10pct)
    
    # PLOT 3: Percentile analysis
    create_percentile_analysis_plot(ax3, design_values_2pct, design_values_10pct)
    
    # PLOT 4: Summary statistics
    create_summary_statistics_plot(ax4, design_values_2pct, design_values_10pct, len(all_hazard_data))
    
    fig.suptitle(f'Multi-Window Hybrid Analysis - {region_name.title()}\n'
                 f'Site: {site_lat:.4f}°N, {site_lon:.4f}°E - {len(all_hazard_data)} Windows', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    plot_file = output_dir / f"hybrid_multi_window_site_{site_lat:.4f}_{site_lon:.4f}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"✅ Hybrid multi-window plot saved: {plot_file}")

def create_detailed_multi_window_plot(site_lat, site_lon, all_window_data, output_dir, region_name):
    """
    Detailed multi-window plot for small numbers of windows (≤20)
    Shows individual curves with readable legend
    """
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Use a better color palette for small numbers
    colors = plt.cm.Set3(np.linspace(0, 1, len(all_window_data)))
    
    design_values_2pct = []
    design_values_10pct = []
    plotted_windows = []
    
    for i, (window_id, window_info) in enumerate(sorted(all_window_data.items())):
        df = window_info['data']
        time_range = window_info['time_range']
        color = colors[i]
        
        site_row = df[
            (np.abs(df['site_lat'] - site_lat) < 0.001) & 
            (np.abs(df['site_lon'] - site_lon) < 0.001)
        ]
        
        if site_row.empty:
            continue
        
        design_2pct = site_row['design_2pct_annual_g'].iloc[0] if 'design_2pct_annual_g' in site_row.columns else 0
        design_10pct = site_row['design_10pct_annual_g'].iloc[0] if 'design_10pct_annual_g' in site_row.columns else 0
        
        if design_2pct > 0:
            design_values_2pct.append(design_2pct)
        if design_10pct > 0:
            design_values_10pct.append(design_10pct)
        
        summary_file = window_info['summary_file']
        results_file = summary_file.parent / summary_file.name.replace('summary_', 'hazard_results_').replace('.csv', '.txt')
        
        hazard_data = extract_hazard_curve_from_results(results_file, site_lat, site_lon)
        if hazard_data:
            ground_motions = np.array(hazard_data['ground_motions'])
            annual_rates = np.array(hazard_data['annual_rates'])
            
            # Clean, shorter label
            label = f"W{window_id}"
            
            # Plot the hazard curve
            ax1.loglog(ground_motions, annual_rates, color=color, linewidth=2, alpha=0.8, label=label)
            
            # Add design point markers
            if design_2pct > 0:
                ax1.loglog([design_2pct], [0.02], 'o', color=color, markersize=8, 
                          markeredgecolor='black', markeredgewidth=1, zorder=10)
            
            if design_10pct > 0:
                ax1.loglog([design_10pct], [0.1], 's', color=color, markersize=8, 
                          markeredgecolor='black', markeredgewidth=1, zorder=10)
            
            plotted_windows.append(window_id)
    
    if not plotted_windows:
        logger.warning(f"No windows could be plotted for detailed view")
        plt.close()
        return
    
    # Add average lines
    if design_values_2pct:
        avg_2pct = np.mean(design_values_2pct)
        ax1.axvline(avg_2pct, color='red', linestyle='--', alpha=0.7, linewidth=2, zorder=5)
        ax1.text(avg_2pct * 1.1, ax1.get_ylim()[0] * 2, f'Avg 2%\n{avg_2pct:.3f}g', 
                bbox=dict(boxstyle='round,pad=0.2', facecolor='red', alpha=0.7),
                fontsize=10, ha='left', va='bottom')
    
    if design_values_10pct:
        avg_10pct = np.mean(design_values_10pct)
        ax1.axvline(avg_10pct, color='orange', linestyle='--', alpha=0.7, linewidth=2, zorder=5)
        ax1.text(avg_10pct * 1.1, ax1.get_ylim()[0] * 5, f'Avg 10%\n{avg_10pct:.3f}g', 
                bbox=dict(boxstyle='round,pad=0.2', facecolor='orange', alpha=0.7),
                fontsize=10, ha='left', va='bottom')
    
    ax1.grid(True, which="major", ls="-", alpha=0.4)
    ax1.grid(True, which="minor", ls=":", alpha=0.2)
    ax1.set_xlabel('Peak Ground Acceleration (g)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Annual Rate of Exceedance', fontsize=12, fontweight='bold')
    ax1.set_title('Individual Hazard Curves', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9, ncol=2)
    
    # Add return periods on right axis
    ax1_twin = ax1.twinx()
    ax1_twin.set_yscale('log')
    ax1_twin.set_ylim(ax1.get_ylim())
    return_periods = [10, 50, 100, 500, 1000, 2500, 10000]
    return_rates = [1/rp for rp in return_periods]
    ax1_twin.set_yticks(return_rates)
    ax1_twin.set_yticklabels([f'{rp}' for rp in return_periods])
    ax1_twin.set_ylabel('Return Period (years)', fontsize=12, fontweight='bold')
    
    # PLOT 2: Design value comparison
    create_design_value_distribution_plot(ax2, design_values_2pct, design_values_10pct)
    
    # Enhanced title
    if design_values_2pct:
        min_2pct, max_2pct = min(design_values_2pct), max(design_values_2pct)
        title = f'Multi-Window Hazard Comparison - {region_name.title()}\n'
        title += f'Site: {site_lat:.4f}°N, {site_lon:.4f}°E\n'
        title += f'Windows: {len(plotted_windows)}, Design Range: {min_2pct:.3f}-{max_2pct:.3f}g (2% annual)'
    else:
        title = f'Multi-Window Hazard Comparison - {region_name.title()}\n'
        title += f'Site: {site_lat:.4f}°N, {site_lon:.4f}°E - {len(plotted_windows)} Windows'
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    plot_file = output_dir / f"detailed_multi_window_site_{site_lat:.4f}_{site_lon:.4f}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"✅ Detailed multi-window plot saved: {plot_file}")

def create_statistical_hazard_plot(ax, all_hazard_data, design_values_2pct, design_values_10pct):
    """
    Create statistical hazard curve plot with confidence bands
    """
    
    # Create common ground motion range
    all_gm_values = []
    for data in all_hazard_data:
        all_gm_values.extend(data['ground_motions'])
    
    if not all_gm_values:
        ax.text(0.5, 0.5, 'No hazard data available', transform=ax.transAxes, 
                ha='center', va='center', fontsize=14)
        return
    
    gm_min = max(min(all_gm_values), 0.001)
    gm_max = min(max(all_gm_values), 2.0)
    
    # Create interpolation grid
    gm_grid = np.logspace(np.log10(gm_min), np.log10(gm_max), 100)
    
    # Interpolate all curves to common grid
    interpolated_rates = []
    for data in all_hazard_data:
        gm = data['ground_motions']
        rates = data['annual_rates']
        
        # Filter valid data
        valid_mask = (gm > 0) & (rates > 0) & np.isfinite(gm) & np.isfinite(rates)
        if not np.any(valid_mask):
            continue
            
        gm_valid = gm[valid_mask]
        rates_valid = rates[valid_mask]
        
        # Sort data for interpolation
        sort_idx = np.argsort(gm_valid)
        gm_sorted = gm_valid[sort_idx]
        rates_sorted = rates_valid[sort_idx]
        
        # Interpolate to grid
        try:
            interp_rates = np.interp(gm_grid, gm_sorted, rates_sorted)
            interpolated_rates.append(interp_rates)
        except:
            continue
    
    if not interpolated_rates:
        ax.text(0.5, 0.5, 'Cannot interpolate hazard data', transform=ax.transAxes, 
                ha='center', va='center', fontsize=14)
        return
    
    # Calculate statistics
    rates_array = np.array(interpolated_rates)
    
    # Percentiles
    rates_5th = np.percentile(rates_array, 5, axis=0)
    rates_25th = np.percentile(rates_array, 25, axis=0)
    rates_50th = np.percentile(rates_array, 50, axis=0)
    rates_75th = np.percentile(rates_array, 75, axis=0)
    rates_95th = np.percentile(rates_array, 95, axis=0)
    rates_mean = np.mean(rates_array, axis=0)
    
    # Plot confidence bands
    ax.fill_between(gm_grid, rates_5th, rates_95th, alpha=0.2, color='blue', label='5th-95th percentile')
    ax.fill_between(gm_grid, rates_25th, rates_75th, alpha=0.4, color='blue', label='25th-75th percentile')
    
    # Plot central tendencies
    ax.loglog(gm_grid, rates_mean, 'r-', linewidth=3, label='Mean', alpha=0.8)
    ax.loglog(gm_grid, rates_50th, 'b-', linewidth=3, label='Median', alpha=0.8)
    
    # Add design value statistics
    if design_values_2pct:
        mean_2pct = np.mean(design_values_2pct)
        std_2pct = np.std(design_values_2pct)
        ax.axvline(mean_2pct, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax.axvline(mean_2pct - std_2pct, color='red', linestyle=':', linewidth=1, alpha=0.5)
        ax.axvline(mean_2pct + std_2pct, color='red', linestyle=':', linewidth=1, alpha=0.5)
        
        ax.text(mean_2pct * 1.05, 0.02, f'2% Annual\n{mean_2pct:.3f}±{std_2pct:.3f}g', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='red', alpha=0.7),
                fontsize=9, ha='left', va='center')
    
    if design_values_10pct:
        mean_10pct = np.mean(design_values_10pct)
        std_10pct = np.std(design_values_10pct)
        ax.axvline(mean_10pct, color='orange', linestyle='--', linewidth=2, alpha=0.7)
        ax.axvline(mean_10pct - std_10pct, color='orange', linestyle=':', linewidth=1, alpha=0.5)
        ax.axvline(mean_10pct + std_10pct, color='orange', linestyle=':', linewidth=1, alpha=0.5)
        
        ax.text(mean_10pct * 1.05, 0.1, f'10% Annual\n{mean_10pct:.3f}±{std_10pct:.3f}g', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='orange', alpha=0.7),
                fontsize=9, ha='left', va='center')
    
    ax.set_xlabel('Peak Ground Acceleration (g)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Annual Rate of Exceedance', fontsize=12, fontweight='bold')
    ax.set_title('Statistical Hazard Curves', fontsize=14, fontweight='bold')
    ax.grid(True, which="major", ls="-", alpha=0.3)
    ax.grid(True, which="minor", ls=":", alpha=0.2)
    ax.legend(loc='center left', bbox_to_anchor=(1.05, 0.5), fontsize=10)
    
    # Add return periods on right axis
    ax2 = ax.twinx()
    ax2.set_yscale('log')
    ax2.set_ylim(ax.get_ylim())
    return_periods = [10, 50, 100, 500, 1000, 2500, 10000]
    return_rates = [1/rp for rp in return_periods]
    ax2.set_yticks(return_rates)
    ax2.set_yticklabels([f'{rp}' for rp in return_periods])
    ax2.set_ylabel('Return Period (years)', fontsize=12, fontweight='bold')

def create_design_value_distribution_plot(ax, design_values_2pct, design_values_10pct):
    """
    Create design value distribution plot with box plots and statistics
    """
    
    if not design_values_2pct and not design_values_10pct:
        ax.text(0.5, 0.5, 'No design values available', transform=ax.transAxes, 
                ha='center', va='center', fontsize=14)
        return
    
    # Create box plots
    data_to_plot = []
    labels = []
    colors = []
    
    if design_values_10pct:
        data_to_plot.append(design_values_10pct)
        labels.append('10% Annual\n(475-yr RP)')
        colors.append('orange')
    
    if design_values_2pct:
        data_to_plot.append(design_values_2pct)
        labels.append('2% Annual\n(2475-yr RP)')
        colors.append('red')
    
    if data_to_plot:
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True, widths=0.6)
        
        # Color the boxes
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
    
    # Add statistics
    stats_text = f"Distribution Statistics:\n"
    stats_text += f"Windows: {len(design_values_2pct) if design_values_2pct else len(design_values_10pct)}\n\n"
    
    if design_values_2pct:
        mean_2pct = np.mean(design_values_2pct)
        std_2pct = np.std(design_values_2pct)
        min_2pct = np.min(design_values_2pct)
        max_2pct = np.max(design_values_2pct)
        cv_2pct = std_2pct / mean_2pct if mean_2pct > 0 else 0
        
        stats_text += f"2% Annual:\n"
        stats_text += f"  Mean: {mean_2pct:.3f}g\n"
        stats_text += f"  Std: {std_2pct:.3f}g\n"
        stats_text += f"  CV: {cv_2pct:.3f}\n"
        stats_text += f"  Range: {min_2pct:.3f}-{max_2pct:.3f}g\n\n"
    
    if design_values_10pct:
        mean_10pct = np.mean(design_values_10pct)
        std_10pct = np.std(design_values_10pct)
        min_10pct = np.min(design_values_10pct)
        max_10pct = np.max(design_values_10pct)
        cv_10pct = std_10pct / mean_10pct if mean_10pct > 0 else 0
        
        stats_text += f"10% Annual:\n"
        stats_text += f"  Mean: {mean_10pct:.3f}g\n"
        stats_text += f"  Std: {std_10pct:.3f}g\n"
        stats_text += f"  CV: {cv_10pct:.3f}\n"
        stats_text += f"  Range: {min_10pct:.3f}-{max_10pct:.3f}g"
    
    # Place text outside the plot area
    ax.text(1.02, 0.98, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    ax.set_ylabel('Ground Motion (g)', fontsize=12, fontweight='bold')
    ax.set_title('Design Value Distributions', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

def create_time_series_plot(ax, all_hazard_data, time_ranges):
    """
    Create time series plot of design values
    """
    
    design_2pct_values = []
    design_10pct_values = []
    window_numbers = []
    
    for data in all_hazard_data:
        window_numbers.append(int(data['window_id']))
        design_2pct_values.append(data['design_2pct'])
        design_10pct_values.append(data['design_10pct'])
    
    # Sort by window number
    sorted_indices = np.argsort(window_numbers)
    window_numbers = np.array(window_numbers)[sorted_indices]
    design_2pct_values = np.array(design_2pct_values)[sorted_indices]
    design_10pct_values = np.array(design_10pct_values)[sorted_indices]
    
    # Plot time series
    ax.plot(window_numbers, design_2pct_values, 'ro-', linewidth=2, markersize=6, 
            alpha=0.7, label='2% Annual')
    ax.plot(window_numbers, design_10pct_values, 'bo-', linewidth=2, markersize=6, 
            alpha=0.7, label='10% Annual')
    
    # Add trend lines
    if len(window_numbers) > 2:
        z_2pct = np.polyfit(window_numbers, design_2pct_values, 1)
        p_2pct = np.poly1d(z_2pct)
        ax.plot(window_numbers, p_2pct(window_numbers), 'r--', alpha=0.5, linewidth=1)
        
        z_10pct = np.polyfit(window_numbers, design_10pct_values, 1)
        p_10pct = np.poly1d(z_10pct)
        ax.plot(window_numbers, p_10pct(window_numbers), 'b--', alpha=0.5, linewidth=1)
    
    ax.set_xlabel('Window Number', fontsize=12, fontweight='bold')
    ax.set_ylabel('Ground Motion (g)', fontsize=12, fontweight='bold')
    ax.set_title('Design Values vs Window Number', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

def create_variability_analysis_plot(ax, design_values_2pct, design_values_10pct, n_windows):
    """
    Create variability analysis plot
    """
    
    # Calculate variability metrics
    cv_2pct = np.std(design_values_2pct) / np.mean(design_values_2pct) if design_values_2pct else 0
    cv_10pct = np.std(design_values_10pct) / np.mean(design_values_10pct) if design_values_10pct else 0
    
    # Create variability bar chart
    metrics = ['Coefficient of Variation']
    cv_2pct_values = [cv_2pct] if design_values_2pct else []
    cv_10pct_values = [cv_10pct] if design_values_10pct else []
    
    x = np.arange(len(metrics))
    width = 0.35
    
    if cv_2pct_values:
        bars1 = ax.bar(x - width/2, cv_2pct_values, width, label='2% Annual', 
                      color='red', alpha=0.7)
    
    if cv_10pct_values:
        bars2 = ax.bar(x + width/2, cv_10pct_values, width, label='10% Annual', 
                      color='orange', alpha=0.7)
    
    ax.set_ylabel('Coefficient of Variation', fontsize=12, fontweight='bold')
    ax.set_title('Variability Analysis', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add interpretation text
    variability_text = f"Variability Interpretation:\n"
    variability_text += f"Windows analyzed: {n_windows}\n\n"
    
    if cv_2pct > 0:
        if cv_2pct < 0.1:
            variability_text += f"2% Annual: Low variability ({cv_2pct:.3f})\n"
        elif cv_2pct < 0.2:
            variability_text += f"2% Annual: Moderate variability ({cv_2pct:.3f})\n"
        else:
            variability_text += f"2% Annual: High variability ({cv_2pct:.3f})\n"
    
    if cv_10pct > 0:
        if cv_10pct < 0.1:
            variability_text += f"10% Annual: Low variability ({cv_10pct:.3f})\n"
        elif cv_10pct < 0.2:
            variability_text += f"10% Annual: Moderate variability ({cv_10pct:.3f})\n"
        else:
            variability_text += f"10% Annual: High variability ({cv_10pct:.3f})\n"
    
    variability_text += f"\nRecommendations:\n"
    if max(cv_2pct, cv_10pct) > 0.3:
        variability_text += "• Consider longer time windows\n"
        variability_text += "• Investigate source of variability\n"
    elif max(cv_2pct, cv_10pct) > 0.1:
        variability_text += "• Acceptable variability\n"
        variability_text += "• Consider epistemic uncertainty\n"
    else:
        variability_text += "• Low variability - good consistency\n"
        variability_text += "• Results are stable\n"
    
    # Place text outside the plot area  
    ax.text(1.02, 0.98, variability_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))

def create_sample_curves_plot(ax, all_hazard_data, design_values_2pct, design_values_10pct):
    """
    Create sample curves plot for hybrid visualization
    Shows representative curves + statistical summary
    """
    
    # Select representative curves (every nth curve to avoid overcrowding)
    n_curves = len(all_hazard_data)
    n_sample = min(10, n_curves)  # Show at most 10 curves
    step = max(1, n_curves // n_sample)
    
    sample_indices = range(0, n_curves, step)
    colors = plt.cm.Set3(np.linspace(0, 1, len(sample_indices)))
    
    # Plot sample curves
    for i, idx in enumerate(sample_indices):
        data = all_hazard_data[idx]
        gm = data['ground_motions']
        rates = data['annual_rates']
        
        # Filter valid data
        valid_mask = (gm > 0) & (rates > 0) & np.isfinite(gm) & np.isfinite(rates)
        if np.any(valid_mask):
            ax.loglog(gm[valid_mask], rates[valid_mask], color=colors[i], 
                     linewidth=1.5, alpha=0.7, label=f'W{data["window_id"]}')
    
    # Add statistical summary (similar to statistical plot)
    # Create common ground motion range
    all_gm_values = []
    for data in all_hazard_data:
        all_gm_values.extend(data['ground_motions'])
    
    if all_gm_values:
        gm_min = max(min(all_gm_values), 0.001)
        gm_max = min(max(all_gm_values), 2.0)
        gm_grid = np.logspace(np.log10(gm_min), np.log10(gm_max), 50)
        
        # Interpolate all curves
        interpolated_rates = []
        for data in all_hazard_data:
            gm = data['ground_motions']
            rates = data['annual_rates']
            
            valid_mask = (gm > 0) & (rates > 0) & np.isfinite(gm) & np.isfinite(rates)
            if not np.any(valid_mask):
                continue
                
            gm_valid = gm[valid_mask]
            rates_valid = rates[valid_mask]
            
            sort_idx = np.argsort(gm_valid)
            gm_sorted = gm_valid[sort_idx]
            rates_sorted = rates_valid[sort_idx]
            
            try:
                interp_rates = np.interp(gm_grid, gm_sorted, rates_sorted)
                interpolated_rates.append(interp_rates)
            except:
                continue
        
        if interpolated_rates:
            rates_array = np.array(interpolated_rates)
            rates_mean = np.mean(rates_array, axis=0)
            rates_25th = np.percentile(rates_array, 25, axis=0)
            rates_75th = np.percentile(rates_array, 75, axis=0)
            
            # Plot statistical summary
            ax.fill_between(gm_grid, rates_25th, rates_75th, alpha=0.3, color='gray', 
                           label='25th-75th percentile')
            ax.loglog(gm_grid, rates_mean, 'k-', linewidth=3, label='Mean', alpha=0.8)
    
    ax.set_xlabel('Peak Ground Acceleration (g)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Annual Rate of Exceedance', fontsize=12, fontweight='bold')
    ax.set_title('Sample Hazard Curves + Statistics', fontsize=14, fontweight='bold')
    ax.grid(True, which="major", ls="-", alpha=0.3)
    ax.grid(True, which="minor", ls=":", alpha=0.2)
    ax.legend(loc='upper right', fontsize=9, ncol=2)

def create_percentile_analysis_plot(ax, design_values_2pct, design_values_10pct):
    """
    Create percentile analysis plot
    """
    
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    
    if design_values_2pct:
        percentile_2pct = np.percentile(design_values_2pct, percentiles)
        ax.plot(percentiles, percentile_2pct, 'ro-', linewidth=2, markersize=6, 
                alpha=0.7, label='2% Annual')
    
    if design_values_10pct:
        percentile_10pct = np.percentile(design_values_10pct, percentiles)
        ax.plot(percentiles, percentile_10pct, 'bo-', linewidth=2, markersize=6, 
                alpha=0.7, label='10% Annual')
    
    ax.set_xlabel('Percentile', fontsize=12, fontweight='bold')
    ax.set_ylabel('Ground Motion (g)', fontsize=12, fontweight='bold')
    ax.set_title('Percentile Analysis', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

def create_summary_statistics_plot(ax, design_values_2pct, design_values_10pct, n_windows):
    """
    Create summary statistics plot
    """
    
    # Calculate statistics
    stats_2pct = {
        'Mean': np.mean(design_values_2pct) if design_values_2pct else 0,
        'Median': np.median(design_values_2pct) if design_values_2pct else 0,
        'Std': np.std(design_values_2pct) if design_values_2pct else 0,
        'Min': np.min(design_values_2pct) if design_values_2pct else 0,
        'Max': np.max(design_values_2pct) if design_values_2pct else 0,
    }
    
    stats_10pct = {
        'Mean': np.mean(design_values_10pct) if design_values_10pct else 0,
        'Median': np.median(design_values_10pct) if design_values_10pct else 0,
        'Std': np.std(design_values_10pct) if design_values_10pct else 0,
        'Min': np.min(design_values_10pct) if design_values_10pct else 0,
        'Max': np.max(design_values_10pct) if design_values_10pct else 0,
    }
    
    # Create table
    ax.axis('off')
    
    # Create table data
    table_data = []
    headers = ['Statistic', '2% Annual (g)', '10% Annual (g)']
    
    for stat in ['Mean', 'Median', 'Std', 'Min', 'Max']:
        row = [stat, f"{stats_2pct[stat]:.3f}", f"{stats_10pct[stat]:.3f}"]
        table_data.append(row)
    
    # Create table
    table = ax.table(cellText=table_data, colLabels=headers, cellLoc='center', 
                    loc='center', bbox=[0, 0, 1, 0.8])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the table
    for i in range(len(headers)):
        table[(0, i)].set_facecolor('#4CAF50')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Add summary text
    summary_text = f"Multi-Window Analysis Summary\n"
    summary_text += f"Total windows: {n_windows}\n"
    summary_text += f"Analysis type: Statistical\n"
    summary_text += f"Variability: {'High' if max(stats_2pct['Std']/stats_2pct['Mean'] if stats_2pct['Mean'] > 0 else 0, stats_10pct['Std']/stats_10pct['Mean'] if stats_10pct['Mean'] > 0 else 0) > 0.2 else 'Moderate' if max(stats_2pct['Std']/stats_2pct['Mean'] if stats_2pct['Mean'] > 0 else 0, stats_10pct['Std']/stats_10pct['Mean'] if stats_10pct['Mean'] > 0 else 0) > 0.1 else 'Low'}"
    
    ax.text(0.5, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    
    ax.set_title('Summary Statistics', fontsize=14, fontweight='bold')