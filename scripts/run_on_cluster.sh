#!/bin/bash

# RSQSim Monte Carlo Ground Motion Simulation - PATTERN FIX
# Fixes the pattern argument passing issue
#
# Usage:
#   ./run_on_cluster.sh san_francisco sequential --pattern "pattern_file.csv"

set -e  # Exit on any error

# =============================================================================
# Configuration Section - FIXED for BC Cluster + Directory Creation
# =============================================================================

# Job scheduler
SCHEDULER="slurm"

# Default resource requests
DEFAULT_NODES=1
DEFAULT_CORES_PER_NODE=48
DEFAULT_MEMORY="64GB"
DEFAULT_WALLTIME="12:00:00"
DEFAULT_QUEUE="short"

# BC Cluster specific settings
PROJECT_ACCOUNT="ebelseismo"
CLUSTER_EMAIL="olawoyiv@bc.edu"

# Storage paths
RSQSIM_HOME_BASE="rsqsim_mc"
RSQSIM_WORK_BASE="/scratch"
RSQSIM_PROJECT_BASE="/projects/ebelseismo"

# =============================================================================
# Functions
# =============================================================================

print_usage() {
    cat << EOF
Usage: $0 REGION MODE [OPTIONS]

Arguments:
  REGION          Region name (e.g., los_angeles, san_francisco)
  MODE            Window mode (sequential or random)

Options:
  --config FILE   Configuration file path
  --nodes N       Number of nodes (default: $DEFAULT_NODES)
  --cores N       Cores per node (default: $DEFAULT_CORES_PER_NODE)
  --memory SIZE   Memory per node (default: $DEFAULT_MEMORY)
  --time TIME     Wall time limit (default: $DEFAULT_WALLTIME)
  --queue QUEUE   Queue/partition name (default: $DEFAULT_QUEUE)
  --account ACC   Project account (default: $PROJECT_ACCOUNT)
  --email EMAIL   Email for notifications (default: $CLUSTER_EMAIL)
  --dry-run       Show job script without submitting
  --create-dirs   Create folder structure before job submission
  --pattern FILE  File pattern to match window files  
  --help          Show this help message

Examples:
  $0 san_francisco sequential
  $0 san_francisco random --config configs/custom_config.py
  $0 san_francisco sequential --pattern "los_angeles_seq_window_0001.csv"
  $0 san_francisco sequential --create-dirs --dry-run
EOF
}

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

create_folder_structure() {
    local region=$1
    local mode=$2
    
    log_message "Creating folder structure for $region/$mode..."
    
    # Try to find the create_folders script
    local create_script=""
    local possible_locations=(
        "$PROJECT_ROOT/create_rsqsim_folders.py"
        "$PROJECT_ROOT/scripts/create_rsqsim_folders.py"
        "$(dirname "$0")/create_rsqsim_folders.py"
        "./create_rsqsim_folders.py"
    )
    
    for location in "${possible_locations[@]}"; do
        if [[ -f "$location" ]]; then
            create_script="$location"
            break
        fi
    done
    
    if [[ -z "$create_script" ]]; then
        log_message "⚠️  create_rsqsim_folders.py not found, creating directories manually..."
        
        # Manual directory creation as fallback
        local base_output="$RSQSIM_WORK_BASE/$USER/rsqsim_data/output"
        local dirs_to_create=(
            "$base_output/$region"
            "$base_output/$region/$mode"
            "$base_output/$region/$mode/regional_summary"
            "$base_output/$region/$mode/visualizations"
            "$base_output/$region/$mode/visualizations/maps"
            "$base_output/$region/$mode/visualizations/plots"
            "$base_output/$region/$mode/visualizations/gis"
        )
        
        for dir in "${dirs_to_create[@]}"; do
            if mkdir -p "$dir" 2>/dev/null; then
                echo "✅ Created: $dir"
            else
                echo "⚠️  Could not create: $dir"
            fi
        done
    else
        log_message "✅ Found create_folders script at: $create_script"
        python "$create_script" "$region" "$mode"
        
        if [[ $? -eq 0 ]]; then
            log_message "✅ Folder structure created successfully"
        else
            log_message "❌ Failed to create folder structure"
            exit 1
        fi
    fi
}

# FIXED: Updated function signature to accept pattern parameter
generate_slurm_script() {
    local region=$1
    local mode=$2
    local config_file=$3
    local job_name="${region}_${mode}_gm_sim"
    local output_dir="$PROJECT_ROOT/logs/$region/cluster"
    
    # Create logs directory
    mkdir -p "$output_dir"
    mkdir -p "$PROJECT_ROOT/logs"
    
    cat << EOF
#!/bin/bash
#SBATCH --job-name=$job_name
#SBATCH --account=$PROJECT_ACCOUNT
#SBATCH --partition=$QUEUE
#SBATCH --nodes=$NODES
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$CORES_PER_NODE
#SBATCH --mem=$MEMORY
#SBATCH --time=$WALLTIME
#SBATCH --output=$output_dir/${job_name}_%j.out
#SBATCH --error=$output_dir/${job_name}_%j.err
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=$CLUSTER_EMAIL

# Job information
echo "=== RSQSim Ground Motion Simulation Job - PATTERN FIX ==="
echo "Job started at: \$(date)"
echo "Job ID: \$SLURM_JOB_ID"
echo "Running on nodes: \$SLURM_NODELIST"
echo "Number of cores: \$SLURM_NTASKS"
echo "Region: $region"
echo "Mode: $mode"
echo "Pattern: ${PATTERN:-"default"}"
echo "Working directory: \$(pwd)"
echo

# MINIMAL FIX: Only fix the conda detection syntax error
echo "=== Setting up conda environment ==="

# Simple conda detection without syntax errors
CONDA_FOUND=false

if [ -f "\$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    echo "✅ Found conda at: \$HOME/miniconda3"
    source "\$HOME/miniconda3/etc/profile.d/conda.sh"
    CONDA_FOUND=true
elif [ -f "\$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    echo "✅ Found conda at: \$HOME/anaconda3"
    source "\$HOME/anaconda3/etc/profile.d/conda.sh"
    CONDA_FOUND=true
elif [ -f "/opt/miniconda3/etc/profile.d/conda.sh" ]; then
    echo "✅ Found conda at: /opt/miniconda3"
    source "/opt/miniconda3/etc/profile.d/conda.sh"
    CONDA_FOUND=true
elif [ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]; then
    echo "✅ Found conda at: /opt/anaconda3"
    source "/opt/anaconda3/etc/profile.d/conda.sh"
    CONDA_FOUND=true
elif command -v conda &> /dev/null; then
    echo "✅ Found conda in PATH"
    eval "\$(conda shell.bash hook)"
    CONDA_FOUND=true
fi

if [ "\$CONDA_FOUND" = false ]; then
    echo "❌ ERROR: Could not find conda installation"
    echo "Please check that conda is installed and accessible"
    exit 1
fi

# Activate the rsqsim-python-tools environment
echo "Activating rsqsim-python-tools environment..."
conda activate rsqsim-python-tools

if [ \$? -eq 0 ]; then
    echo "✅ Successfully activated rsqsim-python-tools environment"
else
    echo "❌ ERROR: Failed to activate rsqsim-python-tools environment"
    echo "Available environments:"
    conda env list
    exit 1
fi

# Verify the environment is working
echo "=== Environment Verification ==="
echo "Conda environment: \$CONDA_DEFAULT_ENV"
echo "Python executable: \$(which python)"
echo "Python version: \$(python --version)"

# Test critical imports to make sure everything works
echo "Testing critical package imports..."
python -c "
import sys
print(f'Python path: {sys.executable}')

# Test all critical packages
critical_packages = ['pandas', 'numpy', 'scipy', 'geopy', 'matplotlib']
all_good = True

for package in critical_packages:
    try:
        module = __import__(package)
        version = getattr(module, '__version__', 'unknown')
        print(f'✅ {package} {version}')
    except ImportError as e:
        print(f'❌ {package}: FAILED - {e}')
        all_good = False

if all_good:
    print('🎉 All critical packages are working!')
else:
    print('❌ Some packages are missing!')
    sys.exit(1)
"

if [ \$? -ne 0 ]; then
    echo "❌ Package import test failed"
    echo "Environment details:"
    conda list | head -20
    exit 1
fi

echo "✅ Environment verification passed"
echo

unset SLURM_MEM_PER_CPU
unset SLURM_MEM_PER_GPU  
unset SLURM_MEM_PER_NODE

# FIXED: Set storage directories with proper environment variables
export RSQSIM_PROJECT_ROOT="$PROJECT_ROOT"
export RSQSIM_HOME="\$HOME/$RSQSIM_HOME_BASE"
export RSQSIM_WORK_DIR="$RSQSIM_WORK_BASE/\$USER/rsqsim_data"
export RSQSIM_PROJECT_DIR="$RSQSIM_PROJECT_BASE/\$USER"
export TMPDIR="$RSQSIM_WORK_BASE/\$USER/tmp"

# Set processing environment
export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK
export PYTHONPATH="\$RSQSIM_PROJECT_ROOT:\$PYTHONPATH"

echo "=== Storage Configuration ==="
echo "  Home (code):    \$RSQSIM_HOME"
echo "  Work (scratch): \$RSQSIM_WORK_DIR"
echo "  Project (long): \$RSQSIM_PROJECT_DIR"
echo "  Temp:           \$TMPDIR"
echo

# FIXED: Create base directory structure
echo "Creating base directory structure..."
mkdir -p "\$RSQSIM_WORK_DIR"/{data,output,logs}
mkdir -p "\$RSQSIM_PROJECT_DIR"/{rsqsim_results,rsqsim_archive}
mkdir -p "\$TMPDIR"

# CRITICAL FIX: Create the complete output folder structure
echo "=== Creating Complete Output Folder Structure ==="
echo "Creating folders for region: $region, mode: $mode"

# Define the complete folder structure
BASE_OUTPUT="\$RSQSIM_WORK_DIR/output"
REGION_DIRS=(
    "\$BASE_OUTPUT/$region"
    "\$BASE_OUTPUT/$region/$mode"
    "\$BASE_OUTPUT/$region/$mode/regional_summary"
    "\$BASE_OUTPUT/$region/$mode/visualizations"
    "\$BASE_OUTPUT/$region/$mode/visualizations/maps"
    "\$BASE_OUTPUT/$region/$mode/visualizations/plots"
    "\$BASE_OUTPUT/$region/$mode/visualizations/gis"
)

# Create each directory
for dir in "\${REGION_DIRS[@]}"; do
    if mkdir -p "\$dir"; then
        echo "✅ Created: \$dir"
    else
        echo "❌ Failed to create: \$dir"
        exit 1
    fi
done

echo "✅ Complete folder structure created successfully"

# Create symlinks if needed
if [ -d "\$RSQSIM_WORK_DIR/data" ] && [ ! -L "\$RSQSIM_HOME/data" ]; then
    echo "Creating data symlink..."
    ln -sf "\$RSQSIM_WORK_DIR/data" "\$RSQSIM_HOME/data"
fi

if [ -d "\$RSQSIM_WORK_DIR/output" ] && [ ! -L "\$RSQSIM_HOME/output" ]; then
    echo "Creating output symlink..."
    ln -sf "\$RSQSIM_WORK_DIR/output" "\$RSQSIM_HOME/output"
fi

# Change to project directory
cd "\$RSQSIM_PROJECT_ROOT"

# FIXED: Test the folder creation by running Python folder creation script
echo "=== Verifying Folder Structure with Python Script ==="
python -c "
import os
from pathlib import Path

# Test that our folder structure exists
base_output = os.environ.get('RSQSIM_WORK_DIR') + '/output'
region = '$region'
mode = '$mode'

required_dirs = [
    f'{base_output}/{region}',
    f'{base_output}/{region}/{mode}',
    f'{base_output}/{region}/{mode}/regional_summary',
    f'{base_output}/{region}/{mode}/visualizations',
    f'{base_output}/{region}/{mode}/visualizations/maps',
    f'{base_output}/{region}/{mode}/visualizations/plots',
    f'{base_output}/{region}/{mode}/visualizations/gis'
]

print('Checking required directories:')
all_exist = True
for dir_path in required_dirs:
    if os.path.exists(dir_path):
        print(f'✅ {dir_path}')
    else:
        print(f'❌ {dir_path}')
        all_exist = False

if all_exist:
    print('🎉 All required directories exist!')
else:
    print('❌ Some directories are missing!')
    import sys
    sys.exit(1)
"

if [ \$? -ne 0 ]; then
    echo "❌ Folder structure verification failed"
    exit 1
fi

echo "✅ Folder structure verification passed"

# Final verification before running simulation
echo "=== Final Pre-flight Check ==="
echo "Current directory: \$(pwd)"
echo "Conda environment: \$CONDA_DEFAULT_ENV"
echo "Python executable: \$(which python)"

# Check if main script exists
if [ ! -f "scripts/run_mc.py" ]; then
    echo "❌ Main script not found: scripts/run_mc.py"
    echo "Current directory contents:"
    ls -la
    exit 1
fi

echo "✅ Pre-flight check complete"
echo

# FIXED: Run the simulation with proper error handling and pattern support
echo "=== Starting Ground Motion Simulation ==="
echo "Region: $region"
echo "Mode: $mode"
echo "Config: ${config_file:-"default"}"
echo "Pattern: ${PATTERN:-"default"}"
echo "Expected output directory: \$RSQSIM_WORK_DIR/output/$region/$mode/"
echo


# FIXED: Always use direct execution instead of srun
# The Python code handles parallelism internally with ProcessPoolExecutor
echo "🚀 Executing simulation with direct command (no srun)"
echo "📌 This prevents multi-node result fragmentation"
echo

# Build the command
SIMULATION_CMD="python scripts/run_mc.py --region $region --mode $mode --log-level ERROR"
if [[ -n "$config_file" ]]; then
    SIMULATION_CMD="\$SIMULATION_CMD --config $config_file"
fi
if [[ -n "$PATTERN" ]]; then
    SIMULATION_CMD="\$SIMULATION_CMD --pattern '$PATTERN'"
    echo "🎯 Using specific pattern: $PATTERN"
fi

echo "🔍 Final command: \$SIMULATION_CMD"
echo

# CRITICAL FIX: Use eval instead of srun for all cases
# This ensures single-node execution with proper result collection
eval "\$SIMULATION_CMD"

# Log the execution method used
echo "✅ Simulation executed using direct eval (single-node mode)"


# Check exit status
SIMULATION_EXIT_CODE=\$?

# FIXED: Post-processing with better result verification
echo
echo "=== Post-processing ==="
if [ \$SIMULATION_EXIT_CODE -eq 0 ]; then
    echo "✅ Simulation completed successfully"
    
    # Verify that output files were actually created
    OUTPUT_DIR="\$RSQSIM_WORK_DIR/output/$region/$mode"
    echo "Checking for output files in: \$OUTPUT_DIR"
    
    # Count output files
    HAZARD_FILES=\$(find "\$OUTPUT_DIR" -name "hazard_results_*.txt" 2>/dev/null | wc -l)
    SUMMARY_FILES=\$(find "\$OUTPUT_DIR" -name "summary_*.csv" 2>/dev/null | wc -l)
    VIZ_FILES=\$(find "\$OUTPUT_DIR/visualizations" -type f 2>/dev/null | wc -l)
    
    echo "Output files found:"
    echo "  Hazard results: \$HAZARD_FILES"
    echo "  Summary CSVs: \$SUMMARY_FILES"
    echo "  Visualization files: \$VIZ_FILES"
    
    if [ \$HAZARD_FILES -gt 0 ] || [ \$SUMMARY_FILES -gt 0 ]; then
        echo "✅ Output files were created successfully"
        
        # Backup important results
        echo "Backing up results to project directory..."
        mkdir -p "\$RSQSIM_PROJECT_DIR/rsqsim_results/$region/$mode"
        
        # Use rsync for reliable copying
        if command -v rsync &> /dev/null; then
            rsync -av "\$OUTPUT_DIR/" "\$RSQSIM_PROJECT_DIR/rsqsim_results/$region/$mode/" 2>/dev/null || {
                echo "Rsync failed, trying cp..."
                cp -r "\$OUTPUT_DIR"/* "\$RSQSIM_PROJECT_DIR/rsqsim_results/$region/$mode/" 2>/dev/null || true
            }
        else
            cp -r "\$OUTPUT_DIR"/* "\$RSQSIM_PROJECT_DIR/rsqsim_results/$region/$mode/" 2>/dev/null || true
        fi
        
        echo "Results backed up to: \$RSQSIM_PROJECT_DIR/rsqsim_results/$region/$mode/"
    else
        echo "⚠️  No output files found - simulation may have failed silently"
        SIMULATION_EXIT_CODE=1
    fi
    
    # Create success summary
    cat > "\$RSQSIM_PROJECT_DIR/rsqsim_results/${job_name}_SUCCESS.txt" << SUMMARY
RSQSim Ground Motion Simulation - SUCCESS
========================================
Job ID: \$SLURM_JOB_ID
Region: $region
Mode: $mode
Pattern: ${PATTERN:-"default"}
Completed: \$(date)
Conda Environment: \$CONDA_DEFAULT_ENV
Python Path: \$(which python)

Output Files Created:
  Hazard Results: \$HAZARD_FILES files
  Summary CSVs: \$SUMMARY_FILES files
  Visualization Files: \$VIZ_FILES files

Output Locations:
  Primary: \$OUTPUT_DIR/
  Backup: \$RSQSIM_PROJECT_DIR/rsqsim_results/$region/$mode/

Final Storage Usage:
  Work: \$(du -sh \$RSQSIM_WORK_DIR 2>/dev/null | cut -f1 || echo "Unknown")
  Project: \$(du -sh \$RSQSIM_PROJECT_DIR 2>/dev/null | cut -f1 || echo "Unknown")
SUMMARY

else
    echo "❌ Simulation failed with exit code: \$SIMULATION_EXIT_CODE"
    
    # Check if any partial results exist
    OUTPUT_DIR="\$RSQSIM_WORK_DIR/output/$region/$mode"
    if [ -d "\$OUTPUT_DIR" ]; then
        echo "Checking for partial results..."
        find "\$OUTPUT_DIR" -type f -ls 2>/dev/null | head -10
    fi
    
    # Create error summary
    cat > "\$RSQSIM_PROJECT_DIR/rsqsim_results/${job_name}_ERROR.txt" << ERROR_SUMMARY
RSQSim Ground Motion Simulation - ERROR
======================================
Job ID: \$SLURM_JOB_ID
Region: $region
Mode: $mode
Pattern: ${PATTERN:-"default"}
Failed: \$(date)
Exit Code: \$SIMULATION_EXIT_CODE
Conda Environment: \$CONDA_DEFAULT_ENV
Python Path: \$(which python)

Check error log: $output_dir/${job_name}_\$SLURM_JOB_ID.err

Debugging info:
- Working directory: \$(pwd)
- Python version: \$(python --version 2>&1)
- Conda environment: \$(conda list | wc -l) packages installed
- Output directory exists: \$([ -d "\$OUTPUT_DIR" ] && echo "Yes" || echo "No")
ERROR_SUMMARY

fi

echo
echo "=== Final Summary ==="
echo "Exit code: \$SIMULATION_EXIT_CODE"
echo "Conda environment used: \$CONDA_DEFAULT_ENV"
echo "Results location: \$RSQSIM_PROJECT_DIR/rsqsim_results/$region/$mode/"
echo "Output directory: \$RSQSIM_WORK_DIR/output/$region/$mode/"
echo "Log files: $output_dir/"
echo "Job completed at: \$(date)"

exit \$SIMULATION_EXIT_CODE
EOF
}

# =============================================================================
# Main Script Logic - ENHANCED with folder creation and pattern support
# =============================================================================

# Parse command line arguments
if [[ $# -lt 2 ]]; then
    print_usage
    exit 1
fi

REGION=$1
MODE=$2
shift 2

# Set defaults
NODES=$DEFAULT_NODES
CORES_PER_NODE=$DEFAULT_CORES_PER_NODE
MEMORY=$DEFAULT_MEMORY
WALLTIME=$DEFAULT_WALLTIME
QUEUE=$DEFAULT_QUEUE
PROJECT_ACCOUNT=$PROJECT_ACCOUNT
CLUSTER_EMAIL=$CLUSTER_EMAIL
CONFIG_FILE=""
PATTERN=""  # Initialize pattern variable
DRY_RUN=false
CREATE_DIRS=false

# Parse options
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --nodes)
            NODES="$2"
            shift 2
            ;;
        --cores)
            CORES_PER_NODE="$2"
            shift 2
            ;;
        --memory)
            MEMORY="$2"
            shift 2
            ;;
        --time)
            WALLTIME="$2"
            shift 2
            ;;
        --queue)
            QUEUE="$2"
            shift 2
            ;;
        --account)
            PROJECT_ACCOUNT="$2"
            shift 2
            ;;
        --email)
            CLUSTER_EMAIL="$2"
            shift 2
            ;;
        --create-dirs)
            CREATE_DIRS=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --pattern)
            PATTERN="$2"
            shift 2
            ;;
        --help)
            print_usage
            exit 0
            ;;
        *)
            echo "❌ Error: Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

# Validate inputs
if [[ "$MODE" != "sequential" && "$MODE" != "random" ]]; then
    echo "❌ Error: MODE must be either 'sequential' or 'random'"
    exit 1
fi

# Auto-detect project root
if [[ -z "$RSQSIM_PROJECT_ROOT" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
    
    if [[ ! -f "$PROJECT_ROOT/scripts/run_mc.py" ]]; then
        PROJECT_ROOT="${HOME}/$RSQSIM_HOME_BASE"
        if [[ ! -f "$PROJECT_ROOT/scripts/run_mc.py" ]]; then
            echo "❌ Error: Could not find project root with run_mc.py"
            echo "Tried: $PROJECT_ROOT"
            exit 1
        fi
    fi
else
    PROJECT_ROOT="$RSQSIM_PROJECT_ROOT"
fi

# ENHANCED: Create folder structure if requested or if it doesn't exist
if [[ "$CREATE_DIRS" == "true" ]] || [[ "$DRY_RUN" == "false" ]]; then
    log_message "Creating folder structure before job submission..."
    create_folder_structure "$REGION" "$MODE"
fi

# Generate and submit job
log_message "Generating job script for BC cluster..."

JOB_SCRIPT="/tmp/rsqsim_${REGION}_${MODE}_$$.sh"
# FIXED: Pass pattern parameter to the function
generate_slurm_script "$REGION" "$MODE" "$CONFIG_FILE" > "$JOB_SCRIPT"
chmod +x "$JOB_SCRIPT"

if [[ "$DRY_RUN" == "true" ]]; then
    log_message "Job script (dry run):"
    echo "================================"
    cat "$JOB_SCRIPT"
    echo "================================"
else
    log_message "Submitting job..."
    echo "  Region: $REGION"
    echo "  Mode: $MODE"
    echo "  Pattern: ${PATTERN:-"default"}"
    echo "  Nodes: $NODES"
    echo "  Cores per node: $CORES_PER_NODE"
    echo "  Memory: $MEMORY"
    echo "  Queue: $QUEUE"
    echo "  Wall time: $WALLTIME"
    echo
    
    sbatch "$JOB_SCRIPT"
    
    if [[ $? -eq 0 ]]; then
        log_message "✅ Job submitted successfully!"
        echo
        echo "Monitor with:"
        echo "  squeue -u $USER"
        echo "  tail -f $PROJECT_ROOT/logs/${REGION}_${MODE}_gm_sim_*.out"
        echo
        echo "Expected output location:"
        echo "  $RSQSIM_WORK_BASE/$USER/rsqsim_data/output/$REGION/$MODE/"
    else
        log_message "❌ Job submission failed"
        exit 1
    fi
    
    rm -f "$JOB_SCRIPT"
fi

log_message "Script completed."