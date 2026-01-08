#!/bin/bash
# File: submit_parallel_windows.sh
# ENHANCED VERSION: Accepts command line arguments for region and mode

set -e

# =============================================================================
# COMMAND LINE ARGUMENT PARSING
# =============================================================================

# Function to show usage
show_usage() {
    cat << EOF
Usage: $0 [REGION] [MODE] [OPTIONS]

Arguments:
  REGION          Region name (e.g., los_angeles, san_francisco)
  MODE            Window mode (sequential or random)

Options:
  --parallel N    Number of parallel jobs (default: 8)
  --cores N       Cores per job (default: 16)
  --memory SIZE   Memory per job (default: 32GB)
  --time TIME     Wall time per job (default: 12:00:00)
  --queue QUEUE   Queue/partition (default: short)
  --help          Show this help

Examples:
  $0 los_angeles sequential
  $0 san_francisco random
  $0 los_angeles sequential --parallel 4 --cores 48
  $0 san_francisco sequential --memory 64GB --time 24:00:00

If no arguments provided, uses defaults from script.
EOF
}

# =============================================================================
# DEFAULT CONFIGURATION
# =============================================================================
DEFAULT_REGION="los_angeles"
DEFAULT_MODE="sequential"
DEFAULT_MAX_PARALLEL=30
DEFAULT_CORES_PER_JOB=48
DEFAULT_MEMORY_PER_JOB="64GB"
DEFAULT_TIME_PER_JOB="2-00:00:00"
DEFAULT_QUEUE="medium"

# =============================================================================
# PARSE COMMAND LINE ARGUMENTS
# =============================================================================

# Set defaults
REGION="$DEFAULT_REGION"
MODE="$DEFAULT_MODE"
MAX_PARALLEL="$DEFAULT_MAX_PARALLEL"
CORES_PER_JOB="$DEFAULT_CORES_PER_JOB"
MEMORY_PER_JOB="$DEFAULT_MEMORY_PER_JOB"
TIME_PER_JOB="$DEFAULT_TIME_PER_JOB"
QUEUE="$DEFAULT_QUEUE"

# Parse positional arguments
if [ $# -ge 1 ]; then
    if [ "$1" = "--help" ] || [ "$1" = "-h" ]; then
        show_usage
        exit 0
    fi
    REGION="$1"
    shift
fi

if [ $# -ge 1 ]; then
    MODE="$1"
    shift
fi

# Parse optional arguments
while [ $# -gt 0 ]; do
    case "$1" in
        --parallel)
            MAX_PARALLEL="$2"
            shift 2
            ;;
        --cores)
            CORES_PER_JOB="$2"
            shift 2
            ;;
        --memory)
            MEMORY_PER_JOB="$2"
            shift 2
            ;;
        --time)
            TIME_PER_JOB="$2"
            shift 2
            ;;
        --queue)
            QUEUE="$2"
            shift 2
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            echo "❌ Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# =============================================================================
# VALIDATE INPUTS
# =============================================================================

# Validate mode
if [ "$MODE" != "sequential" ] && [ "$MODE" != "random" ]; then
    echo "❌ Error: MODE must be 'sequential' or 'random', got: $MODE"
    exit 1
fi

# Validate region (basic check)
if [ -z "$REGION" ]; then
    echo "❌ Error: REGION cannot be empty"
    exit 1
fi

# Set windows directory based on arguments
WINDOWS_DIR="/scratch/$USER/rsqsim_data/data/Catalog_4983/windows/$REGION/$MODE"

echo "🚀 ENHANCED Parallel Window Processing Setup"
echo "Region: $REGION"
echo "Mode: $MODE"
echo "Windows dir: $WINDOWS_DIR"
echo "Max parallel jobs: $MAX_PARALLEL"
echo "Cores per job: $CORES_PER_JOB"
echo "Memory per job: $MEMORY_PER_JOB"
echo "Time per job: $TIME_PER_JOB"
echo "Queue: $QUEUE"
echo ""

# =============================================================================
# VALIDATION AND SETUP
# =============================================================================

# Check if windows directory exists
if [ ! -d "$WINDOWS_DIR" ]; then
    echo "❌ Windows directory not found: $WINDOWS_DIR"
    echo ""
    echo "💡 Available regions and modes:"
    BASE_WINDOWS_DIR="/scratch/$USER/rsqsim_data/data/Catalog_4983/windows"
    if [ -d "$BASE_WINDOWS_DIR" ]; then
        find "$BASE_WINDOWS_DIR" -type d -name "*sequential*" -o -name "*random*" | sort
    fi
    exit 1
fi

# Find window files
WINDOW_FILES=($(find "$WINDOWS_DIR" -name "*.csv" | grep -v checkpoint | sort))
TOTAL_WINDOWS=${#WINDOW_FILES[@]}

if [ $TOTAL_WINDOWS -eq 0 ]; then
    echo "❌ No CSV files found in $WINDOWS_DIR"
    ls -la "$WINDOWS_DIR"
    exit 1
fi

echo "✅ Found $TOTAL_WINDOWS window files"
echo "First 5 files:"
for i in {0..4}; do
    if [ $i -lt $TOTAL_WINDOWS ]; then
        echo "  $(basename "${WINDOW_FILES[$i]}")"
    fi
done
if [ $TOTAL_WINDOWS -gt 5 ]; then
    echo "  ... and $((TOTAL_WINDOWS - 5)) more"
fi
echo ""

# Create tracking directory
TRACKING_DIR="/scratch/$USER/rsqsim_jobs/$REGION/$MODE"
mkdir -p "$TRACKING_DIR"

# =============================================================================
# SUBMIT INDIVIDUAL WINDOW JOBS
# =============================================================================

JOB_IDS=()
echo "📤 Submitting individual window jobs..."

for ((i=0; i<$TOTAL_WINDOWS; i++)); do
    WINDOW_FILE="${WINDOW_FILES[$i]}"
    WINDOW_NAME=$(basename "$WINDOW_FILE")
    
    # Extract window number for job naming
    WINDOW_NUM=$(echo "$WINDOW_NAME" | grep -o 'window_[0-9]*' | grep -o '[0-9]*' || echo "$((i+1))")
    JOB_NAME="${REGION}_${MODE}_w${WINDOW_NUM}"
    
    echo "  Submitting job $((i+1))/$TOTAL_WINDOWS: $WINDOW_NAME"
    
    # FIXED: Submit individual window job with all parameters including --time
    JOB_OUTPUT=$(sbatch \
        --job-name="$JOB_NAME" \
        --nodes=1 \
        --ntasks=1 \
        --ntasks-per-node=$CORES_PER_JOB \
        --mem=$MEMORY_PER_JOB \
        --time=$TIME_PER_JOB \
        --partition=$QUEUE \
        --output="$TRACKING_DIR/${JOB_NAME}_%j.out" \
        --error="$TRACKING_DIR/${JOB_NAME}_%j.err" \
        scripts/run_on_cluster.sh "$REGION" "$MODE" \
        --cores "$CORES_PER_JOB" \
        --memory "$MEMORY_PER_JOB" \
        --time "$TIME_PER_JOB" \
        --queue "$QUEUE" \
        --pattern "$WINDOW_NAME")
    
    # Extract job ID
    JOB_ID=$(echo "$JOB_OUTPUT" | grep -o '[0-9]*')
    JOB_IDS+=("$JOB_ID")
    
    echo "    Job ID: $JOB_ID"
    
    # Limit concurrent jobs
    if (( (i + 1) % MAX_PARALLEL == 0 )) && (( i + 1 < TOTAL_WINDOWS )); then
        echo ""
        echo "⏳ Submitted $MAX_PARALLEL jobs. Waiting 30s before next batch..."
        sleep 30
        echo ""
    fi
done

# =============================================================================
# SAVE JOB TRACKING INFORMATION
# =============================================================================

# Save job IDs
echo "${JOB_IDS[@]}" > "$TRACKING_DIR/job_ids.txt"

# Save comprehensive job information
cat > "$TRACKING_DIR/job_info.txt" << EOF
REGION=$REGION
MODE=$MODE
TOTAL_WINDOWS=$TOTAL_WINDOWS
SUBMITTED_TIME="$(date)"
WINDOWS_DIR=$WINDOWS_DIR
OUTPUT_BASE_DIR=/scratch/$USER/rsqsim_data/output/$REGION/$MODE
TRACKING_DIR=$TRACKING_DIR
CORES_PER_JOB=$CORES_PER_JOB
MEMORY_PER_JOB=$MEMORY_PER_JOB
TIME_PER_JOB=$TIME_PER_JOB
QUEUE=$QUEUE
EOF

echo ""
echo "✅ All $TOTAL_WINDOWS window jobs submitted!"
echo "📊 Job IDs: ${JOB_IDS[@]}"
echo "📂 Tracking dir: $TRACKING_DIR"

# =============================================================================
# CREATE MONITORING SCRIPT
# =============================================================================

cat > "$TRACKING_DIR/check_status.sh" << 'EOFSCRIPT'
#!/bin/bash
# Monitor job progress

TRACKING_DIR="$(dirname "$0")"
source "$TRACKING_DIR/job_info.txt"

echo "==================================="
echo "RSQSim Job Status Monitor"
echo "==================================="
echo "Region: $REGION"
echo "Mode: $MODE"
echo "Total Windows: $TOTAL_WINDOWS"
echo "Optimized: 64 cores, 48GB, short queue"
echo ""

# Check SLURM queue
echo "📊 Current Queue Status:"
squeue -u $USER --format="%.10i %.15j %.8T %.10M %.6D %.4C %.10m" | head -20

echo ""
echo "📈 Job Statistics:"
RUNNING=$(squeue -u $USER -t R | wc -l)
PENDING=$(squeue -u $USER -t PD | wc -l)
echo "  Running: $RUNNING"
echo "  Pending: $PENDING"

echo ""
echo "💾 Completed Windows:"
OUTPUT_DIR="/scratch/$USER/rsqsim_data/output/$REGION/$MODE"
if [ -d "$OUTPUT_DIR" ]; then
    COMPLETED=$(find "$OUTPUT_DIR" -name "hazard_results_*.txt" 2>/dev/null | wc -l)
    echo "  $COMPLETED / $TOTAL_WINDOWS windows completed"
    
    if [ $COMPLETED -gt 0 ]; then
        echo ""
        echo "📁 Recent completions:"
        find "$OUTPUT_DIR" -name "hazard_results_*.txt" -printf "%T@ %p\n" | sort -n | tail -5 | while read timestamp file; do
            echo "  $(basename "$file")"
        done
    fi
else
    echo "  Output directory not yet created"
fi

echo ""
echo "🔍 Recent log entries:"
find "$TRACKING_DIR" -name "*.out" -mmin -10 2>/dev/null | head -3 | while read logfile; do
    echo "  From: $(basename "$logfile")"
    tail -3 "$logfile" 2>/dev/null | sed 's/^/    /'
    echo ""
done

echo "==================================="
echo "💡 Commands:"
echo "  Watch queue: watch -n 30 squeue -u \$USER"
echo "  Cancel all: scancel \$(cat $TRACKING_DIR/job_ids.txt)"
echo "  Check logs: ls -lth $TRACKING_DIR/*.out | head"
echo "==================================="
EOFSCRIPT

chmod +x "$TRACKING_DIR/check_status.sh"

# =============================================================================
# FINAL INSTRUCTIONS
# =============================================================================

echo ""
echo "🔍 Monitor progress:"
echo "  squeue -u $USER"
echo "  $TRACKING_DIR/check_status.sh"
echo ""
echo "⏭️  After completion, run multi-window analysis:"
echo "  $TRACKING_DIR/submit_multiwindow.sh"
echo ""
echo "📂 Results will be saved to:"
echo "  /scratch/$USER/rsqsim_data/output/$REGION/$MODE/visualizations/multi_window_comparisons/"
echo ""
echo "✅ Setup complete!"