#!/bin/bash

# Script to run all experiments and generate plots
# Usage: ./run_all_experiments.sh [--clean]
# 
# Options:
#   --clean    Remove all files in experiments/results before running experiments

set -e  # Exit on any error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse command line arguments
CLEAN_RESULTS=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --clean)
            CLEAN_RESULTS=true
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [--clean]"
            echo ""
            echo "Options:"
            echo "  --clean    Remove all files in experiments/results before running experiments"
            echo "  -h, --help Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Check if we're in the right directory (new structure)
if [[ ! -d "experiments" || ! -d "experiments/scripts" || ! -d "experiments/plot-rendering-code" ]]; then
    print_error "experiments directory structure not found. Please run this script from the project root."
    exit 1
fi

# Initialize conda and activate environment
eval "$(conda shell.bash hook)"
conda activate ordinal-similarity-metrics
print_success "Conda environment activated"

# Clean results directory if requested (new structure)
if [[ "$CLEAN_RESULTS" == true ]]; then
    print_warning "Cleaning results directory..."
    if [[ -d "experiments/results" ]]; then
        rm -rf experiments/results/*
        print_success "Results directory cleaned"
    else
        print_warning "Results directory does not exist, creating it..."
        mkdir -p experiments/results
    fi
fi

# Ensure results directory exists
mkdir -p experiments/results

print_status "Starting experiment pipeline..."

# Run all experiment scripts (new structure)
print_status "Running experiment scripts..."
SCRIPTS_DIR="experiments/scripts"
SCRIPT_COUNT=0
FAILED_SCRIPTS=()

# Run all plot rendering scripts (new structure)
print_status "Running plot rendering scripts..."
PLOT_DIR="experiments/plot-rendering-code"
PLOT_COUNT=0
FAILED_PLOTS=()

for script in "$SCRIPTS_DIR"/*.py; do
    if [[ -f "$script" ]]; then
        script_name=$(basename "$script")
        print_status "Running $script_name..."
        
        if python3 "$script"; then
            print_success "Completed $script_name"
            SCRIPT_COUNT=$((SCRIPT_COUNT+1))
        else
            print_error "Failed to run $script_name"
            FAILED_SCRIPTS+=("$script_name")
        fi
    fi
done

if [[ ${#FAILED_SCRIPTS[@]} -gt 0 ]]; then
    print_error "Some experiment scripts failed:"
    for failed_script in "${FAILED_SCRIPTS[@]}"; do
        echo "  - $failed_script"
    done
    print_warning "Continuing with plot rendering..."
else
    print_success "All $SCRIPT_COUNT experiment scripts completed successfully"
fi

for plot_script in "$PLOT_DIR"/*.py; do
    if [[ -f "$plot_script" ]]; then
        plot_name=$(basename "$plot_script")
        print_status "Running $plot_name..."
        
        if python3 "$plot_script"; then
            print_success "Completed $plot_name"
            PLOT_COUNT=$((PLOT_COUNT+1))
        else
            print_error "Failed to run $plot_name"
            FAILED_PLOTS+=("$plot_name")
        fi
    fi
done

# Final summary
echo ""
print_status "=== EXPERIMENT PIPELINE SUMMARY ==="
echo "Experiment scripts run: $SCRIPT_COUNT"
echo "Plot rendering scripts run: $PLOT_COUNT"

if [[ ${#FAILED_SCRIPTS[@]} -gt 0 || ${#FAILED_PLOTS[@]} -gt 0 ]]; then
    print_warning "Some scripts failed:"
    if [[ ${#FAILED_SCRIPTS[@]} -gt 0 ]]; then
        echo "Failed experiment scripts: ${FAILED_SCRIPTS[*]}"
    fi
    if [[ ${#FAILED_PLOTS[@]} -gt 0 ]]; then
        echo "Failed plot scripts: ${FAILED_PLOTS[*]}"
    fi
    exit 1
else
    print_success "All scripts completed successfully!"
    print_status "Results are available in experiments/results/ and plots in plots/"
fi
