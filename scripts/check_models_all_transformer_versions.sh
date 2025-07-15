#!/bin/bash

# Script to test nnterp with different transformers versions
set -e

# Function to clean up temporary environment
cleanup() {
    if [ -n "$TEMP_VENV" ] && [ -d "$TEMP_VENV" ]; then
        echo "Cleaning up temporary environment: $TEMP_VENV"
        rm -rf "$TEMP_VENV"
    fi
}

# Set up cleanup trap
trap cleanup EXIT

# Create temporary virtual environment
TEMP_VENV=$(mktemp -d)
echo "Creating temporary virtual environment at: $TEMP_VENV"

# Create and activate virtual environment
python -m venv "$TEMP_VENV"
source "$TEMP_VENV/bin/activate"

# Upgrade pip
pip install --upgrade pip

# Install all dependencies from pyproject.toml
pip install uv
uv sync --all-extras --active
uv pip install flash-attn --no-build-isolation



# Get all available transformers versions and extract only the latest patch for each minor version
echo "Getting transformers versions..."
all_versions=$(pip index versions transformers | grep -o '[0-9]\+\.[0-9]\+\.[0-9]\+' | sort -V)

# Extract latest patch version for each x.y combination
declare -A latest_patches
for version in $all_versions; do
    major_minor=$(echo "$version" | grep -o '^[0-9]\+\.[0-9]\+')
    latest_patches["$major_minor"]="$version"
done

# Convert to sorted array of versions to test (newest first)
test_versions=()
for major_minor in $(printf '%s\n' "${!latest_patches[@]}" | sort -Vr); do
    test_versions+=("${latest_patches[$major_minor]}")
done

echo "Will test the following versions (latest patch for each minor version):"
printf '%s\n' "${test_versions[@]}"

# Array to store test results
declare -a results

for version in "${test_versions[@]}"; do
    # Only test versions 4.28.0 and above
    major=$(echo "$version" | cut -d. -f1)
    minor=$(echo "$version" | cut -d. -f2)
    if [ "$major" -lt 4 ] || ([ "$major" -eq 4 ] && [ "$minor" -lt 36 ]); then
        continue
    fi
    
    echo "Testing transformers version: $version"
    
    # Install specific transformers version
    if uv pip install "transformers==$version" --force-reinstall; then
        echo "Successfully installed transformers $version"
        
        # Run tests
        python tests/utils.py
        results+=("$version: RAN")
    else
        echo "⚠️  Failed to install transformers $version"
        results+=("$version: INSTALL_FAILED")
    fi
    
    echo "----------------------------------------"
done

# Print summary
echo ""
echo "==================== TEST SUMMARY ===================="
for result in "${results[@]}"; do
    echo "$result"
done
echo "======================================================="

# Cleanup will be handled by the trap
