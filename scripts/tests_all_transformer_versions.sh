#!/bin/bash

# Script to test nnterp with different transformers versions
set -e

# Function to clean up temporary environment
cleanup() {
    if [ -n "$TEMP_DIR" ] && [ -d "$TEMP_DIR" ]; then
        echo "Cleaning up temporary environment: $TEMP_DIR"
        rm -rf "$TEMP_DIR"
    fi
}

# Set up cleanup trap
trap cleanup EXIT

# create a temporary directory
TEMP_DIR=$(mktemp -d)

# Copy source code to temporary location
cp -r ./nnterp "$TEMP_DIR/"
cp pyproject.toml "$TEMP_DIR/"
cd "$TEMP_DIR"

# Install all dependencies from pyproject.toml
uv venv
uv sync
uv pip install flash-attn --no-build-isolation
uv pip install pytest protobuf sentencepiece
source .venv/bin/activate


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

set +e
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
        
        make clean ; python -m pytest nnterp/tests --cache-clear  --tb=short
        exitcode=$?
        if [ $exitcode -le 1 ]; then
            echo "✅ Tests RAN for transformers $version"
            results+=("$version: PASSED")
        else
            echo "❌ Tests FAILED to run for transformers $version"
            results+=("$version: FAILED_TO_RUN")
        fi
    else
        echo "⚠️  Failed to install transformers $version"
        results+=("$version: INSTALL_FAILED")
    fi
    
    echo "----------------------------------------"
done

set -e

# Print summary
echo ""
echo "==================== TEST SUMMARY ===================="
for result in "${results[@]}"; do
    echo "$result"
done
echo "======================================================="

# Cleanup will be handled by the trap
