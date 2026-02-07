#!/bin/bash
# =============================================================================
# GitHub Repository Setup Script
# Wind Power Ramp Severity Indices (ECSI)
# =============================================================================

echo "========================================"
echo "Setting up GitHub Repository for ECSI"
echo "========================================"

# Navigate to repository directory
cd wind-ramp-severity-indices

# Initialize git repository
echo ""
echo "Step 1: Initializing Git repository..."
git init

# Configure git (replace with your info)
echo ""
echo "Step 2: Configuring Git (update with your info)..."
git config user.name "Julian Cardenas-Barrera"
git config user.email "your.email@unb.ca"

# Add all files
echo ""
echo "Step 3: Adding files to repository..."
git add .

# Create initial commit
echo ""
echo "Step 4: Creating initial commit..."
git commit -m "Initial commit: Wind Power Ramp Severity Indices (ECSI)

This repository contains the implementation and validation code for:

'Beyond Magnitude and Rate: Shape-Based Severity Indices for Wind Power 
Ramp Events with Validated Unique Information Content'

Novel indices:
- RAI: Ramp Acceleration Index (90.1% unique variance)
- RSCI: Ramp Shape Complexity Index (54.0% unique variance)
- OSI: Operational Stress Index (asymmetric operational risk)
- GIP: Grid Impact Potential (context-dependent severity)
- ECSI: Enhanced Composite (50.7% unique variance, 15x improvement)

Includes:
- Core Python implementation (src/)
- Six-test validation framework
- Demonstration notebooks
- Sample data for testing"

# Instructions for GitHub
echo ""
echo "========================================"
echo "NEXT STEPS (Manual)"
echo "========================================"
echo ""
echo "1. Create a new repository on GitHub:"
echo "   - Go to https://github.com/new"
echo "   - Repository name: wind-ramp-severity-indices"
echo "   - Description: Novel shape-based severity indices for wind power ramp events"
echo "   - Choose: Public (for paper supplementary material)"
echo "   - DO NOT initialize with README, .gitignore, or license"
echo ""
echo "2. Connect to GitHub and push:"
echo "   git remote add origin https://github.com/YOUR_USERNAME/wind-ramp-severity-indices.git"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "3. (Optional) Create a release:"
echo "   - Go to your repository > Releases > Create a new release"
echo "   - Tag: v1.0.0"
echo "   - Title: v1.0.0 - Initial Release"
echo "   - Description: First release accompanying paper submission"
echo ""
echo "4. (Optional) Get a DOI via Zenodo:"
echo "   - Go to https://zenodo.org"
echo "   - Connect your GitHub account"
echo "   - Enable the repository"
echo "   - Create a release to trigger DOI minting"
echo ""
echo "========================================"
echo "Repository setup complete!"
echo "========================================"
