#!/bin/bash

# This script sets up the Python virtual environment for the project.
# It creates a virtual environment and installs the required packages.

# Exit immediately if a command exits with a non-zero status.
set -e

VENV_DIR="venv"
PYTHON_CMD="python3"

echo "--- Setting up Python virtual environment ---"

# Check if python3 is installed
if ! command -v $PYTHON_CMD &> /dev/null
then
    echo "Error: '$PYTHON_CMD' could not be found. Please install Python 3."
    exit 1
fi

# Check if the venv module is available
if ! $PYTHON_CMD -c "import venv" &> /dev/null; then
    echo "Error: The 'venv' module is not available for your Python installation."
    echo "On Debian/Ubuntu, you can install it with: sudo apt install python3-venv"
    echo "On other systems, please install the appropriate package for Python virtual environments."
    exit 1
fi

# Create the virtual environment if the pip executable doesn't exist.
# This makes the script robust against an incomplete venv directory.
if [ ! -f "$VENV_DIR/bin/pip" ]; then
    echo "Creating/re-creating virtual environment in '$VENV_DIR'..."
    $PYTHON_CMD -m venv $VENV_DIR
fi

echo "Installing dependencies from requirements.txt..."
./$VENV_DIR/bin/pip install -r requirements.txt

echo -e "\n--- Setup complete! ---\n"
echo "To activate the virtual environment, run:"
echo "source $VENV_DIR/bin/activate"