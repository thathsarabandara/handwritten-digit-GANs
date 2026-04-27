#!/usr/bin/env bash

# ------------------------------------------------------------
# Project setup script for gan_handwritten_digits
# ------------------------------------------------------------
# This script creates a virtual environment, installs required
# Python packages, and then offers the user a choice to either
# launch the Jupyter notebook for training or run the Streamlit
# demo application.
# ------------------------------------------------------------

set -e  # Exit on first error

# Helper function to print a separator line
separator() {
  echo "------------------------------------------------------------"
}

# ------------------------------------------------------------
# Step 1: Create virtual environment
# ------------------------------------------------------------
separator
echo "Creating virtual environment..."
PYTHON_EXEC=python3
if ! command -v $PYTHON_EXEC &> /dev/null; then
  echo "Error: $PYTHON_EXEC not found in PATH. Please install Python 3."
  exit 1
fi
$PYTHON_EXEC -m venv venv

# ------------------------------------------------------------
# Step 2: Activate venv and install requirements
# ------------------------------------------------------------
separator
echo "Activating virtual environment and installing dependencies..."
# shellcheck disable=SC1091
source venv/bin/activate
pip install --upgrade pip
if [ -f requirements.txt ]; then
  pip install -r requirements.txt
else
  echo "Warning: requirements.txt not found. Skipping package installation."
fi

# ------------------------------------------------------------
# Step 3: Prompt user for action
# ------------------------------------------------------------
separator
echo "Setup complete!"
PS3="Please enter your choice (1 or 2): "
options=(
  "1) Open Jupyter notebook (gan_training.ipynb)"
  "2) Launch Streamlit app (app/app.py)"
)
select opt in "${options[@]}"; do
  case $REPLY in
    1)
      echo "Launching Jupyter notebook..."
      jupyter notebook notebooks/gan_training.ipynb
      break
      ;;
    2)
      echo "Launching Streamlit app..."
      streamlit run app/app.py
      break
      ;;
    *)
      echo "Invalid option. Please choose 1 or 2."
      ;;
  esac
done

separator
echo "Done."
