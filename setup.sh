#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "Starting setup..."

# --- Configuration ---
VENV_DIR="venv"
REQUIREMENTS_FILE="requirements.txt"
TRAIN_MODULE="agents.dqn_agent.train_dqn"
EVAL_MODULE="evaluate_agents"

# --- Determine Python command ---
PYTHON_CMD=""
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo "ERROR: Neither 'python3' nor 'python' found in PATH. Please install Python."
    exit 1
fi
echo "INFO: Using '$PYTHON_CMD' for Python commands."

# --- OS Detection and Path Configuration ---
VENV_PYTHON_IN_VENV_PATH="" # Path to python executable INSIDE the venv
ACTIVATE_SCRIPT_PATH=""
USER_ACTIVATE_INSTRUCTION_NIX="source $VENV_DIR/bin/activate"
USER_ACTIVATE_INSTRUCTION_WIN_CMD="$VENV_DIR\\\\Scripts\\\\activate"
USER_ACTIVATE_INSTRUCTION_WIN_BASH="source $VENV_DIR/Scripts/activate"

if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" ]]; then
    echo "INFO: Detected Windows-like environment (msys/cygwin)."
    ACTIVATE_SCRIPT_PATH="$VENV_DIR/Scripts/activate"
    VENV_PYTHON_IN_VENV_PATH="$VENV_DIR/Scripts/python.exe"
    USER_ACTIVATE_INSTRUCTIONS="For CMD/PowerShell: $USER_ACTIVATE_INSTRUCTION_WIN_CMD\n    For Git Bash/MSYS: $USER_ACTIVATE_INSTRUCTION_WIN_BASH"
else
    echo "INFO: Detected Linux/macOS-like environment."
    ACTIVATE_SCRIPT_PATH="$VENV_DIR/bin/activate"
    VENV_PYTHON_IN_VENV_PATH="$VENV_DIR/bin/python"
    USER_ACTIVATE_INSTRUCTIONS="Run: $USER_ACTIVATE_INSTRUCTION_NIX"
fi

# --- Check for requirements file ---
if [ ! -f "$REQUIREMENTS_FILE" ]; then
    echo "ERROR: '$REQUIREMENTS_FILE' not found in the current directory ($(pwd))."
    echo "Please create it before running this script."
    exit 1
fi

# --- Virtual Environment Setup ---
if [ -d "$VENV_DIR" ]; then
    echo "INFO: Existing '$VENV_DIR' directory found."
    # Default to 'n' if user just hits Enter
    read -r -p "Do you want to remove it and create a fresh one? (y/N): " response
    if [[ "${response,,}" =~ ^(y|yes)$ ]]; then
        echo -e "\n==> Removing existing '$VENV_DIR'..."
        rm -rf "$VENV_DIR"
    else
        echo "INFO: Reusing existing '$VENV_DIR'. If issues occur, try removing it manually."
    fi
fi

if [ ! -d "$VENV_DIR" ]; then
    echo -e "\n==> Creating virtual environment in '$VENV_DIR'..."
    "$PYTHON_CMD" -m venv "$VENV_DIR"
fi

echo -e "\n==> Activating virtual environment (for this script session)..."
# shellcheck source=/dev/null
source "$ACTIVATE_SCRIPT_PATH"

echo -e "\n==> Upgrading pip..."
# Use the Python interpreter from the venv to run pip as a module for the upgrade.
python -m pip install --upgrade pip

echo -e "\n==> Installing packages from $REQUIREMENTS_FILE..."
python -m pip install -r "$REQUIREMENTS_FILE"

echo -e "\n==> Starting training module for DQN agent '$TRAIN_MODULE'..."
python -m "$TRAIN_MODULE"
echo -e "\n==> Training complete."

echo -e "\n==> Starting evaluation module '$EVAL_MODULE'..."
python -m "$EVAL_MODULE"
echo -e "\n==> Evaluation complete."

echo "---------------------------------------------------------------------"
echo "Setup and training finished successfully!"
echo "To work in the environment manually, activate it in your terminal:"
echo -e "    $USER_ACTIVATE_INSTRUCTIONS"
echo ""
echo "Then you can run commands like:"
echo "    python -m evaluate_agents"
echo "---------------------------------------------------------------------"