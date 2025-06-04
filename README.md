# SUS Second Global Assignment

This project implements and compares a Deep Q-Network (DQN) agent against baseline Q-learning and naive agents for the given reinforcement learning task. The primary goal is to demonstrate a functional DQN agent and its evaluation.

## Table of Contents

- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Quick Start: Automated Setup and Run](#quick-start-automated-setup-and-run)
- [Manual Operation](#manual-operation)
  - [Environment Setup](#environment-setup)
  - [Training Individual Agents](#training-individual-agents)
  - [Evaluating Agents](#evaluating-agents)

## Project Structure

```
├── agents
│   ├── dqn_agent                   # trained dqn agent (project solution)
│   │   ├── dqn_agent.py
│   │   ├── dqn_model.pth           # pretrained model
│   │   ├── dqn_model.py
│   │   └── train_dqn.py
│   ├── naive_agent                 # naive baseline
│   │   └── my_rl_agent.py
│   └── q_learning_agent            # q-learning baseline
│       ├── q_learning_agent.py
│       ├── q_table.pkl
│       └── train_q_learning.py
├── results                         # results presented in the report
│   ├── ...
├── README.md
├── evaluate_agents.py              # evaluate dqn_agent against baselines
├── reinforcement_task.py           # provided - contains evaluate_agent() function
├── requirements.txt
└── setup.sh                        # automated setup

```


## Prerequisites

Ensure the following are installed on your system:

*   **Python:** Version 3.13 recommended.
*   **pip:** Python package installer.
*   **venv:** Python module for virtual environments.
*   The `requirements.txt` file must be present in the project root directory.

## Quick Start: Automated Setup and Run

The `setup.sh` script is provided to automate the entire process: virtual environment creation, dependency installation, DQN agent training, and evaluation of all agents.

1.  **Navigate to the project root directory in your terminal.**
2.  **Make the script executable (if necessary):**
    ```bash
    chmod +x setup.sh
    ```
3.  **Run the script:**
    ```bash
    bash setup.sh
    ```
    Alternatively:
    ```bash
    ./setup.sh
    ```

**What the script does:**
*   Checks for Python and OS, then creates a virtual environment in `./venv`.
*   Installs all packages from `requirements.txt` into the venv.
*   **Trains the DQN agent** (using `agents.dqn_agent.train_dqn`).
*   **Evaluates all agents** (DQN, Q-learning, Naive) using `evaluate_agents.py`.
*   Provides instructions on how to activate the `venv` for manual use later.

This is the recommended way to run the project for a full demonstration.


## Prerequisites

Ensure the following are installed on your system:

*   **Python:** Version 3.7 or higher.
*   **pip:** Python package installer.
*   **venv:** Python module for virtual environments.
*   The `requirements.txt` file must be present in the project root directory.

## Quick Start: Automated Setup and Run

The `setup.sh` script is provided to automate the entire process: virtual environment creation, dependency installation, DQN agent training, and evaluation of all agents.

1.  **Navigate to the project root directory in your terminal.**
2.  **Make the script executable (if necessary):**
    ```bash
    chmod +x setup.sh
    ```
3.  **Run the script:**
    ```bash
    ./setup.sh
    ```
    Alternatively:
    ```bash
    bash setup.sh
    ```

**What the script does:**
*   Checks for Python and OS, then creates a virtual environment in `./venv`.
*   Installs all packages from `requirements.txt` into the venv.
*   **Trains the DQN agent** (using `agents.dqn_agent.train_dqn`).
*   **Evaluates all agents** (DQN, Q-learning, Naive) using `evaluate_agents.py`.
*   Provides instructions on how to activate the `venv` for manual use later.

This is the recommended way to run the project for a full demonstration.

## Manual Operation

If you wish to perform steps individually (e.g., re-train a specific agent or only run evaluation), follow these instructions.

### Environment Setup

First, ensure the virtual environment is set up and activated. If you haven't run `setup.sh` before, or want to create the venv manually:

1.  **Create the virtual environment (from project root):**
    ```bash
    python3 -m venv venv  # Or `python -m venv venv`
    ```
2.  **Activate the virtual environment:**
    *   **Linux/macOS/Git Bash:**
        ```bash
        source venv/bin/activate
        ```
    *   **Windows (Command Prompt/PowerShell):**
        ```bash
        venv\Scripts\activate
        ```
    *   **Windows (Git Bash/MSYS):**
        ```bash
        source venv/Scripts/activate
        ```
    *(Your prompt should change to indicate the active venv, e.g., `(venv) ...`)*

3.  **Install dependencies (if not already installed by `setup.sh`):**
    ```bash
    python -m pip install --upgrade pip
    python -m pip install -r requirements.txt
    ```

### Training Individual Agents

Ensure the virtual environment is activated.

*   **DQN Agent (Re-training):**
    ```bash
    python -m agents.dqn_agent.train_dqn
    ```
    This will save the trained model to `agents/dqn_agent/dqn_model.pth`.

*   **Q-learning Agent (Re-training):**
    ```bash
    python -m agents.q_learning_agent.train_q_learning
    ```
    This will save the trained Q-table to `agents/q_learning_agent/q_table.pkl`.

### Evaluating Agents

Ensure the virtual environment is activated. This script uses the pre-trained or most recently trained models.

```bash
python -m evaluate_agents
```
This will compare the performance of the DQN agent, Q-learning agent, and the naive agent.