import gymnasium as gym
import numpy as np
import torch
import pickle

from pathlib import Path

from reinforcement_task import evaluate_agent

from agents.naive_agent.my_rl_agent import MyRLAgent
from agents.q_learning_agent.q_learning_agent import QLearningAgent
from agents.dqn_agent.dqn_agent import DQNAgent


SCRIPT_DIR = Path(__file__).resolve().parent

class TrainedQLearningAgent:
    def __init__(self, q_table_path=None):
        if q_table_path is None:
            q_table_path = SCRIPT_DIR / "agents" / "q_learning_agent" / "q_table.pkl"
        else:
            q_table_path = Path(q_table_path)

        if not q_table_path.exists():
            raise FileNotFoundError(f"Q-table not found at: {q_table_path}")

        with open(q_table_path, "rb") as f:
            q_table = pickle.load(f)
        self.agent = QLearningAgent(q_table=q_table)

    def act(self, state):
        return self.agent.act(state)

if __name__ == "__main__":
    print("\nEvaluating a very weak agent (always picks action 0):")
    evaluate_agent(lambda: MyRLAgent())

    print("\nEvaluating the trained Q-learning agent (baseline):")
    evaluate_agent(lambda: TrainedQLearningAgent())

    print("\nEvaluating DQN agent:")
    evaluate_agent(lambda: DQNAgent(model_path=SCRIPT_DIR / "agents" / "dqn_agent" / "dqn_model.pth"))