import torch
import numpy as np
from agents.dqn_agent.dqn_model import DQN

class DQNAgent:
    def __init__(self, model_path=None, state_dim=4, action_dim=2):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(state_dim, action_dim).to(self.device)

        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return int(torch.argmax(q_values).item())
