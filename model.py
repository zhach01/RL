import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def fanin_init(layer):
    """
    Custom weight initializer for fully connected layers.
    We use the 'fan-in' approach: W ~ U[-1/sqrt(n), 1/sqrt(n)] where n = layer's input dimension.
    """
    fan_in = layer.weight.data.size()[0]
    bound = 1.0 / np.sqrt(fan_in)
    nn.init.uniform_(layer.weight.data, -bound, bound)
    nn.init.uniform_(layer.bias.data, -bound, bound)

# =============================================================================
# Updated Actor (Policy) Model for Integrated Muscle Control
# =============================================================================
class EnhancedActor(nn.Module):
    def __init__(
        self,
        state_size=22,    # Updated default to match your environment's 34D observation
        action_size=6,
        seed=0,
        fc1_units=400,
        fc2_units=300,
        fc3_units=200,
        dropout_rate=0.2
    ):
        """
        Initializes the Enhanced Actor network with LayerNorm and fan-in initialization.
        
        Args:
            state_size (int): Dimension of the state (default 34).
            action_size (int): Dimension of the action (default 6).
            seed (int): Random seed.
            fc1_units (int): # neurons in the first hidden layer.
            fc2_units (int): # neurons in the second hidden layer.
            fc3_units (int): # neurons in the third hidden layer.
            dropout_rate (float): Probability for dropout.
        """
        super(EnhancedActor, self).__init__()
        torch.manual_seed(seed)

        # Hidden layers
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.ln1 = nn.LayerNorm(fc1_units)   # using LayerNorm
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.ln2 = nn.LayerNorm(fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.ln3 = nn.LayerNorm(fc3_units)

        # Final output layer -> action in [0,1]
        self.fc4 = nn.Linear(fc3_units, action_size)

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

        # Initialize weights with fan-in approach
        fanin_init(self.fc1)
        fanin_init(self.fc2)
        fanin_init(self.fc3)
        # For final layer, we often initialize more narrowly so we start near 0
        nn.init.uniform_(self.fc4.weight.data, -3e-3, 3e-3)
        nn.init.uniform_(self.fc4.bias.data, -3e-3, 3e-3)

    def forward(self, state):
        """
        Forward pass: maps state -> action in [0,1].
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        x = self.fc1(state)
        x = self.ln1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.ln2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.ln3(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Sigmoid to ensure [0,1] muscle activations
        return torch.sigmoid(self.fc4(x))

# =============================================================================
# Updated Critic (Q-value) Model for Integrated Muscle Control
# =============================================================================
class EnhancedCritic(nn.Module):
    def __init__(
        self,
        state_size=22,
        action_size=6,
        seed=0,
        fcs1_units=400,
        fc2_units=300,
        fc3_units=200,
        dropout_rate=0.2
    ):
        """
        Initializes the Enhanced Critic network with LayerNorm and fan-in init.
        
        Args:
            state_size (int): dimension of state (default 34).
            action_size (int): dimension of action (default 6).
            seed (int): random seed.
            fcs1_units (int): # neurons in the first hidden layer.
            fc2_units (int): # neurons in the second hidden layer.
            fc3_units (int): # neurons in the third hidden layer.
            dropout_rate (float): dropout probability.
        """
        super(EnhancedCritic, self).__init__()
        torch.manual_seed(seed)

        # First fully-connected layer: only state
        self.fcs1 = nn.Linear(state_size, fcs1_units)
        self.ln1 = nn.LayerNorm(fcs1_units)

        # Second layer: combine state features + action
        self.fc2 = nn.Linear(fcs1_units + action_size, fc2_units)
        self.ln2 = nn.LayerNorm(fc2_units)

        # Third hidden layer
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.ln3 = nn.LayerNorm(fc3_units)

        # Q-value output
        self.fc4 = nn.Linear(fc3_units, 1)

        # Dropout
        self.dropout = nn.Dropout(dropout_rate)

        # Initialize weights
        fanin_init(self.fcs1)
        fanin_init(self.fc2)
        fanin_init(self.fc3)
        nn.init.uniform_(self.fc4.weight.data, -3e-4, 3e-4)
        nn.init.uniform_(self.fc4.bias.data, -3e-4, 3e-4)

    def forward(self, state, action):
        """
        Forward pass: maps (state, action) -> Q-value (scalar).
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        xs = self.fcs1(state)
        xs = self.ln1(xs)
        xs = F.relu(xs)
        xs = self.dropout(xs)

        # Concatenate action
        x = torch.cat((xs, action), dim=1)

        x = self.fc2(x)
        x = self.ln2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.ln3(x)
        x = F.relu(x)
        x = self.dropout(x)

        # Q-value (no activation)
        return self.fc4(x)
