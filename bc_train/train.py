import torch
from torch.utils.data import DataLoader, TensorDataset
import minari
import numpy as np
import torch.nn as nn
import torch.optim as optim

# Load the dataset
dataset = minari.load_dataset("Maritime-Expert-v1")

obs_list = []
act_list = []

for episode_idx, episode in enumerate(dataset.iterate_episodes()):
    observations = episode.observations
    actions = episode.actions

    if not isinstance(observations, dict):
        print(f"Unexpected episode observations format: {type(observations)} — skipping episode")
        continue
    
    num_steps = len(actions)
    
    for step in range(num_steps): 
        step_obs = {}
        for key, values in observations.items():
            if isinstance(values[step], str):
                try:
                    step_obs[key] = np.array(eval(values[step]), dtype=np.float32)
                except:
                    print(f"Failed to convert string observation for key {key}")
                    raise ValueError(f"String observation format: {values[step][:50]}...")
            else:
                step_obs[key] = values[step]
        
        ego = np.array(step_obs['ego'])
        neighbors = np.array(step_obs['neighbors']).flatten() if 'neighbors' in step_obs else np.array([])           
        goal = np.array(step_obs['goal']) if 'goal' in step_obs else np.array([])       
        full_obs = np.concatenate([ego.flatten(), neighbors, goal])

        action = actions[step]

        obs_list.append(torch.tensor(full_obs, dtype=torch.float32))
        act_list.append(torch.tensor(action, dtype=torch.float32))

observations = torch.stack(obs_list)
actions = torch.stack(act_list)

print("Observations shape:", observations.shape)
print("Actions shape:", actions.shape)

tensor_dataset = TensorDataset(observations, actions)
dataloader = DataLoader(tensor_dataset, batch_size=64, shuffle=True)
print(len(dataloader.dataset))

class BCPolicy(nn.Module):
    def __init__(self, input_dim=486, hidden_dim=256, output_dim=3):
        super(BCPolicy, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = BCPolicy().to(device)
loss_fn = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

epochs = 80
for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for batch_obs, batch_actions in dataloader:
        batch_obs = batch_obs.to(device)
        batch_actions = batch_actions.to(device)
        pred_actions = model(batch_obs)
        loss = loss_fn(pred_actions, batch_actions)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_obs.size(0)

    avg_loss = total_loss / len(dataloader.dataset)
    print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")

torch.save(model.state_dict(), "./model_weights/bc_weights.pth")
