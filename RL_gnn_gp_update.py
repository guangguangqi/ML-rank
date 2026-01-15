import torch
import torch.nn as nn
import torch.nn.functional as F
import gpytorch
import random
import esm
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch

# --- 1. Oracle Backbone: GNN Feature Extractor ---
class GNNFeatureExtractor(nn.Module):
    def __init__(self, node_features=1280, hidden=512):
        super().__init__()
        self.conv1 = GCNConv(node_features, hidden)
        self.conv2 = GCNConv(hidden, hidden)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return global_mean_pool(x, batch)

# --- 2. Oracle Brain: DKL GP Model ---
class AntibodyDKLGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_y, likelihood, feature_extractor):
        # We initialize with None for train_inputs because we are using a custom graph object
        super().__init__(None, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))
        self.feature_extractor = feature_extractor

    def forward(self, graph_batch):
        # The GNN transforms the graph batch into a latent vector
        projected_x = self.feature_extractor(graph_batch.x, graph_batch.edge_index, graph_batch.batch)
        
        # Now we are back in "Tensor world," so GPyTorch modules work normally
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def __call__(self, *args, **kwargs):
        """
        Overriding __call__ to bypass GPyTorch's internal tensor validation 
        for PyTorch Geometric Batch objects.
        """
        if self.training:
            # During training, we typically handle standard tensors or pre-extracted features
            return super().__call__(*args, **kwargs)
        
        # During inference (RL Reward calculation), we bypass the check
        return self.forward(*args, **kwargs)

#############################################
# --- 3. Prediction and Rewards ---
def gnn_gp_predict(gp_models, graph_batch):
    stats = {}
    property_names = ["stability", "solubility", "aggregation"]
    for i, name in enumerate(property_names):
        gp_models[i].eval()
        gp_models[i].likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = gp_models[i](graph_batch)
            stats[name] = {"mean": posterior.mean, "std": torch.sqrt(posterior.variance)}
    return stats

def calculate_rl_reward(gp_models, sequences, beta=2.0):
    X_residues = embed_sequences_per_residue(sequences)
    graph_batch = Batch.from_data_list([sequence_to_graph(emb) for emb in X_residues])
    stats = gnn_gp_predict(gp_models, graph_batch)
    # Scalarization: Higher is better
    reward = (stats["stability"]["mean"] + beta * stats["stability"]["std"] +
              stats["solubility"]["mean"] + beta * stats["solubility"]["std"] -
              (stats["aggregation"]["mean"] - beta * stats["aggregation"]["std"]))
    return reward

# --- 4. ESM-2 Embeddings (Global and Per-Residue) ---
print("Loading ESM-2...")
model_esm, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
model_esm.eval()
batch_converter = alphabet.get_batch_converter()

def embed_sequences_full(sequences):
    data = [(str(i), s) for i, s in enumerate(sequences)]
    _, _, batch_tokens = batch_converter(data)
    with torch.no_grad():
        results = model_esm(batch_tokens, repr_layers=[33])
    # Returns (Global CLS [B, 1280], Per-Residue [B, L, 1280])
    global_emb = results["representations"][33][:, 0, :]
    per_residue = [results["representations"][33][i, 1:len(s)+1] for i, s in enumerate(sequences)]
    return global_emb, per_residue

def embed_sequences_per_residue(sequences):
    _, res = embed_sequences_full(sequences)
    return res

def sequence_to_graph(seq_embedding):
    L = seq_embedding.shape[0]
    edges = [[i, i+1] for i in range(L-1)] + [[i+1, i] for i in range(L-1)]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(x=seq_embedding, edge_index=edge_index)

# --- 5. RL Agent (Policy Network) ---
class AntibodyActorCritic(nn.Module):
    def __init__(self, emb_dim=1280, hidden=512):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(emb_dim, hidden), nn.ReLU(), nn.Linear(hidden, 20))
        self.critic = nn.Sequential(nn.Linear(emb_dim, hidden), nn.ReLU(), nn.Linear(hidden, 1))

    def forward(self, x):
        return self.actor(x), self.critic(x)

# --- 6. Action Mapping (Discrete Action to Sequence Mutation) ---
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

def apply_rl_actions(sequences, actions):
    # For 2025 Trio: Action is the choice of amino acid for a random mutation site
    new_seqs = []
    for seq, action_idx in zip(sequences, actions):
        s_list = list(seq)
        pos = random.randint(0, len(s_list)-1)
        s_list[pos] = AMINO_ACIDS[action_idx.item()]
        new_seqs.append("".join(s_list))
    return new_seqs

# --- 7. RL Training Loop ---
def rl_optimization_round(agent, gp_models, seed_sequences, epochs=5):
    optimizer = torch.optim.Adam(agent.parameters(), lr=1e-4)
    current_sequences = seed_sequences

    for epoch in range(epochs):
        # 1. Get current state (Embeddings)
        global_emb, _ = embed_sequences_full(current_sequences)
        
        # 2. Agent suggests mutations
        logits, values = agent(global_emb)
        probs = F.softmax(logits, dim=-1)
        actions = torch.multinomial(probs, num_samples=1)
        
        # 3. Environment provides Reward (Oracle)
        new_sequences = apply_rl_actions(current_sequences, actions)
        rewards = calculate_rl_reward(gp_models, new_sequences)
        
        # 4. PPO-style Update
        advantage = rewards - values.detach().squeeze()
        actor_loss = -(torch.log(probs.gather(1, actions).squeeze()) * advantage).mean()
        critic_loss = F.mse_loss(values.squeeze(), rewards)
        
        optimizer.zero_grad()
        (actor_loss + 0.5 * critic_loss).backward()
        optimizer.step()
        
        current_sequences = new_sequences
        print(f"Epoch {epoch} | Avg Reward: {rewards.mean().item():.4f}")
    
    return current_sequences

# --- 8. Execution ---
seed_sequences = [
    "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDGSSGWYVFDYWGQGTLVTVSS",
    "EVQLVESGGGLVQPGRSLRLSCAASGFTFDDYAMHWVRQAPGKGLEWVSGISWNSGSIGYADSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAKVSYLSTASSLDYWGQGTLVTVSS",
    "QVQLVQSGAEVKKPGASVKVSCKASGYTFTGYYMHWVRQAPGQGLEWMGWINPNSGGTNYAQKFQGRVTMTRDTSISTAYMELSRLRSDDTAVYYCARDWGNYGUWFAYWGQGTLVTVSS"
]

print("Initializing GNN+GP+RL Trio...")
shared_extractor = GNNFeatureExtractor()
Y_init = torch.zeros(len(seed_sequences))
likelihoods = [gpytorch.likelihoods.GaussianLikelihood() for _ in range(3)]
gp_models = [AntibodyDKLGPModel(Y_init, likelihoods[i], shared_extractor) for i in range(3)]

agent = AntibodyActorCritic()
optimized_results = rl_optimization_round(agent, gp_models, seed_sequences)



