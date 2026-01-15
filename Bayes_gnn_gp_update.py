import torch
import torch.nn.functional as F
import gpytorch
import random
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
import esm # Make sure this is imported at the top

###### 1 GNN Feature Extractor
class GNNFeatureExtractor(torch.nn.Module):
    def __init__(self, node_features=1280, hidden=512):
        super().__init__()
        self.conv1 = GCNConv(node_features, hidden)
        self.conv2 = GCNConv(hidden, hidden)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return global_mean_pool(x, batch)

## 2 Deep Kernel Learning (DKL) Model
class AntibodyDKLGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_y, likelihood, feature_extractor):
        # We pass None for train_inputs as we handle graph conversion internally
        super().__init__(None, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))
        self.feature_extractor = feature_extractor
        
        # NOTE: GPyTorch expects train_inputs to be set externally for DKL
        self.register_buffer('train_inputs_graphs', None)

    def forward(self, graph_batch):
        # x_graphs is a PyG Batch object containing .x, .edge_index, and .batch
        projected_x = self.feature_extractor(graph_batch.x, graph_batch.edge_index, graph_batch.batch)
        mean_x = self.mean_module(projected_x)
        covar_x = self.covar_module(projected_x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

###3 Updated GP-Based Multi-Task Prediction
def gnn_gp_predict(gp_models, graph_batch):
    stats = {}
    property_names = ["stability", "solubility", "aggregation"]
    for i, name in enumerate(property_names):
        model = gp_models[i]
        model.eval()
        model.likelihood.eval() # Ensure likelihood is in eval mode
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            posterior = model(graph_batch)
            stats[name] = {"mean": posterior.mean, "std": torch.sqrt(posterior.variance)}
    return stats

# 4. Multi-Objective Acquisition (UCB)
def moo_ucb(stats, beta=2.0):
    score = (stats["stability"]["mean"] + beta * stats["stability"]["std"] +
             stats["solubility"]["mean"] + beta * stats["solubility"]["std"] -
             (stats["aggregation"]["mean"] - beta * stats["aggregation"]["std"]))
    return score

# 5. Mutation Operator
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
def mutate_sequence(seq, n_mut=1):
    seq_list = list(seq)
    for _ in range(n_mut):
        i = random.randint(0, len(seq_list) - 1)
        seq_list[i] = random.choice(AMINO_ACIDS)
    return "".join(seq_list)

# 6. ESM-2 Embedding Setup (Per-Residue Extraction)
print("Loading ESM-2 Model...")
# Ensure torch hub load works correctly if esm is installed via pip install fair-esm
try:
    model_esm, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
except AttributeError:
    model_esm, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")

model_esm.eval()
batch_converter = alphabet.get_batch_converter()

def embed_sequences_per_residue(sequences):
    data = [(str(i), s) for i, s in enumerate(sequences)]
    _, _, batch_tokens = batch_converter(data)
    with torch.no_grad():
        # Ensure correct layer indexing (L-1 is usually the last layer)
        results = model_esm(batch_tokens, repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33]
    # Slice off the special start/stop tokens for GNN node features
    return [token_representations[i, 1 : len(s) + 1] for i, s in enumerate(sequences)]

# 7. Sequence-to-Graph Conversion
def sequence_to_graph(seq_embedding):
    L = seq_embedding.shape[0] # Corrected shape access
    edges = []
    for i in range(L - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(x=seq_embedding, edge_index=edge_index)


# 8. Bayesian Optimization Loop (DKL Enabled)
def bayesian_optimization(gp_models, seed_sequences, n_rounds=5, candidates_per_round=40):
    population = seed_sequences
    for round_id in range(n_rounds):
        print(f"\n🔁 BO Round {round_id} (GNN-GP)")
        
        # ... (Mutation and Graph Creation logic remains the same) ...
        candidates = []
        for seq in population:
            for _ in range(candidates_per_round // len(population)):
                candidates.append(mutate_sequence(seq))
        X_residues = embed_sequences_per_residue(candidates) 
        graphs = [sequence_to_graph(emb) for emb in X_residues]
        graph_batch = Batch.from_data_list(graphs)

        # GNN-GP Prediction
        stats = gnn_gp_predict(gp_models, graph_batch)

        # Multi-Objective Acquisition (UCB) and Selection
        scores = moo_ucb(stats)
        top_idx = torch.topk(scores, k=min(10, len(scores))).indices
        population = [candidates[i] for i in top_idx]

        print(f"Round {round_id} complete. Max Score: {scores[top_idx[0]].item():.4f}")
        print(f"Top Lead Sample: {population[0][:30]}...")

    return population

# 9. Execution (Corrected Initialization)
seed_sequences = [
    "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDGSSGWYVFDYWGQGTLVTVSS",
    "EVQLVESGGGLVQPGRSLRLSCAASGFTFDDYAMHWVRQAPGKGLEWVSGISWNSGSIGYADSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAKVSYLSTASSLDYWGQGTLVTVSS",
    "QVQLVQSGAEVKKPGASVKVSCKASGYTFTGYYMHWVRQAPGQGLEWMGWINPNSGGTNYAQKFQGRVTMTRDTSISTAYMELSRLRSDDTAVYYCARDWGNYGUWFAYWGQGTLVTVSS"
]

print("Initializing GNN + GP Multi-Task System (DKL)...")

# 1. Define the shared GNN Feature Extractor
shared_feature_extractor = GNNFeatureExtractor(node_features=1280, hidden=512)

# 2. Prepare initial training data (embeddings of seed sequences)
X_initial_residues = embed_sequences_per_residue(seed_sequences)
graphs_initial = [sequence_to_graph(emb) for emb in X_initial_residues]
graph_batch_initial = Batch.from_data_list(graphs_initial) # Used only for initial shape/data

# 3. Create initial target values (e.g., zeros if starting from scratch)
Y_initial = torch.zeros(len(seed_sequences))

# 4. Initialize 3 GP Models using the shared backbone
likelihoods = [gpytorch.likelihoods.GaussianLikelihood() for _ in range(3)]
gp_models = [
    AntibodyDKLGPModel(Y_initial, likelihoods[i], shared_feature_extractor)
    for i in range(3)
]

# 5. Start the Bayesian Optimization Loop
print("Starting BO Rounds...")
optimized_sequences = bayesian_optimization(
    gp_models, 
    seed_sequences, 
    n_rounds=5, 
    candidates_per_round=40
)


