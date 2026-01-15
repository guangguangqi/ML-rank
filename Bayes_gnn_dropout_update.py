import torch
import torch.nn.functional as F
import random
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch

# 1. GNN Multi-Task Architecture
class GNNMultiTaskAntibodyNet(torch.nn.Module):
    def __init__(self, node_features=1280, hidden=512, dropout=0.2):
        super().__init__()
        self.conv1 = GCNConv(node_features, hidden)
        self.conv2 = GCNConv(hidden, hidden)
        self.dropout = torch.nn.Dropout(dropout)
        self.stability = torch.nn.Linear(hidden, 1)
        self.solubility = torch.nn.Linear(hidden, 1)
        self.aggregation = torch.nn.Linear(hidden, 1)

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = self.dropout(x)
        x = F.relu(self.conv2(x, edge_index))
        x = self.dropout(x)
        x = global_mean_pool(x, batch) 
        return {
            "stability": self.stability(x),
            "solubility": self.solubility(x),
            "aggregation": self.aggregation(x)
        }

# 2. Sequence-to-Graph Conversion
def sequence_to_graph(seq_embedding):
    L = seq_embedding.shape[0]
    edges = []
    for i in range(L - 1):
        edges.append([i, i + 1])
        edges.append([i + 1, i])
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return Data(x=seq_embedding, edge_index=edge_index)

# 3. MC Dropout Inference
def mc_dropout_gnn_predict(model, graph_batch, n_samples=30):
    model.train() 
    preds = {k: [] for k in ["stability", "solubility", "aggregation"]}
    with torch.no_grad():
        for _ in range(n_samples):
            out = model(graph_batch.x, graph_batch.edge_index, graph_batch.batch)
            for k in preds:
                preds[k].append(out[k])
    stats = {}
    for k, v in preds.items():
        v_tensor = torch.stack(v)
        stats[k] = {"mean": v_tensor.mean(dim=0).squeeze(), "std": v_tensor.std(dim=0).squeeze()}
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
model_esm, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
model_esm.eval()
batch_converter = alphabet.get_batch_converter()

def embed_sequences_per_residue(sequences):
    data = [(str(i), s) for i, s in enumerate(sequences)]
    _, _, batch_tokens = batch_converter(data)
    with torch.no_grad():
        results = model_esm(batch_tokens, repr_layers=[33], return_contacts=False)
    
    # Extract full per-residue embeddings, removing start/stop tokens
    token_representations = results["representations"][33]
    return [token_representations[i, 1 : len(s) + 1] for i, s in enumerate(sequences)]

# 7. Bayesian Optimization Loop
def bayesian_optimization(model, seed_sequences, n_rounds=5, candidates_per_round=40):
    population = seed_sequences
    for round_id in range(n_rounds):
        print(f"\n🔁 BO Round {round_id}")
        candidates = []
        for seq in population:
            for _ in range(candidates_per_round // len(population)):
                candidates.append(mutate_sequence(seq))

        # 1. Embed per-residue
        X_residues = embed_sequences_per_residue(candidates) 

        # 2. Convert to graphs and batch
        graphs = [sequence_to_graph(emb) for emb in X_residues]
        graph_batch = Batch.from_data_list(graphs)

        # 3. Predict and Score
        stats = mc_dropout_gnn_predict(model, graph_batch) 
        scores = moo_ucb(stats)
        
        top_idx = torch.topk(scores, k=min(10, len(scores))).indices
        population = [candidates[i] for i in top_idx]
        print(f"Top score in round {round_id}: {scores[top_idx[0]].item():.4f}")
    return population

# 8. Execution
seed_sequences = [
    "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDGSSGWYVFDYWGQGTLVTVSS",
    "EVQLVESGGGLVQPGRSLRLSCAASGFTFDDYAMHWVRQAPGKGLEWVSGISWNSGSIGYADSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAKVSYLSTASSLDYWGQGTLVTVSS",
    "QVQLVQSGAEVKKPGASVKVSCKASGYTFTGYYMHWVRQAPGQGLEWMGWINPNSGGTNYAQKFQGRVTMTRDTSISTAYMELSRLRSDDTAVYYCARDWGNYGUWFAYWGQGTLVTVSS"
]

print("Initializing GNN Multi-Task Network...")
model = GNNMultiTaskAntibodyNet()
optimized = bayesian_optimization(model, seed_sequences)

