import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# 1. Multi-Task Neural Network with Dropout
class MultiTaskAntibodyNet(nn.Module):
    def __init__(self, emb_dim=1280, hidden=512, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, hidden)

        # Multi-task heads for independent properties
        self.stability = nn.Linear(hidden, 1)
        self.solubility = nn.Linear(hidden, 1)
        self.aggregation = nn.Linear(hidden, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        return {
            "stability": self.stability(x),
            "solubility": self.solubility(x),
            "aggregation": self.aggregation(x)
        }

# 2. MC Dropout Inference (Replaces gp_predict)
def mc_dropout_predict(model, x, n_samples=50):
    """
    Forces dropout layers to stay active during inference to 
    approximate Bayesian uncertainty.
    """
    model.train()  # Crucial: Keep dropout ON
    preds = {k: [] for k in ["stability", "solubility", "aggregation"]}

    with torch.no_grad():
        for _ in range(n_samples):
            out = model(x)
            for k in preds:
                preds[k].append(out[k])

    stats = {}
    for k, v in preds.items():
        v_tensor = torch.stack(v) # Shape: [n_samples, n_candidates, 1]
        stats[k] = {
            "mean": v_tensor.mean(dim=0).squeeze(),
            "std": v_tensor.std(dim=0).squeeze()
        }
    return stats

# 3. Multi-Objective Acquisition (Same logic as GP version)
def moo_ucb(stats, beta=2.0):
    stability_score = stats["stability"]["mean"] + beta * stats["stability"]["std"]
    solubility_score = stats["solubility"]["mean"] + beta * stats["solubility"]["std"]
    # Minimize aggregation via LCB negation
    aggregation_lcb = stats["aggregation"]["mean"] - beta * stats["aggregation"]["std"]
    return stability_score + solubility_score - aggregation_lcb

# [Keep Mutation Operator and ESM-2 Embedding functions from previous code]

# 4. Mutation Operator
AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")
def mutate_sequence(seq, n_mut=1):
    seq_list = list(seq)
    for _ in range(n_mut):
        i = random.randint(0, len(seq_list) - 1)
        seq_list[i] = random.choice(AMINO_ACIDS)
    return "".join(seq_list)

# 5. ESM-2 Embedding Setup
print("Loading ESM-2 Model...")
model_esm, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
model_esm.eval()
batch_converter = alphabet.get_batch_converter()

def embed_sequences(sequences):
    data = [(str(i), s) for i, s in enumerate(sequences)]
    _, _, batch_tokens = batch_converter(data)
    with torch.no_grad():
        results = model_esm(batch_tokens, repr_layers=[33])
    return results["representations"][33][:, 0, :] 


# 6. Bayesian Optimization Loop (Updated for MC Dropout)
def bayesian_optimization(model, seed_sequences, n_rounds=10, candidates_per_round=100):
    population = seed_sequences
    for round_id in range(n_rounds):
        print(f"\n🔁 BO Round {round_id}")
        
        candidates = []
        for seq in population:
            for _ in range(candidates_per_round // len(population)):
                candidates.append(mutate_sequence(seq))

        # Vectorized Embedding
        X = embed_sequences(candidates)

        # MC Dropout predictions (Replaces GP call)
        stats = mc_dropout_predict(model, X)

        # Scoring and selection
        scores = moo_ucb(stats)
        top_idx = torch.topk(scores, k=10).indices
        population = [candidates[i] for i in top_idx]

        print(f"Round {round_id} Top score: {scores[top_idx[0]].item():.4f}")
    return population

# 7. Running the Optimization
print("Initializing Multi-Task Network...")
model = MultiTaskAntibodyNet() # No need for X_train/Y_train during init

# --- STEP 1: Define the sequences ---
seed_sequences = [
    "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDGSSGWYVFDYWGQGTLVTVSS", # Trastuzumab
    "EVQLVESGGGLVQPGRSLRLSCAASGFTFDDYAMHWVRQAPGKGLEWVSGISWNSGSIGYADSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAKVSYLSTASSLDYWGQGTLVTVSS", # Adalimumab
    "QVQLVQSGAEVKKPGASVKVSCKASGYTFTGYYMHWVRQAPGQGLEWMGWINPNSGGTNYAQKFQGRVTMTRDTSISTAYMELSRLRSDDTAVYYCARDWGNYGUWFAYWGQGTLVTVSS", # Pembrolizumab
    "QVQLVQSGAEVKKPGSSVKVSCKASGGTFSSYAISWVRQAPGQGLEWMGGIIPIFGTANYAQKFQGRVTITADESTSTAYMELSSLRSEDTAVYYCARAPLRFLEWSTQDYYYYGMDVWGQGTTVTVSS", # Nivolumab
    "EVQLVQSGAEVKKPGASVKVSCKASGYTFTSYGISWVRQAPGQGLEWMGWISAYNGNTNYAQKLQGRVTMTTDTSTSTAYMELRSLRSDDTAVYYCARVGYGNYYFEYWGQGTLVTVSS", # Bevacizumab
    "QVQLVQSGAEVKKPGASVKVSCKASGYTFTNYYMHWVRQAPGQGLEWMGIINPSGGSTSYAQKFQGRVTMTRDTSTSTVYMELSSLRSEDTAVYYCARVAYGMDVWGQGTTVTVSS", # Rituximab
    "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYSMNWVRQAPGKGLEWVSSISSSSSYIYYADSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCARGRWGYGMDVWGQGTLVTVSS", # Infliximab
    "QVQLQESGPGLVKPSETLSLTCTVSGGSVSSGDYYWSWIRQPPGKGLEWIGYIYYSGSTNYNPSLKSRVTISVDTSKNQFSLKLSSVTAADTAVYYCARVGYGMDVWGQGTLVTVSS", # Ustekinumab
    "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYWMHWVRQAPGKGLVWVSRINSDGSSTSYADSVKGRFTISRDNAKNTLYLQMNSLRAEDTAVYYCARDRGYYGDYWGQGTLVTVSS", # Denosumab
    "QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAVISYDGSNKYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARGSGSYYYMDVWGKGTTVTVSS"  # Golimumab
]

# Start BO
optimized_sequences = bayesian_optimization(model, seed_sequences)

