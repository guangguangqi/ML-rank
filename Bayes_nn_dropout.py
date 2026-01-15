
import torch
import torch.nn as nn
import torch.nn.functional as F

###############################################
#1 Multi-Task Predictor with MC Dropout

class MultiTaskAntibodyNet(nn.Module):
    print ("11") 
    def __init__(self, emb_dim=1280, hidden=512, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(emb_dim, hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden, hidden)

        # Three properties
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


#####################################################
#2 MC Dropout Inference (Uncertainty Estimation)

def mc_dropout_predict(model, x, n_samples=50):
    model.train()  # force dropout ON
    print("22")    
    preds = {k: [] for k in ["stability", "solubility", "aggregation"]}

    for _ in range(n_samples):
        out = model(x)
        for k in preds:
            preds[k].append(out[k].detach().cpu())

    stats = {}
    for k, v in preds.items():
        v = torch.stack(v)
        stats[k] = {
            "mean": v.mean(dim=0),
            "std": v.std(dim=0)
        }
    return stats

##############################################
#3 Multi-Objective Acquisition Function (UCB)

def moo_ucb_old(stats, beta=2.0):
    """
    Maximize stability + solubility, minimize aggregation
    """
    score = (
        stats["stability"]["mean"] + beta * stats["stability"]["std"]
        + stats["solubility"]["mean"] + beta * stats["solubility"]["std"]
        - stats["aggregation"]["mean"] + beta * stats["aggregation"]["std"]
    )
    print("333")
    return score

def moo_ucb(stats, beta=2.0):
    """
    Combines three objectives into one score for selection.
    Stability: Maximize (UCB: Mean + Beta * Std)
    Solubility: Maximize (UCB: Mean + Beta * Std)
    Aggregation: Minimize (LCB: Mean - Beta * Std) -> Negate for maximization
    """
    # 1. Stability (Maximize)
    stability_score = stats["stability"]["mean"] + beta * stats["stability"]["std"]
    
    # 2. Solubility (Maximize)
    solubility_score = stats["solubility"]["mean"] + beta * stats["solubility"]["std"]
    
    # 3. Aggregation (Minimize)
    # We use LCB (Mean - Beta * Std) to find the 'best' (lowest) possible value.
    # We then subtract it (or negate it) to convert the goal to maximization.
    aggregation_lcb = stats["aggregation"]["mean"] - beta * stats["aggregation"]["std"]
    
    # Combine: Higher score is better
    total_score = stability_score + solubility_score - aggregation_lcb
    
    return total_score


####################################################
#4 Candidate Sequence Generation (Mutation Operator) 
import random

AMINO_ACIDS = list("ACDEFGHIKLMNPQRSTVWY")

def mutate_sequence(seq, n_mut=1):
    seq = list(seq)
    for _ in range(n_mut):
        i = random.randint(0, len(seq) - 1)
        seq[i] = random.choice(AMINO_ACIDS)
    print("4444")    
    return "".join(seq)

###########################################
###########################################
# 5 Bayesian Optimization Loop

def bayesian_optimization(
    model,
#    embed_fn,
    embed_sequences,
    seed_sequences,
    n_rounds=10,
    candidates_per_round=100
):
    population = seed_sequences

    for round_id in range(n_rounds):
        print(f"\n🔁 BO Round {round_id}")

        candidates = []
        for seq in population:
            for _ in range(candidates_per_round // len(population)):
                candidates.append(mutate_sequence(seq))

        # Embed sequences
        X = torch.stack([embed_fn(seq) for seq in candidates])

        # MC dropout predictions
        stats = mc_dropout_predict(model, X)

        # Acquisition
        scores = moo_ucb(stats)

        # Select Pareto / Top-K
        top_idx = torch.topk(scores.squeeze(), k=10).indices
        population = [candidates[i] for i in top_idx]

        print("Top candidate:", population[0])
        print("555")   
    return population

################################################
# 6  Embedding Function (Placeholder)
def embed_fn(seq):
    # Placeholder: replace with real embedding
    print("6666")
    return torch.randn(1280)

#############################################
import torch

# Load via Torch Hub to ensure the correct model architecture
model_esm, alphabet = torch.hub.load("facebookresearch/esm:main", "esm2_t33_650M_UR50D")
model_esm.eval()  # Set to evaluation mode
batch_converter = alphabet.get_batch_converter()

# Use this function exactly as before
def embed_sequences(sequences):
    data = [(str(i), s) for i, s in enumerate(sequences)]
    _, _, batch_tokens = batch_converter(data)

    with torch.no_grad():
        # Running on CPU by default
        results = model_esm(batch_tokens, repr_layers=[33])
        
    # Extract the CLS token (index 0)
    sequence_embeddings = results["representations"][33][:, 0, :] 
    return sequence_embeddings


###############################################
###############################################
# 7 Running the Optimization
print("777")
model = MultiTaskAntibodyNet()
#seed_sequences = [
#    "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWV",
#    "QVQLVQSGAEVKKPGSSVKVSCKASGDTFSTYWMNWVRQAPGQGLEW"
#]

seed_sequences = [
    "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDGSSGWYVFDYWGQGTLVTVSS", # Trastuzumab (Herceptin)
    "EVQLVESGGGLVQPGRSLRLSCAASGFTFDDYAMHWVRQAPGKGLEWVSGISWNSGSIGYADSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAKVSYLSTASSLDYWGQGTLVTVSS", # Adalimumab (Humira)
    "QVQLVQSGAEVKKPGASVKVSCKASGYTFTGYYMHWVRQAPGQGLEWMGWINPNSGGTNYAQKFQGRVTMTRDTSISTAYMELSRLRSDDTAVYYCARDWGNYGUWFAYWGQGTLVTVSS", # Pembrolizumab (Keytruda)
    "QVQLVQSGAEVKKPGSSVKVSCKASGGTFSSYAISWVRQAPGQGLEWMGGIIPIFGTANYAQKFQGRVTITADESTSTAYMELSSLRSEDTAVYYCARAPLRFLEWSTQDYYYYGMDVWGQGTTVTVSS", # Nivolumab (Opdivo)
    "EVQLVQSGAEVKKPGASVKVSCKASGYTFTSYGISWVRQAPGQGLEWMGWISAYNGNTNYAQKLQGRVTMTTDTSTSTAYMELRSLRSDDTAVYYCARVGYGNYYFEYWGQGTLVTVSS", # Bevacizumab (Avastin)
    "QVQLVQSGAEVKKPGASVKVSCKASGYTFTNYYMHWVRQAPGQGLEWMGIINPSGGSTSYAQKFQGRVTMTRDTSTSTVYMELSSLRSEDTAVYYCARVAYGMDVWGQGTTVTVSS", # Rituximab (Rituxan)
    "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYSMNWVRQAPGKGLEWVSSISSSSSYIYYADSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCARGRWGYGMDVWGQGTLVTVSS", # Infliximab (Remicade)
    "QVQLQESGPGLVKPSETLSLTCTVSGGSVSSGDYYWSWIRQPPGKGLEWIGYIYYSGSTNYNPSLKSRVTISVDTSKNQFSLKLSSVTAADTAVYYCARVGYGMDVWGQGTLVTVSS", # Ustekinumab (Stelara)
    "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYWMHWVRQAPGKGLVWVSRINSDGSSTSYADSVKGRFTISRDNAKNTLYLQMNSLRAEDTAVYYCARDRGYYGDYWGQGTLVTVSS", # Denosumab (Prolia)
    "QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAVISYDGSNKYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARGSGSYYYMDVWGKGTTVTVSS"  # Golimumab (Simponi)
]

optimized = bayesian_optimization(
    model,
#    embed_fn,
    embed_sequences,
    seed_sequences
)

