import torch
import gpytorch
import random
import esm

# 1. Multi-Task Predictor with GP
class AntibodyGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(AntibodyGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=2.5))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# 2. Multi-Objective Acquisition Function (2025 Corrected)
def moo_ucb(stats, beta=2.0):
    stability_score = stats["stability"]["mean"] + beta * stats["stability"]["std"]
    solubility_score = stats["solubility"]["mean"] + beta * stats["solubility"]["std"]
    # Minimize aggregation (subtracting the LCB)
    aggregation_lcb = stats["aggregation"]["mean"] - beta * stats["aggregation"]["std"]
    return stability_score + solubility_score - aggregation_lcb

# 3. GP Prediction Helper
def gp_predict(models, x):
    stats = {}
    property_names = ["stability", "solubility", "aggregation"]
    for i, name in enumerate(property_names):
        model = models[i]
        model.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = model(x)
            stats[name] = {"mean": observed_pred.mean, "std": torch.sqrt(observed_pred.variance)}
    return stats

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

# 6. Bayesian Optimization Loop
def bayesian_optimization(gp_models, seed_sequences, n_rounds=10, candidates_per_round=100):
    population = seed_sequences
    for round_id in range(n_rounds):
        print(f"\n🔁 BO Round {round_id}")
        
        candidates = []
        for seq in population:
            for _ in range(candidates_per_round // len(population)):
                candidates.append(mutate_sequence(seq))

        # Vectorized Embedding for the whole batch
        X = embed_sequences(candidates)

        # GP Predictions for all properties
        stats = gp_predict(gp_models, X)

        # Calculate scores and select top-10
        scores = moo_ucb(stats)
        top_idx = torch.topk(scores, k=10).indices
        population = [candidates[i] for i in top_idx]

        print(f"Round {round_id} complete. Top score: {scores[top_idx[0]].item():.4f}")
        print("Top candidate sample:", population[0][:30] + "...")

    return population

# 7. Running the Optimization
seed_sequences = [
    "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYAMSWVRQAPGKGLEWVSAISGSGGSTYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCAKDGSSGWYVFDYWGQGTLVTVSS",
    "EVQLVESGGGLVQPGRSLRLSCAASGFTFDDYAMHWVRQAPGKGLEWVSGISWNSGSIGYADSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCAKVSYLSTASSLDYWGQGTLVTVSS",
    "QVQLVQSGAEVKKPGASVKVSCKASGYTFTGYYMHWVRQAPGQGLEWMGWINPNSGGTNYAQKFQGRVTMTRDTSISTAYMELSRLRSDDTAVYYCARDWGNYGUWFAYWGQGTLVTVSS",
    "QVQLVQSGAEVKKPGSSVKVSCKASGGTFSSYAISWVRQAPGQGLEWMGGIIPIFGTANYAQKFQGRVTITADESTSTAYMELSSLRSEDTAVYYCARAPLRFLEWSTQDYYYYGMDVWGQGTTVTVSS",
    "EVQLVQSGAEVKKPGASVKVSCKASGYTFTSYGISWVRQAPGQGLEWMGWISAYNGNTNYAQKLQGRVTMTTDTSTSTAYMELRSLRSDDTAVYYCARVGYGNYYFEYWGQGTLVTVSS",
    "QVQLVQSGAEVKKPGASVKVSCKASGYTFTNYYMHWVRQAPGQGLEWMGIINPSGGSTSYAQKFQGRVTMTRDTSTSTVYMELSSLRSEDTAVYYCARVAYGMDVWGQGTTVTVSS",
    "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYSMNWVRQAPGKGLEWVSSISSSSSYIYYADSVKGRFTISRDNAKNSLYLQMNSLRAEDTAVYYCARGRWGYGMDVWGQGTLVTVSS",
    "QVQLQESGPGLVKPSETLSLTCTVSGGSVSSGDYYWSWIRQPPGKGLEWIGYIYYSGSTNYNPSLKSRVTISVDTSKNQFSLKLSSVTAADTAVYYCARVGYGMDVWGQGTLVTVSS",
    "EVQLVESGGGLVQPGGSLRLSCAASGFTFSSYWMHWVRQAPGKGLVWVSRINSDGSSTSYADSVKGRFTISRDNAKNTLYLQMNSLRAEDTAVYYCARDRGYYGDYWGQGTLVTVSS",
    "QVQLVESGGGVVQPGRSLRLSCAASGFTFSSYGMHWVRQAPGKGLEWVAVISYDGSNKYYADSVKGRFTISRDNSKNTLYLQMNSLRAEDTAVYYCARGSGSYYYMDVWGKGTTVTVSS"
]

# Initialize GP models with seed sequence embeddings
print("Initializing GP Models...")
X_initial = embed_sequences(seed_sequences)
Y_initial = torch.zeros(10)  # Starting with zero assumptions
likelihoods = [gpytorch.likelihoods.GaussianLikelihood() for _ in range(3)]
gp_models = [AntibodyGPModel(X_initial, Y_initial, likelihoods[i]) for i in range(3)]

# Start BO
optimized_sequences = bayesian_optimization(gp_models, seed_sequences)

