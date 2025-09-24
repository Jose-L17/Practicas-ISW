import random
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdMolDescriptors

# ============================
# 1. Reference molecule (target)
# ============================
reference_smiles = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin
ref_mol = Chem.MolFromSmiles(reference_smiles)
ref_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(ref_mol, 2)

# ============================
# 2. Fake "GAN generator" (placeholder)
# ============================
vocab = [
    "CCO",                           # ethanol
    "c1ccccc1",                      # benzene
    "CC(=O)Oc1ccccc1C(=O)O",         # aspirin
    "CCN(CC)CC",                     # triethylamine
    "C1CC1C1"                        # invalid
]

def gan_generate_smiles(policy_weights):
    """Choose SMILES based on learned probabilities (policy)."""
    return random.choices(vocab, weights=policy_weights, k=1)[0]

# ============================
# 3. Reward function
# ============================
def score_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return -1.0  # strong penalty for invalid molecules
    
    # Basic properties
    logp = Descriptors.MolLogP(mol)
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2)
    similarity = DataStructs.TanimotoSimilarity(fp, ref_fp)

    # Reward combines similarity + drug-likeness
    reward = similarity - abs(logp - 2) * 0.1
    return reward

# ============================
# 4. Reinforcement Learning loop
# ============================
def train_rl(epochs=50, lr=0.1):
    # Start with uniform policy (all SMILES equally likely)
    policy = np.ones(len(vocab)) / len(vocab)

    reward_history = []

    for epoch in range(epochs):
        # Generate candidate
        smiles = gan_generate_smiles(policy)
        idx = vocab.index(smiles)

        # Evaluate reward
        reward = score_smiles(smiles)
        reward_history.append(reward)

        # Update policy (REINFORCE-style)
        policy[idx] += lr * reward
        policy = np.clip(policy, 0.01, None)  # avoid zero probs
        policy /= policy.sum()  # normalize

        print(f"Epoch {epoch+1:02d} | SMILES: {smiles:25s} | Reward: {reward:.3f}")

    return policy, reward_history

# ============================
# 5. Run training
# ============================
final_policy, rewards = train_rl(epochs=20)

print("\nFinal learned probabilities:")
for s, p in zip(vocab, final_policy):
    print(f"{s:25s} -> {p:.2f}")
