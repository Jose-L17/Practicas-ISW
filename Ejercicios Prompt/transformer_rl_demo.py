import random
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors

# =================================
# 1. Transformer Generator (toy version)
# =================================
# In practice, this would be a trained Transformer model on SMILES.
# Here we simulate it by sampling from a vocabulary with probabilities.
vocab = [
    "CCO",                           # ethanol
    "c1ccccc1",                      # benzene
    "CC(=O)Oc1ccccc1C(=O)O",         # aspirin
    "CCN(CC)CC",                     # triethylamine
    "O=C1NC(=O)C2=C(N1)C=CC=C2"      # barbiturate-like
]

def transformer_generate(policy_weights):
    """Sample SMILES based on Transformer probabilities (policy)."""
    return random.choices(vocab, weights=policy_weights, k=1)[0]

# =================================
# 2. Docking Proxy Score
# =================================
def docking_score(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return -10.0  # invalid = strong penalty
    
    # Basic molecular features (proxy for docking affinity)
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    aromatic_rings = Chem.Lipinski.RingCount(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)

    # Heuristic docking-like score:
    # Favor MW between 150–500, logP between 0–5, aromatic rings, balanced H-bonding
    score = 0
    if 150 < mw < 500: score += 2
    if 0 < logp < 5: score += 2
    score += aromatic_rings * 0.5
    score += min(hbd, 2) * 0.5
    score += min(hba, 5) * 0.2

    return score

# =================================
# 3. RL Training Loop
# =================================
def train_transformer(epochs=30, lr=0.1):
    policy = np.ones(len(vocab)) / len(vocab)  # start uniform
    reward_history = []

    for epoch in range(epochs):
        # Generate candidate
        smiles = transformer_generate(policy)
        idx = vocab.index(smiles)

        # Evaluate docking proxy score
        reward = docking_score(smiles)
        reward_history.append(reward)

        # Update policy (REINFORCE-style)
        policy[idx] += lr * reward
        policy = np.clip(policy, 0.01, None)
        policy /= policy.sum()

        print(f"Epoch {epoch+1:02d} | SMILES: {smiles:30s} | DockingScore: {reward:.2f}")

    return policy, reward_history

# =================================
# 4. Run training
# =================================
final_policy, rewards = train_transformer(epochs=20)

print("\nFinal learned probabilities:")
for s, p in zip(vocab, final_policy):
    print(f"{s:30s} -> {p:.2f}")
