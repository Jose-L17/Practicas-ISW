import random
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdMolDescriptors

# ============================
# 1. Pretend GAN Generator
# ============================
# In practice, replace this with your trained GAN model that outputs SMILES
def gan_generate_smiles(n=10):
    fake_smiles = [
        "CCO",                           # ethanol
        "c1ccccc1",                      # benzene
        "CC(=O)Oc1ccccc1C(=O)O",         # aspirin
        "CCN(CC)CC",                     # triethylamine
        "C1CC1C1",                       # invalid
    ]
    return [random.choice(fake_smiles) for _ in range(n)]


# ============================
# 2. Reference Molecule (target drug or scaffold)
# ============================
reference_smiles = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin
ref_mol = Chem.MolFromSmiles(reference_smiles)
ref_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(ref_mol, 2)


# ============================
# 3. Scoring Function
# ============================
def score_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"SMILES": smiles, "Valid": False}

    # Compute basic properties
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)

    # Similarity to reference (proxy for docking score)
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2)
    similarity = DataStructs.TanimotoSimilarity(fp, ref_fp)

    # Reward (this could be used to train GAN with RL)
    reward = similarity - abs(logp - 2) * 0.1  # bias toward logP ≈ 2

    return {
        "SMILES": smiles,
        "Valid": True,
        "MW": round(mw, 2),
        "LogP": round(logp, 2),
        "HBD": hbd,
        "HBA": hba,
        "Similarity": round(similarity, 2),
        "Reward": round(reward, 2)
    }


# ============================
# 4. Run Pipeline
# ============================
def run_pipeline(n=10):
    smiles_batch = gan_generate_smiles(n)
    results = [score_smiles(s) for s in smiles_batch]
    return results


# ============================
# 5. Example Run
# ============================
import pandas as pd

results = run_pipeline(10)
df = pd.DataFrame(results)
print(df)
