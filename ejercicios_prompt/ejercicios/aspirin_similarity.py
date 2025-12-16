from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors, DataStructs

# Example SMILES (generated + reference drug)
smiles_list = [
    "CCO",                           # ethanol
    "c1ccccc1",                      # benzene
    "CC(=O)Oc1ccccc1C(=O)O",         # aspirin
    "CCN(CC)CC",                     # triethylamine
    "C1CC1C1",                       # invalid
]

reference_smiles = "CC(=O)Oc1ccccc1C(=O)O"  # aspirin
ref_mol = Chem.MolFromSmiles(reference_smiles)
ref_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(ref_mol, 2)

def analyze_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"SMILES": smiles, "Valid": False}
    
    # Calculate basic properties
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    
    # Similarity to reference
    fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2)
    similarity = DataStructs.TanimotoSimilarity(fp, ref_fp)
    
    return {
        "SMILES": smiles,
        "Valid": True,
        "MolecularWeight": round(mw, 2),
        "LogP": round(logp, 2),
        "HBDonors": hbd,
        "HBAcceptors": hba,
        "SimilarityToAspirin": round(similarity, 2)
    }

# Run validation + scoring
results = [analyze_smiles(s) for s in smiles_list]

import pandas as pd
df = pd.DataFrame(results)
print(df)
