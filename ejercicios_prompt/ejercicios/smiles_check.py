from rdkit import Chem
from rdkit.Chem import Descriptors

# Example SMILES (some valid, some invalid)
smiles_list = [
    "CCO",                # ethanol
    "c1ccccc1",           # benzene
    "CC(=O)Oc1ccccc1C(=O)O",  # aspirin
    "C1CC1C1",            # invalid (bad ring closure)
]

def analyze_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"SMILES": smiles, "Valid": False}
    
    # Calculate basic descriptors
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    
    return {
        "SMILES": smiles,
        "Valid": True,
        "MolecularWeight": round(mw, 2),
        "LogP": round(logp, 2),
        "HBDonors": hbd,
        "HBAcceptors": hba
    }

# Run validation and analysis
results = [analyze_smiles(s) for s in smiles_list]

# Show results
import pandas as pd
df = pd.DataFrame(results)
print(df)
