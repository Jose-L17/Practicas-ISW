
import argparse
import random
import joblib
import numpy as np
import pandas as pd
import selfies as sf
from pathlib import Path
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.warning")
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski, DataStructs, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
MODEL_FILE = PROJECT_ROOT / "models" / "rf_bindingdb.joblib"
OUT_DIR = PROJECT_ROOT / "data" / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

HYDROPHOBIC = set("AVLIMFWY")
POLAR = set("STNQC")
POSITIVE = set("KRH")
NEGATIVE = set("DE")
SPECIAL = set("GP")


def read_sequence_from_file(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
    seq_lines = [ln.strip() for ln in text if ln.strip() and not ln.startswith(">")]
    return "".join(seq_lines).strip().upper()


def seq_to_descriptor(seq: str) -> np.ndarray:
    seq = str(seq).strip().upper()

    counts = np.zeros(len(AMINO_ACIDS), dtype=float)
    total = 0

    hydrophobic_count = polar_count = positive_count = negative_count = special_count = 0

    for ch in seq:
        if ch in AA_INDEX:
            counts[AA_INDEX[ch]] += 1
            total += 1
            if ch in HYDROPHOBIC:
                hydrophobic_count += 1
            if ch in POLAR:
                polar_count += 1
            if ch in POSITIVE:
                positive_count += 1
            if ch in NEGATIVE:
                negative_count += 1
            if ch in SPECIAL:
                special_count += 1

    if total > 0:
        freqs = counts / total
        group_feats = np.array(
            [
                hydrophobic_count / total,
                polar_count / total,
                positive_count / total,
                negative_count / total,
                special_count / total,
            ],
            dtype=float,
        )
    else:
        freqs = counts
        group_feats = np.zeros(5, dtype=float)

    length_norm = min(len(seq) / 1000.0, 1.5)
    return np.concatenate([freqs, np.array([length_norm], dtype=float), group_feats])


def smiles_to_morgan_fp(smiles: str, radius: int = 2, n_bits: int = 2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(float)


def featurize_pair(smiles: str, target_seq: str, n_bits: int = 2048):
    fp = smiles_to_morgan_fp(smiles, n_bits=n_bits)
    if fp is None:
        return None
    return np.concatenate([fp, seq_to_descriptor(target_seq)])


def pki_to_ki_nM(pki: float) -> float:
    return 10 ** (9.0 - pki)


ALLOWED_ATOMS = {"C", "N", "O", "S", "F", "Cl", "Br", "I", "H"}


def net_charge(mol: Chem.Mol) -> int:
    return int(sum(a.GetFormalCharge() for a in mol.GetAtoms()))


def allowed_atoms_only(mol: Chem.Mol, allowed: set[str]) -> bool:
    return all(a.GetSymbol() in allowed for a in mol.GetAtoms())


def morgan_bitvect(mol: Chem.Mol, radius: int = 2, n_bits: int = 2048):
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def has_bad_charge_tokens(smiles: str) -> bool:
    bad_tokens = [
        "[C+]", "[C-]", "[N+]", "[N-]", "[O+]", "[O-]", "[S+]", "[S-]", "[P+]", "[P-]",
        "[Na+]", "[K+]", "[Li+]", "[Ca+2]", "[Mg+2]",
    ]
    return any(tok in smiles for tok in bad_tokens)


REACTIVE_SMARTS = [
    "C(=O)Cl",
    "C(=O)Br",
    "C(=O)I",
    "S(=O)(=O)Cl",
    "C1OC1",
    "C1NC1",
    "C=CC(=O)",         
    "C=CS(=O)(=O)",     
    "C=C(C=O)",         
    "N=[N+]=[N-]",       
]

REACTIVE_MOLS = [Chem.MolFromSmarts(s) for s in REACTIVE_SMARTS if Chem.MolFromSmarts(s) is not None]


def has_reactive_substructure(mol: Chem.Mol) -> bool:
    return any(mol.HasSubstructMatch(p) for p in REACTIVE_MOLS)


def basic_filters(
    mol: Chem.Mol,
    seed_fp=None,
    max_abs_charge: int = 0,
    max_sim_to_seed: float = 0.90,
    allowed_atoms: set[str] | None = None,
) -> dict | None:
    """Devuelve dict de propiedades si pasa filtros, si no None."""
    try:
        smi_can = Chem.MolToSmiles(mol, isomericSmiles=True)

        if has_bad_charge_tokens(smi_can):
            return None

        q = net_charge(mol)
        if abs(q) > max_abs_charge:
            return None

        if allowed_atoms is None:
            allowed_atoms = ALLOWED_ATOMS
        if not allowed_atoms_only(mol, allowed_atoms):
            return None

        if has_reactive_substructure(mol):
            return None

        sim_seed = np.nan
        if seed_fp is not None:
            fp = morgan_bitvect(mol)
            sim_seed = float(DataStructs.TanimotoSimilarity(fp, seed_fp))
            if sim_seed > max_sim_to_seed:
                return None

        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        hbd = Lipinski.NumHDonors(mol)
        hba = Lipinski.NumHAcceptors(mol)
        tpsa = Descriptors.TPSA(mol)
        rot = Lipinski.NumRotatableBonds(mol)

        if mw < 120 or mw > 650:
            return None
        if logp < -1.0 or logp > 6.0:
            return None
        if hbd > 8 or hba > 12:
            return None
        if tpsa > 180:
            return None
        if rot > 12:
            return None

        return {
            "smiles": smi_can,
            "MW": mw,
            "logP": logp,
            "HBD": hbd,
            "HBA": hba,
            "tPSA": tpsa,
            "RotB": rot,
            "net_charge": q,
            "sim_to_seed": sim_seed,
        }
    except Exception:
        return None


def murcko_scaffold_smiles(mol: Chem.Mol) -> str:
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        if scaf is None:
            return ""
        return Chem.MolToSmiles(scaf, isomericSmiles=False)
    except Exception:
        return ""


def diverse_greedy_selection(
    sorted_smiles: list[str],
    topk: int,
    max_pair_sim: float = 0.85,
    max_per_scaffold: int = 3,
):
    chosen = []
    chosen_fps = []
    scaffold_counts: dict[str, int] = {}

    for smi in sorted_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        scaf = murcko_scaffold_smiles(mol)
        if scaf:
            scaffold_counts.setdefault(scaf, 0)
            if scaffold_counts[scaf] >= max_per_scaffold:
                continue

        fp = morgan_bitvect(mol)

        ok = True
        for fp2 in chosen_fps:
            if DataStructs.TanimotoSimilarity(fp, fp2) > max_pair_sim:
                ok = False
                break

        if ok:
            chosen.append(smi)
            chosen_fps.append(fp)
            if scaf:
                scaffold_counts[scaf] += 1

        if len(chosen) >= topk:
            break

    return chosen


SELFIES_ALPHABET = list(sf.get_semantic_robust_alphabet())


def mutate_selfies(selfies_str: str, n_mut: int = 1) -> str:
    tokens = list(sf.split_selfies(selfies_str))
    if not tokens:
        return selfies_str

    for _ in range(n_mut):
        op = random.choice(["insert", "replace", "delete"])
        idx = random.randrange(0, len(tokens))

        if op == "insert":
            tok = random.choice(SELFIES_ALPHABET)
            tokens.insert(idx, tok)
        elif op == "replace":
            tok = random.choice(SELFIES_ALPHABET)
            tokens[idx] = tok
        elif op == "delete" and len(tokens) > 1:
            tokens.pop(idx)

    return "".join(tokens)


def generate_candidates(seed_smiles: str, n: int, n_mut_range=(1, 3), max_tries=10):
    seed_mol = Chem.MolFromSmiles(seed_smiles)
    if seed_mol is None:
        raise ValueError("Seed SMILES inválido.")

    seed_selfies = sf.encoder(seed_smiles)
    candidates = set()

    tries = 0
    while len(candidates) < n and tries < n * max_tries:
        tries += 1
        n_mut = random.randint(*n_mut_range)
        mut_s = mutate_selfies(seed_selfies, n_mut=n_mut)
        smi = sf.decoder(mut_s)

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        smi_can = Chem.MolToSmiles(mol, isomericSmiles=True)
        candidates.add(smi_can)

    return list(candidates)


def main():
    parser = argparse.ArgumentParser(description="Genera moléculas y las rankea con el modelo de evaluación.")
    parser.add_argument("--seed", required=True, help="Semilla SMILES")
    parser.add_argument("--seq-file", required=True, help="Archivo FASTA del target")
    parser.add_argument("--n-gen", type=int, default=2000)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--mut-min", type=int, default=1)
    parser.add_argument("--mut-max", type=int, default=3)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--max-sim-to-seed", type=float, default=0.90)
    parser.add_argument("--max-pair-sim", type=float, default=0.85)
    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    seq_path = Path(args.seq_file).expanduser()
    if not seq_path.is_absolute():
        seq_path = (PROJECT_ROOT / seq_path).resolve()
    else:
        seq_path = seq_path.resolve()

    target_seq = read_sequence_from_file(seq_path)
    if not target_seq:
        raise RuntimeError("No se pudo leer secuencia del FASTA.")

    print(f"Target FASTA: {seq_path}")
    print(f"Longitud target: {len(target_seq)} aa")
    print(f"Seed SMILES: {args.seed}")

    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"No se encontró el modelo: {MODEL_FILE}")
    model = joblib.load(MODEL_FILE)

    seed_mol = Chem.MolFromSmiles(args.seed)
    if seed_mol is None:
        raise ValueError("SMILES inválido (seed).")
    seed_fp = morgan_bitvect(seed_mol)

    print(f"\n=== Generando {args.n_gen} candidatos (SELFIES) ===")
    cand_smiles = generate_candidates(
        seed_smiles=args.seed,
        n=args.n_gen,
        n_mut_range=(args.mut_min, args.mut_max),
    )
    print(f"Generados (únicos, sin filtrar): {len(cand_smiles)}")

    rows = []
    print("\n=== Filtrando y evaluando ===")
    for i, smi in enumerate(cand_smiles, start=1):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        props = basic_filters(
            mol,
            seed_fp=seed_fp,
            max_abs_charge=0,
            max_sim_to_seed=args.max_sim_to_seed,
            allowed_atoms=ALLOWED_ATOMS,
        )
        if props is None:
            continue

        smi_can = props["smiles"]
        feat = featurize_pair(smi_can, target_seq, n_bits=2048)
        if feat is None:
            continue

        pki = float(model.predict(feat.reshape(1, -1))[0])
        ki_nm = pki_to_ki_nM(pki)

        row = {
            "smiles": smi_can,
            "pKi_pred": pki,
            "Ki_nM_pred": ki_nm,
            **props,
        }
        rows.append(row)

        if i % 500 == 0:
            print(f"  Procesados {i}/{len(cand_smiles)} ... válidos acumulados: {len(rows)}")

    if not rows:
        print("Ningún candidato pasó filtros. Pruebe aumentar n-gen o relajar filtros.")
        return

    df = pd.DataFrame(rows)
    df.sort_values(by="pKi_pred", ascending=False, inplace=True)

    topk = min(args.topk, len(df))
    ranked_smiles = df["smiles"].tolist()

    diverse_smiles = diverse_greedy_selection(
        ranked_smiles,
        topk=topk,
        max_pair_sim=args.max_pair_sim,
        max_per_scaffold=3,
    )

    df_top = df[df["smiles"].isin(diverse_smiles)].copy()
    df_top.sort_values(by="pKi_pred", ascending=False, inplace=True)
    df_top.reset_index(drop=True, inplace=True)

    out_file = OUT_DIR / "generated_ranked_top.csv"
    df_top.to_csv(out_file, index=False)

    print("\nListo")
    print(f"Total válidos tras filtros: {len(df)}")
    print(f"Top-{len(df_top)} guardado en: {out_file}")
    print("\nTop 5:")
    cols = ["pKi_pred", "Ki_nM_pred", "MW", "logP", "net_charge", "sim_to_seed", "smiles"]
    print(df_top[cols].head(5).to_string(index=False))


if __name__ == "__main__":
    main()
