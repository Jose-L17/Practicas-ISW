
import io
import random
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import selfies as sf
from pathlib import Path
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.warning")
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Crippen, Lipinski, DataStructs, rdMolDescriptors


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
MODEL_FILE = PROJECT_ROOT / "models" / "rf_bindingdb.joblib"


AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

HYDROPHOBIC = set("AVLIMFWY")
POLAR = set("STNQC")
POSITIVE = set("KRH")
NEGATIVE = set("DE")
SPECIAL = set("GP")


def read_sequence_from_text(text: str) -> str:
    lines = text.splitlines()
    seq_lines = [ln.strip() for ln in lines if ln.strip() and not ln.startswith(">")]
    return "".join(seq_lines).strip().upper()


def smiles_to_morgan_fp(smiles: str, radius: int = 2, n_bits: int = 2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(float)


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


def featurize_pair(smiles: str, target_seq: str, n_bits: int = 2048):
    fp = smiles_to_morgan_fp(smiles, n_bits=n_bits)
    if fp is None:
        return None
    return np.concatenate([fp, seq_to_descriptor(target_seq)])


def pki_to_ki_nM(pki: float) -> float:
    return 10 ** (9.0 - pki)


def label_affinity(pki: float) -> str:
    if pki >= 8:
        return "Muy alta"
    if pki >= 7:
        return "Alta"
    if pki >= 6:
        return "Moderada"
    return "Baja"


@st.cache_resource
def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"No se encontró el modelo en: {MODEL_FILE}")
    return joblib.load(MODEL_FILE)


SELFIES_ALPHABET = list(sf.get_semantic_robust_alphabet())

ALLOWED_ATOMS = {"C", "N", "O", "S", "F", "Cl", "Br", "I"}


def net_charge(mol: Chem.Mol) -> int:
    return int(sum(a.GetFormalCharge() for a in mol.GetAtoms()))


def allowed_atoms_only(mol: Chem.Mol, allowed: set[str]) -> bool:
    return all(a.GetSymbol() in allowed for a in mol.GetAtoms())


def has_reactive_patterns(smiles: str) -> bool:
    bad_tokens = ["[S+]", "[P+]", "[P-]", "[C-]", "[O+]", "[S-]"]
    return any(tok in smiles for tok in bad_tokens)


def morgan_bitvect(mol: Chem.Mol, radius: int = 2, n_bits: int = 2048):
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def diverse_greedy_selection(sorted_smiles: list[str], topk: int, max_pair_sim: float = 0.85):
    chosen = []
    chosen_fps = []

    for smi in sorted_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
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

        if len(chosen) >= topk:
            break

    return chosen


def basic_filters(
    mol: Chem.Mol,
    seed_fp=None,
    max_abs_charge: int = 1,
    max_sim_to_seed: float = 0.90,
    allowed_atoms: set[str] | None = None,
) -> dict | None:
    try:
        smi_can = Chem.MolToSmiles(mol, isomericSmiles=True)

        if has_reactive_patterns(smi_can):
            return None

        q = net_charge(mol)
        if abs(q) > max_abs_charge:
            return None

        if allowed_atoms is None:
            allowed_atoms = ALLOWED_ATOMS
        if not allowed_atoms_only(mol, allowed_atoms):
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


def load_target_seq_ui():
    st.markdown("### Proteína (target)")

    tab1, tab2 = st.tabs(["Pegar secuencia / FASTA", "Subir archivo FASTA"])
    seq_text = ""
    uploaded = None

    with tab1:
        seq_text = st.text_area(
            "Pega aquí la secuencia (puede ser FASTA con header > )",
            height=160,
            value="",
        )

    with tab2:
        uploaded = st.file_uploader("Sube un .fasta / .fa / .txt", type=["fasta", "fa", "txt"])

    if uploaded is not None:
        content = uploaded.read().decode("utf-8", errors="ignore")
        return read_sequence_from_text(content)
    return read_sequence_from_text(seq_text)


def main():
    st.set_page_config(page_title="Drug Discovery MVP", layout="wide")
    st.title("Drug Discovery MVP")

    with st.expander("Ruta del modelo", expanded=False):
        st.code(str(MODEL_FILE))

    mode = st.sidebar.radio(
        "Modo",
        ["Predecir afinidad", "Generar + Rankear moléculas"],
        index=0,
    )

    model = load_model()

    if mode == "Predecir afinidad":
        st.subheader("Predicción (1 ligando + 1 proteína)")

        smiles = st.text_input("SMILES", value="CCO")
        target_seq = load_target_seq_ui()

        colA, colB = st.columns(2)
        with colA:
            st.metric("Longitud secuencia", f"{len(target_seq)} aa" if target_seq else "—")
        with colB:
            mol_ok = Chem.MolFromSmiles(smiles.strip()) is not None if smiles.strip() else False
            st.metric("SMILES válido", "Sí" if mol_ok else "No")

        if st.button("Predecir", type="primary"):
            if not smiles.strip():
                st.error("Proporciona un SMILES.")
                st.stop()
            if not target_seq.strip():
                st.error("Proporciona una secuencia de proteína (texto o archivo).")
                st.stop()

            feat = featurize_pair(smiles.strip(), target_seq.strip(), n_bits=2048)
            if feat is None:
                st.error("SMILES inválido: RDKit no pudo interpretarlo.")
                st.stop()

            pki_pred = float(model.predict(feat.reshape(1, -1))[0])
            ki_pred = pki_to_ki_nM(pki_pred)
            qual = label_affinity(pki_pred)

            st.success("Predicción lista")
            c1, c2, c3 = st.columns(3)
            c1.metric("pKi", f"{pki_pred:.3f}")
            c2.metric("Ki estimado (nM)", f"{ki_pred:,.1f}")
            c3.metric("Afinidad", qual)

            report = (
                f"SMILES\t{smiles.strip()}\n"
                f"len_seq\t{len(target_seq)}\n"
                f"pKi\t{pki_pred:.4f}\n"
                f"Ki_nM\t{ki_pred:.2f}\n"
                f"label\t{qual}\n"
            )
            st.download_button(
                "Descargar resultado (TXT)",
                data=report.encode("utf-8"),
                file_name="prediction_result.txt",
                mime="text/plain",
            )

    else:
        st.subheader("Generar + Filtrar + Rankear (seed + proteína → Top-K)")

        col1, col2 = st.columns([2, 1])

        with col1:
            seed = st.text_input("Seed SMILES", value="CC(=O)NC1=NN=C(S1)S(=O)(=O)N")
            target_seq = load_target_seq_ui()

        with col2:
            st.markdown("### Parámetros")
            n_gen = st.number_input("n_gen (candidatos)", min_value=200, max_value=20000, value=3000, step=100)
            topk = st.number_input("topK (salida)", min_value=10, max_value=500, value=50, step=10)
            mut_min = st.number_input("mut_min", min_value=1, max_value=10, value=1, step=1)
            mut_max = st.number_input("mut_max", min_value=1, max_value=10, value=3, step=1)
            rnd_seed = st.number_input("random_seed", min_value=0, max_value=10_000_000, value=42, step=1)

            st.markdown("### Filtros extra")
            max_abs_charge = st.number_input("max_abs_charge", min_value=0, max_value=5, value=1, step=1)
            max_sim_to_seed = st.slider("max_sim_to_seed (anti-clones)", 0.50, 0.99, 0.90, 0.01)
            max_pair_sim = st.slider("max_pair_sim (diversidad topK)", 0.50, 0.99, 0.85, 0.01)

        colA, colB = st.columns(2)
        with colA:
            st.metric("Longitud secuencia", f"{len(target_seq)} aa" if target_seq else "—")
        with colB:
            mol_ok = Chem.MolFromSmiles(seed.strip()) is not None if seed.strip() else False
            st.metric("Seed SMILES válido", "Sí" if mol_ok else "No")

        if st.button("Generar y rankear", type="primary"):
            if not seed.strip() or Chem.MolFromSmiles(seed.strip()) is None:
                st.error("Seed SMILES inválido.")
                st.stop()
            if not target_seq.strip():
                st.error("Proporciona una secuencia de proteína (texto o archivo).")
                st.stop()
            if mut_min > mut_max:
                st.error("mut_min no puede ser mayor que mut_max.")
                st.stop()

            random.seed(int(rnd_seed))
            np.random.seed(int(rnd_seed))

            seed_mol = Chem.MolFromSmiles(seed.strip())
            seed_fp = morgan_bitvect(seed_mol)

            st.info("Generando candidatos…")
            cand_smiles = generate_candidates(
                seed_smiles=seed.strip(),
                n=int(n_gen),
                n_mut_range=(int(mut_min), int(mut_max)),
            )
            st.write(f"Generados (únicos, sin filtrar): **{len(cand_smiles)}**")

            rows = []
            prog = st.progress(0)
            status = st.empty()

            total = len(cand_smiles)
            for i, smi in enumerate(cand_smiles, start=1):
                mol = Chem.MolFromSmiles(smi)
                if mol is None:
                    continue

                props = basic_filters(
                    mol,
                    seed_fp=seed_fp,
                    max_abs_charge=int(max_abs_charge),
                    max_sim_to_seed=float(max_sim_to_seed),
                    allowed_atoms=ALLOWED_ATOMS,
                )
                if props is None:
                    continue

                smi_can = props["smiles"]
                feat = featurize_pair(smi_can, target_seq.strip(), n_bits=2048)
                if feat is None:
                    continue

                pki = float(model.predict(feat.reshape(1, -1))[0])
                ki_nm = pki_to_ki_nM(pki)

                rows.append(
                    {
                        "smiles": smi_can,
                        "pKi_pred": pki,
                        "Ki_nM_pred": ki_nm,
                        **props,
                    }
                )

                if i % 100 == 0 or i == total:
                    prog.progress(int(i / total * 100))
                    status.write(f"Procesados {i}/{total} — válidos: {len(rows)}")

            if not rows:
                st.error("Ningún candidato pasó filtros. Prueba aumentar n_gen o relajar filtros.")
                st.stop()

            df = pd.DataFrame(rows)
            df.sort_values(by="pKi_pred", ascending=False, inplace=True)

            # diversidad en el topK
            ranked_smiles = df["smiles"].tolist()
            diverse_smiles = diverse_greedy_selection(
                ranked_smiles,
                topk=int(min(topk, len(df))),
                max_pair_sim=float(max_pair_sim),
            )

            df_top = df[df["smiles"].isin(diverse_smiles)].copy()
            df_top.sort_values(by="pKi_pred", ascending=False, inplace=True)
            df_top.reset_index(drop=True, inplace=True)

            st.success("Listo")
            st.write(f"Total válidos tras filtros: **{len(df)}**")
            st.write(f"Top-{len(df_top)} (diverso)")

            st.dataframe(df_top, use_container_width=True)

            csv_bytes = df_top.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Descargar Top-K (CSV)",
                data=csv_bytes,
                file_name="generated_ranked_top.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()
