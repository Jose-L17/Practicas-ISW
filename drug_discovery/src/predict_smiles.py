
import argparse
import sys
import numpy as np
import joblib
from pathlib import Path
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.warning")
from rdkit import Chem
from rdkit.Chem import AllChem


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


def read_sequence_from_file(path: Path) -> str:
    """
    Lee una secuencia desde un archivo:
      - Si es FASTA, ignora headers '>'
      - Si es texto plano, junta líneas
    """
    text = path.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
    if not text:
        return ""
    seq_lines = [ln.strip() for ln in text if ln.strip() and not ln.startswith(">")]
    return "".join(seq_lines).strip()


def smiles_to_morgan_fp(smiles: str, radius: int = 2, n_bits: int = 2048):
    """SMILES -> Morgan FP -> np.array float. None si falla."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(float)


def seq_to_descriptor(seq: str) -> np.ndarray:
    """
    Proteína -> vector:
      - 20 frecuencias de aminoácidos
      - longitud normalizada
      - 5 fracciones por grupos (hidrofóbicos, polares, +, -, especiales)
    Total: 26 features
    """
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
    """pKi -> Ki(nM)"""
    return 10 ** (9.0 - pki)


def label_affinity(pki: float) -> str:
    if pki >= 8:
        return "Muy alta"
    if pki >= 7:
        return "Alta"
    if pki >= 6:
        return "Moderada"
    return "Baja"


def main():
    parser = argparse.ArgumentParser(
        description="Predicción -> pKi / Ki(nM)"
    )
    parser.add_argument("smiles", nargs="?", help="SMILES (si no se da, se pedirá por input)")
    parser.add_argument(
        "--seq",
        default=None,
        help="Secuencia de proteína como string (si no se da, se pedirá por input)",
    )
    parser.add_argument(
        "--seq-file",
        default=None,
        help="Archivo con secuencia (FASTA o texto). Si se da, tiene prioridad sobre --seq.",
    )
    args = parser.parse_args()

    smiles = args.smiles or input("Introduce un SMILES: ").strip()
    if not smiles:
        print("No se proporcionó SMILES.")
        sys.exit(1)

    if args.seq_file:
        seq_path = Path(args.seq_file).expanduser().resolve()

        print(f"Ruta fasta resuelta: {seq_path}")

        if not seq_path.exists():
            print(f"No existe el archivo de secuencia: {seq_path}")
            sys.exit(1)

        target_seq = read_sequence_from_file(seq_path)
    elif args.seq:
        target_seq = args.seq.strip()
    else:
        target_seq = input("Introduce la secuencia de la proteína (cadena 1): ").strip()

    if not target_seq:
        print("No se proporcionó secuencia de proteína.")
        sys.exit(1)

    if len(target_seq) < 50:
        print("Aviso: la secuencia es muy corta (<50 aa). "
              "Para resultados realistas usa una proteína completa (FASTA/UniProt).")

    print(f"Cargando modelo desde: {MODEL_FILE}")
    if not MODEL_FILE.exists():
        print("No se encontró el archivo del modelo.")
        print(f"   Coloca el .joblib en: {MODEL_FILE.parent}")
        sys.exit(1)

    model = joblib.load(MODEL_FILE)

    feat = featurize_pair(smiles, target_seq, n_bits=2048)
    if feat is None:
        print("SMILES inválido: RDKit no pudo interpretarlo.")
        sys.exit(1)

    pki_pred = float(model.predict(feat.reshape(1, -1))[0])
    ki_pred_nM = pki_to_ki_nM(pki_pred)

    print("\n================= RESULTADO =================")
    print(f"SMILES:              {smiles}")
    print(f"Longitud secuencia:  {len(target_seq)} aa")
    print(f"pKi predicho:        {pki_pred:.3f}")
    print(f"Ki estimado:         {ki_pred_nM:.1f} nM")
    print(f"Evaluación:          {label_affinity(pki_pred)} afinidad")
    print("============================================================================")


if __name__ == "__main__":
    main()
