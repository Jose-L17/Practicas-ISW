
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.warning")
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.metrics import r2_score, mean_squared_error


THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]

DATA_FILE = PROJECT_ROOT / "data" / "processed" / "bindingdb.csv"
MODEL_FILE = PROJECT_ROOT / "models" / "rf_bindingdb.joblib"

MAX_SAMPLES = 150000

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_INDEX = {aa: i for i, aa in enumerate(AMINO_ACIDS)}

HYDROPHOBIC = set("AVLIMFWY")
POLAR = set("STNQC")
POSITIVE = set("KRH")
NEGATIVE = set("DE")
SPECIAL = set("GP") 


def smiles_to_morgan_fp(smiles: str, radius: int = 2, n_bits: int = 2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros((n_bits,), dtype=np.int8)
    Chem.DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.astype(float)


def seq_to_descriptor(seq: str) -> np.ndarray:
    """
    Convierte una secuencia de aminoácidos en:
      - composición relativa de los 20 aminoácidos (frecuencia)
      - longitud normalizada (len(seq) / 1000, cap a 1.5)
      - fracciones de grupos: hidrofóbicos, polares, positivos,
        negativos y especiales (G, P)
    """
    seq = seq.strip().upper()
    counts = np.zeros(len(AMINO_ACIDS), dtype=float)
    total = 0

    hydrophobic_count = 0
    polar_count = 0
    positive_count = 0
    negative_count = 0
    special_count = 0

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
        frac_hydrophobic = hydrophobic_count / total
        frac_polar = polar_count / total
        frac_positive = positive_count / total
        frac_negative = negative_count / total
        frac_special = special_count / total
    else:
        freqs = counts
        frac_hydrophobic = 0.0
        frac_polar = 0.0
        frac_positive = 0.0
        frac_negative = 0.0
        frac_special = 0.0

    length_norm = min(len(seq) / 1000.0, 1.5)

    group_feats = np.array(
        [
            frac_hydrophobic,
            frac_polar,
            frac_positive,
            frac_negative,
            frac_special,
        ],
        dtype=float,
    )

    return np.concatenate([freqs, np.array([length_norm], dtype=float), group_feats])


def featurize_row(smiles: str, seq: str, n_bits: int = 2048) -> np.ndarray | None:
    fp = smiles_to_morgan_fp(smiles, n_bits=n_bits)
    if fp is None:
        return None
    seq_desc = seq_to_descriptor(seq)
    return np.concatenate([fp, seq_desc])


def featurize_dataframe(df: pd.DataFrame, n_bits: int = 2048):
    features = []
    ys = []

    for i, row in df.iterrows():
        smi = row["smiles"]
        seq = row["target_seq"]
        y = row["pKi"]

        feat = featurize_row(smi, seq, n_bits=n_bits)
        if feat is not None:
            features.append(feat)
            ys.append(y)

        if (i + 1) % 10000 == 0:
            print(f"  Procesadas {i+1} filas...")

    X = np.vstack(features)
    y = np.array(ys, dtype=float)

    print(f"\nTotal de pares featurizados: {len(y)}")
    print(f"Dimensión del vector de características: {X.shape[1]}")
    return X, y


if __name__ == "__main__":
    print(f"Leyendo dataset desde: {DATA_FILE}")
    df = pd.read_csv(DATA_FILE)

    print(f"Filas totales en bindingdb: {len(df)}")

    if len(df) > MAX_SAMPLES:
        df = df.sample(n=MAX_SAMPLES, random_state=42)
        print(f"Usando subconjunto de {MAX_SAMPLES} filas para el prototipo.")

    print("\n=== Featurizando ===")
    X, y = featurize_dataframe(df, n_bits=2048)

    print("\n=== Dividiendo en train/test ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\n=== Entrenando ===")
    model = ExtraTreesRegressor(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=1,
        n_jobs=-1,
        random_state=42,
    )

    model.fit(X_train, y_train)

    print("\n=== Evaluación en test ===")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print(f"R² (test):  {r2:.3f}")
    print(f"RMSE (test): {rmse:.3f} pKi")

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)

    print(f"\nModelo guardado en: {MODEL_FILE}")
