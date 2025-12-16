
import pandas as pd
import numpy as np
from pathlib import Path

SMILES_COL = "Ligand SMILES"
KI_COL = "Ki (nM)"
SEQ_COL = "BindingDB Target Chain Sequence 1"

CHUNK_SIZE = 50000

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]

BINDINGDB_FILE = PROJECT_ROOT / "data" / "raw" / "bindingdb" / "bindingdb.tsv"
OUT_FILE = PROJECT_ROOT / "data" / "processed" / "bindingdb.csv"


def process_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    """
    Se conserva:
      - smiles
      - Ki (nM)
      - secuencia de la proteína (cadena 1)
    Filtros:
      - filas sin SMILES
      - filas sin Ki o Ki <= 0
      - filas sin secuencia de proteína
    Se calcula:
      - pKi = 9 - log10(Ki_nM)
    """
    sub = chunk[[SMILES_COL, KI_COL, SEQ_COL]].copy()
    sub.rename(
        columns={
            SMILES_COL: "smiles",
            KI_COL: "Ki_nM",
            SEQ_COL: "target_seq",
        },
        inplace=True,
    )

    sub["Ki_nM"] = pd.to_numeric(sub["Ki_nM"], errors="coerce")

    sub.dropna(subset=["smiles", "Ki_nM", "target_seq"], inplace=True)
    sub = sub[sub["Ki_nM"] > 0]

    sub["pKi"] = 9.0 - np.log10(sub["Ki_nM"])

    sub = sub[sub["target_seq"].str.len() >= 30]

    return sub[["smiles", "target_seq", "pKi"]]


if __name__ == "__main__":
    path = BINDINGDB_FILE
    print(f"Leyendo BindingDB por chunks desde: {path}")

    chunks = []
    total_rows = 0
    valid_rows = 0

    for i, chunk in enumerate(
        pd.read_csv(
            path,
            sep="\t",
            usecols=[SMILES_COL, KI_COL, SEQ_COL],
            chunksize=CHUNK_SIZE,
            low_memory=False,
        )
    ):
        total_rows += len(chunk)
        df_proc = process_chunk(chunk)
        valid_rows += len(df_proc)
        chunks.append(df_proc)

        print(
            f"Chunk {i+1}: leídas {len(chunk)} filas, "
            f"válidas acumuladas: {valid_rows} (de {total_rows} totales)"
        )

    if not chunks:
        raise RuntimeError("No se procesó ningún chunk. Revisa los nombres de columnas.")

    df_all = pd.concat(chunks, ignore_index=True)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_csv(OUT_FILE, index=False)

    print("\n======================================")
    print(f"Filas totales leídas: {total_rows}")
    print(f"Filas válidas: {len(df_all)}")
    print(f"Dataset guardado en: {OUT_FILE}")
    print("======================================")
