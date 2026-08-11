"""
main.py

CLI entry point: give it a SMILES string, it predicts binding affinity
(kcal/mol, delta-corrected toward exhaustive-docking quality) for the
trained c-Met (PDB 4R1V) model.

Usage:
    python3 main.py "CCOc1ccc2nc(S(N)(=O)=O)sc2c1"
    python3 main.py --file my_smiles_list.txt
"""

import os
import sys
import argparse
import tempfile
import subprocess
import types
import numpy as np
import pandas as pd
import joblib

# --- Compatibility shim: mordred (unmaintained) still uses numpy.product,
# which numpy 2.0 removed. This model bundle needs a modern numpy for
# unpickling, so we patch the missing alias back in rather than downgrade.
if not hasattr(np, "product"):
    np.product = np.prod

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VINA_EXE = os.path.join(BASE_DIR, "vina.exe" if os.name == "nt" else "vina")

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from mordred import Calculator, descriptors

from common_features import build_receptor_typed_index, extract_vina_typed_features_from_file
from stacking_utils import GroupAwareStackingRegressor  # noqa: F401


def _register_fake_mordred():
    if "mordred._base" in sys.modules and hasattr(sys.modules.get("mordred._base"), "pandas_module"):
        return
    import pandas as _pd

    class MordredDataFrame(_pd.DataFrame):
        pass

    sub = types.ModuleType("mordred._base")
    sub2 = types.ModuleType("mordred._base.pandas_module")
    sub2.MordredDataFrame = MordredDataFrame
    sys.modules["mordred._base"] = sub
    sys.modules["mordred._base.pandas_module"] = sub2
    sub.pandas_module = sub2


_register_fake_mordred()

RECEPTOR_PDBQT = os.path.join(BASE_DIR, "receptor", "4R1V_receptor.pdbqt")
CENTER = (108.564, 20.309, 141.504)
SIZE = (20, 20, 20)
CHEAP_EXHAUSTIVENESS = 3
CHEAP_NUM_MODES = 1
VINA_TIMEOUT_SEC = 200

BUNDLE_PATH_TUNED = "ml_prediction_data/tabpred_tuned_bundle.pkl"
BUNDLE_PATH_TYPED = "ml_prediction_data/tabpred_delta_typed_bundle.pkl"


def smiles_to_pdbqt(smiles, out_path):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles}")
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    embed_result = AllChem.EmbedMolecule(mol, params)
    if embed_result != 0:
        params.useRandomCoords = True
        embed_result = AllChem.EmbedMolecule(mol, params)
        if embed_result != 0:
            raise ValueError(f"3D embedding failed for: {smiles}")

    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=1000)
    except Exception:
        AllChem.UFFOptimizeMolecule(mol, maxIters=1000)

    from meeko import MoleculePreparation, PDBQTWriterLegacy
    preparator = MoleculePreparation()
    mol_setups = preparator.prepare(mol)
    pdbqt_string, is_ok, err_msg = PDBQTWriterLegacy.write_string(mol_setups[0])
    if not is_ok:
        raise ValueError(f"Meeko preparation failed for {smiles}: {err_msg}")

    with open(out_path, "w") as f:
        f.write(pdbqt_string)

    return mol


def run_cheap_dock(ligand_pdbqt_path, out_pdbqt_path):
    cmd = [
        VINA_EXE,
        "--receptor", RECEPTOR_PDBQT,
        "--ligand", ligand_pdbqt_path,
        "--center_x", str(CENTER[0]),
        "--center_y", str(CENTER[1]),
        "--center_z", str(CENTER[2]),
        "--size_x", str(SIZE[0]),
        "--size_y", str(SIZE[1]),
        "--size_z", str(SIZE[2]),
        "--exhaustiveness", str(CHEAP_EXHAUSTIVENESS),
        "--num_modes", str(CHEAP_NUM_MODES),
        "--cpu", "1",
        "--out", out_pdbqt_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=VINA_TIMEOUT_SEC)
    if result.returncode != 0:
        raise RuntimeError(f"Vina failed:\n{result.stdout}\n{result.stderr}")


def compute_2d_descriptors(smiles, mordred_columns_kept):
    mol = Chem.MolFromSmiles(smiles)
    calc = Calculator(descriptors, ignore_3D=True)
    result = calc(mol)
    row = pd.Series(dict(result.items()))
    row = pd.to_numeric(row, errors="coerce")
    aligned = row.reindex(mordred_columns_kept).to_numpy(dtype=np.float64)
    return aligned.reshape(1, -1)


def load_resources():
    bundle_path = BUNDLE_PATH_TUNED if os.path.exists(BUNDLE_PATH_TUNED) else BUNDLE_PATH_TYPED
    bundle = joblib.load(bundle_path)
    rec_coords, rec_types, rec_res_idx, num_res = build_receptor_typed_index(RECEPTOR_PDBQT)
    return bundle, rec_coords, rec_types, rec_res_idx, num_res, bundle_path


def predict(smiles, bundle, rec_coords, rec_types, rec_res_idx, num_res):
    with tempfile.TemporaryDirectory() as tmpdir:
        lig_pdbqt = os.path.join(tmpdir, "ligand.pdbqt")
        docked_pdbqt = os.path.join(tmpdir, "ligand_out.pdbqt")

        smiles_to_pdbqt(smiles, lig_pdbqt)
        run_cheap_dock(lig_pdbqt, docked_pdbqt)

        x_3d = extract_vina_typed_features_from_file(
            docked_pdbqt, rec_coords, rec_types, rec_res_idx, num_res, model_index=1
        ).reshape(1, -1)

    x_2d_raw = compute_2d_descriptors(smiles, bundle["mordred_columns_kept"])
    x_2d_imputed = bundle["imputer"].transform(x_2d_raw)

    x_combined = np.hstack([x_2d_imputed, x_3d])
    x_final = bundle["pipeline_pre"].transform(x_combined)

    if "feature_mask" in bundle:
        x_final = x_final[:, bundle["feature_mask"]]

    pred = bundle["model"].predict(x_final)[0]
    return float(pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("smiles", nargs="?", help="A single SMILES string")
    parser.add_argument("--file", help="Path to a text file with one SMILES per line")
    args = parser.parse_args()

    if not args.smiles and not args.file:
        print("Provide a SMILES string or --file path/to/list.txt")
        sys.exit(1)

    print("Loading trained model bundle + receptor index...")
    bundle, rec_coords, rec_types, rec_res_idx, num_res, bundle_path = load_resources()
    print(f"  using: {bundle_path}")

    smiles_list = []
    if args.smiles:
        smiles_list.append(args.smiles)
    if args.file:
        with open(args.file) as f:
            smiles_list.extend([line.strip().split()[0] for line in f if line.strip()])

    for smi in smiles_list:
        try:
            affinity = predict(smi, bundle, rec_coords, rec_types, rec_res_idx, num_res)
            print(f"{smi}\tpredicted_affinity_kcal_mol = {affinity:.3f}")
        except Exception as e:
            print(f"{smi}\tFAILED: {e}")


if __name__ == "__main__":
    main()
