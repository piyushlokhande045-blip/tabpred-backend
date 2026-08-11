"""
common_features.py

Shared feature-extraction logic used by BOTH the training pipeline and the
SMILES->prediction inference pipeline. Keeping this in one module guarantees
that features are computed identically at train time and predict time --
any mismatch here is a silent, hard-to-debug accuracy killer.
"""

import os
import numpy as np
from scipy.spatial.distance import cdist


def build_receptor_residue_index(receptor_pdb_path):
    rec_coords = []
    rec_res_keys = []

    with open(receptor_pdb_path, "r") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                elem = line[76:78].strip()
                if elem == "H" or line[12:16].strip().startswith("H"):
                    continue
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    res_num = line[22:26].strip()
                    res_name = line[17:20].strip()
                    rec_coords.append([x, y, z])
                    rec_res_keys.append(f"{res_name}_{res_num}")
                except ValueError:
                    continue

    rec_coords = np.array(rec_coords, dtype=np.float32)
    unique_residues = sorted(list(set(rec_res_keys)))
    res_to_idx = {r: i for i, r in enumerate(unique_residues)}
    num_res = len(unique_residues)
    return rec_coords, rec_res_keys, res_to_idx, num_res


def parse_ligand_pose_coords(pdbqt_path, model_index=1):
    lig_coords = []
    in_target_model = (model_index == 1)
    seen_any_model_line = False

    with open(pdbqt_path, "r") as pf:
        for line in pf:
            if line.startswith("MODEL"):
                seen_any_model_line = True
                try:
                    this_model_num = int(line.split()[1])
                except (IndexError, ValueError):
                    this_model_num = None
                in_target_model = (this_model_num == model_index)
                continue
            if line.startswith("ENDMDL"):
                if seen_any_model_line and in_target_model:
                    break
                continue
            if in_target_model and (line.startswith("ATOM") or line.startswith("HETATM")):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    lig_coords.append([x, y, z])
                except ValueError:
                    continue

    if len(lig_coords) == 0:
        return None
    return np.array(lig_coords, dtype=np.float32)


def compute_pose_features(lig_coords, rec_coords, rec_res_keys, res_to_idx, num_res):
    feat = np.zeros(num_res * 4, dtype=np.float32)
    if lig_coords is None or len(lig_coords) == 0:
        return feat

    dists = cdist(lig_coords, rec_coords)

    for rec_j, res_key in enumerate(rec_res_keys):
        res_idx = res_to_idx[res_key]
        col_base = res_idx * 4

        min_dists = dists[:, rec_j]
        d3 = np.sum(min_dists < 3.0)
        d45 = np.sum(min_dists < 4.5)
        d6 = np.sum(min_dists < 6.0)

        valid_d = min_dists[min_dists < 6.0]
        inv_d = np.sum(1.0 / (valid_d + 0.1)) if len(valid_d) > 0 else 0.0

        feat[col_base] += d3
        feat[col_base + 1] += d45
        feat[col_base + 2] += d6
        feat[col_base + 3] += inv_d

    return feat


def extract_pose_features_from_file(pdbqt_path, rec_coords, rec_res_keys, res_to_idx, num_res, model_index=1):
    lig_coords = parse_ligand_pose_coords(pdbqt_path, model_index=model_index)
    return compute_pose_features(lig_coords, rec_coords, rec_res_keys, res_to_idx, num_res)


# =====================================================================
# ATOM-TYPED VINA-LIKE FEATURES
# =====================================================================

_VDW_RADII = {
    "C": 1.90, "A": 1.90, "N": 1.75, "NA": 1.75, "NS": 1.75,
    "OA": 1.60, "OS": 1.60, "O": 1.60, "S": 2.00, "SA": 2.00,
    "H": 1.00, "HD": 1.00, "HS": 1.00,
    "F": 1.54, "Cl": 2.04, "CL": 2.04, "Br": 2.16, "BR": 2.16, "I": 2.36,
    "P": 2.10, "Mg": 0.65, "MG": 0.65, "Ca": 0.99, "CA": 0.99,
    "Mn": 0.65, "MN": 0.65, "Fe": 0.65, "FE": 0.65, "Zn": 0.74, "ZN": 0.74,
}
_DEFAULT_RADIUS = 1.80

_HYDROPHOBIC_TYPES = {"C", "A", "F", "Cl", "CL", "Br", "BR", "I"}
_ACCEPTOR_TYPES = {"NA", "OA", "SA", "NS", "OS"}
_DONOR_TYPES = {"HD", "HS"}

_PAIR_CUTOFF = 8.0


def _radius_for_type(t):
    return _VDW_RADII.get(t, _DEFAULT_RADIUS)


def parse_typed_atoms_from_pdbqt(pdbqt_path, model_index=1):
    coords = []
    types = []
    in_target_model = (model_index == 1)
    seen_any_model_line = False

    with open(pdbqt_path, "r") as pf:
        for line in pf:
            if line.startswith("MODEL"):
                seen_any_model_line = True
                try:
                    this_model_num = int(line.split()[1])
                except (IndexError, ValueError):
                    this_model_num = None
                in_target_model = (this_model_num == model_index)
                continue
            if line.startswith("ENDMDL"):
                if seen_any_model_line and in_target_model:
                    break
                continue
            if in_target_model and (line.startswith("ATOM") or line.startswith("HETATM")):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                except ValueError:
                    continue
                parts = line.split()
                atype = parts[-1] if parts else "C"
                coords.append([x, y, z])
                types.append(atype)

    if len(coords) == 0:
        return None, None
    return np.array(coords, dtype=np.float32), types


def build_receptor_typed_index(receptor_pdbqt_path):
    coords = []
    types = []
    res_keys = []

    with open(receptor_pdbqt_path, "r") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                try:
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    res_num = line[22:26].strip()
                    res_name = line[17:20].strip()
                except ValueError:
                    continue
                parts = line.split()
                atype = parts[-1] if parts else "C"
                coords.append([x, y, z])
                types.append(atype)
                res_keys.append(f"{res_name}_{res_num}")

    coords = np.array(coords, dtype=np.float32)
    unique_residues = sorted(list(set(res_keys)))
    res_to_idx = {r: i for i, r in enumerate(unique_residues)}
    res_idx = np.array([res_to_idx[r] for r in res_keys], dtype=np.int64)
    num_res = len(unique_residues)
    return coords, types, res_idx, num_res


def get_num_active_torsions(pdbqt_path):
    try:
        with open(pdbqt_path, "r") as f:
            for line in f:
                if "active torsions" in line:
                    parts = line.split()
                    for p in parts:
                        if p.isdigit():
                            return int(p)
    except FileNotFoundError:
        pass
    return 0


def compute_vina_typed_pose_features(lig_coords, lig_types, rec_coords, rec_types_arr,
                                      rec_res_idx, num_res):
    n_terms = 5
    out = np.zeros((num_res, n_terms), dtype=np.float64)
    if lig_coords is None or len(lig_coords) == 0:
        return out.flatten().astype(np.float32)

    lig_radii = np.array([_radius_for_type(t) for t in lig_types], dtype=np.float64)
    lig_hydrophobic = np.array([t in _HYDROPHOBIC_TYPES for t in lig_types], dtype=bool)
    lig_donor = np.array([t in _DONOR_TYPES for t in lig_types], dtype=bool)
    lig_acceptor = np.array([t in _ACCEPTOR_TYPES for t in lig_types], dtype=bool)

    rec_radii = np.array([_radius_for_type(t) for t in rec_types_arr], dtype=np.float64)
    rec_hydrophobic = np.array([t in _HYDROPHOBIC_TYPES for t in rec_types_arr], dtype=bool)
    rec_donor = np.array([t in _DONOR_TYPES for t in rec_types_arr], dtype=bool)
    rec_acceptor = np.array([t in _ACCEPTOR_TYPES for t in rec_types_arr], dtype=bool)

    dists = cdist(lig_coords.astype(np.float64), rec_coords.astype(np.float64))
    within_cutoff = dists < _PAIR_CUTOFF
    if not within_cutoff.any():
        return out.flatten().astype(np.float32)

    d = dists - lig_radii[:, None] - rec_radii[None, :]

    gauss1 = np.exp(-((d / 0.5) ** 2)) * within_cutoff
    gauss2 = np.exp(-(((d - 3.0) / 2.0) ** 2)) * within_cutoff
    repulsion = np.where(d < 0, d ** 2, 0.0) * within_cutoff

    hydrophobic_mask = np.outer(lig_hydrophobic, rec_hydrophobic) & within_cutoff
    hydrophobic_val = np.clip(1.5 - d, 0.0, 1.0) * hydrophobic_mask

    hbond_mask = (np.outer(lig_donor, rec_acceptor) | np.outer(lig_acceptor, rec_donor)) & within_cutoff
    hbond_val = np.clip(-d / 0.7, 0.0, 1.0) * hbond_mask

    for term_idx, term_matrix in enumerate([gauss1, gauss2, repulsion, hydrophobic_val, hbond_val]):
        per_atom = term_matrix.sum(axis=0)
        np.add.at(out[:, term_idx], rec_res_idx, per_atom)

    return out.flatten().astype(np.float32)


def extract_vina_typed_features_from_file(pose_pdbqt_path, rec_coords, rec_types_arr,
                                           rec_res_idx, num_res, model_index=1):
    lig_coords, lig_types = parse_typed_atoms_from_pdbqt(pose_pdbqt_path, model_index=model_index)
    typed_feats = compute_vina_typed_pose_features(
        lig_coords, lig_types, rec_coords, rec_types_arr, rec_res_idx, num_res
    )
    n_atoms = 0 if lig_coords is None else len(lig_coords)
    n_rot = get_num_active_torsions(pose_pdbqt_path)
    extra = np.array([n_atoms, n_rot], dtype=np.float32)
    return np.concatenate([typed_feats, extra])
