# Copyright 2026 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Synthetic OF3 feature generation.

Produces randomly-valued feature dicts with the same keys, shapes, and dtypes
the real data pipeline emits, so the model can be exercised end-to-end without
any on-disk dataset, alignments, or templates. Used by the dummy data modules
(``openfold3.core.data.framework.dummy_data``) for profiling / benchmarking /
OOM validation, and re-exported from ``openfold3.tests.data_utils`` for the
model shape tests.

The generator here is deliberately dependency-light (no test-suite imports) so
it is safe to ship as library code. ``is_multimer`` is an explicit argument
rather than a global so callers control the chain layout.
"""

from random import randint

import numpy as np
import torch

from openfold3.core.data.primitives.featurization.structure import (
    create_atom_to_token_index,
)
from openfold3.core.data.resources.residues import (
    STANDARD_DNA_RESIDUES,
    STANDARD_PROTEIN_RESIDUES_3,
    STANDARD_RESIDUES_3,
    STANDARD_RESIDUES_WITH_GAP_3,
    STANDARD_RNA_RESIDUES,
)
from openfold3.core.data.resources.token_atom_constants import (
    TOKEN_NAME_TO_ATOM_NAMES,
)
from openfold3.core.utils.atomize_utils import broadcast_token_feat_to_atoms


def random_asym_ids(
    n_res: int,
    is_multimer: bool = False,
    split_chains: bool = True,
    min_chain_len: int = 4,
):
    """Generate random per-token chain (asym) ids.

    Args:
        n_res: Number of tokens.
        is_multimer: If False, a single chain is produced. If True, the tokens
            are split into a random number of chains (each at least
            ``min_chain_len`` long).
        split_chains: If False, returns all-zero ids (single chain, no offset).
        min_chain_len: Minimum length of each chain when splitting.
    """
    n_chain = randint(1, n_res // min_chain_len) if is_multimer else 1

    if not split_chains:
        return [0] * n_res

    assert n_res >= n_chain

    pieces = []
    asym_ids = []
    final_idx = n_chain - 1
    for idx in range(n_chain - 1):
        n_stop = n_res - sum(pieces) - n_chain + idx - min_chain_len
        if n_stop <= min_chain_len:
            final_idx = idx
            break
        piece = randint(min_chain_len, n_stop)
        pieces.append(piece)
        asym_ids.extend(piece * [idx])
    asym_ids.extend((n_res - sum(pieces)) * [final_idx])

    return np.array(asym_ids).astype(np.float32) + 1


def random_of3_features(
    batch_size: int,
    n_token: int,
    n_msa: int,
    n_templ: int,
    is_eval: bool = False,
    is_multimer: bool = False,
) -> dict:
    """Build a single pre-collated OF3 feature batch with random values.

    The returned dict already carries the batch dimension (``batch_size``); it
    is NOT a single dataset sample. Tensors are on CPU; precision conversion and
    device transfer are left to the caller / Lightning.
    """
    restypes_flat = torch.randint(0, len(STANDARD_RESIDUES_3), (n_token,))
    restypes_names = [STANDARD_RESIDUES_3[token_idx] for token_idx in restypes_flat]
    restypes_one_hot = torch.nn.functional.one_hot(
        restypes_flat,
        len(STANDARD_RESIDUES_WITH_GAP_3),
    )

    num_atoms_per_token = torch.Tensor(
        [len(TOKEN_NAME_TO_ATOM_NAMES[name]) for name in restypes_names]
    )

    is_protein = torch.Tensor(
        [1.0 if name in STANDARD_PROTEIN_RESIDUES_3 else 0.0 for name in restypes_names]
    )
    is_rna = torch.Tensor(
        [1.0 if name in STANDARD_RNA_RESIDUES else 0.0 for name in restypes_names]
    )
    is_dna = torch.Tensor(
        [1.0 if name in STANDARD_DNA_RESIDUES else 0.0 for name in restypes_names]
    )

    n_atom = torch.max(torch.sum(num_atoms_per_token, dim=-1)).int().item()

    start_atom_index = torch.concat(
        (torch.zeros((1,)), torch.cumsum(num_atoms_per_token, dim=-1)[:-1]), dim=-1
    )

    token_mask = torch.ones(n_token).float()
    atom_mask = torch.ones(n_atom).float()

    rand_token_mask = torch.randint(0, 2, (n_token,)).float()
    atom_resolved_mask = broadcast_token_feat_to_atoms(
        token_mask=token_mask,
        num_atoms_per_token=num_atoms_per_token,
        token_feat=rand_token_mask,
    )

    atom_to_token_index = create_atom_to_token_index(
        token_mask=token_mask,
        num_atoms_per_token=num_atoms_per_token,
    ).int()

    asym_id = (
        torch.Tensor(random_asym_ids(n_token, is_multimer=is_multimer))
        .unsqueeze(0)
        .repeat((batch_size, 1))
    )

    features = {
        # Input features
        "residue_index": torch.arange(0, n_token)
        .unsqueeze(0)
        .repeat((batch_size, 1))
        .int(),
        "token_index": torch.arange(0, n_token)
        .unsqueeze(0)
        .repeat((batch_size, 1))
        .int(),
        "asym_id": asym_id.int(),
        "entity_id": asym_id.clone().int(),
        "sym_id": torch.ones((batch_size, n_token)).int(),
        "restype": restypes_one_hot.unsqueeze(0).repeat((batch_size, 1, 1)).int(),
        "is_protein": is_protein.unsqueeze(0).repeat((batch_size, 1)).int(),
        "is_dna": is_dna.unsqueeze(0).repeat((batch_size, 1)).int(),
        "is_rna": is_rna.unsqueeze(0).repeat((batch_size, 1)).int(),
        "is_ligand": torch.zeros((batch_size, n_token)).int(),
        "is_atomized": torch.zeros((batch_size, n_token)).int(),
        # Reference conformation features
        "ref_pos": torch.randn((batch_size, n_atom, 3)).float(),
        "ref_mask": torch.ones((batch_size, n_atom)).int(),
        "ref_element": torch.ones((batch_size, n_atom, 119)).int(),
        "ref_charge": torch.ones((batch_size, n_atom)).float(),
        "ref_atom_name_chars": torch.ones((batch_size, n_atom, 4, 64)).int(),
        "ref_space_uid": atom_to_token_index.unsqueeze(0).repeat((batch_size, 1)),
        # MSA features
        "msa": torch.ones((batch_size, n_msa, n_token, 32)).int(),
        "has_deletion": torch.ones((batch_size, n_msa, n_token)).float(),
        "deletion_value": torch.ones((batch_size, n_msa, n_token)).float(),
        "profile": torch.ones((batch_size, n_token, 32)).float(),
        "deletion_mean": torch.ones((batch_size, n_token)).float(),
        # Template features
        "template_restype": torch.ones((batch_size, n_templ, n_token, 32)).int(),
        "template_pseudo_beta_mask": torch.ones((batch_size, n_templ, n_token)).float(),
        "template_backbone_frame_mask": torch.ones(
            (batch_size, n_templ, n_token)
        ).float(),
        "template_distogram": torch.ones(
            (batch_size, n_templ, n_token, n_token, 39)
        ).float(),
        "template_unit_vector": torch.ones(
            (batch_size, n_templ, n_token, n_token, 3)
        ).float(),
        # Bond features
        "token_bonds": torch.ones((batch_size, n_token, n_token)).int(),
        # Additional features
        "token_mask": token_mask.unsqueeze(0).repeat((batch_size, 1)),
        "atom_mask": atom_mask.unsqueeze(0).repeat((batch_size, 1)),
        "start_atom_index": start_atom_index.unsqueeze(0).repeat((batch_size, 1)).int(),
        "num_atoms_per_token": num_atoms_per_token.unsqueeze(0)
        .repeat((batch_size, 1))
        .int(),
        "atom_to_token_index": atom_to_token_index.unsqueeze(0).repeat((batch_size, 1)),
        "msa_mask": torch.ones((batch_size, n_msa, n_token)).float(),
        "num_paired_seqs": torch.randint(
            low=n_msa // 4, high=n_msa // 2, size=(batch_size,)
        ),
        "ground_truth": {
            "atom_positions": torch.randn((batch_size, n_atom, 3)).float(),
            "atom_resolved_mask": atom_resolved_mask.unsqueeze(0).repeat(
                (batch_size, 1)
            ),
        },
        "loss_weights": {
            "bond": torch.Tensor([4.0]).repeat(batch_size),
            "smooth_lddt": torch.Tensor([4.0]).repeat(batch_size),
            "mse": torch.Tensor([4.0]).repeat(batch_size),
            "plddt": torch.Tensor([1e-4]).repeat(batch_size),
            "pde": torch.Tensor([1e-4]).repeat(batch_size),
            "experimentally_resolved": torch.Tensor([1e-4]).repeat(batch_size),
            "pae": torch.Tensor([1e-4]).repeat(batch_size),
            "distogram": torch.Tensor([3e-2]).repeat(batch_size),
        },
    }

    if is_eval:
        features["ground_truth"]["intra_filter_atomized"] = torch.ones(
            batch_size, n_atom
        ).int()
        features["ground_truth"]["inter_filter_atomized"] = torch.ones(
            batch_size, n_atom, n_atom
        ).int()

    return features


def add_permutation_alignment_features(batch: dict) -> list[dict[int, torch.Tensor]]:
    """Synthesize the symmetry features the full multi-chain permutation
    alignment needs, so the training loss runs the real
    ``safe_multi_chain_permutation_alignment`` path instead of falling back to
    ``naive_alignment``.

    ``random_of3_features`` omits these, which degrades the GT-matching kernels
    in the backward (the fallback skips the Kabsch optimal transform +
    per-conformer permutation that the real data pipeline exercises).

    Chains come from ``asym_id``; each chain is a distinct entity
    (``mol_sym_id``=1), which drives the full algorithm (anchor selection,
    optimal transform, per-conformer permutation) without requiring identical
    symmetric chains. The required features (see
    ``single_batch_multi_chain_permutation_alignment`` docstring) are added at
    top level and mirrored into ``ground_truth``.

    Returns ``ref_space_uid_to_perm`` (``list[dict[int -> tensor]]``), which the
    caller should attach separately after any precision/device handling so it is
    not run through the precision converter or the tensor-tree device map.
    """
    B, n_token = batch["asym_id"].shape
    asym = batch["asym_id"].long()
    napt = batch["num_atoms_per_token"].long()

    mol_entity_id = torch.zeros((B, n_token), dtype=torch.int32)
    mol_sym_id = torch.ones((B, n_token), dtype=torch.int32)
    mol_sym_token_index = torch.zeros((B, n_token), dtype=torch.int32)
    mol_sym_component_id = torch.zeros((B, n_token), dtype=torch.int32)
    ref_space_uid_to_perm = []
    for b in range(B):
        for chain in torch.unique(asym[b]):
            idx = (asym[b] == chain).nonzero(as_tuple=True)[0]
            mol_entity_id[b, idx] = int(chain)
            mol_sym_token_index[b, idx] = torch.arange(len(idx), dtype=torch.int32)
            mol_sym_component_id[b, idx] = torch.arange(
                1, len(idx) + 1, dtype=torch.int32
            )
        # ref_space_uid == atom_to_token_index here, so keys are token indices;
        # each token-conformer gets a single identity atom permutation.
        ref_space_uid_to_perm.append(
            {
                t: torch.arange(int(napt[b, t]), dtype=torch.int32).reshape(1, -1)
                for t in range(n_token)
            }
        )

    mol_feats = {
        "mol_entity_id": mol_entity_id,
        "mol_sym_id": mol_sym_id,
        "mol_sym_token_index": mol_sym_token_index,
        "mol_sym_component_id": mol_sym_component_id,
    }
    for k, v in mol_feats.items():
        batch[k] = v
    # Mirror the structural features the multi-chain alignment +
    # get_token_center_atoms read from the GT batch (excludes the big
    # MSA/template tensors). GT-only keys (atom_positions, atom_resolved_mask)
    # are left as-is.
    for k in (
        "token_mask", "atom_mask", "num_atoms_per_token", "is_ligand",
        "start_atom_index", "is_protein", "is_atomized", "is_dna", "is_rna",
        "restype", "atom_to_token_index", "ref_space_uid",
        "mol_entity_id", "mol_sym_id", "mol_sym_token_index", "mol_sym_component_id",
    ):
        if k in batch and k not in batch["ground_truth"]:
            batch["ground_truth"][k] = batch[k].clone()

    return ref_space_uid_to_perm


def build_training_batch(
    n_token: int,
    n_msa: int,
    n_templ: int,
    batch_size: int = 1,
    full_permutation: bool = True,
    is_multimer: bool = False,
) -> dict:
    """A single pre-collated training batch ready for the production
    ``training_step`` (CPU tensors; Lightning handles precision + device).

    Adds the GT keys (``token_index``/``token_mask``/``num_atoms_per_token``)
    and metadata (``pdb_id``, ``preferred_chain_or_interface``) the training
    step reads but ``random_of3_features`` omits. When ``full_permutation`` is
    set, also synthesizes the symmetry features so the real multi-chain
    permutation alignment runs instead of the degenerate naive fallback.
    """
    batch = random_of3_features(
        batch_size=batch_size,
        n_token=n_token,
        n_msa=n_msa,
        n_templ=n_templ,
        is_eval=False,
        is_multimer=is_multimer,
    )
    # GT keys the permutation alignment needs; token_index must be sorted.
    batch["ground_truth"]["token_index"] = batch["token_index"].clone()
    batch["ground_truth"]["token_mask"] = batch["token_mask"].clone()
    batch["ground_truth"]["num_atoms_per_token"] = batch["num_atoms_per_token"].clone()

    if full_permutation:
        perm = add_permutation_alignment_features(batch)
        if perm is not None:
            batch["ref_space_uid_to_perm"] = perm

    # Production-only metadata keys read by training_step (not in random feats).
    batch["pdb_id"] = ["synthetic"] * batch_size
    batch["preferred_chain_or_interface"] = "synthetic"
    return batch


def build_inference_batch(
    n_token: int,
    n_msa: int,
    n_templ: int,
    batch_size: int = 1,
    query_id: str = "dummy",
    seed: int = 42,
    is_multimer: bool = False,
) -> dict:
    """A single pre-collated prediction batch ready for the production
    ``predict_step`` (CPU tensors; Lightning handles precision + device).

    Drops the training-only fields and stamps the metadata ``predict_step`` /
    the timing callback read. ``atom_array`` is a list of ``None`` placeholders:
    the confidence-ranking code only indexes the key per-sample (it does not
    require a real biotite ``AtomArray``), and the structure writer that would
    dereference it is removed for dummy runs.
    """
    batch = random_of3_features(
        batch_size=batch_size,
        n_token=n_token,
        n_msa=n_msa,
        n_templ=n_templ,
        # is_eval would only add ground_truth filter tensors (incl. an
        # n_atom x n_atom one) that we drop below, so leave it off.
        is_eval=False,
        is_multimer=is_multimer,
    )
    # Inference does not need these, and the permutation-alignment fallback
    # paths choke on partial GT dicts.
    batch.pop("ground_truth", None)
    batch.pop("loss_weights", None)
    # Metadata predict_step + PredictTimer read.
    batch["query_id"] = [query_id] * batch_size
    batch["seed"] = torch.tensor([seed] * batch_size)
    batch["valid_sample"] = True
    batch["repeated_sample"] = False
    batch["atom_array"] = [None] * batch_size
    return batch
