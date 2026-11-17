from __future__ import annotations

import os
import warnings

import numpy as np
import pytest
from phonopy import Phonopy
from phonopy.file_IO import write_FORCE_CONSTANTS
from phonopy.interface.vasp import write_vasp
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.unfolding.core import Unfolding

from puppy.unfolder import PhononUnfoldingandProjection, _load_phonopy

EX = os.path.join(os.path.dirname(__file__), "..", "examples", "CuAg")
HOST = os.path.join(EX, "host") + "/"
SS = os.path.join(EX, "solid_solution") + "/"
QPATH = [[[0, 0, 0], [0.5, 0, 0.5], [0.5, 0.25, 0.75]], [[0.5, 0.5, 0.5], [0, 0, 0]]]
NO_BAR = {"disable": True}

warnings.filterwarnings("ignore")


def _run(supercell_dir, **kws):
    p = PhononUnfoldingandProjection(
        supercell_directory=supercell_dir, host_directory=HOST, line_density=7
    )
    p.run_unfold(qpaths=QPATH, tqdm_kwargs=NO_BAR, **kws)
    return p


def _write_cell(tmp_path, name, cell, fc):
    d = tmp_path / name
    d.mkdir()
    write_vasp(str(d / "SPOSCAR"), cell, direct=True)
    write_FORCE_CONSTANTS(fc, filename=str(d / "FORCE_CONSTANTS"))
    return str(d) + "/"


def _degsum(f, w, tol=1e-4):
    """sum weights over degenerate modes (per-mode weights are basis dependent)"""
    f = f.reshape(-1, f.shape[-1])
    w = w.reshape(-1, w.shape[-1])
    same = np.abs(f[:, :, None] - f[:, None, :]) < tol
    return np.einsum("qab,qb->qa", same, w)


def _phonopy_reference(p, supercell_dir, atom_mapping):
    ph = _load_phonopy(supercell_dir)
    u = Unfolding(
        ph, p.unfolding_matrix, p.ideal_positions, atom_mapping, np.vstack(p.qpaths)
    )
    u.run()
    return u.frequencies, u.unfolding_weights


def test_host_unfolds_onto_primitive_bands():
    p = _run(HOST)
    assert p.unfolding_matrix.tolist() == [[-2, 2, 2], [2, -2, 2], [2, 2, -2]]
    w = p.weights.reshape(-1, p.weights.shape[-1])
    f = p.frequencies.reshape(-1, p.frequencies.shape[-1])
    prim = np.vstack(p.primitive_frequencies)
    # weights are 0 or 1 and sum to the 3 primitive branches
    assert np.allclose(w.sum(axis=1), 3)
    wd = _degsum(f, w)
    for fq, wq, pq in zip(f, wd, prim):
        # every mode carrying weight sits exactly on a primitive band
        on_band = np.abs(fq[:, None] - pq[None, :]).min(axis=1) < 1e-4
        assert np.allclose(wq[~on_band], 0, atol=1e-8)
        assert np.allclose(np.unique(fq[on_band].round(4)), np.unique(pq.round(4)))


def test_solid_solution_matches_phonopy_and_mapping_is_by_position():
    p = _run(SS)
    # SPOSCAR is species-sorted, so the mapping must not be the identity
    assert p.site_mapping != list(range(len(p.site_mapping)))
    f_ref, w_ref = _phonopy_reference(p, SS, p.site_mapping)
    n = p.weights.shape[-1]
    assert np.allclose(p.frequencies.reshape(-1, n), f_ref, atol=1e-6)
    assert np.allclose(_degsum(p.frequencies, p.weights), _degsum(f_ref, w_ref), atol=1e-8)
    assert np.allclose(p.weights.sum(axis=-1), 3)


def test_atom_order_does_not_matter(tmp_path):
    ph = _load_phonopy(SS)
    # shuffle within each species block (POSCAR keeps species contiguous)
    rng = np.random.default_rng(0)
    symbols = np.array(ph.supercell.symbols)
    perm = np.concatenate(
        [rng.permutation(np.where(symbols == el)[0]) for el in dict.fromkeys(symbols)]
    )
    cell = PhonopyAtoms(
        symbols=[ph.supercell.symbols[i] for i in perm],
        cell=ph.supercell.cell,
        scaled_positions=ph.supercell.scaled_positions[perm],
    )
    fc = ph.force_constants[np.ix_(perm, perm)]
    d = _write_cell(tmp_path, "perm", cell, fc)
    a, b = _run(SS), _run(d)
    assert np.allclose(a.frequencies, b.frequencies, atol=1e-6)
    assert np.allclose(_degsum(a.frequencies, a.weights), _degsum(b.frequencies, b.weights), atol=1e-8)


def test_vacancy_matches_phonopy(tmp_path):
    host = _load_phonopy(HOST)
    vac = 5
    keep = [i for i in range(len(host.supercell)) if i != vac]
    fc = host.force_constants[np.ix_(keep, keep)].copy()
    for i in range(len(keep)):  # restore the acoustic sum rule
        fc[i, i] -= fc[i].sum(axis=0)
    cell = PhonopyAtoms(
        symbols=[host.supercell.symbols[i] for i in keep],
        cell=host.supercell.cell,
        scaled_positions=host.supercell.scaled_positions[keep],
    )
    d = _write_cell(tmp_path, "vac", cell, fc)
    p = _run(d)
    expected = list(range(vac)) + [None] + list(range(vac, len(keep)))
    assert p.site_mapping == expected
    f_ref, w_ref = _phonopy_reference(p, d, expected)
    assert np.allclose(_degsum(p.frequencies, p.weights), _degsum(f_ref, w_ref), atol=1e-8)
    # one of 32 sites is empty
    assert np.allclose(p.weights.sum(axis=-1), 3 * 31 / 32)


def test_partial_weights_sum_to_total():
    p = _run(SS, groups="elements")
    assert p.groups == ["Cu", "Ag"]
    assert np.allclose(p.partial_weights.sum(axis=(-1, -2)), p.weights, atol=1e-10)
    diag = np.einsum("...gg->...g", p.partial_weights)
    assert (diag > -1e-12).all()


def test_spectral_function_normalisation():
    p = _run(SS, groups="elements")
    w = np.linspace(-20, 30, 20001)
    _, A, A_p = p.get_spectral_function(w, width=0.05, function="gaussian", partial=True)
    integral = np.trapezoid(A, w, axis=-1)
    assert np.allclose(integral, 3, atol=1e-3)
    assert np.allclose(A_p.sum(axis=(2, 3)), A)


def test_star_average_host_is_unchanged():
    p = _run(HOST, average_star=True)
    assert np.allclose(p.weights.sum(axis=-1), 3)


def test_projected_sites_shape():
    p = _run(SS, projected_sites=[0, 1, 2])
    assert p.eigendisplacements.shape == p.weights.shape
    assert (p.eigendisplacements >= 0).all()


@pytest.mark.parametrize("bad", [np.eye(3)])
def test_explicit_primitive_matrix_array(bad):
    # explicit primitive matrix (unit cell -> primitive) as an array must not crash
    p = PhononUnfoldingandProjection(
        supercell_directory=SS,
        host_directory=HOST,
        smatrix=np.diag([2, 2, 2]),
        primitive_matrix=bad,
        line_density=3,
    )
    p.run_unfold(qpaths=QPATH, tqdm_kwargs=NO_BAR)
    # unfolding onto the 4-atom conventional cell: 12 branches
    assert p.unfolding_matrix.tolist() == [[2, 0, 0], [0, 2, 0], [0, 0, 2]]
    assert np.allclose(p.weights.sum(axis=-1), 12)
