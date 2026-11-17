"""
Phonon band unfolding, projections and spectral functions.

The unfolding follows P. B. Allen et al., Phys. Rev. B 87, 085322 (2013), in the
form used by phonopy (``phonopy.unfolding.core.Unfolding``) and by
Y. Ikeda et al., Phys. Rev. B 95, 024305 (2017). The unfolded spectral function is

    A(k, w) = sum_J W_J(k) delta(w - w_J(K)),       W_J(k) = |P^k v_J(K)|^2

where K = k (mod supercell reciprocal lattice), v_J(K) are the supercell phonon
eigenvectors and P^k is the projection operator onto the Bloch character k of the
underlying (host primitive) lattice. The partial spectral functions of Ikeda et al.
for a set of atom groups X (e.g. chemical elements) are

    A^{XX'}(k, w) = sum_J Re{[P^k P^X v_J]^dagger [P^k P^X' v_J]} delta(w - w_J)

which satisfy sum_{X,X'} A^{XX'} = A when the groups partition the supercell atoms.
"""

from __future__ import annotations

import os
import warnings
from typing import Optional

import numpy as np
import tqdm
from phonopy import load
from phonopy.harmonic.dynmat_to_fc import get_commensurate_points
from phonopy.phonon.band_structure import (
    get_band_qpoints_and_path_connections,
    get_band_qpoints_by_seekpath,
)
from phonopy.structure.atoms import PhonopyAtoms
from phonopy.structure.cells import get_supercell
from pymatgen.core import Structure


def _find_file(directory: str, name: str) -> Optional[str]:
    for fname in (name, name + ".gz", name + ".xz", name + ".bz2"):
        path = os.path.join(directory, fname)
        if os.path.exists(path):
            return path
    return None


def _load_phonopy(directory: str, primitive_matrix="P", **kws):
    """Load SPOSCAR + FORCE_SETS/FORCE_CONSTANTS with SPOSCAR as the cell itself."""
    sposcar = _find_file(directory, "SPOSCAR")
    if sposcar is None:
        raise FileNotFoundError(f"no SPOSCAR found in {directory}")
    force_sets = _find_file(directory, "FORCE_SETS")
    force_constants = _find_file(directory, "FORCE_CONSTANTS")
    if force_sets is None and force_constants is None:
        raise FileNotFoundError(f"no FORCE_SETS or FORCE_CONSTANTS in {directory}")
    # SPOSCAR is read as the *unit cell* with an identity supercell matrix so that
    # phonopy never rebuilds / reorders the cell (passing `supercell_filename` makes
    # phonopy silently ignore `supercell_matrix`).
    common = dict(
        unitcell_filename=sposcar,
        supercell_matrix=np.eye(3, dtype=int),
        primitive_matrix=primitive_matrix,
        is_compact_fc=False,
        log_level=0,
        **kws,
    )
    if force_sets is not None:
        return load(force_sets_filename=force_sets, **common)
    return load(force_constants_filename=force_constants, **common)


class PhononUnfoldingandProjection:
    def __init__(
        self,
        supercell_directory: str,
        host_directory: str,
        smatrix: Optional[np.ndarray] = None,
        primitive_matrix=None,
        defect_site_index: Optional[int] = None,
        defect_site_coords: Optional[np.ndarray] = None,
        line_density: int = 100,
        nearest_neighbour_tolerance: float = 4,
        ideal_positions: Optional[np.ndarray] = None,
        site_mapping: Optional[list] = None,
        mapping_tolerance: float = 1.0,
        **kws,
    ):
        """
        Args:
            supercell_directory: directory with the defect/disordered SPOSCAR and
                FORCE_SETS or FORCE_CONSTANTS.
            host_directory: directory with the perfect host SPOSCAR and
                FORCE_SETS or FORCE_CONSTANTS (same supercell as the defect cell).
            smatrix: supercell matrix of the host SPOSCAR relative to the host unit
                cell (phonopy convention). Only needed together with
                ``primitive_matrix``.
            primitive_matrix: primitive matrix relative to the host *unit cell*
                (phonopy convention, needs ``smatrix``), or "auto" / None to find
                the primitive cell of the host automatically. The phonons are
                unfolded onto this cell's Brillouin zone.
            defect_site_index, defect_site_coords: defect site (used for the
                neighbour search only).
            line_density: number of q-points per path segment.
            nearest_neighbour_tolerance: shell radius (Angstrom) for the
                neighbour search around the defect.
            ideal_positions: fractional positions of the perfect sites in the
                supercell. Defaults to the host SPOSCAR positions.
            site_mapping: list of length len(ideal_positions) mapping each ideal
                site to a supercell atom index (None = vacancy). Determined
                automatically from positions if not given.
            mapping_tolerance: max distance (Angstrom) between a supercell atom and
                an ideal site for the atom to be assigned to that site. Atoms that
                cannot be assigned are treated as interstitials.
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        self.supercell_directory = supercell_directory
        self.host_directory = host_directory
        self.line_density = line_density
        self.nearest_neighbour_tolerance = nearest_neighbour_tolerance
        self.defect_site_index = defect_site_index
        self.defect_site_coords = defect_site_coords
        self.smatrix = None if smatrix is None else np.array(smatrix)
        if primitive_matrix is None or (
            isinstance(primitive_matrix, str) and primitive_matrix == "auto"
        ):
            self.primitive_matrix = "auto"
        else:
            self.primitive_matrix = primitive_matrix
        self.ideal_positions = ideal_positions
        self.site_mapping = site_mapping
        self.mapping_tolerance = mapping_tolerance

        self.labels = None
        self.path_connections = None
        self.eigendisplacements = None
        self.frequencies = None
        self.weights = None
        self.partial_weights = None
        self.groups = None

        self.__dict__.update(kws)

    # ------------------------------------------------------------------ sites
    def get_defect_neighbour_sites(
        self, nearest_neighbour_tolerance=None, **kws
    ) -> dict:
        """
        get the nearest neighbour sites of the defect site given a defect supercell

        Args:
            nearest_neighbour_tolerance (float): the tolerance for the nearest neighbour search
        Returns:
            dict: {element: [supercell atom indices]}
        """
        self.__dict__.update(kws)

        if self.defect_site_coords is None:
            raise AttributeError("no defect_site_coords given")

        if nearest_neighbour_tolerance:
            self.nearest_neighbour_tolerance = nearest_neighbour_tolerance

        struct = Structure.from_file(_find_file(self.supercell_directory, "SPOSCAR"))
        nearest_neighbours = struct.get_neighbors_in_shell(
            origin=self.defect_site_coords,
            r=0,
            dr=self.nearest_neighbour_tolerance,
        )

        unique_elements = list(dict.fromkeys([x.species_string for x in struct.sites]))

        neighbours = {}
        for elem in unique_elements:
            neighbours[elem] = [
                int(x.index) for x in nearest_neighbours if x.species_string == elem
            ]

        return neighbours

    def get_all_sites(self) -> dict:
        """
        get the supercell atom indices of every element in the supercell

        Returns:
            dict: {element: [supercell atom indices]}
        """
        struct = Structure.from_file(_find_file(self.supercell_directory, "SPOSCAR"))
        elements = list(dict.fromkeys(struct.symbol_set))
        return {
            element: [i for i, site in enumerate(struct) if site.species_string == element]
            for element in elements
        }

    # ------------------------------------------------------------- phonons
    def _host_primitive_matrix(self):
        """primitive matrix relative to the host SPOSCAR (as loaded)"""
        if isinstance(self.primitive_matrix, str):
            return self.primitive_matrix  # "auto" or "P"
        if self.smatrix is None:
            raise ValueError(
                "smatrix (host supercell matrix relative to the unit cell) is "
                "required when primitive_matrix is given explicitly"
            )
        return np.linalg.inv(self.smatrix) @ np.array(self.primitive_matrix, dtype=float)

    def get_primitive_phonons(self, qpaths="auto", labels=None):
        """
        get the host primitive phonons which the defect phonons are unfolded onto
        and set the unfolding supercell matrix.

        Args:
            qpaths: "auto" (seekpath) or a list of band paths in reduced coordinates
                of the host primitive reciprocal lattice, e.g.
                [[[0, 0, 0], [0.5, 0, 0.5]], [[0.5, 0.5, 0.5], [0, 0, 0]]].
            labels: labels of the path end points when qpaths is given.
        Returns:
            phonopy.Phonopy of the host
        """
        ph = _load_phonopy(self.host_directory, primitive_matrix=self._host_primitive_matrix())

        # matrix of the supercell (SPOSCAR) relative to the primitive cell
        umat = np.linalg.inv(ph.primitive_matrix)
        if not np.allclose(umat, np.rint(umat), atol=1e-5):
            raise ValueError(
                f"the supercell is not an integer multiple of the primitive cell: {umat}"
            )
        self.unfolding_matrix = np.rint(umat).astype(int)
        if self.smatrix is not None and isinstance(self.primitive_matrix, str):
            n_unit = round(abs(np.linalg.det(self.smatrix)))
            if n_unit != round(abs(np.linalg.det(self.unfolding_matrix))):
                warnings.warn(
                    "unfolding onto the primitive cell of the host "
                    f"(unfolding matrix {self.unfolding_matrix.tolist()}), not onto the "
                    f"unit cell implied by smatrix {self.smatrix.tolist()}. Pass "
                    "primitive_matrix to choose a different cell."
                )

        if isinstance(qpaths, str) and qpaths == "auto":
            qpaths, labels, path_connections = get_band_qpoints_by_seekpath(
                primitive=ph.primitive,
                npoints=self.line_density,
                is_const_interval=False,
            )
        else:
            qpaths, path_connections = get_band_qpoints_and_path_connections(
                qpaths, npoints=self.line_density
            )
            if labels is None:
                labels = self.labels
            if labels is None:
                n_points = len(qpaths) + sum(1 for c in path_connections if not c)
                labels = [""] * n_points
        self.labels = labels
        self.path_connections = path_connections
        self.qpaths = [np.array(q) for q in qpaths]

        ph.run_band_structure(
            self.qpaths, path_connections=path_connections, labels=labels
        )

        if self.ideal_positions is None:
            self.ideal_positions = ph.supercell.scaled_positions
        self.host_lattice = ph.supercell.cell
        self.host_rotations = ph.primitive_symmetry.pointgroup_operations
        self.distances = ph.band_structure.distances

        self.special_points = [x[0] for x in self.qpaths]
        self.special_points.append(self.qpaths[-1][-1])
        self.primitive_frequencies = ph.band_structure.frequencies

        return ph

    def get_supercell_phonons(self, qpaths=None):
        """
        get the supercell phonons which will be unfolded back to the primitive phonons

        Returns:
            phonopy.Phonopy with the supercell as its primitive cell
        """
        # symmetry must stay on: FORCE_SETS are usually symmetry-reduced
        # displacement sets (is_symmetry=False makes phonopy fail on them)
        ph = _load_phonopy(self.supercell_directory, primitive_matrix="P")
        self.supercell_atom_coords = ph.supercell.scaled_positions
        self.masses = ph.supercell.masses
        self.symbols = list(ph.supercell.symbols)
        return ph

    # --------------------------------------------------------------- mapping
    def get_site_mapping(self, supercell_positions, lattice=None):
        """
        map the ideal (host) sites onto the supercell atoms.

        Vacancies are ideal sites with no atom (None). Atoms that cannot be assigned
        to an ideal site (interstitials) are added as extra ideal sites, together with
        their images in every other virtual primitive cell (as vacancies), following
        the phonopy convention.

        Returns:
            ideal_positions (np.ndarray), site_mapping (list)
        """
        if lattice is None:
            lattice = self.host_lattice
        ideal = np.array(self.ideal_positions, dtype=float)
        pos = np.array(supercell_positions, dtype=float)

        diff = pos[:, None, :] - ideal[None, :, :]
        diff -= np.rint(diff)
        dist = np.linalg.norm(diff @ lattice, axis=2)  # (n_atoms, n_ideal)

        mapping = [None] * len(ideal)
        interstitials = []
        # greedy assignment, closest pairs first
        order = np.argsort(dist.min(axis=1))
        for atom in order:
            for site in np.argsort(dist[atom]):
                if dist[atom, site] > self.mapping_tolerance:
                    interstitials.append(atom)
                    break
                if mapping[site] is None:
                    mapping[site] = int(atom)
                    break

        if interstitials:
            trans = self._translations()[0]
            new_ideal = [ideal]
            for atom in interstitials:
                images = pos[atom] + trans
                images -= np.floor(images)
                new_ideal.append(images)
                mapping += [int(atom)] + [None] * (len(trans) - 1)
            ideal = np.vstack(new_ideal)

        return ideal, mapping

    def _translations(self):
        """primitive translations in supercell (trans_s) and primitive (trans_p) coords"""
        pcell = PhonopyAtoms(
            symbols=["H"], scaled_positions=[[0, 0, 0]], cell=np.eye(3)
        )
        trans_s = get_supercell(pcell, self.unfolding_matrix).scaled_positions
        trans_p = trans_s @ self.unfolding_matrix.T
        return trans_s, trans_p

    def _setup_unfolding(self, supercell_phonons):
        """build the translation index maps T(r_i)^-1 used by the projection P^k"""
        if self.site_mapping is None or len(self.site_mapping) != len(
            self.ideal_positions
        ):
            self.ideal_positions, self.site_mapping = self.get_site_mapping(
                supercell_phonons.supercell.scaled_positions
            )
        ideal = np.array(self.ideal_positions)
        lattice = self.host_lattice
        trans_s, trans_p = self._translations()
        n_trans = len(trans_s)
        if len(ideal) % n_trans:
            raise ValueError(
                "ideal_positions are not commensurate with the primitive cell"
            )

        index_map_inv = np.zeros((n_trans, len(ideal)), dtype=int)
        for i, shift in enumerate(trans_s):
            p = ideal - shift
            d = ideal[None, :, :] - p[:, None, :]
            d -= np.rint(d)
            dist = np.linalg.norm(d @ lattice, axis=2)
            k = dist.argmin(axis=1)
            if (dist[np.arange(len(p)), k] > 1e-3).any():
                raise ValueError(
                    "ideal_positions are not invariant under the primitive translations"
                )
            index_map_inv[i] = k

        n_atoms = len(supercell_phonons.supercell)
        # vacancies point at three zero rows appended to the eigenvectors
        atom_mapping = np.array(
            [n_atoms if m is None else m for m in self.site_mapping], dtype=int
        )
        mapped = atom_mapping[index_map_inv]  # (n_trans, n_ideal)
        self._row_index = (mapped[:, :, None] * 3 + np.arange(3)).reshape(n_trans, -1)
        self._trans_p = trans_p
        self._comm_points = get_commensurate_points(self.unfolding_matrix)
        self._n_atoms = n_atoms

    def _project(self, eigvecs, q_p, q_s, masks=None):
        """
        apply P^k to (masked) supercell eigenvectors.

        Returns:
            e: (n_groups or 1, 3 * n_ideal, n_bands) complex projected vectors
        """
        diff = q_p - q_s @ np.linalg.inv(self.unfolding_matrix)
        d = diff[None, :] - self._comm_points
        d -= np.rint(d)
        G = self._comm_points[np.argmin(np.abs(d).sum(axis=1))]
        phases = np.exp(2j * np.pi * self._trans_p @ G)

        padded = np.zeros((eigvecs.shape[0] + 3, eigvecs.shape[1]), dtype=complex)
        padded[:-3] = eigvecs
        if masks is None:
            stacks = padded[None]
        else:
            stacks = padded[None] * masks[:, :, None]

        e = np.zeros((len(stacks), self._row_index.shape[1], eigvecs.shape[1]), dtype=complex)
        for phase, rows in zip(phases, self._row_index):
            e += stacks[:, rows, :] * phase
        e /= len(phases)
        return e

    def get_star(self, qpoint):
        """symmetrically equivalent q-points (star of q) of the host primitive cell"""
        star = []
        for r in self.host_rotations:
            q = np.dot(qpoint, r)
            if not any(np.allclose(q - s, np.rint(q - s), atol=1e-6) for s in star):
                star.append(q)
        return np.array(star)

    # ------------------------------------------------------------- unfolding
    def run_unfold(
        self,
        qpaths="auto",
        projected_sites=None,
        groups=None,
        average_star=False,
        labels=None,
        tqdm_kwargs={},
    ):
        """
        unfold the supercell phonons onto the host primitive Brillouin zone

        Args:
            qpaths: "auto" (seekpath) or list of band paths (see get_primitive_phonons)
            projected_sites: list of supercell atom indices; the mean mass-weighted
                eigendisplacement norm of these sites is returned per mode.
            groups: partial spectral functions (Ikeda et al. PRB 95, 024305):
                "elements", or a dict {name: [supercell atom indices]}. Stores
                self.partial_weights with shape (path, q, band, n_groups, n_groups).
            average_star: average over the star of each q (Ikeda et al. Eq. 40).
                The modes of all star members are kept, each with weight / |star|.
            labels: labels of the path end points when qpaths is given.

        Returns:
            frequencies, weights[, eigendisplacements]
            each with shape (n_paths, n_qpoints_per_path, n_modes)
        """
        self.eigendisplacements = None
        self.frequencies = None
        self.weights = None
        self.partial_weights = None

        _ = self.get_primitive_phonons(qpaths=qpaths, labels=labels)
        supercell_phonons = self.get_supercell_phonons()
        self._setup_unfolding(supercell_phonons)

        masks = None
        if groups is not None:
            if isinstance(groups, str) and groups == "elements":
                groups = {
                    el: [i for i, s in enumerate(self.symbols) if s == el]
                    for el in dict.fromkeys(self.symbols)
                }
            self.groups = list(groups.keys())
            masks = np.zeros((len(groups), 3 * self._n_atoms + 3))
            for g, sites in enumerate(groups.values()):
                for s in sites:
                    masks[g, 3 * s : 3 * s + 3] = 1.0

        if projected_sites is not None:
            projected_sites = np.array(sorted(set(projected_sites)), dtype=int)
            inv_sqrt_mass = 1 / np.sqrt(np.array(self.masses)[projected_sites])

        qpoint_list = np.vstack(self.qpaths)
        n_bands = 3 * self._n_atoms
        n_star_max = 1
        if average_star:
            stars = [self.get_star(q) for q in qpoint_list]
            n_star_max = max(len(s) for s in stars)
        else:
            stars = [q[None, :] for q in qpoint_list]

        n_q = len(qpoint_list)
        n_modes = n_bands * n_star_max
        frequencies = np.zeros((n_q, n_modes))
        weights = np.zeros((n_q, n_modes))
        eigdisp = np.zeros((n_q, n_modes)) if projected_sites is not None else None
        partial = (
            np.zeros((n_q, n_modes, len(self.groups), len(self.groups)))
            if masks is not None
            else None
        )

        for iq, star in enumerate(
            tqdm.tqdm(stars, desc="Unfolding phonons...", **tqdm_kwargs)
        ):
            for istar, q_p in enumerate(star):
                q_s = q_p @ self.unfolding_matrix
                q_s = q_s - np.rint(q_s)
                freqs, eigvecs = supercell_phonons.get_frequencies_with_eigenvectors(q_s)
                sl = slice(istar * n_bands, (istar + 1) * n_bands)
                e = self._project(eigvecs, q_p, q_s)[0]
                frequencies[iq, sl] = freqs
                weights[iq, sl] = np.einsum("rb,rb->b", e.conj(), e).real / len(star)
                if masks is not None:
                    eg = self._project(eigvecs, q_p, q_s, masks=masks)
                    partial[iq, sl] = (
                        np.einsum("grb,hrb->bgh", eg.conj(), eg).real / len(star)
                    )
                if projected_sites is not None:
                    vec = eigvecs.T.reshape(n_bands, -1, 3)[:, projected_sites, :]
                    eigdisp[iq, sl] = np.mean(
                        np.linalg.norm(vec, axis=2) * inv_sqrt_mass[None, :], axis=1
                    )

        shape = (len(self.qpaths), len(self.qpaths[0]), n_modes)
        self.frequencies = frequencies.reshape(shape)
        self.weights = weights.reshape(shape)
        if partial is not None:
            self.partial_weights = partial.reshape(shape + partial.shape[2:])

        if projected_sites is not None:
            self.eigendisplacements = eigdisp.reshape(shape)
            return (self.frequencies, self.weights, self.eigendisplacements)
        return (self.frequencies, self.weights)

    # ------------------------------------------------------ spectral function
    def get_spectral_function(
        self,
        frequency_points=None,
        width=0.05,
        function="lorentzian",
        partial=False,
        num_points=500,
    ):
        """
        unfolded spectral function A(k, w) (Ikeda et al. PRB 95, 024305, Eq. 32)

            A(k, w) = sum_J W_J(k) delta(w - w_J)

        with the delta functions broadened by a Lorentzian (half-width at half
        maximum ``width``, as in the paper) or a Gaussian (standard deviation
        ``width``). Units are 1/THz; integrating over w gives the number of
        primitive-cell branches (3 * atoms per primitive cell).

        Args:
            frequency_points: frequencies (THz) to evaluate A on. Defaults to
                ``num_points`` points spanning the unfolded frequencies.
            width: broadening (THz).
            function: "lorentzian" or "gaussian".
            partial: if True also return the partial spectral functions
                A^{XX'}(k, w) for the groups passed to run_unfold.

        Returns:
            frequency_points (n_w,), A (n_paths, n_q, n_w)
            [, A_partial (n_paths, n_q, n_groups, n_groups, n_w)]
        """
        if self.weights is None:
            raise AttributeError("run run_unfold() first")
        if frequency_points is None:
            fmin = min(self.frequencies.min(), 0.0) - 5 * width
            fmax = self.frequencies.max() + 5 * width
            frequency_points = np.linspace(fmin, fmax, num_points)
        frequency_points = np.asarray(frequency_points, dtype=float)

        x = frequency_points[None, None, None, :] - self.frequencies[..., None]
        if function == "lorentzian":
            kernel = width / np.pi / (x**2 + width**2)
        elif function == "gaussian":
            kernel = np.exp(-0.5 * (x / width) ** 2) / (width * np.sqrt(2 * np.pi))
        else:
            raise ValueError("function must be 'lorentzian' or 'gaussian'")

        spectral = np.einsum("pqb,pqbw->pqw", self.weights, kernel)
        self.spectral_function = spectral
        self.spectral_frequencies = frequency_points
        if partial:
            if self.partial_weights is None:
                raise AttributeError("run run_unfold(groups=...) first")
            spectral_partial = np.einsum("pqbgh,pqbw->pqghw", self.partial_weights, kernel)
            self.partial_spectral_function = spectral_partial
            return frequency_points, spectral, spectral_partial
        return frequency_points, spectral

    def as_dict(self):
        return self.__dict__
