1. can now do solid solutions or disordered systems 
2. makes a line plot rather than a scatter pot (can use less points which is faster) 
3. code is cleaner so easier to use 
4. a lot of old code is moved to _deprecated for now but will be deleted in future commits 
5. analysis.py is untouched but will be implemented later 
### unfolding fixes + spectral functions

1. **fix:** q-points (host primitive basis) and the unfolding matrix (`smatrix`, unit-cell basis) were inconsistent when the host primitive cell is smaller than the unit cell (e.g. fcc). The unfolding matrix is now taken from the host primitive cell; `primitive_matrix` (relative to the unit cell, with `smatrix`) selects a different target cell
2. **fix:** supercell atoms are now mapped onto the ideal host sites by position (species-sorted / reordered SPOSCARs were previously mapped wrongly); vacancies and interstitials are detected automatically (`site_mapping` / `mapping_tolerance` to override)
3. **fix:** vacancy mapping was shifted by one after the vacancy and also removed the last atom
4. **fix:** projected eigendisplacements were evaluated at the wrong q-point (primitive q used in the supercell basis) and so did not correspond to the unfolded modes
5. **fix:** `FORCE_SETS` were always silently ignored (loading with `is_symmetry=False` fails for symmetry-reduced displacement sets) in favour of `FORCE_CONSTANTS`
6. **fix:** `get_defect_neighbour_sites` (undefined `file_unzip`), `primitive_matrix` given as an array, and custom `qpaths`
7. **new:** unfolded spectral function `A(k, w)` and partial spectral functions `A^{XX'}(k, w)` for elements or any groups of sites (Ikeda et al., PRB 95, 024305 (2017)), with optional averaging over the star of k: `run_unfold(groups=..., average_star=True)`, `get_spectral_function()`, `PuppyPlotter.plot_spectral_function()`
8. **new:** tests in `tests/` (checked against `phonopy.unfolding.core.Unfolding`)
