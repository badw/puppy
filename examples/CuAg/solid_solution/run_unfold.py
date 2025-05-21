import numpy as np

from ase.dft.kpoints import *
from ase.io import read 
import matplotlib.pyplot as plt
from unfolding.phonopy_unfolder import phonopy_unfold

def run_unfolding():
#(kpts, x, X) = bandpath(...).  Use path = bandpath(...) and then kpts = path.kpts and (x, X, labels) = path.get_linear_kpoint_axis().

    # Generate k-path for FCC structure
    from ase.build import bulk
    atoms = read('POSCAR')
    points = get_special_points('fcc', atoms.cell, eps=0.01)
    #kpts, x, X 
    path = bandpath(points, atoms.cell, 300)
    kpts = path.kpts
    x,X,names = path.get_linear_kpoint_axis()
    print(names)
    #names = ['Gamma', 'X', 'W', 'Gamma', 'L']

    # here is the unfolding. Here is an example of 3*3*3 fcc cell. 
    ax=phonopy_unfold(sc_mat=np.diag([1,1,1]),      # supercell matrix for phonopy.
                          unfold_sc_mat=np.diag([2,2,2]),   # supercell matrix for unfolding
                          force_constants='FORCE_CONSTANTS', # FORCE_CONSTANTS file path
                          sposcar='SPOSCAR',                 # SPOSCAR file for phonopy
                          qpts=kpts,                         # q-points. In primitive cell!
                          qnames=names,                      # Names of high symmetry q-points in the q-path.
                          xqpts=x,                           # x-axis, should have the same length of q-points.
                          Xqpts=X                            # x-axis
    )
    plt.show()

run_unfolding()
