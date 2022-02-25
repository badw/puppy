import os
from phonopy import load
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from pymatgen.core import Structure
import itertools as it
import tqdm
from ase import io
from ase.dft.kpoints import *
import numpy as np 
from phonopy.file_IO import parse_FORCE_SETS
from phonopy.interface.vasp import read_vasp 
from phonopy.structure.cells import get_supercell
from phonopy import Phonopy
from phonopy.unfolding.core import Unfolding
from phonopy.interface.calculator import read_crystal_structure
from pymatgen.io.phonopy import eigvec_to_eigdispl

class PhononUnfolder:
    
    def __init__(self,dict_of_locations,expansion,**kwargs):
        self.data = dict_of_locations
        self.expansion = expansion
        self.progress = kwargs['tqdm_disable']
        
    def get_possible_path(self,tol):
        atoms = io.read(filename=os.path.join(self.data['host_directory'],'POSCAR')) 
        points = get_special_points(atoms.cell, eps=tol)
        return(points)
        
    
    def get_phonopy_data(self,kpath,line_density=100): # rehash to not use phonopy but pymatgen
        '''generates a phonopy band structure object from the defect supercell'''
        qpoints, connections = get_band_qpoints_and_path_connections(kpath,npoints=line_density)
        ph = load(supercell_filename=os.path.join(self.data['defect_directory'],'SPOSCAR'),
                  force_sets_filename=os.path.join(self.data['defect_directory'],'FORCE_SETS'),
                  log_level=0)
        ph.run_band_structure(qpoints,
                                 path_connections=connections,
                                 with_eigenvectors='True') # can add with_group_velocities = True
        band_data = ph.get_band_structure_dict()
        band_data['connections'] = connections
        return(band_data)
    
    def get_primitive_qpts(self,kpath,tol=0.01,line_density=100):
        atoms = io.read(filename=os.path.join(self.data['host_directory'],'POSCAR')) 
        points = get_special_points(atoms.cell, eps=tol)
        Q = [x for k in kpath for x,y in points.items() if all(y==k)]
        path = bandpath(Q,atoms.cell,line_density) 
        qpts = path.kpts
        (q,line,label) = path.get_linear_kpoint_axis()        
        return({'q':q,'line':line,'label':label,'qpts':qpts})
    
    def get_neighbour_sites(self,tol=3):
        struct = Structure.from_file(os.path.join(self.data['host_directory'],'SPOSCAR'))
        nearest_neighbours = struct.get_neighbor_list(tol,
                                                      sites=[struct.sites[self.data['defect_index']]])[1]
        unique_elements = list(dict.fromkeys([x.species_string for x in struct.sites]))
        neighbours = {}
        for elem in unique_elements:
            neighbours[elem] = [x for x in nearest_neighbours 
                                if struct.sites[x].species_string == elem]
        return(neighbours)
    
    def get_eigendisplacements(self,band_data,sites):
    
        supercell, _ = read_crystal_structure(os.path.join(self.data['defect_directory'],'SPOSCAR'),
                                              interface_mode='vasp')
        atom_coords = supercell.get_scaled_positions()
        num_atoms = supercell.get_number_of_atoms()
        masses = supercell.get_masses()
        qpoints = band_data['qpoints'] # what is this all about? surely we should do more?
        distances = band_data['distances']
        frequencies = band_data['frequencies']
        eigenvectors = band_data['eigenvectors']
        connections = band_data['connections']
        true_q = [i for i in connections if i]        
        iterations = int(np.product(np.shape(eigenvectors)[0:3]))
        

        
        total = [] # we want one "displacement" per qpoint and frequency based on sites in radius around defect
        with tqdm(total=iterations,disable=self.progress) as pbar:
            for i in range(len(true_q)):
                total.append([])
                for q in range(len(qpoints[i])):
                    total[i].append([])
                    for w in range(len(frequencies[i][q])):
                
                        mean = [[np.linalg.norm(eigvec_to_eigdispl(eigenvectors[i][q][w][at],
                                                                              q=qpoints[i][q],
                                                                              frac_coords=atom_coords[at],
                                                                              mass=masses[at])) for elem in sites if elem == at] for at in range(num_atoms)]
                        total[i][q].append(np.mean(list(it.chain(*mean))))
                        pbar.update(1)
   
        return(total)

    def phonon_unfolder(self,prim_data): # could make it allow alloys in future
        #preamble setup 
        qpts = prim_data['qpts']
        line = prim_data['line']
        labels = prim_data['label']
        dim = self.expansion

        if len(dim) == 9:
            smatrix = np.reshape(dim,[3,3])
        elif len(dim) == 3:
            smatrix = np.zeros(9)
            for i in range(len(dim)):
                smatrix[i*4] = dim[i]
            smatrix = np.reshape(smatrix,[3,3])
        
        defect_index = self.data['defect_index']
        forcesets = parse_FORCE_SETS(filename=os.path.join(self.data['defect_directory'],'FORCE_SETS'))  
        
        pmatrix = np.linalg.inv(smatrix) 
        # setting up phonopy object
        prim_cell = read_vasp(os.path.join(self.data['host_directory'],'POSCAR')) 
        perf_supercell = get_supercell(prim_cell,smatrix)
        def_supercell = read_vasp(os.path.join(self.data['defect_directory'],'SPOSCAR'))
        phonon = Phonopy(def_supercell,np.diag([1,1,1]))
        phonon.dataset = forcesets
        phonon.produce_force_constants()
        mapping = [x for x in range(perf_supercell.get_number_of_atoms())]
        mapping[defect_index] = None
        # unfolding
        unfold = Unfolding(phonon=phonon, 
                           supercell_matrix=smatrix, 
                           ideal_positions=perf_supercell.get_scaled_positions(),
                           atom_mapping=mapping,
                           qpoints=qpts)
        unfold.run()
        weights = unfold.get_unfolding_weights()
        freqs = unfold.get_frequencies()

        return({'f':freqs,'w':weights})
    
    def run_all(self,kpath=None,site_tol=3,sym_tol=0.01,line_density=100,eigendisplacement_atom=None):
        # could be worth having an automatic kpath generator if not defined
        bs = self.get_phonopy_data(kpath,line_density)
        s = self.get_neighbour_sites(site_tol)
        pq = self.get_primitive_qpts(kpath,sym_tol,line_density)
        u = self.phonon_unfolder(pq)
        if not eigendisplacement_atom == None:
            e = self.get_eigendisplacements(bs,s[eigendisplacement_atom])
            return({'bs':bs,'sites':s,'prim_data':pq,'unfolded_data':u,'eigendisplacements':e})
        else:
            return({'bs':bs,'sites':s,'prim_data':pq,'unfolded_data':u})
