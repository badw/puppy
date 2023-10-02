import os
from phonopy import load
from phonopy.phonon.band_structure import get_band_qpoints_and_path_connections
from pymatgen.core import Structure
import itertools as it
from tqdm import tqdm
from ase import io
from ase.dft.kpoints import *
import numpy as np 
from phonopy.file_IO import parse_FORCE_SETS
from phonopy.interface.vasp import read_vasp 
from phonopy.structure.cells import get_supercell
from phonopy import Phonopy
from phonopy.unfolding.core import Unfolding
from phonopy.interface.calculator import read_crystal_structure
from phonopy.phonon.band_structure import BandStructure, get_band_qpoints_by_seekpath
from pymatgen.io.phonopy import eigvec_to_eigdispl

class PhononUnfolder:
    
    def __init__(self,dict_of_locations,expansion,**kwargs):
        self.data = dict_of_locations
        self.expansion = expansion
        self.progress = kwargs['tqdm_disable']
        
    def get_possible_path(self,tol=0.01):
        atoms = io.read(filename=os.path.join(self.data['host_directory'],'POSCAR')) 
        points = get_special_points(atoms.cell, eps=tol)
        kpath  = {}
        lists = list(points.keys())
        for i,k in enumerate(lists):
            if not i == len(lists)-1:
                k1,k2 = lists[i],lists[i+1]
                kpath['{}-{}'.format(k1,k2)] = [points[k1],points[k2]]
        return(kpath)
    
    def get_phonopy_defect_data_2(self,line_density=101): # rehash to not use phonopy but pymatgen
        '''generates a phonopy band structure object from the defect supercell
        todo:
        * make it so you can use .gz files
        * add custom kpoint path without the kernel crashing'''
        bands,labels,path_connections = get_band_qpoints_by_seekpath(primitive=self.primitive,
                                                             npoints=line_density,
                                                             is_const_interval=True)
        ph = load(supercell_filename=os.path.join(self.data['defect_directory'],'SPOSCAR'),
                  force_sets_filename=os.path.join(self.data['defect_directory'],'FORCE_SETS'),
                  log_level=0)
        ph.run_band_structure(bands,
                              with_eigenvectors=True,
                              path_connections=path_connections,
                              labels=labels)
        band_data = ph.get_band_structure_dict()
        self.defect_band_data = band_data
        self.defect_band_data['connections'] = path_connections
        self.supercell = ph.supercell
        self.defect_phonons = ph
        self.labels = labels
        self.special_points = [x[0] for x in ph.band_structure.qpoints]
        self.special_points.append(ph.band_structure.qpoints[-1][-1])
    
    def get_phonopy_primitive_data_2(self,line_density=101):
        '''todo:
        * make it so you can use .gz files
        * add custom kpoint path without the kernel crashing'''
        ph = load(supercell_filename=os.path.join(self.data['host_directory'],'SPOSCAR'),
                  force_sets_filename=os.path.join(self.data['host_directory'],'FORCE_SETS'),
                  log_level=0)
        ph.auto_band_structure(with_eigenvectors='False')
        band_data=ph.get_band_structure_dict()
        self.primitive_band_data = band_data
        self.primitive = ph.primitive
        self.primitive_phonons = ph
    
    @staticmethod   
    def label_formatter(labels,special_points):
        labs = []
        for i,j in enumerate(labels):
            if j == '$\\Gamma$':
                labs.append({'G':special_points[i]})
            else:
                clean = j.split('$')[1].split('{')[1].split('}')[0]            
                labs.append({clean:special_points[i]})
        return(labs)

    def get_primitive_qpts(self,tol=0.01,line_density=100):
        points = get_special_points(self.primitive.cell, eps=tol)
        kpath = self.label_formatter(self.labels,self.special_points)
        Q = [x for k in kpath for x,y in points.items() if all(y==k)]
        Q = list(points)
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
        
    def run_all(self,kpaths=None,site_tol=3,sym_tol=0.01,line_density=100,eigendisplacement_atom=None):
        # could be worth having an automatic kpath generator if not defined
        bs_p = self.get_phonopy_primitive_data([kpaths[kpath] for kpath in kpaths],line_density) # do the primitive kpoints all in one
        
        data = {}
        for kpath in tqdm(kpaths): # this can be multiprocessed in future
            bs_d = self.get_phonopy_defect_data(kpaths[kpath],line_density)
            s = self.get_neighbour_sites(site_tol)
            pq = self.get_primitive_qpts(kpaths[kpath],sym_tol,line_density)
            u = self.phonon_unfolder(pq)
            if not eigendisplacement_atom == None:
                e = self.get_eigendisplacements(bs_d,s[eigendisplacement_atom])
                data[kpath] = {'bs':bs_d,'sites':s,'prim_data':pq,'unfolded_data':u,'eigendisplacements':e}
            else:
                data[kpath] = {'bs':bs_d,'sites':s,'prim_data':pq,'unfolded_data':u}
        return({'data':data,'prim':bs_p})