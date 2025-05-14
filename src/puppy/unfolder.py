# get primitive data and band structure 
import gzip
import os
from tqdm import tqdm
import copy 
from phonopy import load
import numpy as np 
import pandas as pd 
from phonopy.unfolding.core import Unfolding
from phonopy.phonon.band_structure import get_band_qpoints_by_seekpath
from pymatgen.io.phonopy import eigvec_to_eigdispl
from pymatgen.core import Structure
from puppy.file_io import file_unzip 
import warnings 

class PhononUnfoldingandProjection:

    def __init__(
        self,
            defect_directory: str = None,
            host_directory: str = None,
            defect_site_index: int = None,
            defect_site_coords: np.ndarray = None,
            line_density: int = 100,
            nearest_neighbour_tolerance: int = 4,
            matrix: np.ndarray = None
    ):
        
        self.defect_directory = defect_directory
        self.host_directory = host_directory
        self.line_density = line_density
        self.nearest_neighbour_tolerance = nearest_neighbour_tolerance
        self.defect_site_index = defect_site_index
        self.defect_site_coords = defect_site_coords
        self.eigendisplacements = None
        self.matrix = matrix 
        self.unfold_data = {}


    def return_cage_structure(self,
                              use_dummy_atom='K',
                              with_vectors=False,
                              vector_args=None,
                              scale=2,
                              chosen_index=None):
        
        struct = Structure.from_file(self.defect_directory+'SPOSCAR')
        
        def _grab_chosen_vectors(nearest_neighbours,chosen_index=None):
        
            from pymatgen.io.phonopy import eigvec_to_eigdispl    

            #vector_args = {'qpt':0,'line':0,'band':5,'threshold':0.1,'tolerance':0.05}        
            if not vector_args['tolerance']:
                tolerance = 0.1
            else:
                tolerance = vector_args['tolerance']    

            if not vector_args['threshold']:
                threshold = 0.1
            else:
                threshold = vector_args['threshold']    
            host_frequencies = self.host_band_data['frequencies']
            chosen_frequency = host_frequencies[vector_args['qpt']][vector_args['line']][vector_args['freq']]            

            unfolded_frequencies = self.unfold_data['f'] 
            unfolded_weights = self.unfold_data['w']
            eigenvectors = self.defect_band_data['eigenvectors']            

            chosen_indexes = {}
            for i,(f,w) in enumerate(zip(unfolded_frequencies[vector_args['qpt']][vector_args['line']],
                           unfolded_weights[vector_args['qpt']][vector_args['line']])):
                if w>=threshold:
                    if np.isclose(f,chosen_frequency,tolerance):
                        chosen_indexes[i] = {'f':f,'w':w}        

            df = pd.DataFrame(chosen_indexes).T.sort_values(by='w',ascending=False)     
            print("possible defect cell modes are:")
            print(df)       
            print("choosing mode {}".format(df.index[0]))
            if not chosen_index:
                chosen_index = df.index[0]            

            atom_coords = self.defect_phonons.supercell.get_scaled_positions()
            masses = self.defect_phonons.supercell.get_masses()
       
            eigendisplacements = []
            for i in range(len(masses)):
                eigendisplacements.append(
                    [np.real(x) for x in eigenvectors[vector_args['qpt']][vector_args['line']].T[chosen_index][i*3:i*3+3]] 
                )            

            chosen_vectors = []
            max_disp = np.max([np.linalg.norm(x) for x in eigendisplacements])
            for atom in range(len(atom_coords)):
                chosen_vectors.append([(np.real(x)/max_disp)*scale for x in eigendisplacements[atom]]) 
            return(chosen_vectors)
        
        nn = self.get_neighbour_sites()    

        indexes = []
        for x in nn.values():
            for ix in x:
                indexes.append(ix)
        non_indexes = [x for x in range(len(struct)) if x not in list(indexes)]
        if with_vectors and vector_args:
            chosen_vectors = _grab_chosen_vectors(nn,chosen_index=chosen_index)
            struct.add_site_property('magmom',chosen_vectors)    

        struct.remove_sites(non_indexes)
        if use_dummy_atom:
            struct.append(species=use_dummy_atom,
                          coords=self.defect_site_coords,
                          coords_are_cartesian=True) #self.defect_site.frac_coords
        return (struct)

    def get_neighbour_sites(self):
            
        file_unzip(files=[self.defect_directory+'SPOSCAR.gz'])
        struct = Structure.from_file(self.defect_directory+'SPOSCAR')
        nearest_neighbours = struct.get_neighbors_in_shell(origin=self.defect_site_coords,
                                                           r=0,
                                                           dr=self.nearest_neighbour_tolerance,
                                                           )
        unique_elements = list(dict.fromkeys([x.species_string for x in struct.sites]))
        neighbours = {}
        for elem in unique_elements:
            neighbours[elem] = [x.index for x in nearest_neighbours
                                if x.species_string == elem]
        
        return(neighbours)
    
    def get_all_atoms_of_a_type(self,atom_type=None):
        struct = Structure.from_file(self.defect_directory+'SPOSCAR')
        return({atom_type:
                [i for i,index in enumerate(struct) 
                 if index.species_string == atom_type]})

        

    def get_host_phonons(self,eigenvectors=False):
        '''get the host or primitive phonons which the defect phonons will be unfolded back towards
        * currently assumes seekpath and has no manual kpoints mode, but this can be rectified in the future'''

        file_unzip([self.host_directory+'SPOSCAR.gz',
                         self.host_directory+'FORCE_SETS.gz'])
        
        ph = load(supercell_filename=self.host_directory+'SPOSCAR',
                  force_sets_filename=self.host_directory+'FORCE_SETS',
                  log_level=0)
        
        
        bands,labels,path_connections = get_band_qpoints_by_seekpath(primitive=ph.primitive,
                                                                     npoints=self.line_density,
                                                                     is_const_interval=False)
                
        ph.run_band_structure(bands,
                              with_eigenvectors=eigenvectors,
                              path_connections=path_connections,
                              labels=labels) #old - needs to remove get_band_qpoints

        band_data = ph.get_band_structure_dict()
        self.host_band_data = band_data
        self.path_connections = path_connections
        self.labels=labels
        self.host = ph.primitive
        self.host_phonons = ph
        if not np.any(self.matrix):
            warnings.warn('No matrix provided, guessing...')
            self.matrix = np.abs(np.linalg.inv(self.host_phonons.primitive_matrix).round(0)) 

        self.unfold_data['host_band_data'] = band_data
        self.unfold_data['path_connections'] = path_connections
        self.unfold_data['labels'] = labels
    

    def eigenvectors_to_eigendisplacements(self, all_atoms=None, project_specific_sites=None, direction=None):
        if not all_atoms:
            if not project_specific_sites:
                nn = self.get_neighbour_sites()
            else:
                nn = project_specific_sites
        else:
            if all_atoms == True:
                cell = self.host_phonons.supercell
                atom_types = list(dict.fromkeys(cell.get_chemical_symbols()))
                nn = self.get_all_atoms_of_a_type(str(atom_types[0]))
                for atom in atom_types[1:]:
                    nn.update(self.get_all_atoms_of_a_type(str(atom)))
            else:
                nn = self.get_all_atoms_of_a_type(all_atoms)    


        atom_coords = self.defect_phonons.supercell.get_scaled_positions()
        masses = self.defect_phonons.supercell.get_masses()
        eigenvecs = np.array(self.defect_band_data['eigenvectors'])
        # reformat eigenvecs
        new_eigenvectors = []
        for i,q in enumerate(eigenvecs):
            new_eigenvectors.append([])
            for ii,l in enumerate(q):
                new_eigenvectors[i].append(l.T)

        eigenvecs = eigenvecs.swapaxes(3,2)
        

        #eigenvecs = new_eigenvectors 
        qpts = self.defect_band_data['qpoints']    

        if not direction:
            eigendisplacements = {}
            for atom, sites in tqdm(nn.items(),desc='generating_eigendisplacements...'):
                eigendisplacements[atom] = []
                for i, group in enumerate(eigenvecs):
                    eigendisplacements[atom].append([])
                    for ii, line in enumerate(group):
                        eigendisplacements[atom][i].append([])
                        for iii, freq in enumerate(line): # this is wrong....
                            eigdispl = [
                                np.linalg.norm(
                                    eigvec_to_eigdispl(
                                        freq[site*3:site*3+3], # [site:site+3]
                                        q=qpts[i][ii],
                                        frac_coords=atom_coords[site],
                                        mass=masses[site])
                                ) for site in sites if site]
                            if eigdispl:
                                mean_eigdispl = np.mean(eigdispl)
                            else:
                                mean_eigdispl = 0
                            eigendisplacements[atom][i][ii].append(
                                mean_eigdispl)    

        else:
            if direction in ['x', 'a']:
                _direc = 0
            elif direction in ['y', 'b']:
                _direc = 1
            elif direction in ['z', 'c']:
                _direc = 2
            eigendisplacements = {}
            for atom, sites in tqdm(nn.items(),desc='generating_eigendisplacements...'):
                eigendisplacements[atom] = []
                for i, group in enumerate(eigenvecs):
                    eigendisplacements[atom].append([])
                    for ii, line in enumerate(group):
                        eigendisplacements[atom][i].append([])
                        for iii, freq in enumerate(line):
                            eigdispl = [np.linalg.norm(
                                eigvec_to_eigdispl(
                                        freq[site*3:site*3+3],
                                        q=qpts[i][ii],
                                        frac_coords=atom_coords[site],
                                        mass=masses[site])[_direc]
                                        )
                                for site in sites if site]
                            if eigdispl:
                                mean_eigdispl = np.mean(eigdispl)
                            else:
                                mean_eigdispl = 0
                            eigendisplacements[atom][i][ii].append(
                                mean_eigdispl)    

        self.eigendisplacements = eigendisplacements

        self.unfold_data['eigendisplacements'] = eigendisplacements

    def get_defect_phonons(self,with_eigenvectors=True):
        '''get the defect phonons which will be unfolded'''
        
        file_unzip(
            [self.defect_directory+'SPOSCAR.gz',
             self.defect_directory+'FORCE_SETS.gz',
             self.defect_directory+'FORCE_CONSTANTS.gz']
        )

        
        bands,labels,path_connections = get_band_qpoints_by_seekpath(primitive=self.host,
                                                             npoints=self.line_density,
                                                             is_const_interval=False)
        
        try:
            ph = load(supercell_filename=self.defect_directory+'SPOSCAR',
                      force_sets_filename=self.defect_directory+'FORCE_SETS',
                      log_level=0)
        except Exception:
            ph = load(supercell_filename=self.defect_directory+'SPOSCAR',
                      force_constants_filename=self.defect_directory+'FORCE_CONSTANTS',
                      log_level=0)

                
        ph.run_band_structure(bands,
                              with_eigenvectors=with_eigenvectors,
                              path_connections=path_connections,
                              labels=labels)
        

        band_data = ph.get_band_structure_dict()
        self.defect_band_data = band_data
        self.supercell = ph.supercell
        self.defect_phonons = ph
        self.special_points = [x[0] for x in ph.band_structure.qpoints]
        self.special_points.append(ph.band_structure.qpoints[-1][-1])

        self.unfold_data['defect_band_data'] = band_data

    def unfold(self):

        def mp_function(qpoints):
            if not self.matrix.any():
                self.matrix = np.abs(np.linalg.inv(self.defect_phonons.primitive_matrix).round(0))
            mapping = [x for x in range(self.host_phonons.get_supercell().get_number_of_atoms())]
            mapping[self.defect_site_index] = None
            unfold = Unfolding(phonon = self.defect_phonons,
                   supercell_matrix = self.matrix,
                   ideal_positions=self.host_phonons.get_supercell().get_scaled_positions(),
                   atom_mapping = mapping,
                   qpoints = qpoints
                   )
            unfold.run()
            weights = unfold.get_unfolding_weights()
            freqs = unfold.get_frequencies()
            return([freqs,weights])
        
        frequencies = []
        weights = []
        for q in tqdm(self.host_band_data['qpoints'],desc='unfolding phonons...'):
            freqs,wts = mp_function(q)
            frequencies.append(freqs)
            weights.append(wts)

        self.unfold_data['f'] = frequencies
        self.unfold_data['w'] = weights 

