# get primitive data and band structure 
import gzip
import os
from tqdm import tqdm
import copy 
from phonopy import load
import numpy as np 
import pandas as pd 
from phonopy.unfolding.core import Unfolding
from phonopy.phonon.band_structure import get_band_qpoints_by_seekpath,get_band_qpoints
from pymatgen.io.phonopy import eigvec_to_eigdispl
from pymatgen.core import Structure
from puppy.file_io import file_unzip 

import tqdm_pathos 
import tqdm 

class PhononUnfoldingandProjection:

    def __init__(
        self,
            supercell_directory: str,
            host_directory: str,
            defect_site_index: int,
            defect_site_coords: np.ndarray,
            matrix: np.ndarray,
            line_density: int = 100,
            nearest_neighbour_tolerance: int = 4,
            **kws
    ):
        
        self.supercell_directory = supercell_directory
        self.host_directory = host_directory
        self.line_density = line_density
        self.nearest_neighbour_tolerance = nearest_neighbour_tolerance
        self.defect_site_index = defect_site_index
        self.defect_site_coords = defect_site_coords
        self.eigendisplacements = None
        self.matrix = matrix 
        self.unfold_data = {}
        
        self.__dict__.update(kws)

    def get_defect_neighbour_sites(
            self,
            nearest_neighbour_tolerance:float = None,
            **kws
            )->dict:
        """
        get the nearest neighbour sites of the defect site given a defect supercell

        Args:
            nearest_neighbour_tolerance (float): the tolerance for the nearest neighbour search
        Returns:
            dict: a dictionary of the nearest neighbour sites
        """
        #######
        self.__dict__.update(kws)

        if self.defect_site_coords is None: 
            raise AttributeError('no defect_site_coords given')

        file_unzip(
            files=[self.supercell_directory+'SPOSCAR.gz']
            ) 
        
        if nearest_neighbour_tolerance:
            self.nearest_neighbour_tolerance = nearest_neighbour_tolerance
        
        struct = Structure.from_file(self.supercell_directory+'SPOSCAR')
        #######
        nearest_neighbours = struct.get_neighbors_in_shell(
            origin=self.defect_site_coords,
            r=0,
            dr=self.nearest_neighbour_tolerance,
            )
        
        unique_elements = list(
            dict.fromkeys([x.species_string for x in struct.sites])
            )
        
        neighbours = {}
        for elem in unique_elements:
            neighbours[elem] = [
                int(x.index) for x in nearest_neighbours if x.species_string == elem
            ]
        
        return(neighbours)
    
    def get_all_atoms_of_a_specie(
            self,
            atom_type: str
    ):
        """
        get all atoms of a specific specie in the structure
        Args:
            defect_structure (pymatgen.core.Structure): the defect structure
            atom_type (str): the type of atom to get i.e. "Li"

        Returns:
            dict: a dictionary with a list of all the atoms of a specific type
        """
        ###### need to pass a Structure object 
        struct = Structure.from_file(self.supercell_directory+'SPOSCAR')
        ######
        
        return (
            {
                atom_type: [
                    i for i, index in enumerate(struct)
                    if index.species_string == atom_type
                ]
            }
        )

    def get_host_phonons(
            self,
            calculate_eigenvectors: bool = False,
            **kws
    ):
        """
        get the host or primitive phonons which the defect phonons will be unfolded back towards
        Args:
            calculate_eigenvectors (bool): whether to include eigenvectors or not
        Returns:
            None
        """

        self.__dict__.update(kws)

        #check files are unzipped
        file_unzip(
                [
                    self.host_directory+'SPOSCAR.gz',
                    self.host_directory+'FORCE_SETS.gz',
                    self.host_directory+'FORCE_CONSTANTS.gz'
                ]
            )
        try:
            ph = load(
            supercell_filename=self.host_directory+'SPOSCAR',
            force_sets_filename=self.host_directory+'FORCE_SETS',
            log_level=0
            )
        except Exception:
            ph = load(
                supercell_filename=self.host_directory+'SPOSCAR',
                force_constants_filename=self.host_directory+'FORCE_CONSTANTS',
                log_level=0
            )
        
        bands, labels, path_connections = get_band_qpoints_by_seekpath(
                primitive=ph.primitive,
                npoints=self.line_density,
                is_const_interval=False
            )

        ph.run_band_structure(
            bands,
            with_eigenvectors=calculate_eigenvectors,
            path_connections=path_connections,
            labels=labels
        )  

        band_data = ph.get_band_structure_dict()

        self.host_phonons = ph
        self.unfold_data['host_band_data'] = band_data
        self.unfold_data['path_connections'] = path_connections
        self.unfold_data['labels'] = labels

    def get_defect_phonons(
            self,
            calculate_eigenvectors:bool=True
            ):
        
        try: 
            self.host_phonons
        except AttributeError:
            raise AttributeError('no host phonons found, please run Puppy.get_host_phonons()')
        
        
        file_unzip(
            [self.supercell_directory+'SPOSCAR.gz',
             self.supercell_directory+'FORCE_SETS.gz',
             self.supercell_directory+'FORCE_CONSTANTS.gz']
        )

        bands, labels, path_connections = get_band_qpoints_by_seekpath(
            primitive=self.host_phonons.primitive,
            npoints=self.line_density,
            is_const_interval=False
        )

        try:
            ph = load(
                supercell_filename=self.supercell_directory+'SPOSCAR',
                force_sets_filename=self.supercell_directory+'FORCE_SETS',
                log_level=0,
                primitive_matrix='auto',
                supercell_matrix=[[1,0,0],[0,1,0],[0,0,1]]
            )
        except Exception as e:
            print(e)
            ph = load(
                supercell_filename=self.supercell_directory+'SPOSCAR',
                force_constants_filename=self.supercell_directory+'FORCE_CONSTANTS',
                log_level=0,
                primitive_matrix='auto',
                supercell_matrix=[[1,0,0],[0,1,0],[0,0,1]]
            )

        ph.run_band_structure(
            bands,
            with_eigenvectors=calculate_eigenvectors,
            path_connections=path_connections,
            labels=labels
        )

        band_data = ph.get_band_structure_dict()
        self.defect_band_data = band_data
        self.supercell = ph.supercell
        self.defect_phonons = ph
        self.special_points = [x[0] for x in ph.band_structure.qpoints]
        self.special_points.append(ph.band_structure.qpoints[-1][-1])

        self.unfold_data['defect_band_data'] = band_data
    

    def eigenvectors_to_eigendisplacements(
            self, 
            all_atoms=None, 
            project_specific_sites=None, 
            direction=None
            ):
        if not all_atoms:
            if not project_specific_sites:
                nn = self.get_defect_neighbour_sites()
            else:
                nn = project_specific_sites
        else:
            if all_atoms:
                cell = self.host_phonons.supercell
                atom_types = list(dict.fromkeys(cell.get_chemical_symbols()))
                nn = self.get_all_atoms_of_a_specie(str(atom_types[0]))
                for atom in atom_types[1:]:
                    nn.update(self.get_all_atoms_of_a_specie(str(atom)))
            else:
                nn = self.get_all_atoms_of_a_specie(all_atoms)    


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
            for atom, sites in tqdm.tqdm(nn.items(),desc='generating_eigendisplacements...'):
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

    def _unfold_function(self,qpoints,**kws):
        self.__dict__.update(kws)
        
        unfold = Unfolding(
            phonon=self.defect_phonons,
            supercell_matrix=self.matrix,
            ideal_positions=self.host_phonons.get_supercell().get_scaled_positions(),
            atom_mapping=self.mapping,
            qpoints=qpoints # can we make this 1 qpoint?
        )

        unfold.run()
        weights = unfold.unfolding_weights
        freqs = unfold.frequencies

        return([freqs,weights])

    def unfold(self, **kws):
        #set the atom mappings - defect vacancy = None 
        self.mapping = [
            x for x in range(
                self.host_phonons.get_supercell().get_number_of_atoms()
                )
            ]
        if self.defect_site_index:
            self.mapping[self.defect_site_index] = None

        qpoints = self.unfold_data['host_band_data']['qpoints']

        frequencies = []
        weights = []
        for q_set in tqdm.tqdm(qpoints,desc='unfolding phonons...'):
            freqs,wts = self._unfold_function(q_set)
            frequencies.append(freqs)
            weights.append(wts)

        self.unfold_data['f'] = frequencies
        self.unfold_data['w'] = weights 

        return([frequencies,weights])
