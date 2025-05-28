# get primitive data and band structure 
import numpy as np 
from phonopy import load 
from phonopy.unfolding.core import Unfolding
from phonopy.phonon.band_structure import get_band_qpoints_by_seekpath,get_band_qpoints
from pymatgen.io.phonopy import eigvec_to_eigdispl
from pymatgen.core import Structure
import warnings 
from typing import Optional
import tqdm 

class PhononUnfoldingandProjection:

    def __init__(
        self,
            supercell_directory: str,
            host_directory: str,
            smatrix: np.ndarray,
            primitive_matrix = 'P',
            defect_site_index: Optional[int] = None,
            defect_site_coords: Optional[np.ndarray] = None,
            line_density: int = 100,
            nearest_neighbour_tolerance: int = 4,
            **kws
    ):
        warnings.filterwarnings("ignore", category=DeprecationWarning)         
        
        self.supercell_directory = supercell_directory
        self.host_directory = host_directory
        self.line_density = line_density
        self.nearest_neighbour_tolerance = nearest_neighbour_tolerance
        self.defect_site_index = defect_site_index
        self.defect_site_coords = defect_site_coords
        self.eigendisplacements = None
        self.smatrix = smatrix 
        self.primitive_matrix = primitive_matrix
        
        self.__dict__.update(kws)

    def get_defect_neighbour_sites(
            self,
            nearest_neighbour_tolerance=None,
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
    
    def get_all_sites(
            self,
    )->dict:
        """
        get all atoms of a specific specie in the structure
        Args:
            defect_structure (pymatgen.core.Structure): the defect structure
            atom_type (str): the type of atom to get i.e. "Li"

        Returns:
            dict: {element:list}
        """
        ###### need to pass a Structure object 
        struct = Structure.from_file(self.supercell_directory+'SPOSCAR')
        ######
        elements = struct.symbol_set
        site_dict = {
            element: [
                i for i, index in enumerate(struct) if index.species_string == element
            ]
            for element in elements
        }
        
        return (site_dict)

    def get_primitive_phonons(
            self,
            qpaths='auto',
    ):
        """
        get the host or primitive phonons which the defect phonons will be unfolded back towards
        Args:
            calculate_eigenvectors (bool): whether to include eigenvectors or not
        Returns:
            phonopy bandstructure object 
        """

        try:
            ph = load(
                supercell_filename=self.host_directory + 'SPOSCAR',
                force_sets_filename=self.host_directory + 'FORCE_SETS',
                log_level=0,
                supercell_matrix=self.smatrix,
                primitive_matrix="auto",
                is_compact_fc=False,
            )
        except Exception:
            ph = load(
                supercell_filename=self.host_directory+'SPOSCAR',
                force_constants_filename=self.host_directory+'FORCE_CONSTANTS',
                log_level=0,
                supercell_matrix=self.smatrix,
                primitive_matrix='auto',
                is_compact_fc=False,
            )
        if qpaths == 'auto':
            qpaths, labels, path_connections = get_band_qpoints_by_seekpath(
                primitive=ph.primitive,
                npoints=self.line_density,
                is_const_interval=False,
            )
            self.labels = labels
            self.path_connections = path_connections
        else:
            qpaths = get_band_qpoints(qpaths,
                                      npoints=self.line_density)
            
            if not self.labels or self.path_connections:# change this?
                self.labels = None 
                self.path_connections = None 

        self.qpaths = qpaths

        ph.run_band_structure(
            qpaths,
            with_eigenvectors=False,
            #path_connections=self.path_connections,
            #labels=self.labels
        )  

        self.ideal_positions = ph.supercell.scaled_positions
        self.site_mapping = [i for i in range(len(self.ideal_positions))]
        if self.defect_site_index:
            if self.defect_site_index > len(self.site_mapping):
                self.site_mapping.append(self.defect_site_index)
            else:
                self.site_mapping[self.defect_site_index] = None 
        self.distances = ph.band_structure.distances

        self.special_points = [x[0] for x in qpaths]
        self.special_points.append(qpaths[-1][-1])
        self.primitive_frequencies = ph.band_structure.frequencies

        return(ph)

    def get_supercell_phonons(
            self,
            qpaths,
            ):
        """
        get the supercell phonons which will be unfolded back to the primitive phonons
        Args:
            qpaths: list, list of qpoints to unfold the phonons for, if 'auto' then uses the seekpath method in Phonopy
        Returns:
            phonopy bandstructure object 
        """
        
        try:
            ph = load(
                supercell_filename=self.supercell_directory+'SPOSCAR',
                force_sets_filename=self.supercell_directory+'FORCE_SETS',
                log_level=0,
                primitive_matrix=self.primitive_matrix,
                supercell_matrix=[[1,0,0],[0,1,0],[0,0,1]],
                is_symmetry=False,
                symprec=1e-10,
                is_compact_fc=False,
            )
        except Exception:
            ph = load(
                supercell_filename=self.supercell_directory+'SPOSCAR',
                force_constants_filename=self.supercell_directory+'FORCE_CONSTANTS',
                log_level=0,
                primitive_matrix=self.primitive_matrix,
                supercell_matrix=[[1,0,0],[0,1,0],[0,0,1]],
                is_compact_fc=False,
            )

        ph.run_band_structure(
            qpaths,
            #path_connections=path_connections,
            #labels=labels,
        )
        
        self.supercell_atom_coords = ph.supercell.scaled_positions
        self.masses = ph.supercell.masses

        return(ph)

    def get_eigendisplacements_from_qpoint(
            self,
            supercell_phonons,
            qpoint:list,
            projected_sites:list
            ):
        """
        Get the eigendisplacements from a given qpoint from a supercell qpoint     

        Args: 
        supercell_phonons: phonopy.phonon,  object of a calculated supercell with band structure already set 
        qpoint: list, qpoint in reciprocal space [0.0,0.0,0.0]
        chosen_indexes:list, atom site indexes intended to be plotted, i.e. [0,1,2,3,4]     

        Returns: 
        averaged eigendisplacements for those sites 
        """    
        
        # chosen atoms for projection
        sites = [
            i for i in range(len(self.supercell_atom_coords)) if i in projected_sites
            ]    

        frequencies,eigenvectors = supercell_phonons.get_frequencies_with_eigenvectors(qpoint)
                
        eigenvectors = eigenvectors.swapaxes(1,0)     

        # band x band    

        eigendisplacements = [
            [
                np.linalg.norm(
                    eigvec_to_eigdispl(
                        eig_vec=eigenvectors[freq][site * 3 : site * 3 + 3],
                        q=qpoint,
                        frac_coords=self.supercell_atom_coords[site],
                        mass=self.masses[site],
                    )
                )
                for site in sites
            ]
            for freq in range(len(frequencies))
        ]    

        return(np.mean(eigendisplacements,axis=1)) # frequencies * atoms     #average 

    def unfold_per_qpoint_phonopy(
            self,
            qpoint,
            supercell_phonons,
            projected_sites=None
            ):
        """
        unfolds the phonon dispersion per qpoint using phonopy.unfolding.core.Unfolding
        Args: 
        qpoint: list, chosen qpoint that you will calculate (i.e. [0.0,0.0,0.0])
        supercell_phonon: phonopy.Phonon, object of a calculated supercell with band structure already set 
        smatrix: np.ndarray(3x3), supercell matrix used to generate the host phonons 
        projected_sites: list or None, sites you wish to project onto the band structure 
        ideal_positions: list, ideal positions of sites in the perfect supercell
        site_mapping: list, mapping of the ideal positions used for unfolding (vacancy = None)    

        Returns:
        frequencies, weights, eigendisplacements (if projected_sites specified )
        """
        unfold = Unfolding(
            phonon=supercell_phonons,  # defect phonons
            supercell_matrix=self.smatrix,  # supercell matrix
            ideal_positions=self.ideal_positions,
            atom_mapping=self.site_mapping,  # should be better...
            qpoints=[qpoint],  # np.asarray(qpoints).reshape(-1,3),
        )
        
        unfold.run()

        weights = unfold.unfolding_weights[0]
        frequencies = unfold.frequencies[0]
            

        if projected_sites:
            eigendisplacements = self.get_eigendisplacements_from_qpoint(
                qpoint=qpoint,
                supercell_phonons=supercell_phonons,
                projected_sites=projected_sites
            ) 
            return(frequencies,weights,eigendisplacements)
        else:
            return(frequencies,weights)
        

    def run_unfold(
            self,
            qpaths='auto',
            projected_sites=None,
            tqdm_kwargs={},
            **kws
            ):
        """
        runs the phonopy.Unfolding algorithm for each qpoint in turn 

        Args: 
        qpaths: list, list of qpoints to unfold the phonons for, if 'auto' then uses the seekpath method in Phonopy 
        projected_sites: list, indexes of sites to project onto the band structure, if None then no projection is done 

        Returns: 
        frequencies: np.ndarray, frequencies of the unfolded phonons
        weights: np.ndarray, weights of the unfolded phonons
        eigendisplacements: np.ndarray, eigendisplacements of the unfolded phonons (if projected_sites specified)
        """
        self.eigendisplacements = None
        self.frequencies = None
        self.weights = None 

        _ = self.get_primitive_phonons(qpaths=qpaths)
        supercell_phonons = self.get_supercell_phonons(qpaths=self.qpaths)

        qpoint_list = np.array(self.qpaths).flatten().reshape(-1,3)
#       import tqdm_pathos
#       unfold = tqdm_pathos.map(
#            self.unfold_per_qpoint_phonopy,
#            qpoint_list,
#            supercell_phonons=supercell_phonons,
#            projected_sites=projected_sites,
#            **kws
#            )

        unfold = np.array(
            [
                self.unfold_per_qpoint_phonopy(
                    qpoint=qpoint,
                    supercell_phonons=supercell_phonons,
                    projected_sites=projected_sites,
                    **kws,
                )
                for qpoint in tqdm.tqdm(qpoint_list,desc='Unfolding phonons...',**tqdm_kwargs)
            ]
        )
        #self.primitive_phonons = primitive_phonons
        #self.supercell_phonons = supercell_phonons
        frequencies = unfold[:,0].reshape(
            np.shape(self.qpaths)[0],
            np.shape(self.qpaths)[1],
            -1
            )
        weights = unfold[:,1].reshape(
            np.shape(self.qpaths)[0],
            np.shape(self.qpaths)[1],
            -1
            )
        
        self.frequencies = frequencies 
        self.weights = weights 

        if projected_sites:
            eigendisplacements = unfold[:, 2].reshape(
                np.shape(self.qpaths)[0], np.shape(self.qpaths)[1], -1
            )
            self.eigendisplacements = eigendisplacements 

            return(frequencies,weights,eigendisplacements)
        else:
            return(frequencies,weights)
        
    def as_dict(self):
        return(self.__dict__)