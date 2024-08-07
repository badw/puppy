# get primitive data and band structure 
import gzip
import os
from tqdm import tqdm
import itertools as it 
import copy 
from phonopy import load
import numpy as np 
from phonopy.unfolding.core import Unfolding
from phonopy.phonon.band_structure import get_band_qpoints_by_seekpath
from pymatgen.io.phonopy import eigvec_to_eigdispl
from pymatgen.core import Structure

class PhononUnfoldingandProjection:

    def __init__(self,
                 defect_directory,
                 host_directory,
                 line_density,
                 expansion,
                 defect_index,
                 nearest_neighbour_tolerance):
        self.defect_directory = defect_directory
        self.host_directory = host_directory
        self.line_density = line_density
        self.expansion = expansion
        self.defect_index = defect_index
        self.nearest_neighbour_tolerance = nearest_neighbour_tolerance

    def file_unzip(self, 
                   files):
        '''unzips a .gz file if present
            TODO: remove unzipped files afterwards'''
        for file in files:
            if os.path.exists(file):
                with gzip.open(file, 'rb') as f_in, open(file.replace('.gz', ''), 'wb') as f_out:
                    f_out.writelines(f_in)

    def get_neighbour_sites(self):
        self.file_unzip([self.host_directory+'SPOSCAR.gz'])

        struct = Structure.from_file(self.host_directory+'SPOSCAR')
        nearest_neighbours = struct.get_neighbor_list(self.nearest_neighbour_tolerance,
                                                      sites=[struct.sites[self.defect_index]])[1]
        unique_elements = list(dict.fromkeys([x.species_string for x in struct.sites]))
        neighbours = {}
        for elem in unique_elements:
            neighbours[elem] = [x for x in nearest_neighbours 
                                if struct.sites[x].species_string == elem]
        return(neighbours)
        

    def get_host_phonons(self):
        '''get the host or primitive phonons which the defect phonons will be unfolded back towards
        * currently assumes seekpath and has no manual kpoints mode, but this can be rectified in the future'''

        self.file_unzip([self.host_directory+'SPOSCAR.gz',
                         self.host_directory+'FORCE_SETS.gz'])
        
        ph = load(supercell_filename=self.host_directory+'SPOSCAR',
                  force_sets_filename=self.host_directory+'FORCE_SETS',
                  log_level=0)
        
        
        bands,labels,path_connections = get_band_qpoints_by_seekpath(primitive=ph.primitive,
                                                                     npoints=self.line_density,
                                                                     is_const_interval=False)
                
        ph.run_band_structure(bands,
                              with_eigenvectors=False,
                              path_connections=path_connections,
                              labels=labels) #old - needs to remove get_band_qpoints

        band_data = ph.get_band_structure_dict()
        self.host_band_data = band_data
        self.path_connections = path_connections
        self.labels=labels
        self.host = ph.primitive
        self.host_phonons = ph
    
    def eigenvectors_to_eigendisplacements(self):
        nn = self.get_neighbour_sites()
        atom_coords = self.defect_phonons.supercell.get_scaled_positions()
        num_atoms = self.defect_phonons.supercell.get_number_of_atoms()
        masses = self.defect_phonons.supercell.get_masses()
        #
        eigenvecs = self.defect_band_data['eigenvectors']
        qpts = self.host_band_data['qpoints']        
        
        eigendisplacements = {}
        for atom,sites in tqdm(nn.items(),desc='generating eigendisplacements...'):
            eigendisplacements[atom] = []
            for i,group in enumerate(eigenvecs):
                eigendisplacements[atom].append([])
                for ii,line in enumerate(group):
                    eigendisplacements[atom][i].append([])
                    for iii,freq in enumerate(line):
                        mean = [
                            [
                                np.linalg.norm(
                                    eigvec_to_eigdispl(
                                        freq[at],
                                        q=qpts[i][ii],
                                        frac_coords=atom_coords[at],
                                        mass=masses[at]
                                    )
                                )
                                for elem in sites if elem == at
                            ]
                            for at in range(num_atoms)
                        ]
                        eigendisplacements[atom][i][ii].append(np.mean(list(it.chain        (*mean))))
        self.eigendisplacements = eigendisplacements

    def get_defect_phonons(self):
        '''get the defect phonons which will be unfolded'''

        self.file_unzip([self.defect_directory+'SPOSCAR.gz',
                         self.defect_directory+'FORCE_SETS.gz'])
        
        bands,labels,path_connections = get_band_qpoints_by_seekpath(primitive=self.host,
                                                             npoints=self.line_density,
                                                             is_const_interval=False)
        
        ph = load(supercell_filename=self.defect_directory+'SPOSCAR',
                  force_sets_filename=self.defect_directory+'FORCE_SETS',
                  log_level=0)
                
        ph.run_band_structure(bands,
                              with_eigenvectors=True,
                              path_connections=path_connections,
                              labels=labels)
        

        band_data = ph.get_band_structure_dict()
        self.defect_band_data = band_data
        self.supercell = ph.supercell
        self.defect_phonons = ph
        self.special_points = [x[0] for x in ph.band_structure.qpoints]
        self.special_points.append(ph.band_structure.qpoints[-1][-1])

    def unfold(self):

        def mp_function(qpoints):
            mapping = [x for x in range(self.host_phonons.get_supercell().get_number_of_atoms())]
            mapping[self.defect_index] = None
            unfold = Unfolding(phonon = self.defect_phonons,
                   supercell_matrix = np.abs(np.linalg.inv(self.host_phonons.primitive_matrix).round(0)),
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

        self.unfold_data = {'f':frequencies,'w':weights}

    def plot_unfold(self,base_colour=(0.1,0.1,0.1),with_prim=False,threshold=0.1,atom='Li'):
                
        import matplotlib.pyplot as plt 
        import matplotlib.colors as mcolors
        from sumo.plotting import sumo_base_style
        plt.style.use(sumo_base_style)


        unfolded_weights = copy.deepcopy(self.unfold_data['w'])
        unfolded_freq = self.unfold_data['f']


        for i in range(len(unfolded_weights)):
            unfolded_weights[i][unfolded_weights[i]<threshold] = 0

        norm = mcolors.Normalize(vmin=np.min(unfolded_weights),vmax=np.max(unfolded_weights))

        line = self.host_band_data['distances']
        path_connections = self.path_connections
        labels =self.labels
        distances = self.host_band_data['distances']

        axiscount = 1
        for i,x in enumerate(path_connections[:-1]):
            if i > 0 :
                if not path_connections[i] == path_connections[i-1]:
                    axiscount+=1

        import collections

        sizing = collections.Counter(path_connections[:-1]).values()

        fig,axes = plt.subplots(ncols=axiscount,figsize=(6,6),dpi=300,sharey=True,gridspec_kw={'width_ratios':sizing})

        if with_prim:
            for dist,freq in zip(self.host_band_data['distances'],self.host_band_data['frequencies']):
                axes[0].plot(dist,freq,color='tab:blue',alpha=0.5)
                axes[1].plot(dist,freq,color='tab:blue',alpha=0.5)

        axisvlines = [0]

        totallen = len(distances)
        count = 0 
        fig.axes[count].axvline(axisvlines[0])

        for i,(l,connect,label) in enumerate(zip(distances,path_connections,labels)):
            
            if not l[0] in axisvlines:
                fig.axes[count].axvline(l[0],color='k')
                axisvlines.append(l[0])
            if not l[-1] in axisvlines:
                fig.axes[count].axvline(l[0],color='k')
                axisvlines.append(l[-1])

            qpts = [[q for x in range(len(unfolded_freq[i][0]))] for q in line[i]]
            if self.eigendisplacements and atom:
                ed = self.eigendisplacements[atom]
                max_disp = np.max(ed)

                cols = [[mcolors.to_rgba([(ed[i][w1][w2]/max_disp)*base_colour[0],
                                          (ed[i][w1][w2]/max_disp) *
                                          base_colour[1],
                                          (ed[i][w1][w2]/max_disp)*base_colour[2]], alpha=unfolded_weights[i][w1][w2])
                         for w2 in range(len(unfolded_weights[i][w1]))]
                        for w1 in range(len(unfolded_weights[i]))]

            else:
                cols = [[mcolors.to_rgba(base_colour, alpha=unfolded_weights[i][w1][w2])
                         for w2 in range(len(unfolded_weights[i][w1]))]
                        for w1 in range(len(unfolded_weights[i]))]
                
            for ii,qq in enumerate(qpts):
                fig.axes[count].scatter(x=qq,y=unfolded_freq[i][ii],c=cols[ii],edgecolor=None,linewidths=0,norm=norm,s=5)
            
            if not connect:
                if not i == totallen:
                    count+=1    


        lefts = [0]
        rights = []
        for i, c in enumerate(path_connections):
            if not c:
                lefts.append(i + 1)
                rights.append(i)
            seg_indices = [list(range(lft, rgt + 1)) for lft, rgt in zip(lefts, rights)]
            special_points = []
            for indices in seg_indices:
                pts = [distances[i][0] for i in indices]
                pts.append(distances[indices[-1]][-1])
                special_points.append(pts)        

        l_count = 0         

        for ax, spts in zip(axes,special_points):
            ax.set_xticks(spts)
            ax.set_xlim(spts[0],spts[-1])
            ax.set_xticklabels(labels[l_count : (l_count + len(spts))])
            l_count += len(spts)        
        
        axes[0].set_ylabel('Frequency (THz)')
        plt.tight_layout()   
        plt.show() 

        return(fig)