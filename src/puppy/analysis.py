###working progress. 

def return_cage_structure(,
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