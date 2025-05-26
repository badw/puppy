### kept for posterity
# 
#     def eigenvectors_to_eigendisplacements(
#            self, 
#            all_atoms=None, 
#            project_specific_sites=None, 
#            direction=None
#            ):
#        if not all_atoms:
#            if not project_specific_sites:
#                nn = self.get_defect_neighbour_sites()
#            else:
#                nn = project_specific_sites
#        else:
#            if all_atoms:
#                cell = self.host_phonons.supercell
#                atom_types = list(dict.fromkeys(cell.get_chemical_symbols()))
#                nn = self.get_all_sites_of_an_element(str(atom_types[0]))
#                for atom in atom_types[1:]:
#                    nn.update(self.get_all_sites_of_an_element(str(atom)))
#            else:
#                nn = self.get_all_sites_of_an_element(all_atoms)    #

#        print(nn)
#        atom_coords = self.defect_phonons.supercell.get_scaled_positions()
#        masses = self.defect_phonons.supercell.get_masses()
#        eigenvecs = np.array(self.defect_band_data['eigenvectors'])
#        # reformat eigenvecs
#        #new_eigenvectors = []
#        #for i,q in enumerate(eigenvecs):
#        #    new_eigenvectors.append([])
#        #    for ii,l in enumerate(q):
#        #        new_eigenvectors[i].append(l.T)#

#        eigenvecs = eigenvecs.swapaxes(3,2)
#        #

#        #eigenvecs = new_eigenvectors 
#        qpts = self.defect_band_data['qpoints']    #

#        if not direction:
#            eigendisplacements = {}
#            for atom, sites in tqdm.tqdm(nn.items(),desc='generating_eigendisplacements...'):
#                eigendisplacements[atom] = []
#                for i, group in enumerate(eigenvecs):
#                    eigendisplacements[atom].append([])
#                    for ii, line in enumerate(group):
#                        eigendisplacements[atom][i].append([])
#                        for iii, freq in enumerate(line): # this is wrong....
#                            eigdispl = [
#                                np.linalg.norm(
#                                    eigvec_to_eigdispl(
#                                        freq[site*3:site*3+3], # [site:site+3]
#                                        q=qpts[i][ii],
#                                        frac_coords=atom_coords[site],
#                                        mass=masses[site])
#                                ) for site in sites if site]
#                            if eigdispl:
#                                mean_eigdispl = np.mean(eigdispl)
#                            else:
#                                mean_eigdispl = 0
#                            eigendisplacements[atom][i][ii].append(
#                                mean_eigdispl)    #

#        else:
#            if direction in ['x', 'a']:
#                _direc = 0
#            elif direction in ['y', 'b']:
#                _direc = 1
#            elif direction in ['z', 'c']:
#                _direc = 2
#            eigendisplacements = {}
#            for atom, sites in tqdm(nn.items(),desc='generating_eigendisplacements...'):
#                eigendisplacements[atom] = []
#                for i, group in enumerate(eigenvecs):
#                    eigendisplacements[atom].append([])
#                    for ii, line in enumerate(group):
#                        eigendisplacements[atom][i].append([])
#                        for iii, freq in enumerate(line):
#                            eigdispl = [np.linalg.norm(
#                                eigvec_to_eigdispl(
#                                        freq[site*3:site*3+3],
#                                        q=qpts[i][ii],
#                                        frac_coords=atom_coords[site],
#                                        mass=masses[site])[_direc]
#                                        )
#                                for site in sites if site]
#                            if eigdispl:
#                                mean_eigdispl = np.mean(eigdispl)
#                            else:
#                                mean_eigdispl = 0
#                            eigendisplacements[atom][i][ii].append(
#                                mean_eigdispl)    #

#        self.eigendisplacements = eigendisplacements#

#        self.unfold_data['eigendisplacements'] = eigendisplacements#
#
#

#    def _unfold_function(self,qpoints,**kws):
#        self.__dict__.update(kws)
#        
#        unfold = Unfolding(
#            phonon=self.defect_phonons,
#            supercell_matrix=self.matrix,
#            ideal_positions=self.host_phonons.get_supercell().get_scaled_positions(),
#            atom_mapping=self.mapping,
#            qpoints=qpoints # can we make this 1 qpoint?
#        )#

#        unfold.run()
#        weights = unfold.unfolding_weights
#        freqs = unfold.frequencies#

#        return([freqs,weights])#

#    def unfold(self, **kws):
#        #set the atom mappings - defect vacancy = None 
#        self.mapping = [
#            x for x in range(
#                self.host_phonons.get_supercell().get_number_of_atoms()
#                )
#            ]
#        if self.defect_site_index:
#            self.mapping[self.defect_site_index] = None#

#        qpoints = self.unfold_data['host_band_data']['qpoints']#

#        frequencies = []
#        weights = []
#        for q_set in tqdm.tqdm(qpoints,desc='unfolding phonons...'):
#            freqs,wts = self._unfold_function(q_set)
#            frequencies.append(freqs)
#            weights.append(wts)#

#        self.unfold_data['f'] = frequencies
#        self.unfold_data['w'] = weights #

#        return([frequencies,weights])