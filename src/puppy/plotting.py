import matplotlib.pyplot as plt 
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import matplotlib as mpl 
import numpy as np 
import copy 

class PuppyPlotter:

    def __init__(self,unfold_data:dict):
        try:
            self.eigendisplacements = unfold_data['eigendisplacements']
        except Exception:
            self.eigendisplacements = None 
        self.host_band_data = unfold_data['host_band_data']
        self.defect_band_data = unfold_data['defect_band_data']
        self.unfold_data = {'f':unfold_data['f'],
                            'w':unfold_data['w']}
        self.path_connections = unfold_data['path_connections']
        self.labels = unfold_data['labels'] 

    @staticmethod
    def axes_sizing(path_connections,with_colourbar=True):
        lefts = [0] 
        rights = []
        for i, c in enumerate(path_connections):
            if not c:
                lefts.append(i + 1)
                rights.append(i)

        seg_indices = [list(range(lft, rgt + 1)) for lft, rgt in zip(lefts, rights)]
        sizing = [len(x) for x in seg_indices]
        if with_colourbar:
            sizing.append(0.2) # for the colourbar
        return(sizing)

    def plot_unfold(self,
                    custom_axes=None,
                    base_colour=(0.1,0.1,0.1),
                    with_prim=False,
                    prim_colour='tab:Blue',
                    cmap='viridis',
                    threshold=0.1,
                    atom='Li',
                    ylim=None,
                    show_lines=True,
                    plot_kws=None,
                    legend_kws=None,
                    figsize=(8,8),
                    show_colourbar=True):
                

        
        if not plot_kws:
            plot_kws = {'edgecolor':None,
                        'linewidths':0,
                        's':2,
                        'rasterized':True}
            
        if not legend_kws:
            legend_kws = {'bbox_to_anchor':[0.5,0.9],
                          'edgecolor':'black',
                          'loc':'upper right',
                          'framealpha':1,
                          'facecolor':'white'}

        legend_lines = [
            Line2D([0], [0], color=base_colour, alpha=0.5, lw=2),
            Line2D([0], [0], color=(0.1, 0.1, 0.1), lw=2)
        ]
        legend_handles = ['chosen {} atoms'.format(atom),
                        'defect cell']

        
        unfolded_weights = copy.deepcopy(self.unfold_data['w'])
        unfolded_freq = self.unfold_data['f']

        #unfolded_weights = unfolded_weights / unfolded_weights.max()

        for i in range(len(unfolded_weights)):
            unfolded_weights[i][unfolded_weights[i]< threshold] = 0


        line = self.host_band_data['distances']
        path_connections = self.path_connections
        labels =self.labels
        distances = self.host_band_data['distances']

        if not np.any(custom_axes):
            sizing = self.axes_sizing(self.path_connections,with_colourbar=show_colourbar)
            axiscount = len(sizing)
            fig,axes = plt.subplots(ncols=axiscount,figsize=figsize,dpi=300,gridspec_kw={'width_ratios':sizing})
        else:
            axes = custom_axes

        if with_prim:
            for dist,freq in zip(self.host_band_data['distances'],self.host_band_data['frequencies']):
                [ax.plot(dist,freq,color=prim_colour,zorder=1) for ax in axes[:-1]]

            legend_lines.append(Line2D([0],[0],color=prim_colour,lw=2))
            legend_handles.append('primitive cell')

        axisvlines = [0]

        totallen = len(distances)
        count = 0 
        if show_lines:
            axes[count].axvline(axisvlines[0])
        colourmap = mpl.colormaps[cmap]
        for i,(l,connect,label) in enumerate(zip(distances,path_connections,labels)):
            
            if not l[0] in axisvlines:
                if show_lines:
                    axes[count].axvline(l[0],color='k')
                axisvlines.append(l[0])
            if not l[-1] in axisvlines:
                if show_lines:
                    axes[count].axvline(l[0],color='k')
                axisvlines.append(l[-1])

            qpts = [[q for x in range(len(unfolded_freq[i][0]))] for q in line[i]]
            if self.eigendisplacements and atom:
                ed = self.eigendisplacements[atom]
                max_disp = np.max(ed)
                

                norm = mcolors.Normalize(vmin=np.min(ed/max_disp),vmax=np.max(ed/max_disp))

                cols = [
                    [
                        colourmap(ed[i][w1][w2]/max_disp,
                                  alpha=unfolded_weights[i][w1][w2])
                        for w2 in range(len(unfolded_weights[i][w1]))
                    ]
                    for w1 in range(len(unfolded_weights[i]))
                ]

            else:
                norm = mcolors.Normalize(vmin=0,vmax=1)

                cols = [
                    [
                        colourmap(
                            unfolded_weights[i][w1][w2], alpha=unfolded_weights[i][w1][w2])
                        for w2 in range(len(unfolded_weights[i][w1]))
                    ]
                    for w1 in range(len(unfolded_weights[i]))
                ]
                
            for ii,qq in enumerate(qpts):
                axes[count].scatter(x=qq,
                                        y=unfolded_freq[i][ii],
                                        c=cols[ii],
                                        norm=norm,
                                        **plot_kws)
            
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
        
        if show_colourbar:
            for ax, spts in zip(axes[:-1],special_points):
                ax.set_xticks(spts)
                ax.set_xlim(spts[0],spts[-1])
                ax.set_xticklabels(labels[l_count : (l_count + len(spts))])
                l_count += len(spts)  

        else:
            for ax, spts in zip(axes,special_points):
                ax.set_xticks(spts)
                ax.set_xlim(spts[0],spts[-1])
                ax.set_xticklabels(labels[l_count : (l_count + len(spts))])
                l_count += len(spts)  

        

        if not ylim:
            mi = np.min(self.defect_band_data['frequencies'])
            ma = np.max(self.defect_band_data['frequencies'])
            if show_colourbar:
                [ax.set_ylim(np.round(mi)-2,np.round(ma)+2) for ax in axes[0:-1]]
            else:
                [ax.set_ylim(np.round(mi)-2,np.round(ma)+2) for ax in axes]
                
        else:
            if show_colourbar:
                [ax.set_ylim(ylim[0],ylim[1]) for ax in axes[0:-1]]
            else:
                [ax.set_ylim(ylim[0],ylim[1]) for ax in axes]
        
        if show_colourbar:
            mpl.colorbar.Colorbar(axes[-1],cmap=cmap,norm=norm)
            axes[-1].set_ylabel('normalised displacement (arb. units)')

        if not np.any(custom_axes):
            axes[0].set_ylabel('Frequency (THz)')
            fig.legend(legend_lines,legend_handles,**legend_kws)
        else:
            axes[0].legend(legend_lines,legend_handles,**legend_kws)

        if not np.any(custom_axes):
            fig.tight_layout()
            #plt.tight_layout()   
            #plt.show() 
            return(fig,axes)