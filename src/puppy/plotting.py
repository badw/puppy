from matplotlib.collections import LineCollection
from matplotlib.colors import colorConverter, Normalize
from matplotlib import colormaps 
import matplotlib as mpl 
import matplotlib.pyplot as plt 
import numpy as np 
from puppy.unfolder import PhononUnfoldingandProjection
from matplotlib.colorbar import Colorbar


class PuppyPlotter:
    def __init__(self,puppy,**kws):
        if isinstance(puppy,PhononUnfoldingandProjection):
            self.__dict__.update(puppy.as_dict())
        elif isinstance(puppy,dict):
            self.__dict__.update(puppy)
        else:
            raise TypeError("puppy must be a PhononUnfoldingandProjection object or a dictionary representation of it.")
        

        self.__dict__.update(kws)
        
    def axes_sizing(self,with_colourbar=True):
        lefts = [0]
        rights = []
        for i, c in enumerate(self.path_connections):
            if not c:
                lefts.append(i + 1)
                rights.append(i)    

        seg_indices = [list(range(lft, rgt + 1)) for lft, rgt in zip(lefts, rights)]
        sizing = [len(x) for x in seg_indices]
        if with_colourbar:
            sizing.append(0.2)  # for the colourbar
        return sizing

    def get_special_points(self):
        lefts = [0]
        rights = []
        for i, c in enumerate(self.path_connections):
            if not c:
                lefts.append(i + 1)
                rights.append(i)
            seg_indices = [list(range(lft, rgt + 1)) for lft, rgt in zip(lefts, rights)]
            special_points = [] 
            for indices in seg_indices:
                pts = [self.distances[i][0] for i in indices]
                pts.append(self.distances[indices[-1]][-1])
                special_points.append(pts)   
        
        return(special_points)

    def plot(
            self,
            figsize=(6,6),
            dpi=300,
            plot_primitive=False,
            primitive_colour='tab:grey',
            colourmap='cividis',
            linewidth=1,
            with_colourbar=True,
            ):
        
        #cmap = colormaps[colourmap]

        axes_sizes = self.axes_sizing(with_colourbar=with_colourbar)

        fig,axes = plt.subplots(
            ncols=len(axes_sizes),figsize=figsize,dpi=dpi,gridspec_kw={'width_ratios':axes_sizes},
            )
        # colour based on eigendisplacements, or if no eigendisplacements are given then on unfolding weights
        if np.any(self.eigendisplacements):
            #normalised
            colouring = self.eigendisplacements / self.eigendisplacements.max()
            axes[-1].set_ylabel('Normalised Eigendisplacement (arb. units)')

        else:
            colouring = self.weights / self.weights.max()
            axes[-1].set_ylabel('Normalised Weights (arb. units)')

        
        norm = Normalize(colouring.min(),colouring.max())

        special_points = self.get_special_points()

        if plot_primitive: # this needs to be better

            for dist,freq in zip(
                self.distances,
                self.primitive_frequencies
                ):
                [
                    ax.plot(
                        dist,
                        freq,
                        zorder=1,
                        color=primitive_colour,
                        alpha=0.3,
                        linestyle='dotted'
                        ) for ax in axes[:-1]
                    ]


        linewidth = 1
        count = 0
        ii=0
        for (
            subdistances,
            path_connection,
            label,
            subfrequencies,
            subweights,
            subcolouring,
        ) in zip(
            self.distances,
            self.path_connections,
            self.labels,
            self.frequencies,
            self.weights,
            colouring,
        ):  
        
            dist = np.array(
            [subdistances]*subfrequencies.shape[1]
            )

    
            for i in range(len(dist)):
                x = dist[i]
                y = subfrequencies.T[i]
                points = np.array(
                    [x,y]
                    ).T.reshape(-1,1,2)    

                lwidths = subweights[:,:].T[i]*linewidth #.flatten()    

                segments = np.concatenate([points[:-1], points[1:]], axis=1)
            
                colors = [norm(col) for col in subcolouring.T[i]]    

                alphas = np.abs(lwidths.T / (linewidth + 0.001))    

                l = LineCollection(
                    segments,
                    cmap=colourmap,
                    alpha=alphas,
                    linewidths=lwidths.T,
                    antialiased=True,
                    )
                l.set_array(colors)    

                axes[count].add_collection(l)

            axes[count].set_ylim(np.min(self.frequencies),np.max(self.frequencies))
            axes[count].set_xlim(special_points[count][0],special_points[count][-1])
            if count>=1:
                axes[count].set_yticklabels([])
                axes[count].set_yticks([])

            ii+=1
            if not path_connection: 
                if not ii == len(self.qpaths):
                    count+=1

        cbar = Colorbar(axes[-1],cmap=colourmap,norm=norm)
        l_count = 0 
        for ax, spts in zip(axes[:-1],special_points):
            ax.set_xticks(spts)
            ax.set_xlim(spts[0],spts[-1])
            ax.set_xticklabels(self.labels[l_count : (l_count + len(spts))])
            l_count += len(spts)    

        axes[0].set_ylabel('Frequency (THz)')    
    

        return(fig,ax) #plt.show()