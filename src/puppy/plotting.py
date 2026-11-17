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
            style='image',
            num_points=600,
            alpha_gamma=0.5,
            vmax=None,
            background='white',
            ):
        """
        plot the unfolded band structure.

        style='image' (default): the unfolding weights are deposited on a fine
            (q, frequency) grid (``num_points`` frequencies, smoothed by ~1.5 grid
            spacings for display only) and drawn as an image. If eigendisplacements
            were projected, the colour shows the (normalised) eigendisplacement and
            the opacity the unfolding weight; otherwise the colour shows the weight.
        style='lines': the original line plot (line width/alpha = weight).
        """
        if style == 'image':
            return self._plot_image(
                figsize=figsize, dpi=dpi, plot_primitive=plot_primitive,
                primitive_colour=primitive_colour, colourmap=colourmap,
                with_colourbar=with_colourbar, num_points=num_points,
                alpha_gamma=alpha_gamma, vmax=vmax, background=background,
                )
        elif style != 'lines':
            raise ValueError("style must be 'image' or 'lines'")

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


        return(fig,axes) #plt.show()

    def _plot_image(self, num_points, colourmap, alpha_gamma, vmax, background,
                    **kws):
        freqs = np.asarray(self.frequencies)
        weights = np.asarray(self.weights)
        has_proj = self.eigendisplacements is not None and np.any(self.eigendisplacements)

        pad = 0.02 * (freqs.max() - freqs.min())
        grid = np.linspace(min(freqs.min(), 0) - pad, freqs.max() + pad, num_points)
        sigma0 = 1.5 * (grid[1] - grid[0])  # minimum display smoothing (~1.5 px)

        if has_proj:
            colour_value = np.asarray(self.eigendisplacements) / np.max(self.eigendisplacements)
            label = 'Normalised Eigendisplacement (arb. units)'
        else:
            colour_value = weights / weights.max()
            label = 'Normalised Weights (arb. units)'

        intensity = np.zeros(freqs.shape[:2] + (num_points,))
        colour = np.zeros_like(intensity)
        for p in range(freqs.shape[0]):
            # how far the host bands move between neighbouring q-points: used to
            # smear each mode so that steep bands are drawn as continuous lines
            prim = np.asarray(self.primitive_frequencies[p])
            step = np.abs(np.gradient(prim, axis=0)) if len(prim) > 1 else np.zeros_like(prim)
            for q in range(freqs.shape[1]):
                f = freqs[p, q]
                nearest = np.abs(f[:, None] - prim[q][None, :]).argmin(axis=1)
                sigma = np.maximum(sigma0, 0.6 * step[q][nearest])
                # peak-normalised: pixel brightness ~ unfolding weight
                g = np.exp(-0.5 * ((grid[None, :] - f[:, None]) / sigma[:, None]) ** 2)
                intensity[p, q] = weights[p, q] @ g
                colour[p, q] = (weights[p, q] * colour_value[p, q]) @ g
        colour = np.divide(colour, intensity, out=np.zeros_like(colour),
                           where=intensity > 1e-8 * intensity.max())

        fig, axes = self.plot_spectral_function(
            spectral=intensity, frequency_points=grid, colourmap=colourmap,
            vmax=1.0 if vmax is None else vmax, colour_by=colour,
            colour_by_limits=(0, 1), colour_by_colourmap=colourmap,
            alpha_gamma=alpha_gamma, background=background,
            colour_by_label=label, **kws,
            )
        return fig, axes

    def plot_spectral_function(
            self,
            spectral=None,
            frequency_points=None,
            figsize=(6, 6),
            dpi=300,
            colourmap='viridis',
            vmax=None,
            log=False,
            plot_primitive=False,
            primitive_colour='white',
            with_colourbar=True,
            ylim=None,
            colour_by=None,
            colour_by_limits=None,
            colour_by_colourmap='coolwarm',
            colour_by_label=None,
            alpha_gamma=0.5,
            background='black',
            ):
        """
        plot the unfolded spectral function A(k, w) as a colour map (Ikeda et al.
        PRB 95, 024305). By default uses the total spectral function computed by
        PhononUnfoldingandProjection.get_spectral_function; pass ``spectral``
        (shape (n_paths, n_q, n_w)) to plot e.g. a partial spectral function
        ``A_partial[:, :, g, h]``.

        colour_by: optional array with the same shape as ``spectral`` (e.g. the
            neighbour enhancement). The plot is then coloured by ``colour_by``
            (``colour_by_colourmap``, limits ``colour_by_limits=(vmin, vmax)``)
            while the opacity follows ``spectral`` (scaled to ``vmax`` and raised to
            ``alpha_gamma``; smaller gamma shows weaker features), so the colour is
            only visible where there is spectral weight, drawn on ``background``.
            Needs evenly spaced frequency_points.
        """
        from matplotlib.colors import LogNorm

        if spectral is None:
            spectral = self.spectral_function
        if frequency_points is None:
            frequency_points = self.spectral_frequencies
        spectral = np.asarray(spectral)

        axes_sizes = self.axes_sizing(with_colourbar=with_colourbar)
        fig, axes = plt.subplots(
            ncols=len(axes_sizes), figsize=figsize, dpi=dpi,
            gridspec_kw={'width_ratios': axes_sizes},
            )
        axes = np.atleast_1d(axes)
        plot_axes = axes[:-1] if with_colourbar else axes

        vmax = spectral.max() if vmax is None else vmax
        if log:
            positive = spectral[spectral > 0]
            norm = LogNorm(vmin=max(positive.min(), vmax * 1e-4), vmax=vmax)
        else:
            norm = Normalize(vmin=0, vmax=vmax)

        if colour_by is not None:
            colour_by = np.asarray(colour_by)
            if colour_by.shape != spectral.shape:
                raise ValueError("colour_by must have the same shape as spectral")
            if colour_by_limits is None:
                colour_by_limits = (np.nanmin(colour_by), np.nanmax(colour_by))
            cnorm = Normalize(*colour_by_limits)
            ccmap = colormaps[colour_by_colourmap]

        special_points = self.get_special_points()
        count = 0
        for i, (dist, A, connected) in enumerate(
            zip(self.distances, spectral, self.path_connections)
            ):
            if colour_by is None:
                plot_axes[count].pcolormesh(
                    dist, frequency_points, np.clip(A.T, norm.vmin if log else 0, None),
                    shading='gouraud', cmap=colourmap, norm=norm, rasterized=True,
                    )
            else:
                rgba = ccmap(cnorm(colour_by[i].T))
                rgba[..., 3] = np.clip(A.T / vmax, 0, 1) ** alpha_gamma
                plot_axes[count].set_facecolor(background)
                plot_axes[count].imshow(
                    rgba, origin='lower', aspect='auto', interpolation='bilinear',
                    extent=(dist[0], dist[-1], frequency_points[0], frequency_points[-1]),
                    )
            if plot_primitive:
                plot_axes[count].plot(
                    dist, self.primitive_frequencies[i], color=primitive_colour,
                    lw=0.5, ls='dotted', alpha=0.6,
                    )
            if not connected and i != len(self.distances) - 1:
                count += 1

        l_count = 0
        for i, (ax, spts) in enumerate(zip(plot_axes, special_points)):
            ax.set_xticks(spts)
            ax.set_xlim(spts[0], spts[-1])
            ax.set_xticklabels(self.labels[l_count:(l_count + len(spts))])
            l_count += len(spts)
            ax.set_ylim(ylim if ylim else (frequency_points[0], frequency_points[-1]))
            for x in spts[1:-1]:
                ax.axvline(x, color='white' if colour_by is None else 'grey',
                           lw=0.5, alpha=0.5)
            if i > 0:
                ax.set_yticks([])
        plot_axes[0].set_ylabel('Frequency (THz)')

        if with_colourbar and colour_by is not None:
            Colorbar(axes[-1], cmap=ccmap, norm=cnorm)
            axes[-1].set_ylabel(colour_by_label or '')
        elif with_colourbar:
            Colorbar(axes[-1], cmap=colourmap, norm=norm)
            axes[-1].set_ylabel(r'$A(\mathbf{k}, \omega)$ (THz$^{-1}$)')
        return (fig, axes)