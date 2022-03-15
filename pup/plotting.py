import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import colors as mcolors
import numpy as np 

class UnfolderPlotting:
    
    def __init__(self,data):
        self.data = data
        
    def single_plot(self,ax,i,threshold,s):
        ax = ax
        
        labels = self.data[i]['prim_data']['label']
        #q_line = self.data[i]['prim_data']['qpts']
        line = self.data[i]['prim_data']['line']
        pq = self.data[i]['prim_data']['q']
        
        uf = self.data[i]['unfolded_data']['f']
        wts = self.data[i]['unfolded_data']['w']
        
        norm = Normalize(vmin=0.0,vmax=np.max(wts))
        
        qpts = [[q for x in range(len(uf[0]))] for q in pq]
        col = [[mcolors.to_rgba([0.0,0.0,0.5],alpha=wts[w1][w2]*threshold)
                for w2 in range(len(wts[w1]))] 
               for w1 in range(len(wts))]            
        
        for j,q in enumerate(qpts):
            ax.scatter(q,uf[j],c=col[j],s=s,edgecolor=None,linewidths=0,norm=norm)

        formatted_labels = ['$\\Gamma$' if x == 'G' else x for x in labels]        
        
        print(np.min(pq),np.max(pq))
        print(line)
        ax.set_xlim(np.min(pq),np.max(pq))
        ax.set_ylim(np.min(uf)-2,np.max(uf)+2)
        ax.set_xticks(line)
        ax.set_xticklabels(formatted_labels)
        vline = [ax.axvline(x,color='k') for x in line[1:-1]]
        print(formatted_labels)
        
        
    def single_plot_with_weight(self,ax,threshold,s):
        ax = ax
        
        labels = self.data['prim_data']['label']
        line = self.data['prim_data']['line']
        pq = self.data['prim_data']['q']
        
        uf = self.data['unfolded_data']['f']
        wts = self.data['unfolded_data']['w']
        eigendisp = self.data['eigendisplacements']
        max_disp = np.max(eigendisp) 
        
        norm = Normalize(vmin=0.0,vmax=np.max(wts))
        
        qpts = [[q for x in range(len(uf[0]))] for q in pq]
        col = [[mcolors.to_rgba([eigendisp[1][w1][w2]/max_disp,0.0,0.0],alpha=wts[w1][w2]*threshold)
                for w2 in range(len(wts[w1]))] 
               for w1 in range(len(wts))]            
        
        for j,q in enumerate(qpts):
            ax.scatter(q,uf[j],c=col[j],s=s,edgecolor=None,linewidths=0,norm=norm)

        formatted_labels = ['$\\Gamma$' if x == 'G' else x for x in labels]   
        
        
        ax.set_xlim(np.min(pq),np.max(pq))
        #ax.set_ylim(np.min(uf)-2,np.max(uf)+2)
        ax.set_xticks(line)
        ax.set_xticklabels(formatted_labels)
        vline = [ax.axvline(x,color='k') for x in line[1:-1]]
