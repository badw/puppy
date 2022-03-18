import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib import colors as mcolors
import numpy as np
import itertools as it

class UnfolderPlotting:
    
    def __init__(self,data):
        self.data = data
        
    def single_plot(self,ax,i,threshold,s,primitive_bs=False):
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
        
        
    def _single_plot_with_weight_legacy(self,ax,threshold,s,primitive_bs=False,primitive_color='tab:blue'):
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
        
        if primitive_bs == True:
            distances  = self.data['bs_p']['distances']
            frequencies = self.data['bs_p']['frequencies']
            connections = self.data['bs_p']['connections']
            for i, (d,f,c) in enumerate(zip(distances,frequencies,connections)):
                if i == 0:
                    ax.plot(d,f[::-1],color=primitive_color,label='bulk')
                else:
                    ax.plot(d,f[::-1],color=primitive_color,label='bulk')
            
            
            #bs_p_f = self.data['bs_p']['frequencies'][-1]
            #ax.plot(pq,bs_p_f[::-1],color=primitive_color,label='bulk')
        
        
        ax.set_xlim(np.min(pq),np.max(pq))
        #ax.set_ylim(np.min(uf)-2,np.max(uf)+2)
        ax.set_xticks(line)
        ax.set_xticklabels(formatted_labels)
        vline = [ax.axvline(x,color='k') for x in line[1:-1]]
        
        
    def single_plot_with_weight(self,ax,threshold,s,primitive_bs=False,primitive_color='tab:blue'):
        ax = ax
        labels = []
        lines = []
        maxpq = None
        for i,path in enumerate(self.data['data']):
            if i == 0:
                labels.append(self.data['data'][path]['prim_data']['label'])
                lines.append(self.data['data'][path]['prim_data']['line'])
                pq = self.data['data'][path]['prim_data']['q']
                maxpq = np.max(pq)
                
                uf = self.data['data'][path]['unfolded_data']['f']
                wts = self.data['data'][path]['unfolded_data']['w']
                eigendisp = self.data['data'][path]['eigendisplacements']
                max_disp = np.max(eigendisp) 
                
                norm = Normalize(vmin=0.0,vmax=np.max(wts)) # this probably needs to be combined over the plot
        
                qpts = [[q for x in range(len(uf[0]))] for q in pq]
                col = [[mcolors.to_rgba([eigendisp[1][w1][w2]/max_disp,0.0,0.0],alpha=wts[w1][w2]*threshold)
                        for w2 in range(len(wts[w1]))] 
                       for w1 in range(len(wts))]            
        
                for j,q in enumerate(qpts):
                    ax.scatter(q,uf[j],c=col[j],s=s,edgecolor=None,linewidths=0,norm=norm)
                
            else:
                labels.append(self.data['data'][path]['prim_data']['label'])
                lines.append(self.data['data'][path]['prim_data']['line'])
                pq = self.data['data'][path]['prim_data']['q']+maxpq
                maxpq = np.max(pq)
                
                uf = self.data['data'][path]['unfolded_data']['f']
                wts = self.data['data'][path]['unfolded_data']['w']
                eigendisp = self.data['data'][path]['eigendisplacements']
                max_disp = np.max(eigendisp) 
                
                norm = Normalize(vmin=0.0,vmax=np.max(wts)) # this probably needs to be combined over the whole plot
        
                qpts = [[q for x in range(len(uf[0]))] for q in pq]
                col = [[mcolors.to_rgba([eigendisp[1][w1][w2]/max_disp,0.0,0.0],alpha=wts[w1][w2]*threshold)
                        for w2 in range(len(wts[w1]))] 
                       for w1 in range(len(wts))]            
        
                for j,q in enumerate(qpts):
                    ax.scatter(q,uf[j],c=col[j],s=s,edgecolor=None,linewidths=0,norm=norm)

         
        
        if primitive_bs == True:
            distances  = self.data['prim']['distances'] 
            frequencies = self.data['prim']['frequencies']
            connections = self.data['prim']['connections']
            scale = maxpq / np.max(distances)
            
            for i, (d,f,c) in enumerate(zip(distances,frequencies,connections)):
                if i == 0:
                    ax.plot(d*scale,f,color=primitive_color,label='bulk',zorder=0,alpha=0.5)
                else:
                    ax.plot(d*scale,f,color=primitive_color,label='bulk',zorder=0,alpha=0.5)
        
        labels = list(it.chain(*labels))
        labels = [x for i,x in enumerate(labels) if not labels[i-1] == x]
        formatted_labels = ['$\\Gamma$' if x == 'G' else x for x in labels]  
        lines = list(dict.fromkeys(np.cumsum(lines)))
        
        ax.set_xlim(0,maxpq)
        ax.set_xticks(lines)
        ax.set_xticklabels(formatted_labels)
        vline = [ax.axvline(x,color='k') for x in lines[1:-1]]
