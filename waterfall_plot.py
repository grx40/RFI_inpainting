import hera_sim
import matplotlib.pyplot as plt
import numpy as np
import uvtools
import time


def waterfall_separate(vis, freq, lsts,n, **kwargs):
    """
        A wrapper around the uvtools' waterfall function providing some
        extra labelling and plot adjustment.
        """
    vrange=None
    title=None
    masked = kwargs.pop('masked', False)

    #freq=Data.sim.data.freq_array[0]/1e6
    #lsts=Data.sim.data.lst_array
    vmax = np.log10(np.max(np.where(np.abs(vis) != np.inf  ))/0.5)
    fig, ax = plt.subplots(figsize=(12,5))

    if title is not None:
        ax.set_title(title, fontsize=12)
    plt.sca(ax)
    uvtools.plot.waterfall(vis, mode='log', mx=vmax, drng=vrange,extent=(freq.min(), freq.max(), lsts.min(), lsts.max()))
    #plt.colorbar(label=r'log$_{10}$(Vis/Jy)')
    #plt.ylabel("LST", fontsize=12)
    if masked:
        plt.savefig('Images/Masked/'+str(int(n)) + '.png')
    else:
        plt.savefig('Images/'+str(int(n)) + '.png')
    
    plt.show()
    plt.close()

    fig, ax = plt.subplots(figsize=(12,5))
    plt.sca(ax)
    uvtools.plot.waterfall( vis, mode='phs', extent=(freq.min(), freq.max(), lsts.min(), lsts.max()))
    #plt.colorbar(label='Phase [rad]')
    #plt.xlabel("Frequency [MHz]", fontsize=12)
    #plt.ylabel("LST", fontsize=12)
    if masked:
        plt.savefig('Images/Masked/'+str(int(n)) + '.png')
    else:
        plt.savefig('Images/'+str(int(n)) + '.png')
    plt.show()
    plt.close()




