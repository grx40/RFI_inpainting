import hera_sim
import matplotlib.pyplot as plt
import numpy as np
import uvtools
import time


def waterfall_separate(vis, Data, vrange=None, title=None,):
    """
        A wrapper around the uvtools' waterfall function providing some
        extra labelling and plot adjustment.
        """
    freq=Data.sim.data.freq_array[0]/1e6
    lsts=Data.sim.data.lst_array
    vmax = np.log10(np.max(np.where(np.abs(vis) != np.inf  ))/0.5)
    fig, ax = plt.subplots(figsize=(12,5))
    print(lsts.max(), freq.max())
    
    if title is not None:
        ax.set_title(title, fontsize=12)
    plt.sca(ax)
    uvtools.plot.waterfall(vis, mode='log', mx=vmax, drng=vrange,extent=(freq.min(), freq.max(), lsts.min(), lsts.max()))
    #plt.colorbar(label=r'log$_{10}$(Vis/Jy)')
    #plt.ylabel("LST", fontsize=12)
    plt.show()
    plt.close()

    fig, ax = plt.subplots(figsize=(12,5))
    plt.sca(ax)
    uvtools.plot.waterfall( vis, mode='phs', extent=(freq.min(), freq.max(), lsts.min(), lsts.max()))
    #plt.colorbar(label='Phase [rad]')
    #plt.xlabel("Frequency [MHz]", fontsize=12)
    #plt.ylabel("LST", fontsize=12)
    plt.show()
    plt.close()




