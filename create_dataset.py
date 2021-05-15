from hera_sim.simulate import Simulator
import numpy as np
import uvtools
from hera_sim.noise import HERA_Tsky_mdl
from hera_sim.data import DATA_PATH
from hera_sim import eor
import copy
import h5py
import os
import waterfall_plot


class Dataset(Simulator):
    '''
    This class is responsible for creating the data, which comes in the form of waterfall plots
    of the foreground + 21cm signal. This class is the interface between the ML code and HERA_sim.
    Running this file returns the data and saves the dataa in the base directory. Besides the parameters
    that HERA sim needs, we also need to know which pixels to mask.
    '''
    def __init__(self, **kwargs):
        
        #these are the necessary hera_sim parameters required to make a basic waterfall plot
        #please see HERA_sim documenation for these parameter details. Here we accpet the acceptable
        #parameter range to generate random visibilities. Certain parameters must remain fixed, for example,
        #the observation frequencies for fair comparison
        
        #this is just the frequency axis which must remain fixed for plotting purposes
        self.n_freq = kwargs.pop('n_freq', 500)
        #observation time must also be fixed for the randomly visibility dataset
        self.n_times = kwargs.pop('n_times', 500)
        #this can be randomized, but we shall keep this fixed for now
        antennas = {0: (20.0, 20.0, 0), 1: (50.0, 50.0, 0) }
        self.antennas = kwargs.pop('antennas', antennas )
        #this must also remain fixed
        self.integration_time =  kwargs.pop('integration_time', 3600*12./1000.)
        #same, where can we generate random foreground noise?
        self.no_autos = kwargs.pop('no_autos', True)

        #set number of items in the randomized dataset
        self.ndata = kwargs.pop('ndata', 3)
    
    def random_mask(self, array):
        '''
            Takes in an array and masks it
        '''
        #make a mask of the same dimension as our data array, we need only do this once.
        bool_array = np.full((self.sim.data.data_array.shape[0], self.sim.data.data_array.shape[1], self.sim.data.data_array.shape[2], self.sim.data.data_array.shape[3]) , False)
        
        #pick a random index to start
        nu_mask_i = int(np.random.uniform(1,500))
        nu_mask_f= int(nu_mask_i + np.random.uniform(10, 100))
        bool_array[:,0,nu_mask_i:nu_mask_f,0] = True
        if np.random.normal(0,1) > 0:
            nu_mask_ti = int(np.random.uniform(1,300))
            nu_mask_tf= int(nu_mask_ti + np.random.uniform(1, 20))
            bool_array[nu_mask_ti:nu_mask_tf,0,:,0] = True
        
        array = np.ma.array(array, mask = bool_array)
        return array
    
    
    #genereate dataset
    def generate(self, **kwargs):
        #make a directory to store all the images
        directory = 'Images'
        if not os.path.exists(directory):
            os.mkdir('Images')
            os.mkdir('Images/Masked')

        data_masked = []
        data_non_masked =[]
        for n in range(self.ndata):
            #run hera sim
            #instantiate the simulator class and pass all the info into the create_dataset class
            self.sim = Simulator(
                                 n_freq = self.n_freq,
                                 n_times = self.n_times,
                                 antennas = self.antennas,
                                 no_autos = self.no_autos,
                                 integration_time = self.integration_time,)
            
            self.sim.add_foregrounds("diffuse_foreground", Tsky_mdl = HERA_Tsky_mdl['xx'])
            if np.random.normal(0,1) > 1.5:
                self.sim.add_foregrounds("pntsrc_foreground", nsrcs = 5000)
            self.sim.add_eor(model=eor.noiselike_eor)
            self.sim.add_noise("thermal_noise", Tsky_mdl = HERA_Tsky_mdl['xx'], Trx = 100.0)
            #self.sim.add_rfi("rfi_stations",)
            #self.sim.add_rfi("rfi_impulse", chance=0.01, strength=100.0)
            self.sim.add_rfi("rfi_scatter", chance=0.001, strength=20, std = 5)
            #generate a random mask
            masked = self.random_mask(self.sim.data.data_array.copy())
            #data_masked.append(masked[:,0,:,0])
            #data_non_masked.append(self.sim.data.data_array[:,0,:,0])
            waterfall_plot.waterfall_separate(masked[:,0,:,0] , self.sim.data.freq_array[0]/1e6, self.sim.data.lst_array, n, masked = True )
            waterfall_plot.waterfall_separate(np.ma.filled(np.ma.array(self.sim.data.data_array[:,0,:,0]),np.inf) , self.sim.data.freq_array[0]/1e6, self.sim.data.lst_array, n )

        #save info
        #with h5py.File('Waterfall_X_Y_sets.hdf5', 'w') as h:
        #    h.create_dataset('data_masked' , data = np.ma.filled(np.ma.array(data_masked),1))
        #    h.create_dataset('data_non_masked' , data = np.array(data_non_masked))
        #    h.create_dataset('mask_array' , data = np.ma.getmask(np.ma.array(data_masked)))
                
        return np.ma.filled(np.ma.array(data_masked),np.inf), np.array(data_non_masked)




        
myData = Dataset(ndata = 20000)
masked, not_masked = myData.generate()






