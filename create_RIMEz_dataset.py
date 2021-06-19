
import numpy as np
import uvtools
import copy
import h5py
import os
from pyuvdata import UVData
from tqdm import tqdm

class Dataset:
    '''
    This class is responsible for creating the visibility data, which comes in the form of waterfall plots 
    of the foreground + 21cm signal. This class in the interface between the simulation data from RIMEz and the ML code. We're reading in the UVH5 file for a particular baseline, taking the data/masks and storing them into an array. We will then cut up the large array into smaller pieces
    '''
    def __init__(self, **kwargs):
        
        #these are the necessary hera_sim parameters required to make a basic waterfall plot
        #please see HERA_sim documenation for these parameter details. Here we accpet the acceptable
        #parameter range to generate random visibilities. Certain parameters must remain fixed, for example,
        #the observation frequencies
        
        #this is just the frequency axis which must remain fixed for consistency with observations
        self.n_freq = kwargs.pop('n_freq', 512)
        
        #how many times are we cutting the data array?
        self.n_times = kwargs.pop('n_times', 512)
        
        #where do we find the uvh5 file?
        self.filename = kwargs.pop('filename', '/home/grx40/scratch/HERA_ML/RIMEz/all_sim_data.uvh5')
        
        #which baselines are we using?
        #this needs to be a list
        self.bls = kwargs.pop('bls', [(9, 10, 'ee')] )
        
        #how many baselines are we using?
        self.n_baselines = len(self.bls)
        
        #for now just fix which baseline we would like to use
        #self.bls = [(84, 85, 'nn')]
        
        #create a UVData object
        self.uvd = UVData()
        
        print('finished init, ',  flush = True)
    
    def random_mask(self, array):
        '''
            Takes in an array (visibilities) with the 4 default hera_sim shape and masks it
        '''
        #make a mask of the same dimension as our data array, we need only do this once.
        binary_mask = np.full((array.shape[0], array.shape[1]) , 1)

        #pick a random index to start
        #nu_mask_i = int(np.random.uniform(1,100))
        #nu_mask_f= int(nu_mask_i + np.random.uniform(10, 100))
        
        #zero corresponds to the masked value (because we are including the mask in the input channel and we
        #cannot take a gradient of the boolean values. Please ignore the name of this array, it is really binary
        #will change that later
        #binary_mask[:,nu_mask_i:nu_mask_f] = 0
            #if np.random.normal(0,1) > 0:
        #nu_mask_ti = int(np.random.uniform(1,300))
        #nu_mask_tf= int(nu_mask_ti + np.random.uniform(1, 20))
        
        #we are doing fixes masked locations for now
        nu_mask_ti = 60
        nu_mask_tf = 100
        
        binary_mask[nu_mask_ti:nu_mask_tf,:] = 0
        
        array[:,:,0] = np.multiply(array[:,:,0],binary_mask)
        array[:,:,1] = np.multiply(array[:,:,1],binary_mask)

        array[:,:,2] = binary_mask
        
        #the third channel will have the real flags from now on
        #array[:,:,3] = binary_mask
        return array
    
    
    #genereate dataset
    def generate(self, **kwargs):
        #make a directory to store all the arrays
        directory = 'ML_RIMEz_Data'
        if not os.path.exists(directory):
            os.mkdir('ML_RIMEz_Data')
        
        data_masked = []
        data_non_masked = []
    
        for i in tqdm(range(self.n_baselines) , desc = 'Making Data'):
            self.uvd.read(self.filename, bls = self.bls[i])
            key = self.bls[i]
            print('Doing key/bls ' , key, self.bls[i], flush = True )
            
            # check the dimension of the input and determine how many times we can cut up the time axis
            dim_t , dim_f = self.uvd.get_data(key).shape[0] , self.uvd.get_data(key).shape[1]
            n_cuts_t = int(float(dim_t)/float(self.n_times))
            n_cuts_f = int(float(dim_f)/float(self.n_freq))
            delta_t = self.n_times
            delta_f = self.n_freq

            print('we are using a data set with dimension' , dim_t, ' by ' , dim_f,  ' and will be segmented in chunks of ' , delta_t, delta_f, flush = True)
            
            ctr_t = 0
            while ((ctr_t+1)*delta_t < dim_t):
                ctr_f = 0
                while (ctr_f*delta_f < dim_f):
                    #re-retrieve the data array and store the data
                    self.data= self.uvd.get_data(key)[int(ctr_t*delta_t):int(ctr_t*delta_t +delta_t), int(ctr_f*delta_f):int(ctr_f*delta_f +delta_f)   ]
                    self.flags = self.uvd.get_flags(key)[int(ctr_t*delta_t):int(ctr_t*delta_t +delta_t), int(ctr_f*delta_f):int(ctr_f*delta_f +delta_f)]
                
                    print('the shape of the cut up data and flags are',self.data.shape, self.flags.shape )
                
                    #split the visibilities
                    amplitude = np.abs(self.data)
                    phase = np.angle(self.data)
                
                    #put them in a single array
                    #the 3 channels are due amplitude, phase and mask
                    visibilities = np.zeros((amplitude.shape[0], amplitude.shape[1], 4))
                
                    #put the amplitude and phase in the first and 2nd channel
                    visibilities[:,:,0] = amplitude
                    visibilities[:,:,1] = phase
                
                    #generate a random mask
                    masked = self.random_mask(visibilities.copy())
                
                    #take the log of the magnitude
                    masked[:,:,0] = np.log10(masked[:,:,0])
                    masked[:,:,0][masked[:,:,0] == -np.inf] = 0

                    #put both the unmasked (below) and masked (above) version of the visibilities in the same units
                    visibilities[:,:,0] = np.log10(visibilities[:,:,0])
                
                    #assign the true values to be the masks on the 3rd channel but we want 0s to the flags and 1s to not be flags in both cases (not bool type)
                    visibilities[:,:,2] = masked[:,:,2]
                    visibilities[:,:,3] = np.where(self.flags == True, 0 , 1)
                
                    print('On iteration ' , ctr_t , ' of t and  ' ,ctr_f, ' of f , the final shape of our data is ' , visibilities.shape, flush = True)
                    data_masked.append(masked)
                    data_non_masked.append(visibilities)
            
                    ctr_f += 1
                ctr_t += 1

        #i really don't want to run this everytime
        np.save('masked_RIMEz_dataset.npy', np.array(data_masked))
        np.save('not_masked_RIMEz_dataset.npy', np.array(data_non_masked))
        return np.array(data_masked), np.array(data_non_masked)
        




#bls_to_use = [(9, 10, 'ee'), (9, 10, 'nn'), (9, 10, 'en'), (9, 10, 'ne') ]
#bls_to_use = [(9, 10, 'ee')], (9, 10, 'nn')]
#myData = Dataset(bls = bls_to_use  )
#masked, not_masked = myData.generate()

#np.save('masked_RIMEz_dataset.npy', masked)
#np.save('not_masked_RIMEz_dataset.npy', not_masked)







