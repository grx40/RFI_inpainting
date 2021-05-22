import numpy as np
from tensorflow.keras import backend as K

def custom_loss_wrapper(mask_array):
    '''
        inverse mask out the portions on the arrays that are not RFI.
        Compute the chi^2 between the maps
    '''
    #here we must apply the mask to the arrays
    
    #draw a random_number
    if np.random.normal(0, 1) > 1:
        print('MASK IS ' , mask, flush = True)

    def loss_function(Y_pred, Y_true):
        '''
        This function computes the loss from the masked arrays from above
        '''

        #apply the mask to the input values
        Y_pred_mask = np.ma.array(Y_pred, mask = mask_array )
        Y_true_mask = np.ma.array(Y_true, mask = mask_array )

        meanchisquare = np.mean(np.square(np.subtract(Y_pred_mask,Y_true_mask)))

        return loss_function




