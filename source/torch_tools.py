import os
import torch
import numpy as np
import torch.nn as nn

class Module(nn.Module):

    def __init__(self):

        super(Module, self).__init__()

        self.t_step = 0

    def save_training(self, folder):
        
        if not os.path.isdir( folder ):
            os.mkdir( folder )
        
        checkpoint = {
            'epoch': self.t_step,
            'model_state': self.state_dict()
        }

        files = os.listdir( folder )
        chpts = [ int( f.split('_')[1].replace('.pth', '') ) for f in files if f.startswith('checkpoint') ]
        if len( chpts ) > 50:
            if chpts[0] > self.t_step:
                for f in chpts:
                    file_path = os.path.join( folder, f'checkpoint_{f}.pth' )
                    os.remove( file_path )
                    print("\n\n=======================================================================================\n")
                    print("Deleting : {}, by maximum number of checkpoints".format( f ) )
                    print("\n=======================================================================================\n\n")
            else:
                chpts.sort()
                for f in chpts[:len(chpts) - 5]:
                    file_path = os.path.join( folder, f'checkpoint_{f}.pth' )
                    os.remove( file_path )
                    print("\n\n=======================================================================================\n")
                    print("Deleting : {}, by maximum number of checkpoints".format( f ) )
                    print("\n=======================================================================================\n\n")
        
        torch.save( checkpoint, folder + f'/checkpoint_{self.t_step}.pth' )
        print("\n\n=======================================================================================\n")
        print("Saved to: {} - {}".format(folder, self.t_step))
        print("\n=======================================================================================\n\n")

    def load_training(self, folder, chpt=None, eval=True):

        def get_max_checkpoint():
            files = os.listdir( folder )
            chpt = np.max( [ int( f.split('_')[1].replace('.pth', '') ) for f in files if f.startswith('checkpoint') ] )
            return chpt
        
        if not os.path.exists( folder ):

            print("\n\n=======================================================================================\n")
            print("Model not found, initializing randomly!")
            print("\n=======================================================================================\n\n")

        else:

            if not chpt is None:
                if os.path.isfile( folder + f'/checkpoint_{chpt}.pth' ): chpt = chpt
                else: chpt = get_max_checkpoint()
            else: chpt = get_max_checkpoint()
            
            print( folder + f'/checkpoint_{chpt}.pth' )

            cpt = torch.load( folder + f'/checkpoint_{chpt}.pth' )

            model_dict = self.state_dict()
            pretrained_dict = cpt['model_state']

            # Filter out matched parameters
            pretrained_dict_filtered = {k: v for k, v in pretrained_dict.items() if k in model_dict}

            # Identify unmatched parameters
            unmatched_pretrained = {k for k in pretrained_dict if k not in model_dict}
            unmatched_current = {k for k in model_dict if k not in pretrained_dict}

            # Print unmatched parameters
            print("\nUnmatched parameters in the pretrained model:")
            for param in unmatched_pretrained:
                print(param)

            print("\nUnmatched parameters in the current model:")
            for param in unmatched_current:
                print(param)

            print("\n\n=======================================================================================\n")
            model_dict.update( pretrained_dict_filtered )
            print( self.load_state_dict( model_dict ) )
            
            self.t_step = cpt['epoch']
            
            print("\n\n=======================================================================================\n")
            print("Loaded from: {} - {}".format(folder, self.t_step))
            print("\n=======================================================================================\n\n")
