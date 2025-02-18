import torch
import torch.optim as optim
import numpy as np
from sklearn.metrics import precision_score
from sklearn.model_selection import KFold
import optuna
import time
from tqdm import tqdm

from source.kmnist_model import KMNISTModel
from source.optmizers import SAM, NovoGrad, Lamb
from source.adopt import ADOPT

class KMNISTTrainer:

    def __init__(self, cfg):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print( f'Using device: {self.device}' )
        print( f'Max memory allocated: {torch.cuda.max_memory_allocated() / 1024 ** 2:.2f} MB' )
    
        self.model = KMNISTModel().to(self.device)
        self.cfg = cfg
        self.optimizer = None
        self.setup_optimizer()

    def setup_optimizer(self, params=None, trial=None):
    
        """Set up the optimizer based on self.cfg.optimizer."""
        params = params or self.model.parameters()

        opt_type = self.cfg.optimizer.lower()
        
        if opt_type == 'rmsprop':
            
            if trial:
                lr = trial.suggest_float('rmsprop_lr',
                                         self.cfg.get_optimizer_config().ranges['lr_range'][0],
                                         self.cfg.get_optimizer_config().ranges['lr_range'][1],
                                         log=True)
                alpha = trial.suggest_float('rmsprop_alpha',
                                            self.cfg.get_optimizer_config().ranges['alpha_range'][0],
                                            self.cfg.get_optimizer_config().ranges['alpha_range'][1])
                momentum = trial.suggest_float('rmsprop_momentum',
                                               self.cfg.get_optimizer_config().ranges['momentum_range'][0],
                                               self.cfg.get_optimizer_config().ranges['momentum_range'][1])
                weight_decay = trial.suggest_float('rmsprop_weight_decay',
                                                   self.cfg.get_optimizer_config().ranges['weight_decay_range'][0],
                                                   self.cfg.get_optimizer_config().ranges['weight_decay_range'][1],
                                                   log=True)
            else:
                lr = self.cfg.get_optimizer_config().learning_rate
                alpha = self.cfg.get_optimizer_config().alpha
                momentum = self.cfg.get_optimizer_config().momentum
                weight_decay = self.cfg.get_optimizer_config().weight_decay

            self.optimizer = optim.RMSprop(
                params,
                lr=lr,
                alpha=alpha,
                eps=self.cfg.get_optimizer_config().eps,
                weight_decay=weight_decay,
                momentum=momentum
            )
        elif opt_type in ['novograd', 'lamb', 'adopt', 'adam','adamw']:
            
            if trial:

                lr = trial.suggest_float(f'{opt_type}_lr',
                                         self.cfg.get_optimizer_config().ranges['lr_range'][0],
                                         self.cfg.get_optimizer_config().ranges['lr_range'][1],
                                         log=True)

                beta1 = trial.suggest_float(f'{opt_type}_beta1',
                                              self.cfg.get_optimizer_config().ranges['beta1_range'][0],
                                              self.cfg.get_optimizer_config().ranges['beta1_range'][1])
                
                beta2 = trial.suggest_float(f'{opt_type}_beta2',
                                              self.cfg.get_optimizer_config().ranges['beta2_range'][0],
                                              self.cfg.get_optimizer_config().ranges['beta2_range'][1])
                
                weight_decay = trial.suggest_float(f'{opt_type}_weight_decay',
                                                   self.cfg.get_optimizer_config().ranges['weight_decay_range'][0],
                                                   self.cfg.get_optimizer_config().ranges['weight_decay_range'][1],
                                                   log=True)

            else:
                
                lr = self.cfg.get_optimizer_config().learning_rate
                beta1, beta2 = self.cfg.get_optimizer_config().beta1, self.cfg.get_optimizer_config().beta2
                weight_decay = self.cfg.get_optimizer_config().weight_decay
            
            if opt_type == 'novograd':
                self.optimizer = NovoGrad( params, lr = lr, betas = ( beta1, beta2 ), weight_decay = weight_decay )
            elif opt_type == 'lamb':
                self.optimizer = Lamb( params, lr = lr, betas = ( beta1, beta2 ), weight_decay = weight_decay )
            elif opt_type == 'adopt':
                self.optimizer = ADOPT( params, lr = lr, betas = ( beta1, beta2 ), weight_decay = weight_decay )
            elif opt_type == 'adam':
                self.optimizer = optim.Adam( params, lr = lr, betas = ( beta1, beta2 ), weight_decay = weight_decay )
            elif opt_type == 'adamw':
                self.optimizer = optim.AdamW( params, lr = lr, betas = ( beta1, beta2 ), weight_decay = weight_decay )
            
        elif opt_type == 'sam':

            # SAM with base SGD
            if trial:

                lr = trial.suggest_float('sam_lr',
                                         self.cfg.get_optimizer_config().ranges['lr_range'][0],
                                         self.cfg.get_optimizer_config().ranges['lr_range'][1],
                                         log=True)
                momentum = trial.suggest_float('sam_momentum',
                                               self.cfg.get_optimizer_config().ranges['momentum_range'][0],
                                               self.cfg.get_optimizer_config().ranges['momentum_range'][1])
                rho = trial.suggest_float('sam_rho',
                                          self.cfg.get_optimizer_config().ranges['rho_range'][0],
                                          self.cfg.get_optimizer_config().ranges['rho_range'][1])
                weight_decay = trial.suggest_float('sam_weight_decay',
                                                   self.cfg.get_optimizer_config().ranges['weight_decay_range'][0],
                                                   self.cfg.get_optimizer_config().ranges['weight_decay_range'][1],
                                                   log=True)
            else:
                lr = self.cfg.get_optimizer_config().learning_rate
                momentum = self.cfg.get_optimizer_config().momentum
                rho = self.cfg.get_optimizer_config().rho
                weight_decay = self.cfg.get_optimizer_config().weight_decay

            self.optimizer = SAM( params, optim.SGD, rho = rho, lr = lr, momentum = momentum, weight_decay = weight_decay )

        else:
            raise ValueError(f"Optimizer '{self.cfg.optimizer}' not supported or not implemented.")

    def objective(self, trial, train_data, val_data):

        # Reset model
        self.model = KMNISTModel().to( self.device )
        # Setup optimizer with trial parameters
        self.setup_optimizer( trial = trial )
        
        # Create data loaders
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size = self.cfg.batch_size,
            shuffle = True
        )
        val_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size = self.cfg.batch_size,
            shuffle = False
        )
        
        # Train for a few epochs (HPO)
        results = self.train( train_loader, val_loader, epochs = self.cfg.hpo_epochs, eval_first = False )
        
        return results['val_precisions'][-1]  # Return the final precision

    def hyperparameter_tuning( self, train_data, val_data ):
        
        study = optuna.create_study(
            direction = "maximize",
            sampler = optuna.samplers.TPESampler( seed = self.cfg.seed )
        )
        
        study.optimize(
            lambda trial: self.objective( trial, train_data, val_data ),
            n_trials = self.cfg.n_trials,
            timeout = self.cfg.timeout
        )
        
        # Get the best parameters
        best_params = study.best_params
        
        # Update the config with best parameters
        opt_type = self.cfg.optimizer.lower()
        
        if opt_type == 'adam':
            self.cfg.get_optimizer_config().learning_rate = best_params['adam_lr']
            self.cfg.get_optimizer_config().beta1 = best_params['adam_beta1']
            self.cfg.get_optimizer_config().beta2 = best_params['adam_beta2']
            self.cfg.get_optimizer_config().weight_decay = best_params['adam_weight_decay']

        elif opt_type == 'adamw':
            self.cfg.get_optimizer_config().learning_rate = best_params['adamw_lr']
            self.cfg.get_optimizer_config().beta1 = best_params['adamw_beta1']
            self.cfg.get_optimizer_config().beta2 = best_params['adamw_beta2']
            self.cfg.get_optimizer_config().weight_decay = best_params['adamw_weight_decay']

        elif opt_type == 'novograd':
            self.cfg.get_optimizer_config().learning_rate = best_params['novograd_lr']
            self.cfg.get_optimizer_config().beta1 = best_params['novograd_beta1']
            self.cfg.get_optimizer_config().beta2 = best_params['novograd_beta2']
            self.cfg.get_optimizer_config().weight_decay = best_params['novograd_weight_decay']

        elif opt_type == 'lamb':
            self.cfg.get_optimizer_config().learning_rate = best_params['lamb_lr']
            self.cfg.get_optimizer_config().beta1 = best_params['lamb_beta1']
            self.cfg.get_optimizer_config().beta2 = best_params['lamb_beta2']
            self.cfg.get_optimizer_config().weight_decay = best_params['lamb_weight_decay']

        elif opt_type == 'adopt':
            self.cfg.get_optimizer_config().learning_rate = best_params['adopt_lr']
            self.cfg.get_optimizer_config().beta1 = best_params['adopt_beta1']
            self.cfg.get_optimizer_config().beta2 = best_params['adopt_beta2']
            self.cfg.get_optimizer_config().weight_decay = best_params['adopt_weight_decay']

        elif opt_type == 'rmsprop':
            self.cfg.get_optimizer_config().learning_rate = best_params['rmsprop_lr']
            self.cfg.get_optimizer_config().alpha = best_params['rmsprop_alpha']
            self.cfg.get_optimizer_config().momentum = best_params['rmsprop_momentum']
            self.cfg.get_optimizer_config().weight_decay = best_params['rmsprop_weight_decay']

        elif opt_type == 'sam':
            self.cfg.get_optimizer_config().learning_rate = best_params['sam_lr']
            self.cfg.get_optimizer_config().momentum = best_params['sam_momentum']
            self.cfg.get_optimizer_config().rho = best_params['sam_rho']
            self.cfg.get_optimizer_config().weight_decay = best_params['sam_weight_decay']
            
        else:
            pass

        # Reset model and optimizer with the best parameters
        self.model = KMNISTModel().to( self.device )
        self.setup_optimizer()

        return study

    def cross_validate(self, train_data, n_splits=5):

        kf = KFold( n_splits = n_splits, shuffle = True, random_state = self.cfg.seed )
        scores = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': [],
            'val_precisions': []
        }

        for fold, (train_idx, val_idx) in enumerate(kf.split(train_data)):

            print( f'Cross-validation fold {fold + 1} of {n_splits}' )
            
            # Reset model for each fold
            self.model = KMNISTModel().to( self.device )
            self.setup_optimizer()

            # Create data loaders for this fold
            train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
            val_subsampler = torch.utils.data.SubsetRandomSampler(val_idx)
            
            train_loader = torch.utils.data.DataLoader(
                train_data, 
                batch_size=self.cfg.batch_size, 
                sampler=train_subsampler
            )
            val_loader = torch.utils.data.DataLoader(
                train_data,
                batch_size=self.cfg.batch_size,
                sampler=val_subsampler
            )

            # Train for a few epochs
            fold_results = self.train( train_loader, val_loader, epochs = self.cfg.cv_epochs, eval_first = False )
            
            scores['train_loss'].append(fold_results['train_losses'][-1])
            scores['val_loss'].append(fold_results['val_losses'][-1])
            scores['val_accuracy'].append(fold_results['val_accuracy'][-1])
            scores['val_precisions'].append(fold_results['val_precisions'][-1])

        return scores

    def train(self, train_data, val_data, epochs, eval_first=True):
        
        train_losses = []
        val_losses = []
        val_accuracies = []
        val_precisions = []
        train_times = []
        epoch_logs = []

        # Check if train_data and val_data are already DataLoader
        if isinstance( train_data, torch.utils.data.DataLoader ):
            train_loader = train_data
        else:
            train_loader = torch.utils.data.DataLoader(
                train_data,
                batch_size = self.cfg.batch_size,
                shuffle = True
            )

        # Check if val_data is already DataLoader
        if isinstance( val_data, torch.utils.data.DataLoader ):
            val_loader = val_data
        else:
            val_loader = torch.utils.data.DataLoader(
                val_data,
                batch_size = self.cfg.batch_size,
                shuffle = False
            )

        def eval():

            nonlocal val_losses, val_accuracies, val_precisions

            self.model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            all_preds = []
            all_targets = []

            with torch.no_grad():

                for data, target in val_loader:

                    data, target = data.to( self.device ), target.to( self.device )
                    output = self.model( data )
                    val_loss += self.model.criterion( output, target ).item()
                    
                    _, predicted = output.max(1)
                    total += target.size(0)
                    correct += predicted.eq( target ).sum().item()
                    
                    # For precision, collect all preds and targets
                    all_preds.extend( predicted.cpu().numpy() )
                    all_targets.extend( target.cpu().numpy() )

            val_loss /= len( val_loader )
            val_losses.append( val_loss )
            accuracy = 100.0 * correct / total
            precision = precision_score( all_targets, all_preds, average = 'macro', zero_division = 0 )
            val_accuracies.append( accuracy )
            val_precisions.append( precision )

            return val_loss, accuracy, precision

        opt_type = self.cfg.optimizer.lower()
        epoch_bar = tqdm( range( epochs ), desc = 'Epochs', leave = False )
        for epoch in epoch_bar:

            if eval_first:
                val_loss, accuracy, precision = eval()

            # Training phase
            self.model.train()
            train_loss = 0.0
            start_time = time.time()
                                    
            batch_bar = tqdm( train_loader, desc = 'Batches', leave = False )
            for data, target in batch_bar:

                data, target = data.to( self.device ), target.to( self.device )
                
                if opt_type == 'sam':
                    # SAM: two-step update
                    loss_val = self.model.forward_backward( data, target )
                    self.optimizer.first_step( zero_grad = True )
                    
                    # second step
                    _ = self.model.forward_backward( data, target )
                    self.optimizer.second_step( zero_grad = True )
                    train_loss += loss_val

                else:
                    
                    # Normal single-step
                    loss_val = self.model.forward_backward( data, target )
                    self.optimizer.step()
                    train_loss += loss_val

                batch_bar.set_postfix( train_loss = train_loss )

            epoch_time = time.time() - start_time
            train_times.append( epoch_time )
        
            # Calculate average training loss
            train_loss /= len( train_loader )
            train_losses.append( train_loss )

            if not eval_first:
                val_loss, accuracy, precision = eval()

            # Store logs for this epoch
            epoch_log = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'val_accuracy': accuracy,
                'val_precision': precision,
                'epoch_time': epoch_time
            }
            epoch_logs.append(epoch_log)

            epoch_bar.set_postfix( train_loss = train_loss, val_loss = val_loss, val_accuracy = accuracy, val_precision = precision, epoch_time = epoch_time )

        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'val_accuracy': val_accuracies,
            'val_precisions': val_precisions,
            'train_times': train_times,
            'epoch_logs': epoch_logs
        }

    def test(self, test_data):

        test_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size = self.cfg.batch_size,
            shuffle = False
        )

        self.model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():

            for data, target in test_loader:

                data, target = data.to( self.device ), target.to( self.device )
                output = self.model( data )
                test_loss += self.model.criterion( output, target ).item()
                
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq( target ).sum().item()
                
                all_preds.extend( predicted.cpu().numpy() )
                all_targets.extend( target.cpu().numpy() )

        test_loss /= len( test_loader )
        accuracy = 100.0 * correct / total
        precision = precision_score( all_targets, all_preds, average = 'macro', zero_division = 0 )

        return {
            'test_loss': test_loss,
            'accuracy': accuracy,
            'precision': precision
        }
        
