from torchvision import datasets, transforms
from trainer import KMNISTTrainer
from configs import Config
import numpy as np
import torch
import torch.utils.data
import pandas as pd
import os
import argparse

def run_experiment( optimizer_name, batch_size, experiment_name ):

    # Check if the experiment name already exists
    if not os.path.exists( f'./results/{experiment_name}' ):
        # Create the results directory if it doesn't exist
        os.makedirs( f'./results/{experiment_name}', exist_ok = True )

    #--------------------------------
    # Download and transform the KMNIST dataset
    #--------------------------------
    print( 'Downloading and transforming the KMNIST dataset' )
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    # Download or load the KMNIST dataset
    train_data = datasets.KMNIST(
        root = './data',
        train = True,
        download = True,
        transform = transform
    )

    test_data = datasets.KMNIST(
        root = './data',
        train = False,
        download = True,
        transform = transform
    )

    # Split the test_data data into test_data and validation sets
    test_data, val_data = torch.utils.data.random_split( test_data, [ 5000, 5000 ] )

    # Create config
    print( f'Creating config for {optimizer_name} with batch size {batch_size}' )
    cfg = Config( optimizer_name = optimizer_name, batch_size = batch_size )

    # Create trainer
    print( f'Creating trainer for {optimizer_name} with batch size {batch_size}' )
    trainer = KMNISTTrainer( cfg )

    #--------------------------------
    # Hyperparameter tuning
    #--------------------------------
    print( '\n\n\n\n--------------------------------' )
    print( f'Hyperparameter tuning for {optimizer_name} with batch size {batch_size}' )
    print( '--------------------------------' )
    study = trainer.hyperparameter_tuning( train_data, val_data )
    print( "Best hyperparameters:", study.best_params )
    df_study = study.trials_dataframe()
    df_study.to_csv( f'./results/{experiment_name}/{cfg.optimizer}_study.csv', index = False )

    #--------------------------------
    # Cross-validation
    #--------------------------------
    print( '\n\n\n\n--------------------------------' )
    print( f'Cross-validation for {optimizer_name} with batch size {batch_size}' )
    print( '--------------------------------' )
    cv_results = trainer.cross_validate( train_data, n_splits = cfg.cv_splits )
    mean_val_loss = np.mean( cv_results['val_loss'] )
    mean_val_acc = np.mean( cv_results['val_accuracy'] )
    print( f"Mean CV Loss: {mean_val_loss:.4f}, Mean CV Accuracy: {mean_val_acc:.2f}" )

    #--------------------------------
    # Train the model with the best hyperparameters
    #--------------------------------
    print( '\n\n\n\n--------------------------------' )
    print( f'Training the model with the best hyperparameters for {optimizer_name} with batch size {batch_size}' )
    print( '--------------------------------' )
    history = trainer.train( train_data, val_data, epochs = cfg.epochs )

    #--------------------------------
    # Testing the model
    #--------------------------------
    print( '\n\n\n\n--------------------------------' )
    print( f'Testing the model for {optimizer_name} with batch size {batch_size}' )
    print( '--------------------------------' )
    test_results = trainer.test( test_data )
    print( f"Test Loss: {test_results['test_loss']:.4f}, Test Accuracy: {test_results['accuracy']:.2f}, Test Precision: {test_results['precision']:.4f}" )

    #--------------------------------
    # Save epoch_logs to a status.csv file
    #--------------------------------
    print( '\n\n\n\n--------------------------------' )
    print( f'Saving epoch_logs to a status.csv file for {optimizer_name} with batch size {batch_size}' )
    print( '--------------------------------' )

    # Check if files exist and load them, otherwise initialize DataFrames
    if os.path.exists( f'./results/{experiment_name}/status.csv' ):
        df_status = pd.read_csv( f'./results/{experiment_name}/status.csv' )
        df_metrics = pd.read_csv( f'./results/{experiment_name}/metrics.csv' )
        df_cv_results = pd.read_csv( f'./results/{experiment_name}/cv_results.csv' )
    else:
        df_status = pd.DataFrame( columns = [ 'optimizer', 'epoch', 'train_loss', 'val_loss', 'val_accuracy', 'val_precision', 'epoch_time', 'batch_size' ] )
        df_metrics = pd.DataFrame( columns = [ 'optimizer', 'test_loss', 'accuracy', 'precision', 'batch_size' ] )
        df_cv_results = pd.DataFrame( columns = [ 'optimizer', 'train_loss', 'val_loss', 'val_accuracy', 'val_precisions', 'batch_size' ] )

    # Append new epoch logs
    new_epoch_log_df = pd.DataFrame.from_dict( history['epoch_logs'] )
    new_epoch_log_df['optimizer'] = optimizer_name
    new_epoch_log_df['batch_size'] = batch_size

    df_status = pd.concat( [ df_status, new_epoch_log_df ], ignore_index = True )

    # Append new CV results
    new_cv_results_df = pd.DataFrame.from_dict( cv_results )
    new_cv_results_df['optimizer'] = optimizer_name
    new_cv_results_df['batch_size'] = batch_size

    df_cv_results = pd.concat( [ df_cv_results, new_cv_results_df ], ignore_index = True )

    # Append new test results (test_results is a single dictionary)
    test_results['optimizer'] = optimizer_name
    test_results['optimizer_config'] = str( cfg.get_optimizer_config() )
    test_results['batch_size'] = batch_size

    df_metrics = pd.concat( [ df_metrics, pd.DataFrame( [ test_results ] ) ], ignore_index = True )

    # Save the updated CSV files
    df_status.to_csv( f'./results/{experiment_name}/status.csv', index = False )
    df_metrics.to_csv( f'./results/{experiment_name}/metrics.csv', index = False )
    df_cv_results.to_csv( f'./results/{experiment_name}/cv_results.csv', index = False )

    print( '\n\n\n\n--------------------------------' )
    print( f'Experiment {experiment_name} completed successfully' )
    print( '--------------------------------' )

def main():

    parser = argparse.ArgumentParser( description = 'Run KMNIST experiments' )
    parser.add_argument( '--optimizer', type = str, default = 'adam', help = 'Optimizer to use' )
    parser.add_argument( '--batch_size', type = int, default = 128, help = 'Batch size' )
    parser.add_argument( '--experiment_name', type = str, default = 'default', help = 'Experiment name' )
    args = parser.parse_args()

    run_experiment( args.optimizer, args.batch_size, args.experiment_name )

if __name__ == '__main__':
    main()


