#!/bin/bash

sbatch -o testing_new_compelte_siamese_network.out -e testing_new_compelte_siamese_network.out slurm/proposed/test_new_complete --params_path configs_cineca/new_training/training.yaml 
sbatch -o testing_new_compelte_siamese_network_expand.out -e testing_new_compelte_siamese_network_expand.out slurm/proposed/test_new_complete --params_path configs_cineca/new_training/training_expand.yaml