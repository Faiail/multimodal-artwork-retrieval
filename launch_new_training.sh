#!/bin/bash

sbatch -o new_base_training.out -e new_base_training.out test_new_training --params_path configs_cineca/new_training/training.yaml 
sbatch -o new_base_training_expand.out -e new_base_training_expand.out test_new_training --params_path configs_cineca/new_training/training_expand.yaml 