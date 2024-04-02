#!/bin/bash

sbatch -o new_full_emb_training.out -e new_full_emb_training.out test_new_training --params_path configs_cineca/new_training/full_embedding/training.yaml 
sbatch -o new_full_emb_training_expand.out -e new_full_emb_training_expand.out test_new_training --params_path configs_cineca/new_training/full_embedding/training_expand.yaml 