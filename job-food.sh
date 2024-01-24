#!/bin/bash
#SBATCH -J food
#SBATCH -p local
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --time=infinite
#SBATCH --output=.slurm_output/output_%j.log
#SBATCH --error=.slurm_error/error_%j.log

# Activate the virtual environment
# cd /home/pgajo/working/food/TASTEset
# pwd
# source /home/pgajo/working/food/food-env/bin/activate
python -c 'import time; print("Current time: ", time.asctime())'
python -c 'print("Running job...")'
# python -m spacy train config_transformer.cfg --output ./output_transformer_0 --paths.train ./nerfr_train.spacy --paths.dev ./nerfr_train.spacy -g 0 # NER training
python /home/pgajo/working/food/src/translate/translate_tasteset_json.py