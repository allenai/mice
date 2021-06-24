# Minimal Contrastive Editing (MiCE) 🐭

This repository contains code for our paper, [Explaining NLP Models via Minimal Contrastive Editing (MiCE)](https://arxiv.org/pdf/2012.13985.pdf).

## Citation
```bibtex
@inproceedings{Ross2020ExplainingNM,
    title = "Explaining NLP Models via Minimal Contrastive Editing (MiCE)",
    author = "Ross, Alexis  and Marasovi{\'c}, Ana  and Peters, Matthew E.",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2021",
    publisher = "Association for Computational Linguistics",
    url= "https://arxiv.org/abs/2012.13985",
}
```
## Installation

1.  Clone the repository.
    ```bash
    git clone https://github.com/allenai/mice.git
    cd mice
    ```

2.  [Download and install Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html).

3.  Create a Conda environment.

    ```bash
    conda create -n mice python=3.7
    ```
 
4.  Activate the environment.

    ```bash
    conda activate mice
    ```
    
5.  Download the requirements.

    ```bash
    pip3 install -r requirements.txt
    ```

## Quick Start

1. **Download Task Data**: If you want to work with the RACE dataset, download it here: [Link](https://www.cs.cmu.edu/~glai1/data/race/). 
The commands below assume that this data, after downloaded, is stored in `data/RACE/`. 
All other task-specific datasets are automatically downloaded by the commands below.
2. **Download Pretrained Models**: You can download pretrained models by running:

    ```bash
    bash download_models.sh
    ```

      For each task (IMDB/Newsgroups/RACE), this script saves the:
      
      - Predictor model to: `trained_predictors/{TASK}/model/model.tar.gz`.
      - Editor checkpoint to: `results/{TASK}/editors/mice/{TASK}_editor.pth`.

4. **Generate Edits**: Run the following command to generate edits for a particular task with our pretrained editor. It will write edits to `results/{TASK}/edits/{STAGE2EXP}/edits.csv`.

        python run_stage_two.py -task {TASK} -stage2_exp {STAGE2EXP} -editor_path results/{TASK}/editors/mice/{TASK}_editor.pth
        
      
      For instance, to generate edits for the IMDB task, the following command will save edits to `results/imdb/edits/mice_binary/edits.csv`:
      
      ```bash
      python run_stage_two.py -task imdb -stage2_exp mice_binary -editor_path results/imdb/editors/mice/imdb_editor.pth
      ```
      
      
4. **Inspect Edits**: Inspect these edits with the demo notebook `notebooks/evaluation.ipynb`.

## More Information

`run_all.sh` contains commands for recreating the main experiments in our paper.

### Training Predictors

We use AllenNLP to train our Predictor models. Code for training Predictors can be found in `src/predictors/`. 
See `run_all.sh` for commands used to train Predictors, which will save models to subfolders in `trained_predictors`.

Alternatively, you can work with our pretrained models, which you can download with `download_models.sh`.


### Training Editors
The following command will train an editor (i.e. run Stage 1 of MiCE) for a particular task. It saves checkpoints to `results/{TASK}/editors/{STAGE1EXP}/checkpoints/`.

    python run_stage_one.py -task {TASK} -stage1_exp {STAGE1EXP}


### Generating Edits
The following command will find MiCE edits (i.e. run Stage 2 of MiCE) for a particular task. It saves edits to `results/{TASK}/edits/{STAGE2EXP}/edits.csv`. `-editor_path` determines the Editor model to use. Defaults to our pretrained Editor.

    python run_stage_two.py -task {TASK} -stage2_exp {STAGE2EXP} -editor_path results/{TASK}/editors/mice/{TASK}_editor.pth


### Inspecting Edits
  The notebook `notebooks/evaluation.ipynb` contains some code to inspect edits.
  To compute fluency of edits, see the `EditEvaluator` class in `src/edit_finder.py`.

## Adding a Task
Follow the steps below to extend this repo for your own task.

1.  Create a subfolder within `src/predictors/{TASK}`

2.  **Dataset reader**: Create a task specific dataset reader in a file `{TASK}_dataset_reader.py` within that subfolder. It should have methods: `text_to_instance()`, `_read()`, and `get_inputs()`.

3.  **Train Predictor**: Create a training config (see `src/predictors/imdb/imdb_roberta.json` for an example). Then train the Predictor using AllenNLP (see above commands or commands in `run_all.sh` for examples).

4.  **Train Editor Model**: Depending on the task, you may have to create a new `StageOneDataset` subclass (see `RaceStageOneDataset` in `src/dataset.py` for an example of how to inherit from `StageOneDataset`). 
    - For classification tasks, the existing base `StageOneDataset` class should work.
    - For new multiple-choice QA tasks with dataset readers patterned after the `RaceDatasetReader` (`src/predictors/race/race_dataset_reader.py`), the existing `RaceStageOneDataset` class should work.

5.  **Generate Edits**: Depending on the task, you may have to create a new `Editor` subclass (see `RaceEditor` in `src/editor.py` for an example of how to inherit from `Editor`). 
    - For classification tasks, the existing base `Editor` class should work. 
    - For multiple-choice QA with dataset readers patterned after `RaceDatasetReader`, the existing `RaceEditor` class should work. 

