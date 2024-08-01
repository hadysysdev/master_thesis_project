# Master Thesis Project Setup

This guide provides step-by-step instructions for setting up the project environment using Conda and Docker on a Linux system.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Enviroment Setup](#enviroment-setup)
3. [Configuration](#configuration)
4. [Running NoteBooks](#running-notebooks)


## Prerequisites

Before setting up TimeEval, ensure you have the following prerequisites installed:

- **Linux Operating System:** TimeEval is tested and supported only on Unix-based systems, specifically Linux.
- **Docker:** Docker is required to manage and run algorithms in isolated containers. Install Docker by following the instructions for your specific Linux distribution on the [official Docker website](https://docs.docker.com/engine/install/linux/).
- **Conda:** A Conda environment is used to manage dependencies. Install Miniconda or Anaconda from the [official Conda website](https://docs.conda.io/en/latest/miniconda.html).
- **Git:** Version control is essential for managing the project repository. Download Git from the [official Git website](https://git-scm.com/downloads).

## Enviroment Setup

 Follow these steps to set up the Project enviroment on your Linux machine:

### 1. Clone the Repository

Open a terminal and run the following command to clone the repository:

```bash
 git clone https://github.com/hadysysdev/master_thesis_project.git
 ```

```bash
cd master_thesis_project
```

```bash
conda env create -f environment.yml 
```

```bash
conda activate timeeval
```

## Configuration

Pull all the required eleven Algorithm Docker images using the docker command `docker pull`.

E.g `docker pull ghcr.io/timeeval/lof:0.3.1`

Used Images are:

- LOF: ghcr.io/timeeval/lof:0.3.1
- KNN: ghcr.io/timeeval/knn:0.3.1
- KMeans: ghcr.io/timeeval/kmeans:0.3.0
- PCC: ghcr.io/timeeval/pcc:0.3.1
- iForest: ghcr.io/timeeval/iforest:0.3.1
- IF-LOF: ghcr.io/timeeval/if_lof:0.3.0
- LSTM-AD: ghcr.io/timeeval/lstm_ad:0.3.0
- Roburst PCA: ghcr.io/timeeval/robust_pca:0.3.0
- FastMCD: ghcr.io/timeeval/fast_mcd:0.3.0
- EncDec-AD: ghcr.io/timeeval/encdec_ad:0.3.0
- DeepAnT: ghcr.io/timeeval/deepant:0.3.0

Other Unsupervise or semi-supervise algorithms can also be pulled from the [TimeEval Algorithm Reposition](https://github.com/TimeEval/TimeEval-algorithms), this would however require some configuration to evaluate them
see [Running NoteBooks](#running-notebooks) for Details.

## Running NoteBooks

The three provided Notebooks can be used to run the already preconfigured algorithms evaulation. (This assumes that the algorithm images have already being pulled see [Configuration](#configuration))

First start the jupyter notebook server (which should already be installed during the conda evn installation)

```bash
jupyter notebook
```
Open the jupyter url **http://localhost:8888/** and navigate to the cloned folder.

#### Details of the NoteBooks

##### algorithms_evaluation
 
This notebook can be used to run the evaulation as presented in the thesis paper.

Adding an additional Algorithm will require for the file to be modified specifically the following sections:

First check if the algorithm requires a post processing function ? (which is the case for most semi-supervised algorithm).
If that is the case add the required function to the notebook.

Next, add the configuration to the 
*algorithms* array in the notebook.

The configuration looks as follows:

```python
algorithms = [
        Algorithm(
        name="LOF",
        main=DockerAdapter(
            image_name="ghcr.io/timeeval/lof",
            tag="0.3.1",
            skip_pull=True
        ),
        data_as_file=True,
        training_type=TrainingType.UNSUPERVISED,
        input_dimensionality=InputDimensionality.MULTIVARIATE
    ),
        Algorithm(
        name="EncDec-AD",
        main=DockerAdapter(
            image_name="ghcr.io/timeeval/encdec_ad",
            tag="0.3.0",
            skip_pull=True
        ),
        data_as_file=True,
        training_type=TrainingType.SEMI_SUPERVISED,
        input_dimensionality=InputDimensionality.MULTIVARIATE,
        # If post processing is needed
        postprocess=post_encdec_ad
    ),]
```

##### experiment-report

This notebook can be used to visualize the report of the evaluation.

This requires the folder where the evaluation report is located (this is created authomatically inside a folder called *result* in the parent directory by TimeEval if one is not given explicitly, the folder usualy have the following pattern "2024_07_04_20_27_46" )


##### data_pre_proccing (optional)

This notebook can be used to stage the MOVE II data. 
