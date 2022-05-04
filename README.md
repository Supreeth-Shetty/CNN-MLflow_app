# MLflow-project-template
MLflow project template

## STEPS -

### STEP 01- Create a repository by using template repository

### STEP 02- Clone the new repository

### STEP 03- Create a conda environment after opening the repository in VSCODE

```bash
conda create --prefix ./env python=3.7 -y
```

```bash
conda activate ./env
```
OR
```bash
source activate ./env
```

### STEP 04- install the requirements
```bash
pip install -r requirements.txt
```

### STEP 05 - Create conda.yaml file -
```bash
conda env export > conda.yaml
```

### STEP 06- commit and push the changes to the remote repository

### STEP 07 - To run the project -
```bash
init_setup.sh #for initial env and requirement setup
mlflow run . --no-conda #for not creating conda env and running in same env
mlflow run . #for creating conda env
```

### STEP 07 - To run the project by passing custom stages and config file-
```bash
mlflow run . -e stage_name -P cofigs=customs config file path --no-conda