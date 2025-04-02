# pybreathe: a python package for respiratory air flow rates analysis

## Get started

First of all, we highly recommend that you install [miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install) on your machine.

To set up the virtual environment required for respiratory flow analysis, open **Anaconda prompt**, copy the full package path and move to it with `$ cd package_path`. Then, please run (only the first time):

```bash
$ conda env create -f environment.yml
$ conda activate venv_pybreathe
$ python -m ipykernel install --user --name=venv_pybreathe
```

Then, every time you want to analyze data, activate the virtual environment with:

```bash
$ conda activate venv_pybreathe
```

and run the example script (or any other) with:

```bash
$ jupyter lab example.ipynb
```

> /!\ You must either be in the same directory as your .ipynb file, or specify the absolute path.

Finally, to quit the analysis, please follow these steps:

- Right clic on your .ipynb file in the tree, then "Shut Down Kernel"
- Clic on "File", the "Shut Down"
- When you regain control in your terminal, run:

```bash
$ conda deactivate
```
