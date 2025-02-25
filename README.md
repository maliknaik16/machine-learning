# Machine Learning Journey

This repository contains the notebooks and scripts I've developed throughout my exploration of Machine Learning concepts and frameworks. It serves as a personal log of my learning experiences, revisiting foundational topics, and delving into new areas within the field.

[badge_url]: https://img.shields.io/badge/Open_Notebook-gray?style=flat&logo=jupyter&logoColor=red&logoSize=auto

## Getting Started
To get started with running these notebooks. Follow these steps:

Step 1: Clone the repo

```bash
git clone https://github.com/maliknaik16/machine-learning.git
```

Step 2: Activate the virtual environment
```bash
poetry shell
```

Step 3: Install all the dependencies
```bash
poetry install
```

Step 4: Add the environment to notebook

```bash
poetry run python -m ipykernel install --user --name=ml-notebooks
```

Step 5: Run jupyter notebook and use the `ml-notebooks` kernel

```bash
poetry run jupyter notebook
```


## Notebooks 

**Glossary/Notes**: [![Open in Notebook](https://img.shields.io/badge/Open_Notebook-gray?style=flat&logo=jupyter&logoColor=red&logoSize=auto)](/Notes.ipynb)

### Basics

| **Title**                                                                                                       | **Notebook**                                                                                                                                                                             | **Date**       |
|-----------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|----------------|
| [PyTorch Tensors](/foundations/01_Basics/pytorch_tensors.ipynb)                                                 | [![Open in Notebook](https://img.shields.io/badge/Open_Notebook-gray?style=flat&logo=jupyter&logoColor=red&logoSize=auto)](/foundations/01_Basics/pytorch_tensors.ipynb)                 | 02/19/2025     |
| [PyTorch - Datasets & DataLoaders](/foundations/01_Basics/pytorch_dataloaders.ipynb)                  | [![Open in Notebook](https://img.shields.io/badge/Open_Notebook-gray?style=flat&logo=jupyter&logoColor=red&logoSize=auto)](/foundations/01_Basics/pytorch_dataloaders.ipynb)             | 02/20/2025     |
| [PyTorch - Neural Network](/foundations/01_Basics/pytorch_neural_network.ipynb)                       | [![Open in Notebook](https://img.shields.io/badge/Open_Notebook-gray?style=flat&logo=jupyter&logoColor=red&logoSize=auto)](/foundations/01_Basics/pytorch_neural_network.ipynb)          | 02/20/2025     |
| [PyTorch - Automatic Differentiation](/foundations/01_Basics/pytorch_automatic_differentiation.ipynb) | [![Open in Notebook](https://img.shields.io/badge/Open_Notebook-gray?style=flat&logo=jupyter&logoColor=red&logoSize=auto)](/foundations/01_Basics/pytorch_automatic_differentiation.ipynb) | 02/21/2025     |
| [PyTorch - Model Optimization](/foundations/01_Basics/pytorch_model_optimization.ipynb)               | [![Open in Notebook](https://img.shields.io/badge/Open_Notebook-gray?style=flat&logo=jupyter&logoColor=red&logoSize=auto)](/foundations/01_Basics/pytorch_model_optimization.ipynb) | 02/22/2025     |


### Intermediate

| **Title**                                                                                                                            | **Notebook**                                                                                                                                                                             | **Date**   |
|--------------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|------------|
| [Convolutional Neural Networks (CNNs)](/foundations/02_Intermediate/pytorch_cnn.ipynb)                                               | [![Open in Notebook](https://img.shields.io/badge/Open_Notebook-gray?style=flat&logo=jupyter&logoColor=red&logoSize=auto)](/foundations/02_Intermediate/pytorch_cnn.ipynb)                 | 02/22/2025 |
| [Transformers - Attention Is All You Need from Scratch](/foundations/02_Intermediate/attention_is_all_you_need.ipynb) | [![Open in Notebook](https://img.shields.io/badge/Open_Notebook-gray?style=flat&logo=jupyter&logoColor=red&logoSize=auto)](/foundations/02_Intermediate/attention_is_all_you_need.ipynb)                 | 02/24/2025 |

