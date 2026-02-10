# CQD-SHAP

This repository contains the code to reproduce the results from the paper "[CQD-SHAP: Explainable Complex Query Answering with Shapley Values](https://arxiv.org/abs/2510.15623)".

For reproducing the exact results of the current arXiv paper, please refer to the [Release v1.0](https://github.com/ds-jrg/CQD-SHAP/releases/tag/v1.0). The code in the main branch contains the latest version of our work, which includes some improvements and additional datasets.

**Google Colab Notebook:**

You can test CQD-SHAP directly in the Google Colab environment if you want to quickly see how it works. Colab environment has already been set up with all the necessary packages we used in our experiments and its GPU is sufficient for our evaluation.

## Prerequisites

### Environment Setup

We recommend using a conda environment with python `3.10`. You can use the following commands to set up the environment:

```bash
conda create -n xcqa python=3.10
```

To activate the environment, use:

```bash
conda activate xcqa
```

The list of required packages is provided in the `requirements.txt` file. You can install them using pip:

```bash
pip install -r requirements.txt
```

### Data Preparation

You can download all of the benchmark datasets we used in our experiments using the following commands. The datasets will be stored in a `data` directory. The original *FB15k-237* and *NELL995* datasets are based on the [CQD](https://github.com/uclnlp/cqd/) repository, with some slight modifications to have a unified format for data loading. The two other datasets, *FB15k-237+H* and *FB15k-237-betae*, are based on the resource from *"Is Complex Query Answering Really Complex?"* paper, which can be found at [here](https://github.com/april-tools/is-cqa-complex/). Furthermore, we enriched Freebase datasets with titles of entities based on [KNN-KG repository](https://github.com/zjunlp/KNN-KG/tree/main/dataset/FB15k-237).

```bash
wget https://groups.uni-paderborn.de/fg-ds-jrg/projects/cqd-shap/datasets/data_v2.zip
```

```bash
unzip data_v2.zip
```

### Pre-trained Models

We also use pre-trained models from [CQD](https://github.com/uclnlp/cqd/). We've provided a new file that contains only the necessary models to reduce the download size. You can download the models using the following command:

```bash
wget https://groups.uni-paderborn.de/fg-ds-jrg/projects/cqd-shap/models/models.zip
```

After downloading, run the following command to extract the models (this will create a `models` directory):

```bash
unzip models.zip
```

**Note:** We use the following pre-trained models for our experiments:
- FB15k-237: `models/FB15k-model-rank-1000-epoch-100-1602520745.pt`
- NELL995: `models/NELL-model-rank-1000-epoch-100-1602499096.pt`

## Necessary and Sufficient Explanations

The result for necessary and sufficient explanations evaluation can be reproduced by the `evaluation.py` script. The script takes the following arguments:

| Argument | Description | Value |
|----------|-------------|-------|
| `--kg` | The knowledge graph to use | `Freebase` (default) or `NELL` |
| `--benchmark` | The benchmark dataset to use | `1` for original, `2` for `+H` version (default) |
| `--query_type` | The type of query to evaluate(all if not specified) | `2p`, `3p`, `2i`, `3i`, `2u`, `pi` (i.e., 1p2i), `ip` (i.e., 2i1p), `up` (i.e., 2u1p) |
| `--method` | The method to use for generating explanations | `shapley` (default), `score`, `random`, `last`, `first` |
| `--k` | Value of k for top-k beam search | Default is `10` |
| `--t-norm` | The t-norm to use for evaluation | `prod` (default), `min`, `max` |
| `--t-conorm` | The t-conorm to use for evaluation | `prod` (default), `max`, `min` |
| `--split` | The data split to use for evaluation | `test` (default), `valid` |
| `--output_path` | The path to save the evaluation results | Default is `evaluation` |
| `--log_file` | The path to save the log file | Default is `{output_path}/bench_{benchmark}_{query_type}_{method}.log` |
| `--data_dir` | The directory where the data is stored (not required if using default KGs) | e.g. `data/FB15k-237`, `data/NELL`, `data/FB15k-237+H`, `data/NELL+H` |
| `--model_path` | The path to the pre-trained model (not required if using default KGs) | e.g. `models/FB15k-model-rank-1000-epoch-100-1602520745.pt` or `models/NELL-model-rank-1000-epoch-100-1602499096.pt` |
| `--normalize` | Whether to normalize CQD scores | | `True` (default) or `False` |

To produce the CQD-SHAP rows in Table 2 of the paper, you can run the following commands for each evaluation scenario and dataset combination:

For example, to compute the results for all query types in the Freebase dataset with the `+H` version using the Shapley method, you can run:

```bash
python evaluation.py --kg Freebase --benchmark 2 --method shapley
```

## Citing This Work

```bibtex
@misc{abbasi2025cqdshapexplainablecomplexquery,
      title={CQD-SHAP: Explainable Complex Query Answering via Shapley Values}, 
      author={Parsa Abbasi and Stefan Heindorf},
      year={2025},
      eprint={2510.15623},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.15623}, 
}
```
