# Explainable Graph Query Answering

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

### Data Preparation

We use the same data as in [CQD](https://github.com/uclnlp/cqd/). You can use the following command to download the data inside the project directory:

```bash
wget http://data.neuralnoise.com/cqd-data.tgz
```

After downloading, run the following command to extract the data (this will create a `data` directory):

```bash
tar -xvzf cqd-data.tgz
```

After downloading, run the following command to extract the data (this will create a `data` directory):

```bash
tar -xvzf cqd-data.tgz
```

### Pre-trained Models

We also use pre-trained models from [CQD](https://github.com/uclnlp/cqd/). You can download the pre-trained models using the following command:

```bash
wget http://data.neuralnoise.com/cqd-models.tgz
```

After downloading, run the following command to extract the models (this will create a `models` directory):

```bash
tar -xvzf cqd-models.tgz
```

**Note:** There are multiple checkpoints available in the `models` directory. The ones we used in our experiments are as follows:

- FB15k-237: `models/FB15k-model-rank-1000-epoch-100-1602520745.pt`
- NELL995: `models/NELL-model-rank-1000-epoch-100-1602499096.pt`

## Necessary and Sufficient Explanations

The result for necessary and sufficient explanations evaluation can be reproduced by the `evaluation.py` script. The script takes the following arguments:

| Argument | Description | Value |
|----------|-------------|-------|
| `query_type` | The type of query to evaluate | `2p`, `3p`, `2i`, `3i`, `2u`, `pi` (i.e., 1p2i), `ip` (i.e., 2i1p), `up` (i.e., 2u1p) |
| `--data_dir` | The directory where the data is stored | e.g. `data/FB15k-237` (default) or `data/NELL995` |
| `--model_path` | The path to the pre-trained model | e.g. `models/FB15k-model-rank-1000-epoch-100-1602520745.pt` (default) or `models/NELL-model-rank-1000-epoch-100-1602499096.pt` |
| `--k` | Value of k for top-k beam search | Default is `10` |
| `--t-norm` | The t-norm to use for evaluation | `prod` (default), `min`, `max` |
| `--t-conorm` | The t-conorm to use for evaluation | `prob` (default), `max`, `min` |
| `--split` | The data split to use for evaluation | `test` (default), `valid` |
| `--method` | The method to use for generating explanations | `shapley` (default), `score`, `random`, `last`, `first` |
| `--explanation` | The type of explanation to evaluate | `necessary` (default), `sufficient` |
| `--output_path` | The path to save the evaluation results | Default is `output.json` |

An example command to run the evaluation for necessary explanations on 2p queries using the NELL dataset is as follows:

```bash
python evaluation.py 2p --k 10 --method shapley --explanation necessary --output_path eval/nell/necessary_2p_shapley.json --data_dir data/NELL995 --model_path models/NELL-model-rank-1000-epoch-100-1602499096.pt
```