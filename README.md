## Summary

This repository is using Hyena-DNA to score terminator sequences. First, follow the steps below to setup the environment, local repo, and data structure. You will be able to fine-tune the model and see the evaluation metrics. The different checkpoints will be saved, so you can load whichever one you would like. 


### Resources
- **Hyena-DNA**: [GitHub Repository](https://github.com/instadeepai/nucleotide-transformer/tree/main)

- **Dataset Source Paper 1**: [PLOS Genetics Article](https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1005147)

- **Dataset Source Paper 2**: [Springer Genetics Article](https://link.springer.com/article/10.1186/s13059-021-02509-6)

> **Note**: Conda environment is ready for use, though Dockerfile is not production-ready :)
## Setup Instructions

To set up a quick conda environment for this project, follow these steps:

### Step 1: Clone Hyena-DNA repo and Setup the Environment


```bash
git clone --recurse-submodules https://github.com/HazyResearch/hyena-dna.git && cd hyena-dna
```

```bash
conda create -n ./envs/hyena-dna python=3.8
conda activate hyena-dna
conda install pytorch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install -r requirements.txt
```

```bash
cd hyena-dna
git submodule update --init
cd flash-attention
git submodule update --init
pip install -e . --no-build-isolation
```


### Step 2: Download my repo
---
#### This will contain all the scripts used to fine-tune and evaluate Hyena-DNA for transcription terminator analysis in Yeast, specifically Saccharomyces cerevisiae, the paper with the training data can be found at: this [link](https://journals.plos.org/plosgenetics/article?id=10.1371/journal.pgen.1005147)
---

```bash
git clone https://github.com/zbates1/hyena-dna-terminator.git
```


#### You will need to delete the native Hyena DNA script, 'huggingface.py'. Though it will be replaced with a custom script, after you move the scripts from this repo over

```bash
rm ./huggingface.py && rm ./Dockerfile
mv /hyena-dna-terminator/* ./
```


##### NOTE: If you find that you are getting errors with the transformer library, specifically the tokenizer module, then remove and re-install the transformers==4.34.0 version

## Usage:

Go through the [1], [2] and [3] notebooks. These will allow to you either load your own datasets, or use the standard datasets I used during this experimentation. Make sure that you check the graphs created in [1] by the utils.checkpoint_evals.py script, which shows the 10-Fold Cross Validation results of each epoch by creating embeddings of the training dataset for each fine-tuned epoch. For my datasets, I found somewhere around 50 epochs (58 chosen in [1]) to be optimal. See the graph below.

<img src="pics/kfold_epoch_[1ab].png" alt= "" width="600" height="400">


##### Results from 10-Fold Cross Validation on a the MPRA dataset using embeddings generated from our fine-tuned Hyena DNA model. Hyena DNA does a great job of predicting gene expression based on 3' sequences, when trained on MPRA experimental data.

<img src="pics/kfold_epoch58_[3].png" alt= "" width="600" height="400">


##### Before these more complex FM's were available, predicting gene expression with nucleotide sequences was very challenging, especially for terminator sequences. These sequences are integral to mRNA stability. The graph below illustrates that mRNA abundance correlates very well with protein production for promoter sequences, in Figure D, and poorly for varying terminator sequences (Figure C). The high variance/low correlation coefficient seen for terminators reaffirms their effect on protein production post-translation. This offers us the opportunity to create a more predictive model to understand how terminator sequences affect gene expression. Paper [LINK](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1002934#pcbi-1002934-g003)

<img src="pics/shalem_13_gene_expression.PNG" alt= "" width="800" height="400">