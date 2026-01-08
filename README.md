## Adaptive Immune Profiling Challenge 2025 - Rank#4 Solution

This is the official implementation of **Rank#4 solution** in the AIRR challenge. Briefly speaking, our approach uses three different models to make predictions for datasets with different characteristics. By evaluating how the three methods perform on different datasets, we selectively integrate them on specific datasets to achieve the best results.

### 1. Environment

Download the docker image from [Google Drive](https://drive.google.com/drive/folders/1AtQJw2JaDeuWSNr9h5S0QR1fEgfiIgdV?usp=sharing), then load it use:

```
docker load -i kaggle-ml-env.tar
```

After loading the docker image, you can run it to use bash and run the scripts below:

```
docker run --rm -it \
  -v /path/to/datasets:/data:ro \
  -w /workspace \
  kaggle-ml-env
```

### 2. How to run
If you want to reproduce the Kaggle results quickly, please use:

```
python3 -m submission.main --train_dir /path/to/train_dir --test_dir /path/to/test_dir
# e.g.
python3 -m submission.main --train_dir /data/train_datasets/train_datasets/train_dataset_1 --test_dir /data/test_datasets/test_datasets/test_dataset_1
python3 -m submission.main --train_dir /data/train_datasets/train_datasets/train_dataset_8 --test_dir /data/test_datasets/test_datasets/test_dataset_8_1 /data/test_datasets/test_datasets/test_dataset_8_2 /data/test_datasets/test_datasets/test_dataset_8_3
```

OR you want to use our method in new datasets, please use:

```
# use single kmer
python3 -m submission.main --train_dir /path/to/train_dir --test_dir /path/to/test_dir --predictor kmer

# use multi-kmer
python3 -m submission.main --train_dir /path/to/train_dir --test_dir /path/to/test_dir --predictor kmer

# use emerson with compairr
python3 -m submission.main --train_dir /path/to/train_dir --test_dir /path/to/test_dir --predictor emerson
```
**Emerson+CoMPAIRR** is most appropriate when you want to quantify repertoire similarity based on exact or near-exact clonotype sharing between samples (e.g. comparing TCR repertoires across patients, conditions, or time points) and your data are already annotated at the CDR3/clone level. 

**Single k-mer approach** is better when you care about local sequence motifs and want to capture similarity between clonotypes that are not identical but share short subsequences; this is useful for detecting convergent patterns or antigen-associated motifs in relatively small to medium-sized datasets. 

**Multi-k-mer methods** are most suitable for large or heterogeneous repertoires where signal may appear at different spatial scales along the sequence: by combining multiple k values, they can jointly capture fine-grained local motifs and broader sequence patterns, making them preferable for complex prediction tasks (e.g. antigen specificity or disease classification) where both short motifs and longer-range context may matter.


### 3. Generate the submissions.csv file

As shown in the command line above, we assume that the implementations will adhere to a uniform interface. Regardless of whether one has a loop around the `main` function or not, we assume the output directory will contain multiple `*_test_predictions.tsv` files, one per each training dataset, and multiple `*_important_sequences.tsv` files, one per each training dataset. We provide one utility function named `concatenate_output_files` under `utils.py` that can be used to generate the final `submissions.csv` file from these individual prediction files.

```
results_dir = "/path/to/results"
concatenate_output_files(out_dir=results_dir) # provided utility function
```