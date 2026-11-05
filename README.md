# PanBART

PanBART is a long-context encoder-decoder model for pangenome data, based on the [LED (Longformer Encoder-Decoder)](https://huggingface.co/docs/transformers/model_doc/led) architecture. Pre-trained models for *E. coli* and *S. pneumoniae* are available from [Zenodo](https://zenodo.org/records/19823911).

## 1. Prepare the data

Use [WTBCluster](https://github.com/samhorsfield96/WTBcluster) to generate tokenized genome data. The training scripts read one genome per line. Each sequence should contain space-separated pangenome tokens; `_` separates contigs. If a line has tab-separated columns, PanBART uses the final column as the sequence.

## 2. Set up the environment

The Conda environment files in [`environments/`](environments/) document the dependencies for standard and DeepSpeed training. They pin a Linux/CUDA-based setup. Their final `prefix:` entry is machine-specific: remove it (or change it to a valid local path) before creating the environment on another machine.

For example:

```bash
conda env create --name panBART --file environments/environment_panBART.yml
conda activate panBART
```

For DeepSpeed training, use `environments/environment_deepspeed.yml` instead and activate that environment. DeepSpeed training requires a CUDA-capable system and the `deepspeed` launcher.

Run commands below from the repository root. Replace the example paths with paths to your data and output directories.

Create the output directory before training; the scripts do not create the checkpoint's parent directory:

```bash
mkdir -p output
```

## 3. Train a model

### Standard training

Pass one input file to `panBART.py`. By default, it splits the data into 80% training, 10% validation, and 10% test sets. The tokenizer is created from the input data and saved at `--tokenizer_path` alongside the model checkpoint at `--model_save_path`.

```bash
python panBART.py \
  --input_file /path/to/tokenized_genomes.txt \
  --model_save_path output/panbart.chk \
  --tokenizer_path output/tokenizer.json \
  --attention_window 512 \
  --max_seq_length 8192 \
  --num_heads 8 \
  --num_layers 8 \
  --embed_dim 256 \
  --batch_size 1 \
  --epochs 400 \
  --prop_masked 0.15 \
  --learning_rate 0.000001 \
  --early_stop_patience 10 \
  --min_delta 0.001 \
  --model_dropout_rate 0.4 \
  --num_workers 1 \
  --log_dir output/logs
```

You can instead provide pre-split data with `--train_file`, `--val_file`, and `--test_file`. `--max_seq_length` must be a multiple of `--attention_window`; choose the sequence length and batch size to fit available GPU memory.

### DeepSpeed training

DeepSpeed expects separate training, validation, and test files. First build a tokenizer from those files with the Python interpreter (do not launch this step through `deepspeed`):

```bash
python panBART_deepspeed.py \
  --train_file /path/to/train.txt \
  --val_file /path/to/validation.txt \
  --test_file /path/to/test.txt \
  --model_save_path output/panbart_deepspeed.chk \
  --tokenizer_path output/tokenizer_deepspeed.json \
  --attention_window 512 \
  --max_seq_length 7200 \
  --num_heads 8 \
  --num_layers 8 \
  --embed_dim 256 \
  --batch_size 1 \
  --epochs 300 \
  --prop_masked 0.15 \
  --early_stop_patience 10 \
  --min_delta 0.01 \
  --model_dropout_rate 0.4 \
  --num_workers 1 \
  --log_dir output/deepspeed_logs \
  --deepspeed_config deepspeed_ZS2_fp16.json \
  --generate_tokenizer
```

Then launch training with the same model, data, and tokenizer settings, omitting `--generate_tokenizer`:

```bash
deepspeed panBART_deepspeed.py \
  --train_file /path/to/train.txt \
  --val_file /path/to/validation.txt \
  --test_file /path/to/test.txt \
  --model_save_path output/panbart_deepspeed.chk \
  --tokenizer_path output/tokenizer_deepspeed.json \
  --attention_window 512 \
  --max_seq_length 7200 \
  --num_heads 8 \
  --num_layers 8 \
  --embed_dim 256 \
  --batch_size 1 \
  --epochs 300 \
  --prop_masked 0.15 \
  --early_stop_patience 10 \
  --min_delta 0.01 \
  --model_dropout_rate 0.4 \
  --num_workers 1 \
  --log_dir output/deepspeed_logs \
  --deepspeed_config deepspeed_ZS2_fp16.json
```

## 4. Run a trained or pre-trained model

Use an analysis script with the model checkpoint and its matching tokenizer. For example, generate sequence embeddings:

```bash
python analysis_scripts/compute_sequence_embedding.py \
  --model_path /path/to/panbart.chk \
  --tokenizer_path /path/to/tokenizer.json \
  --prompt_file /path/to/genomes.txt \
  --embed_dim 256 \
  --num_heads 8 \
  --num_layers 8 \
  --max_seq_length 8192 \
  --attention_window 512 \
  --batch_size 8 \
  --pooling mean \
  --outpref output/embeddings
```

Use the model architecture settings used during training (or supplied with the pre-trained model), particularly `embed_dim`, `num_heads`, `num_layers`, `max_seq_length`, and `attention_window`. For pre-trained weights, download both the checkpoint and its corresponding tokenizer from [Zenodo](https://zenodo.org/records/19823911).

To evaluate a held-out test set, run:

```bash
python analysis_scripts/test_model_accuracy.py \
  --test_file /path/to/test_genomes.txt \
  --model_save_path /path/to/panbart.chk \
  --tokenizer_path /path/to/tokenizer.json \
  --embed_dim 256 \
  --num_heads 8 \
  --num_layers 8 \
  --max_seq_length 8192 \
  --attention_window 512 \
  --batch_size 8 \
  --num_workers 1
```

The [`analysis_scripts/README.md`](analysis_scripts/README.md) includes examples for embeddings, clustering, sequence generation, likelihoods, attention plots, and other analyses.
