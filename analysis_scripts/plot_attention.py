import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import os
import random
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tokenizers import Tokenizer
from transformers import LEDConfig, LEDForConditionalGeneration, logging
from torch.utils.data import DataLoader

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from panBART import setup, cleanup, GenomeDataset

logging.set_verbosity_error()


def parse_args():
    parser = argparse.ArgumentParser(description="Plot attention matrices when passing a genome through panBART.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint file.")
    parser.add_argument("--tokenizer_path", type=str, required=True, help="Path to the tokenizer file.")
    parser.add_argument("--prompt_file", type=str, required=True, help="Tab separated file of genome ids and genomes.")
    parser.add_argument("--genome_id", type=str, default=None, help="Genome id to plot. Default = first genome in file.")
    parser.add_argument("--attention_type", type=str, default="encoder", choices=["encoder", "decoder", "cross"],
                        help="Which attention to plot: encoder self-attention, decoder self-attention or decoder-to-encoder cross-attention. Default = encoder")
    parser.add_argument("--layer", type=int, default=-1, help="Layer to plot (negative indexes from last). Default = -1")
    parser.add_argument("--head", type=int, default=None, help="Head to plot. Default = mean over heads.")
    parser.add_argument("--start", type=int, default=0, help="First token position to plot. Default = 0")
    parser.add_argument("--end", type=int, default=None, help="Last token position (exclusive) to plot. Default = end of sequence.")
    parser.add_argument("--log_scale", action="store_true", default=False, help="Plot attention on a log colour scale.")
    parser.add_argument("--label_tokens", action="store_true", default=False, help="Label axes with tokens (best for short windows).")
    parser.add_argument("--save_matrix", action="store_true", default=False, help="Also save the attention matrix as .npy")
    parser.add_argument("--outpref", type=str, default="attention", help="Output prefix. Default = 'attention'")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI. Default = 300")
    parser.add_argument("--embed_dim", type=int, default=256)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--num_layers", type=int, default=8)
    parser.add_argument("--max_seq_length", type=int, default=16384)
    parser.add_argument("--model_dropout_rate", type=float, default=0.2)
    parser.add_argument("--attention_window", type=int, default=512)
    parser.add_argument("--global_contig_breaks", action="store_true", default=False, help="Attend globally to contig breaks.")
    parser.add_argument("--device", type=str, default=None, help="Device (e.g. 'cpu', 'cuda', 'mps'). Default = auto.")
    parser.add_argument("--DDP", action="store_true", default=False, help="Load/wrap the model with DistributedDataParallel (single process), as in compute_SHAP.py.")
    parser.add_argument("--port", type=str, default="12356", help="Port for DDP. Default = 12356")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    args.max_seq_length = max(args.max_seq_length, args.attention_window)
    args.max_seq_length = (args.max_seq_length // args.attention_window) * args.attention_window
    return args


def read_prompt_file(path, genome_id=None):
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            gid, genome = line.rstrip("\n").split("\t")[:2]
            if genome_id is None or gid == genome_id:
                return gid, genome
    raise ValueError(f"Genome id {genome_id} not found in {path}")


def longformer_to_dense(local_attn, global_attn, global_idx, window):
    """
    Convert LED sliding-window encoder attention to a dense (heads, seq, seq) matrix.

    local_attn: (heads, seq, n_global + window + 1); first n_global columns are
        attention to global tokens, the rest are the local window centred on each query.
    global_attn: (heads, n_global, seq) attention from global tokens to all tokens, or None.
    global_idx: positions of global tokens.
    """
    heads, seq, _ = local_attn.shape
    n_global = len(global_idx)
    one_sided = window // 2
    dense = np.zeros((heads, seq, seq), dtype=np.float32)

    local = np.nan_to_num(local_attn[:, :, n_global:])
    for j in range(2 * one_sided + 1):
        offset = j - one_sided
        q_start, q_end = max(0, -offset), min(seq, seq - offset)
        if q_start >= q_end:
            continue
        q = np.arange(q_start, q_end)
        dense[:, q, q + offset] += local[:, q, j]

    if n_global > 0:
        glob_cols = np.nan_to_num(local_attn[:, :, :n_global])
        dense[:, :, global_idx] = glob_cols
        # global tokens attend to every token
        if global_attn is not None:
            dense[:, global_idx, :] = np.nan_to_num(global_attn)
    return dense


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.device is not None:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    DDP_active = args.DDP

    tokenizer = Tokenizer.from_file(args.tokenizer_path)
    config = LEDConfig(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=args.embed_dim,
        encoder_layers=args.num_layers,
        decoder_layers=args.num_layers,
        encoder_attention_heads=args.num_heads,
        decoder_attention_heads=args.num_heads,
        decoder_ffn_dim=4 * args.embed_dim,
        encoder_ffn_dim=4 * args.embed_dim,
        max_encoder_position_embeddings=args.max_seq_length,
        max_decoder_position_embeddings=args.max_seq_length,
        dropout=args.model_dropout_rate,
        attention_window=args.attention_window,
    )

    # DDP runs as a single process on a single device (rank 0, world_size 1)
    rank = 0
    world_size = 1
    if DDP_active:
        if not torch.cuda.is_available():
            raise RuntimeError("--DDP requires a CUDA GPU (nccl backend).")
        if device.type != "cuda":
            device = torch.device("cuda")
        if device.index is None:
            device = torch.device("cuda", 0)
        print("Using DDP on {}".format(device))
        setup(rank, world_size, args.port)

    model = LEDForConditionalGeneration(config)
    model = model.to(device)
    if DDP_active:
        model = DDP(model, device_ids=[device.index], find_unused_parameters=True)
        map_location = {"cuda:0": "cuda:%d" % device.index}
        dist.barrier()
    else:
        map_location = "cpu"

    checkpoint = torch.load(args.model_path, map_location=map_location)
    state_dict = checkpoint["model_state_dict"]
    if not DDP_active:
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    elif not any(k.startswith("module.") for k in state_dict):
        state_dict = {"module." + k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    model.eval()
    # DDP's forward clones every output tensor (all layers' attentions), causing CUDA OOM;
    # run the forward pass through the wrapped module instead.
    forward_model = model.module if DDP_active else model

    genome_id, genome = read_prompt_file(args.prompt_file, args.genome_id)

    dataset = GenomeDataset([genome], tokenizer, args.max_seq_length, 0, args.global_contig_breaks, False)
    dataset.attention_window = args.attention_window
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    decoder_input, encoder_input, labels, decoder_mask, encoder_mask, global_mask = next(iter(loader))

    decoder_input, encoder_input = decoder_input.to(device), encoder_input.to(device)
    decoder_mask, encoder_mask, global_mask = decoder_mask.to(device), encoder_mask.to(device), global_mask.to(device)

    with torch.no_grad():
        output = forward_model(
            input_ids=encoder_input,
            attention_mask=encoder_mask,
            decoder_input_ids=decoder_input,
            decoder_attention_mask=decoder_mask,
            global_attention_mask=global_mask,
            output_attentions=True,
            use_cache=False,
        )

    layer = args.layer
    if args.attention_type == "encoder":
        local = output.encoder_attentions[layer][0].float().cpu().numpy()
        glob_t = output.encoder_global_attentions[layer] if output.encoder_global_attentions else None
        glob = glob_t[0].float().cpu().numpy() if glob_t is not None else None
        global_idx = torch.nonzero(global_mask[0]).flatten().cpu().numpy()
        # LED pads the sequence to a multiple of attention_window; crop back
        seq_len = int(encoder_mask[0].sum().item())
        matrix = longformer_to_dense(local, glob, global_idx, args.attention_window)
        matrix = matrix[:, :seq_len, :seq_len]
        row_ids = col_ids = encoder_input[0, :seq_len].cpu().numpy()
        ylabel, xlabel = "Query token position", "Key token position"
    elif args.attention_type == "decoder":
        matrix = output.decoder_attentions[layer][0].float().cpu().numpy()
        seq_len = int(decoder_mask[0].sum().item())
        matrix = matrix[:, :seq_len, :seq_len]
        row_ids = col_ids = decoder_input[0, :seq_len].cpu().numpy()
        ylabel, xlabel = "Decoder query token position", "Decoder key token position"
    else:
        matrix = output.cross_attentions[layer][0].float().cpu().numpy()
        dec_len = int(decoder_mask[0].sum().item())
        enc_len = int(encoder_mask[0].sum().item())
        matrix = matrix[:, :dec_len, :enc_len]
        row_ids = decoder_input[0, :dec_len].cpu().numpy()
        col_ids = encoder_input[0, :enc_len].cpu().numpy()
        ylabel, xlabel = "Decoder token position", "Encoder token position"

    if args.head is not None:
        matrix = matrix[args.head]
        head_label = f"head {args.head}"
    else:
        matrix = matrix.mean(axis=0)
        head_label = "mean of heads"

    # CSV covers all tokens (ignores --start/--end); labels are "position:token"
    layer_str = layer % args.num_layers
    head_str = f"head{args.head}" if args.head is not None else "meanheads"
    stem = f"{args.outpref}_{genome_id}_{args.attention_type}_layer{layer_str}_{head_str}"
    row_names = [f"{i}:{tokenizer.id_to_token(int(t))}" for i, t in enumerate(row_ids)]
    col_names = [f"{i}:{tokenizer.id_to_token(int(t))}" for i, t in enumerate(col_ids)]
    pd.DataFrame(matrix, index=row_names, columns=col_names).to_csv(stem + ".csv", float_format="%.6g")
    print(f"Saved {stem}.csv")

    end_r = args.end if args.end is not None else matrix.shape[0]
    end_c = args.end if args.end is not None else matrix.shape[1]
    matrix = matrix[args.start:end_r, args.start:end_c]
    row_ids, col_ids = row_ids[args.start:end_r], col_ids[args.start:end_c]

    fig, ax = plt.subplots(figsize=(10, 9))
    norm = None
    if args.log_scale:
        positive = matrix[matrix > 0]
        vmin = positive.min() if positive.size else 1e-8
        norm = matplotlib.colors.LogNorm(vmin=vmin, vmax=max(matrix.max(), vmin * 10))
    im = ax.imshow(matrix, cmap="viridis", aspect="auto", interpolation="nearest", norm=norm,
                   extent=(args.start, args.start + matrix.shape[1], args.start + matrix.shape[0], args.start))
    fig.colorbar(im, ax=ax, label="Attention weight")

    if args.label_tokens:
        tick_r = [tokenizer.id_to_token(int(i)) for i in row_ids]
        tick_c = [tokenizer.id_to_token(int(i)) for i in col_ids]
        ax.set_yticks(np.arange(len(tick_r)) + args.start + 0.5)
        ax.set_yticklabels(tick_r, fontsize=6)
        ax.set_xticks(np.arange(len(tick_c)) + args.start + 0.5)
        ax.set_xticklabels(tick_c, fontsize=6, rotation=90)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(f"{genome_id}: {args.attention_type} attention, layer {layer % args.num_layers}, {head_label}")
    fig.tight_layout()

    fig.savefig(stem + ".png", dpi=args.dpi)
    print(f"Saved {stem}.png")
    if args.save_matrix:
        np.save(stem + ".npy", matrix)
        print(f"Saved {stem}.npy")
    if args.DDP:
        cleanup()


if __name__ == "__main__":
    main()
