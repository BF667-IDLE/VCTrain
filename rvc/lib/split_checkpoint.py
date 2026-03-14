#!/usr/bin/env python3
"""
Utility for splitting a unified checkpoint into separate G/D files.

Usage:
    python split_checkpoint.py path/to/checkpoint.pth output/folder/
    
Will create:
    output/folder/G_pretrain.pth
    output/folder/D_pretrain.pth
"""

import argparse
import os
import sys

import torch


def split_checkpoint(checkpoint_path: str, output_dir: str):
    """
    Splits a unified checkpoint into separate G and D files.
    
    Args:
        checkpoint_path: Path to the unified checkpoint (checkpoint.pth)
        output_dir: Folder for saving G_pretrain.pth and D_pretrain.pth
    """
    if not os.path.exists(checkpoint_path):
        print(f"Error: file '{checkpoint_path}' not found!")
        sys.exit(1)

    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading checkpoint: {checkpoint_path}")

    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except Exception:
        print("Loading in unsafe mode...")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Check format
    if "generator" not in checkpoint or "discriminator" not in checkpoint:
        print("Error: file is not a unified checkpoint of the new format!")
        print("Expected structure with 'generator' and 'discriminator' keys.")
        sys.exit(1)

    epoch = checkpoint.get("epoch", 0)
    learning_rate = checkpoint.get("learning_rate", 1e-4)

    # Extract generator
    g_checkpoint = {
        "model": checkpoint["generator"]["model"],
        "optimizer": checkpoint["generator"]["optimizer"],
        "iteration": epoch,
        "learning_rate": learning_rate,
    }

    # Extract discriminator
    d_checkpoint = {
        "model": checkpoint["discriminator"]["model"],
        "optimizer": checkpoint["discriminator"]["optimizer"],
        "iteration": epoch,
        "learning_rate": learning_rate,
    }

    # Save
    g_path = os.path.join(output_dir, "G_pretrain.pth")
    d_path = os.path.join(output_dir, "D_pretrain.pth")

    torch.save(g_checkpoint, g_path)
    print(f"Saved: {g_path}")

    torch.save(d_checkpoint, d_path)
    print(f"Saved: {d_path}")

    print(f"\nDone! Checkpoint epoch: {epoch}")
    print(f"Files saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Splitting a unified checkpoint into separate G/D files for pretraining.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Usage examples:
            - python split_checkpoint.py logs/MyModel/checkpoint.pth ./pretrain_output/
            - python split_checkpoint.py checkpoint.pth .
        """,
    )
    parser.add_argument("checkpoint_path", type=str, help="Path to the unified checkpoint (checkpoint.pth)")
    parser.add_argument("output_dir", type=str, help="Folder for saving G_pretrain.pth and D_pretrain.pth")

    args = parser.parse_args()
    split_checkpoint(args.checkpoint_path, args.output_dir)


if __name__ == "__main__":
    main()
