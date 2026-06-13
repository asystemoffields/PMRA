"""
DSmolLM Gradient Refinement — Knowledge Distillation on Compressed Model

Compresses the model via the allocation, then fine-tunes the compressed weights
using KL divergence from the original (teacher) model. Tests whether gradient
refinement can close the compound error gap.
"""

import argparse
import gc
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from structured_search import get_data, eval_perplexity
from compress_model import apply_allocation, parse_family

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(write_through=True)
    except Exception:
        pass


def get_training_data(tokenizer, n_samples=2048, seq_len=512):
    """Stream C4 training data for KD."""
    dataset = load_dataset("allenai/c4", "en", split="train", streaming=True)
    samples = []
    for doc in dataset:
        if len(samples) >= n_samples:
            break
        text = doc["text"].strip()
        if len(text) < 200:
            continue
        tokens = tokenizer(text, return_tensors="pt", truncation=True,
                           max_length=seq_len)["input_ids"][0]
        if len(tokens) >= seq_len:
            samples.append(tokens[:seq_len])
    return torch.stack(samples)


def train_kd(student, teacher, train_data, device, n_steps=5000,
             batch_size=8, lr=1e-4, temperature=2.0, alpha=0.7,
             warmup_steps=100, eval_every=500, eval_data=None, base_loss=0.0,
             checkpoint_dir=None, commit_fn=None):
    """Knowledge distillation: train student to match teacher's output distribution."""
    teacher.eval()
    student.train()

    optimizer = torch.optim.AdamW(student.parameters(), lr=lr, weight_decay=0.01)

    def lr_schedule(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, n_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    n_train = len(train_data)
    step = 0
    epoch = 0
    history = []

    print(f"    Training: {n_steps} steps, batch={batch_size}, lr={lr}, "
          f"T={temperature}, alpha={alpha}")

    while step < n_steps:
        perm = torch.randperm(n_train)
        epoch += 1

        for i in range(0, n_train - batch_size + 1, batch_size):
            if step >= n_steps:
                break

            idx = perm[i:i + batch_size]
            batch = train_data[idx].to(device)

            with torch.no_grad():
                teacher_out = teacher(input_ids=batch)
                teacher_logits = teacher_out.logits

            student_out = student(input_ids=batch, labels=batch)
            student_logits = student_out.logits
            nll_loss = student_out.loss

            kd_loss = F.kl_div(
                F.log_softmax(student_logits / temperature, dim=-1),
                F.softmax(teacher_logits / temperature, dim=-1),
                reduction='batchmean',
            ) * (temperature ** 2)

            loss = alpha * kd_loss + (1 - alpha) * nll_loss

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            step += 1

            if step % 100 == 0:
                print(f"      step {step:>5d}/{n_steps}  loss={loss.item():.4f}  "
                      f"kd={kd_loss.item():.4f}  nll={nll_loss.item():.4f}  "
                      f"lr={scheduler.get_last_lr()[0]:.6f}")

            if eval_data is not None and step % eval_every == 0:
                student.eval()
                ppl, lss = eval_perplexity(student, eval_data, device)
                dloss = lss - base_loss
                history.append({"step": step, "ppl": ppl, "loss": lss, "dloss": dloss})
                print(f"      >>> eval step {step}: PPL={ppl:.2f}, dloss={dloss:+.4f}")

                if checkpoint_dir is not None:
                    ckpt_path = Path(checkpoint_dir) / f"checkpoint_step{step}.pt"
                    torch.save({
                        "step": step, "ppl": ppl, "dloss": dloss,
                        "model_state_dict": student.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                    }, ckpt_path)
                    print(f"      >>> saved checkpoint: {ckpt_path}")
                    if commit_fn:
                        try:
                            commit_fn()
                        except Exception:
                            pass
                student.train()

    student.eval()
    return history


def main(commit_fn=None):
    parser = argparse.ArgumentParser(description="DSmolLM Gradient Refinement")
    parser.add_argument("--allocation", type=str, required=True)
    parser.add_argument("--model", type=str, default="HuggingFaceTB/SmolLM-135M")
    parser.add_argument("--teacher", type=str, default=None,
                        help="Teacher model for KD (default: same as --model). "
                             "Use a bigger model for cross-size distillation.")
    parser.add_argument("--output", type=str, default="results/dsmollm_refined")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--n-steps", type=int, default=5000)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--temperature", type=float, default=2.0)
    parser.add_argument("--alpha", type=float, default=0.7)
    parser.add_argument("--n-eval", type=int, default=128)
    parser.add_argument("--n-calib", type=int, default=64)
    parser.add_argument("--n-train", type=int, default=2048)
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to checkpoint .pt to resume training from")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"  DSmolLM — Gradient Refinement via Knowledge Distillation")
    print(f"  Model: {args.model}")
    print(f"  Device: {device}")
    print(f"{'=' * 70}")

    with open(args.allocation) as f:
        alloc_data = json.load(f)

    allocation = alloc_data["optimal"]["allocations"]
    est_dloss = alloc_data["optimal"]["total_dloss"]
    target_ratio = alloc_data["target_layer_ratio"]
    n_to_compress = sum(1 for a in allocation if a["family"] != "original")

    print(f"\n  Allocation: {target_ratio}x, {n_to_compress} matrices to compress")
    print(f"  Additive estimate: dloss={est_dloss:+.4f}")
    print(f"  Refinement: {args.n_steps} steps, lr={args.lr}, T={args.temperature}")

    # Load teacher (frozen)
    teacher_name = args.teacher or args.model
    print(f"\n  Loading teacher model: {teacher_name}...")
    teacher = AutoModelForCausalLM.from_pretrained(teacher_name, torch_dtype=torch.float32).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    if teacher_name != args.model:
        print(f"    Cross-size distillation: {teacher_name} -> {args.model}")

    # Load student (will be compressed)
    print(f"  Loading student model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    student = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32).to(device)
    student.eval()

    # Load data
    print(f"\n  Loading evaluation data...")
    calib_data = get_data(tokenizer, "train", n_samples=args.n_calib, skip=0)
    eval_data = get_data(tokenizer, "test", n_samples=args.n_eval, skip=args.n_calib)

    print(f"  Loading training data ({args.n_train} samples)...")
    train_data = get_training_data(tokenizer, n_samples=args.n_train, seq_len=512)
    print(f"    Got {len(train_data)} training samples")

    # Baseline (from teacher since it's the clean model)
    print(f"\n  Measuring baseline...")
    base_ppl, base_loss = eval_perplexity(teacher, eval_data, device)
    print(f"    Baseline: PPL={base_ppl:.2f}, loss={base_loss:.4f}")

    if args.resume:
        # Resume from checkpoint — skip compression
        print(f"\n  Resuming from checkpoint: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        student.load_state_dict(ckpt["model_state_dict"])
        orig = sum(a["original_params"] for a in allocation)
        compressed = sum(a["compressed_params"] for a in allocation)
        pre_ppl, pre_loss = eval_perplexity(student, eval_data, device)
        pre_dloss = pre_loss - base_loss
        print(f"    Resumed at step {ckpt.get('step', '?')}: PPL={pre_ppl:.2f}, "
              f"dloss={pre_dloss:+.4f}")
    else:
        # Compress student from scratch
        print(f"\n  Compressing student ({n_to_compress} matrices)...")
        t0 = time.time()
        orig, compressed = apply_allocation(student, allocation, calib_data, device)
        print(f"    Compression took {time.time()-t0:.1f}s")

        # Evaluate pre-refinement
        print(f"\n  Pre-refinement evaluation...")
        pre_ppl, pre_loss = eval_perplexity(student, eval_data, device)
        pre_dloss = pre_loss - base_loss
        print(f"    Pre-refinement: PPL={pre_ppl:.2f}, dloss={pre_dloss:+.4f}")

    if commit_fn:
        try:
            commit_fn()
        except Exception:
            pass

    # Knowledge distillation
    checkpoint_dir = output_path / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Starting knowledge distillation...")
    t0 = time.time()
    history = train_kd(
        student, teacher, train_data, device,
        n_steps=args.n_steps, batch_size=args.batch_size,
        lr=args.lr, temperature=args.temperature, alpha=args.alpha,
        eval_every=500, eval_data=eval_data, base_loss=base_loss,
        checkpoint_dir=checkpoint_dir, commit_fn=commit_fn,
    )
    train_time = time.time() - t0
    print(f"    Training took {train_time:.1f}s")

    # Save final model
    final_ckpt = output_path / f"dsmollm_{target_ratio}x_final.pt"
    torch.save({"model_state_dict": student.state_dict()}, final_ckpt)
    print(f"    Final model saved: {final_ckpt}")
    if commit_fn:
        try:
            commit_fn()
        except Exception:
            pass

    # Final evaluation
    print(f"\n  Final evaluation...")
    final_ppl, final_loss = eval_perplexity(student, eval_data, device)
    final_dloss = final_loss - base_loss

    # Report
    compound_gap = pre_dloss - est_dloss
    recovery = pre_dloss - final_dloss
    gap_pct = recovery / compound_gap * 100 if compound_gap > 0 else 0

    print(f"\n{'=' * 70}")
    print(f"  RESULTS — Gradient Refinement")
    print(f"{'=' * 70}")
    print(f"  Baseline PPL:         {base_ppl:>12.2f}")
    print(f"  Pre-refinement PPL:   {pre_ppl:>12.2f}  (dloss={pre_dloss:+.4f})")
    print(f"  Post-refinement PPL:  {final_ppl:>12.2f}  (dloss={final_dloss:+.4f})")
    print(f"  Additive estimate:    {'':>12s}  (dloss={est_dloss:+.4f})")
    print(f"")
    print(f"  Compound gap:         {compound_gap:>+12.4f}  (pre - additive)")
    print(f"  Recovery:             {recovery:>+12.4f}  ({gap_pct:.0f}% of gap closed)")
    print(f"  Remaining gap:        {final_dloss - est_dloss:>+12.4f}")
    print(f"")
    print(f"  Compression ratio:    {orig / compressed:>12.2f}x")
    print(f"  Training: {args.n_steps} steps in {train_time:.0f}s "
          f"({len(train_data) * args.n_steps // len(train_data):.0f} epochs)")

    results = {
        "method": "gradient_refinement_kd",
        "model": args.model,
        "target_ratio": target_ratio,
        "baseline_ppl": base_ppl,
        "baseline_loss": base_loss,
        "pre_refinement_ppl": pre_ppl,
        "pre_refinement_dloss": pre_dloss,
        "post_refinement_ppl": final_ppl,
        "post_refinement_dloss": final_dloss,
        "estimated_dloss": est_dloss,
        "compound_gap": compound_gap,
        "recovery": recovery,
        "gap_closed_pct": gap_pct,
        "n_steps": args.n_steps,
        "lr": args.lr,
        "temperature": args.temperature,
        "alpha": args.alpha,
        "train_time_s": train_time,
        "history": history,
        "original_params": orig,
        "compressed_params": compressed,
    }

    results_path = output_path / f"refined_{target_ratio}x.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved to {results_path}")

    if commit_fn:
        try:
            commit_fn()
        except Exception:
            pass


if __name__ == "__main__":
    main()
