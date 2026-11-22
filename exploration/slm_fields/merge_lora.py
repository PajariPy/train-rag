"""Merge the field LoRA into Qwen3.5-2B and save a full model vLLM can load (runs on the GPU).

The adapter was trained on the multimodal model class (train.py), so the merge loads the same
class; the processor and tokenizer files are saved next to the weights.

Usage (on the machine, from vast_score.sh):
    python merge_lora.py adapter merged
"""

import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForMultimodalLM, AutoProcessor, AutoTokenizer

MODEL = "Qwen/Qwen3.5-2B"


def main() -> None:
    """Load base + adapter, merge, save."""
    adapter, out = Path(sys.argv[1]), Path(sys.argv[2])
    model = AutoModelForMultimodalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
    model.save_pretrained(out)
    AutoProcessor.from_pretrained(MODEL).save_pretrained(out)
    AutoTokenizer.from_pretrained(MODEL).save_pretrained(out)
    print("merged into", out, flush=True)


if __name__ == "__main__":
    main()
