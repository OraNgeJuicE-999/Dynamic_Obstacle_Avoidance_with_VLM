"""Fine-tune the VLM arm's teacher model (7B) on oracle labels.

Two-stage pipeline per plan.md, both stages done by the same deployed model
(not a separate perception model + separate non-VLM scorer, which would make
"VLM compute cost" impossible to attribute cleanly):
  1. VQA: is there a person on screen, and if so, predict their movement.
  2. Candidate selection: given that context, choose the best candidate path.

This script fine-tunes stage 2 directly against oracle_scorer.py's labels --
the shared answer key every arm trains against (plan.md: "MLP and VLM train
against the exact same answer key"). The camera image plus a text-serialized
candidate list is the input; the oracle-best candidate's id is the target.
Evaluation always scores against the shared oracle rubric / real outcomes,
never "agreement with a human label" -- that's a separate concern
(evaluate.py), not this script's job.

Requires camera_capture.py to actually be implemented first -- every
TrainingExample needs a real camera_image_path. VLMSelectionDataset should
filter out any example missing an image; training on a dataset built before
camera_capture.py exists should fail loudly rather than silently train on
nothing (see main()'s TODO).

Model: Qwen2-VL-7B-Instruct by default (plan.md: "a larger teacher (7B
Qwen)") -- swap via --model-name for whatever's actually available/licensed.
Uses LoRA (via peft) rather than full fine-tuning to keep this runnable on
realistic hardware; --no-lora should full-fine-tune instead.

Usage:
    python -m training.arms.train_vlm_teacher --dataset training_data/dataset.json

TEMPLATE -- fill in the bodies below.
"""

import argparse
from typing import List

import torch
from torch.utils.data import Dataset as TorchDataset
from transformers import AutoModelForVision2Seq, AutoProcessor, Trainer, TrainingArguments

from training.common.dataset_schema import TrainingExample, load_dataset

SYSTEM_PROMPT = (
    "You are the path-selection module of an autonomous JetRacer. Given the "
    "camera view and a list of candidate paths (as waypoint lists), first "
    "note whether a person is visible and predict their movement, then "
    "choose the single best candidate id."
)


def _format_candidates(example: TrainingExample) -> str:
    """Serialize example.candidates into the text the prompt shows the
    model -- candidate_id + waypoints for each."""
    lines = [f"{c.candidate_id}: {c.waypoints}" for c in example.candidates]
    return "Candidates:\n" + "\n".join(lines)


def _best_candidate_id(example: TrainingExample) -> str:
    """The candidate_id with the highest oracle_scorer score among
    example.labels -- the supervised fine-tuning target."""
    return max(example.labels, key=lambda label: label.score).candidate_id


class VLMSelectionDataset(TorchDataset):
    """Wraps dataset_schema.Dataset's TrainingExamples into (image, prompt,
    target) rows for supervised fine-tuning. Should skip any example
    missing a camera image -- see module docstring on camera_capture.py's
    status.
    """

    def __init__(self, examples: List[TrainingExample], processor):
        self.examples = [e for e in examples if e.camera_image_path]
        self.processor = processor

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        """Build the chat-template prompt (SYSTEM_PROMPT + _format_candidates),
        load the image, tokenize via self.processor, and set labels =
        input_ids (causal LM fine-tuning on the assistant's target text)."""
        from PIL import Image

        example = self.examples[idx]
        image = Image.open(example.camera_image_path).convert("RGB")
        prompt = SYSTEM_PROMPT + "\n" + _format_candidates(example)
        target = _best_candidate_id(example)

        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
            {"role": "assistant", "content": target},
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False)
        inputs = self.processor(text=text, images=image, return_tensors="pt", padding=True)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="/home/itri/jetracer_nav/training_data/dataset.json")
    parser.add_argument("--model-name", default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--output-dir", default="/home/itri/jetracer_nav/training_data/vlm_teacher")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lora", action="store_true", default=True)
    parser.add_argument("--no-lora", dest="lora", action="store_false")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = AutoModelForVision2Seq.from_pretrained(args.model_name, torch_dtype=torch.bfloat16, device_map="auto")

    if args.lora:
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"]))

    train_dataset = VLMSelectionDataset(dataset.examples, processor)
    if len(train_dataset) == 0:
        raise RuntimeError(
            "No examples with a camera_image_path -- camera_capture.py needs to be implemented "
            "and the dataset rebuilt (training.data_pipeline.dataset_builder) before this can train."
        )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        save_strategy="epoch",
        bf16=True,
        report_to=[],
    )

    trainer = Trainer(model=model, args=training_args, train_dataset=train_dataset)
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
