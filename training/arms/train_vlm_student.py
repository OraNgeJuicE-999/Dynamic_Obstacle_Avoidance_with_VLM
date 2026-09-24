"""Distill the VLM arm's deployable student (3B) from the fine-tuned teacher
(7B) + oracle labels.

Per plan.md: "train the deployable student against both the oracle label
and the fine-tuned 7B's output (combined loss), not distillation alone --
pure distillation has no error-correction if the teacher itself has blind
spots." Evaluation for both teacher and student always scores against the
shared oracle rubric / real outcomes, never "agreement with the teacher" --
that check belongs in evaluate.py, not here.

One way to implement "combined loss" without needing token/logit-level
alignment between two differently-sized models (which would additionally
assume a shared tokenizer/vocabulary): mix two copies of the training data --
one with the oracle-best candidate as the target, one with the teacher's own
predicted candidate as the target -- at a ratio controlled by
--teacher-weight, rather than a literal per-example weighted loss term.
Coarser than true distillation, but simple and correct. A proper KL-based
loss is worth it later if the teacher and student end up sharing a
tokenizer and this approximation proves too coarse.

Usage:
    python -m training.arms.train_vlm_student --dataset training_data/dataset.json \
        --teacher-model-dir training_data/vlm_teacher

TEMPLATE -- fill in the bodies below.
"""

import argparse
from typing import List

import torch
from torch.utils.data import ConcatDataset, Dataset as TorchDataset
from transformers import AutoModelForVision2Seq, AutoProcessor, Trainer, TrainingArguments

from training.arms.train_vlm_teacher import SYSTEM_PROMPT, _best_candidate_id, _format_candidates
from training.common.dataset_schema import TrainingExample, load_dataset


class TargetOverrideDataset(TorchDataset):
    """Same (image, prompt) inputs as train_vlm_teacher.VLMSelectionDataset,
    but with an explicit per-example target string instead of always using
    the oracle-best candidate -- lets this script build a second copy of the
    data with the teacher's predicted target instead, then mix the two."""

    def __init__(self, examples: List[TrainingExample], processor, targets: List[str]):
        assert len(examples) == len(targets)
        self.examples = examples
        self.processor = processor
        self.targets = targets

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        """Same tokenization as VLMSelectionDataset.__getitem__, but the
        assistant target comes from self.targets[idx] instead of
        _best_candidate_id."""
        from PIL import Image

        example = self.examples[idx]
        image = Image.open(example.camera_image_path).convert("RGB")
        prompt = SYSTEM_PROMPT + "\n" + _format_candidates(example)

        messages = [
            {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]},
            {"role": "assistant", "content": self.targets[idx]},
        ]
        text = self.processor.apply_chat_template(messages, tokenize=False)
        inputs = self.processor(text=text, images=image, return_tensors="pt", padding=True)
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"].clone()
        return inputs


@torch.no_grad()
def _teacher_predict(teacher_model, processor, example: TrainingExample, device: str) -> str:
    """Run the teacher model's own generate() on this example's (image,
    prompt) and return its predicted candidate id as text."""
    from PIL import Image

    image = Image.open(example.camera_image_path).convert("RGB")
    prompt = SYSTEM_PROMPT + "\n" + _format_candidates(example)
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=text, images=image, return_tensors="pt").to(device)
    output_ids = teacher_model.generate(**inputs, max_new_tokens=16)
    return processor.batch_decode(output_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="/home/itri/jetracer_nav/training_data/dataset.json")
    parser.add_argument("--teacher-model-dir", default="/home/itri/jetracer_nav/training_data/vlm_teacher")
    parser.add_argument("--student-model-name", default="Qwen/Qwen2-VL-3B-Instruct")
    parser.add_argument("--output-dir", default="/home/itri/jetracer_nav/training_data/vlm_student")
    parser.add_argument("--teacher-weight", type=float, default=0.3, help="fraction of training rows using the teacher's predicted target instead of the oracle's")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-5)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    examples = [e for e in dataset.examples if e.camera_image_path]
    if not examples:
        raise RuntimeError("No examples with a camera_image_path -- see train_vlm_teacher.py's same caveat.")

    print(f"Loading teacher from {args.teacher_model_dir}...")
    teacher_processor = AutoProcessor.from_pretrained(args.teacher_model_dir)
    teacher_model = AutoModelForVision2Seq.from_pretrained(
        args.teacher_model_dir, torch_dtype=torch.bfloat16
    ).to(device).eval()

    print(f"Running teacher inference over {len(examples)} examples...")
    teacher_targets = [_teacher_predict(teacher_model, teacher_processor, e, device) for e in examples]
    del teacher_model  # free before loading the student

    oracle_targets = [_best_candidate_id(e) for e in examples]

    student_processor = AutoProcessor.from_pretrained(args.student_model_name)
    student_model = AutoModelForVision2Seq.from_pretrained(
        args.student_model_name, torch_dtype=torch.bfloat16, device_map="auto"
    )

    from peft import LoraConfig, get_peft_model

    student_model = get_peft_model(student_model, LoraConfig(r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"]))

    oracle_dataset = TargetOverrideDataset(examples, student_processor, oracle_targets)
    teacher_dataset = TargetOverrideDataset(examples, student_processor, teacher_targets)

    # Mix the two target sets by repeating each dataset in proportion to
    # teacher_weight (see module docstring) -- e.g. teacher_weight=0.3 means
    # roughly 30% of the combined training signal is the teacher's
    # predictions, 70% the oracle's; exact ratio depends on rounding at
    # small repeat counts.
    oracle_repeats = max(1, round((1.0 - args.teacher_weight) * 10))
    teacher_repeats = max(1, round(args.teacher_weight * 10))
    train_dataset = ConcatDataset([oracle_dataset] * oracle_repeats + [teacher_dataset] * teacher_repeats)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        save_strategy="epoch",
        bf16=True,
        report_to=[],
    )

    trainer = Trainer(model=student_model, args=training_args, train_dataset=train_dataset)
    trainer.train()
    trainer.save_model(args.output_dir)
    student_processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
