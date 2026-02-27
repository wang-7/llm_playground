import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from peft import get_peft_model, LoraConfig
from trl import SFTTrainer, SFTConfig
from huggingface_hub import snapshot_download
import torch

BASE_MODEL = "Qwen/Qwen3-0.6B"
OUTPUT_DIR = "../outputs/sft_qwen3_0.6b"
DATASET_DIR = "../data/alpaca_gpt4"

def main():
    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF token is missing. Set HF_TOKEN (or HUGGINGFACE_HUB_TOKEN) in env/.env before training."
        )

    # Resolve to a local snapshot path first to avoid network-only tokenizer metadata calls.
    # If the model is not cached locally, this raises a clear error instead of a late httpx traceback.
    local_model_dir = snapshot_download(
        repo_id=BASE_MODEL,
        token=hf_token,
        local_files_only=True,
    )

    # load base model/tokenizer from local path
    base_model = AutoModelForCausalLM.from_pretrained(
        local_model_dir,
        token=hf_token,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        local_model_dir,
        token=hf_token,
        use_fast=False,
        local_files_only=True,
    )

    # load dataset and preprocess
    ds = load_dataset(DATASET_DIR)
    def to_messages(example):
        user_text = example['instruction'].strip()
        if example['input'].strip():
            user_text += "\n\n" + example['input'].strip()
        return {
            "messages":[
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": example['output']},
            ]
        }
    
    ds_messages = ds.map(to_messages, remove_columns=ds['train'].column_names)

    def apply_chat_template(example):
        text = tokenizer.apply_chat_template(
            example['messages'],
            tokenize=False,
            # return_tensors='pt',
            add_generation_prompt=False,
        )
        return {'text':text}
    
    inputs = ds_messages.map(apply_chat_template)

    # lora config
    lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM",
    )
    model = get_peft_model(base_model, lora_cfg)

    # trainer config
    train_cfg = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    num_train_epochs=2,
    logging_steps=20,
    eval_steps=50,
    save_steps=200,
    save_total_limit=10,
    max_length=512,
    bf16=True,
    gradient_checkpointing=True,
    dataset_text_field="text",
    report_to=['tensorboard'],
    )

    trainer = SFTTrainer(
    model=model,
    processing_class=tokenizer,
    args=train_cfg,
    train_dataset=inputs["train"],
    eval_dataset=inputs["validation"],
    # dataset_text_field="text",
    )

    trainer.train()
    trainer.save_model(os.path.join(OUTPUT_DIR, "final"))


if __name__ == "__main__":
    main()
