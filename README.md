# 大模型后训练完整教程（可自己实现代码并复现实验）

本教程目标是让你在本目录下，从 0 到 1 完成一条可复现的 LLM 后训练流程：

1. 明确任务和基线  
2. 构建 SFT 数据并训练 SFT 模型  
3. 构建偏好数据并做 DPO 对齐  
4. 用统一评测脚本对比基线/SFT/DPO  
5. 做系统化实验分析（不只看 loss）

教程默认你使用 Hugging Face + TRL + PEFT（LoRA/QLoRA）路线，先做“小模型+小数据跑通”，再扩规模。

---

## 0. 先理解后训练全景

后训练通常是三层：

1. `SFT`（监督微调）  
   目标：让模型学会“按你想要的格式/风格/任务”回答。
2. `Preference Alignment`（偏好对齐：DPO/ORPO/PPO 等）  
   目标：在多个可行回答里，更偏向你期望的回答。
3. `Evaluation + Iteration`（评测与迭代）  
   目标：通过统一指标和误差分析，明确下一轮该改数据、改训练，还是改解码策略。

建议路线：`Base Model -> SFT -> DPO -> 评测分析 -> 新一轮数据闭环`

---

## 1. 项目目录建议

你可以按下面结构组织：

```text
llm_playground/
├─ README.md
├─ pyproject.toml
├─ main.py
├─ configs/
│  ├─ sft.yaml
│  ├─ dpo.yaml
│  └─ eval.yaml
├─ data/
│  ├─ raw/
│  ├─ processed/
│  │  ├─ sft_train.jsonl
│  │  ├─ sft_valid.jsonl
│  │  ├─ pref_train.jsonl
│  │  └─ pref_valid.jsonl
│  └─ eval/
│     └─ heldout_eval.jsonl
├─ scripts/
│  ├─ build_sft_dataset.py
│  ├─ build_pref_dataset.py
│  ├─ train_sft.py
│  ├─ train_dpo.py
│  ├─ evaluate.py
│  └─ analyze_results.py
└─ outputs/
   ├─ sft/
   ├─ dpo/
   └─ reports/
```

---

## 2. 环境准备（先保证可运行）

### 2.1 Python 与依赖

建议先用小规模实验验证流程，再上多卡和大模型。

```bash
cd /media/a100/data/WQ/private/llm_playground

# 如果你使用 uv
uv venv
source .venv/bin/activate

uv pip install torch transformers datasets trl peft accelerate evaluate scikit-learn pandas pyyaml tqdm

# 可选（量化/分布式）
uv pip install bitsandbytes deepspeed
```

### 2.2 HF Token（安全方式）

不要在代码里写死 token。使用环境变量：

```bash
cd /media/a100/data/WQ/private/llm_playground
cp .env.example .env
# 编辑 .env，把 HF_TOKEN 改成你自己的 read token
```

训练时，`scripts/train.sh` 会自动加载 `.env` 并检查 `HF_TOKEN`。

### 2.3 accelerate 配置

```bash
accelerate config
```

关键点：

1. 先单机单卡跑通（最容易定位问题）。  
2. 再切单机多卡。  
3. 最后再考虑 Deepspeed/ZeRO。

---

## 3. 第一步：定义任务、基线和验收标准

很多训练失败其实不是训练脚本问题，而是“目标不清”。

你先写清楚这 4 件事（建议放进 `configs/*.yaml`）：

1. `目标能力`：比如“代码解释”“客服问答”“数学推理”。  
2. `约束`：响应长度、风格、拒答策略、安全边界。  
3. `关键指标`：准确率/通过率、拒答正确率、格式合规率、延迟。  
4. `基线模型`：未训练基座模型在同一测试集上的结果。

示例（`configs/eval.yaml`）：

```yaml
task_name: "domain_qa"
base_model: "Qwen/Qwen2.5-1.5B"
max_new_tokens: 512
temperature: 0.0
metrics:
  - exact_match
  - rougeL
  - format_accuracy
```

---

## 4. 第二步：准备 SFT 数据

### 4.1 数据格式（推荐 JSONL）

每行一条样本：

```json
{"system":"你是一个严谨的助手","prompt":"解释什么是过拟合","response":"过拟合是...","source":"manual_v1","tags":["ml","definition"]}
```

### 4.2 数据质量检查清单

1. 去重：相同 prompt-response 重复要清。  
2. 格式统一：字段名、编码、换行风格一致。  
3. 长度控制：极短/极长异常样本单独排查。  
4. 事实风险：高风险领域样本单独标记。  
5. 污染检查：评测集样本不能混进训练集。

### 4.3 切分策略

1. `train`：80%~95%  
2. `valid`：5%~10%（调参与早停）  
3. `test/heldout`：固定不动，仅用于最终评估

建议先用 2k~10k 样本跑通，再扩到更大规模。

---

## 5. 第三步：自己写 SFT 训练代码

核心是 3 件事：

1. 把样本转成聊天模板文本  
2. 只训练可训练参数（LoRA）  
3. 记录可比较的训练日志

下面是 `scripts/train_sft.py` 的最小骨架（你按此自己实现）：

```python
import os
from dataclasses import dataclass

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig


BASE_MODEL = "Qwen/Qwen2.5-1.5B"
TRAIN_FILE = "data/processed/sft_train.jsonl"
VALID_FILE = "data/processed/sft_valid.jsonl"
OUTPUT_DIR = "outputs/sft"


def format_example(ex):
    # 你可以替换成自己的 chat template
    system = ex.get("system", "")
    prompt = ex["prompt"]
    response = ex["response"]
    text = f"<|system|>\n{system}\n<|user|>\n{prompt}\n<|assistant|>\n{response}"
    return {"text": text}


def main():
    ds = load_dataset("json", data_files={"train": TRAIN_FILE, "validation": VALID_FILE})
    ds = ds.map(format_example, remove_columns=ds["train"].column_names)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype="auto")

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    train_cfg = SFTConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=2,
        logging_steps=20,
        eval_steps=200,
        save_steps=200,
        save_total_limit=2,
        max_seq_length=1024,
        bf16=True,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=train_cfg,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        dataset_text_field="text",
    )
    trainer.train()
    trainer.save_model(os.path.join(OUTPUT_DIR, "final"))


if __name__ == "__main__":
    main()
```

启动训练：

```bash
accelerate launch scripts/train_sft.py
```

---

## 6. 第四步：构建偏好数据并做 DPO

SFT 之后进入偏好优化。DPO 数据格式通常是：

```json
{"prompt":"解释梯度下降","chosen":"梯度下降是...","rejected":"梯度下降就是把梯度变小"}
```

`chosen` 是你偏好的回答，`rejected` 是较差但仍可读的回答（不要太垃圾，否则训练信号弱）。

DPO 训练骨架（`scripts/train_dpo.py`）：

```python
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from trl import DPOTrainer, DPOConfig


BASE_MODEL = "Qwen/Qwen2.5-1.5B"
SFT_ADAPTER = "outputs/sft/final"
TRAIN_FILE = "data/processed/pref_train.jsonl"
VALID_FILE = "data/processed/pref_valid.jsonl"
OUTPUT_DIR = "outputs/dpo"


def main():
    ds = load_dataset("json", data_files={"train": TRAIN_FILE, "validation": VALID_FILE})

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype="auto")
    model = PeftModel.from_pretrained(base, SFT_ADAPTER, is_trainable=True)

    dpo_args = DPOConfig(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=5e-6,
        num_train_epochs=1,
        logging_steps=20,
        eval_steps=100,
        save_steps=100,
        beta=0.1,
        bf16=True,
        report_to=[],
    )

    trainer = DPOTrainer(
        model=model,
        args=dpo_args,
        processing_class=tokenizer,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
    )
    trainer.train()
    trainer.save_model(f"{OUTPUT_DIR}/final")


if __name__ == "__main__":
    main()
```

启动训练：

```bash
accelerate launch scripts/train_dpo.py
```

---

## 7. 第五步：统一评测（必须同一套设置）

比较 `Base vs SFT vs DPO` 时，必须保证：

1. 同一批测试样本  
2. 同样解码参数（temperature/top_p/max_new_tokens）  
3. 同样打分脚本

评测脚本至少输出：

1. 每条样本的模型回答  
2. 每条样本的自动指标  
3. 总体均值 + 分位数（P50/P90）  
4. 失败样本清单（便于人工复盘）

示例结果文件：

```text
outputs/reports/
├─ base_eval.jsonl
├─ sft_eval.jsonl
├─ dpo_eval.jsonl
└─ summary.csv
```

---

## 8. 第六步：实验分析怎么做（重点）

不要只看训练 loss。至少从 4 个维度分析：

1. `效果`：核心任务指标是否提升？提升幅度是否稳定？  
2. `泛化`：训练外样本是否也提升？还是只记住模板？  
3. `安全与拒答`：危险请求是否更稳健？误拒是否变多？  
4. `代价`：训练耗时、显存、推理速度是否可接受？

建议做这个消融矩阵（Ablation Matrix）：

1. 数据规模：2k / 10k / 50k  
2. LoRA rank：8 / 16 / 32  
3. 学习率：1e-4 / 2e-4（SFT），1e-6 / 5e-6（DPO）  
4. 模板：模板 A / 模板 B  
5. 偏好数据构造：人工标注 / 模型打分筛选

记录模板（`outputs/reports/experiment_log.csv`）建议字段：

```text
run_id,model_stage,base_model,data_version,train_size,lr,epoch,lora_r,beta,
eval_exact_match,eval_rougeL,format_acc,refusal_acc,toxicity_rate,
train_hours,max_gpu_mem_gb,notes
```

如何解读：

1. `train loss 下降但 eval 不升`：通常是数据分布或模板问题。  
2. `SFT 升，DPO 降`：常见于偏好对质量差、beta 过大、学习率偏高。  
3. `指标升但人工体验差`：自动指标与真实体验错位，需补人工 rubric。

---

## 9. 最小可复现执行清单（建议按顺序）

1. 准备 2k 条高质量 SFT 样本，跑完 1 次 SFT。  
2. 准备 1k 条偏好对，跑完 1 次 DPO。  
3. 固定 300 条 heldout 样本，分别评测 Base/SFT/DPO。  
4. 输出 `summary.csv` + 20 条失败案例分析。  
5. 基于失败类型回流数据，开始第二轮迭代。

---

## 10. 常见坑位

1. `训练集和评测集泄漏`：会导致“虚高提升”。  
2. `偏好对质量差`：DPO 经常会把模型拉偏。  
3. `只看平均指标`：必须看分布和失败样本。  
4. `一次改太多变量`：无法定位真正有效因素。  
5. `直接大规模训练`：建议先小规模验证脚本与数据闭环。

---

## 11. 你下一步可以直接开始写的代码

按优先级：

1. `scripts/build_sft_dataset.py`：清洗、去重、切分、导出 JSONL  
2. `scripts/train_sft.py`：加载数据、模板化、LoRA SFT  
3. `scripts/build_pref_dataset.py`：生成/整理 chosen-rejected 对  
4. `scripts/train_dpo.py`：从 SFT 模型继续做 DPO  
5. `scripts/evaluate.py`：统一评测与结果导出  
6. `scripts/analyze_results.py`：聚合 CSV、绘图、失败样本统计

如果你愿意，我下一步可以直接在这个项目里把以上 6 个脚本的“可运行最小版本”全部搭好，你只需要填入自己的数据即可开始训练。
