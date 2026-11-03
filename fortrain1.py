# ==========================================
# [AstraGo / MIG 3g.40gb] LLaMA 3.1 8B Instruct 한국어 뉴스 요약 - 학습 전용
# 1만 학습 / 500개 검증, 16-bit LoRA, 3 epoch, Early Stopping
# ★ 원본 대비 변경점은 "# ★" 주석으로 표시 (길이 유도 + 1만 건 규모 조정만)
# 1. 워크로드 변수에 HF_TOKEN(write 권한) 추가, 명령어: python train_llama_summary_10k_astrago.py
# 2. train_original.json은 input/dataset에서 자동 탐색
# 3. ※ 영구 볼륨이 없어 결과는 로컬(/tmp)에 두고 Hugging Face Hub 업로드로 보존
# ※ 대조군(길이 유도 없음): EOS_WEIGHT=1.0, MAX_SUMMARY_CHARS=None, REPO_NAME/final_dir 이름 변경
# ==========================================
import sys
import subprocess

subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"])
subprocess.run([
    sys.executable, "-m", "pip", "install", "-q",
    "transformers==4.57.1", "peft<0.18", "accelerate<1.11",
    "huggingface_hub<1.0", "datasets"
], check=True)

import json
import torch
import torch.nn.functional as F   # ★ 길이: EOS 가중치 loss 계산용
import pandas as pd

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, get_peft_model
from huggingface_hub import login
import glob, shutil

assert torch.cuda.is_available(), "GPU가 없습니다. 리소스 프리셋에서 GPU(MIG)를 선택했는지 확인해주세요."
print("GPU:", torch.cuda.get_device_name(0))

# ==========================================
# [1] 설정 및 로그인
# ==========================================
HF_TOKEN = os.environ["HF_TOKEN"]
HF_USERNAME = "chaeryu"
REPO_NAME = "llama-3.1-8b-instruct-summary-ko-10k"   # ★ 1만 건
login(token=HF_TOKEN)

def find_train_file(name="train_original.json", max_depth=4):
    """input/dataset 우선 탐색 → 실패 시 주요 경로를 깊이 제한(4)으로 탐색"""
    for d in ["/input/dataset", "input/dataset", "/input", "input", "/dataset", "."]:
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    roots = ["/input", "/dataset", "/data", "/mnt", "/workspace", os.getcwd()]
    for root in roots:
        if root in ("", "/") or not os.path.isdir(root):
            continue
        base = root.rstrip("/").count("/")
        for cur, dirs, files in os.walk(root):
            if cur.count("/") - base >= max_depth:
                dirs[:] = []
            if name in files:
                return os.path.join(cur, name)
    raise FileNotFoundError(f"{name}을(를) 찾지 못했습니다. 탐색 경로: input/dataset, {roots}")

TRAIN_PATH = find_train_file()
print("TRAIN_PATH:", TRAIN_PATH)
OUT_DIR = "/tmp/summary_train"      # 영구 볼륨 없음 → 로컬 임시 경로 (결과는 Hub에 업로드됨)
os.makedirs(OUT_DIR, exist_ok=True)
print(f"디스크 여유(/tmp): {shutil.disk_usage('/tmp').free / 1e9:.0f}GB (모델 16GB + 체크포인트 여유 필요)")

MODEL_ID = "meta-llama/Meta-Llama-3.1-8B-Instruct"
MAX_OUTPUT_LEN = 256
MAX_LEN = 1100              # 학습 시 프롬프트 + 요약 전체 길이 상한
KEEP_TAIL = 12              # 잘릴 때 보존할 끝 토큰 수 (요약: + assistant 헤더)
N_TRAIN = 10000             # ★ 1만 건 (원본 100000)
N_VAL = 10000               # 분리만 1만 개, 실제 평가는 500개만 사용
N_VAL_EVAL = 500
SEED = 42
DATE_STR = "05 Oct 2026"    # 채팅 템플릿의 Today Date 고정 (추론 시에도 동일하게 사용)

# ★ 길이 유도 설정
EOS_WEIGHT = 5.0            # <|eot_id|> 토큰 loss 가중치 (1.0이면 효과 없음, 3~10 사이 조절)
MIN_SUMMARY_CHARS = 20      # 너무 짧은 정답 제외
MAX_SUMMARY_CHARS = 200     # 정답 p90(약 180자) 근처. None이면 필터 없음

# ==========================================
# [2] 데이터 로드
# ==========================================
def extract_text(text_field) -> str:
    return " ".join(s["sentence"] for p in text_field for s in p)

def load_data(path: str) -> pd.DataFrame:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    rows = []
    for doc in raw["documents"]:
        if not doc.get("abstractive"):
            continue
        content = extract_text(doc["text"])
        if len(content) < 100:
            continue
        summary = doc["abstractive"][0]                       # ★ 길이 필터용 변수
        if len(summary) < MIN_SUMMARY_CHARS:                  # ★
            continue
        if MAX_SUMMARY_CHARS is not None and len(summary) > MAX_SUMMARY_CHARS:   # ★
            continue
        rows.append({
            "content": content,
            "summary": summary,
            "title": doc["title"],
            "category": doc.get("category", ""),
        })
    return pd.DataFrame(rows)

print("▶ 데이터 로드 중...")
df_all = load_data(TRAIN_PATH).sample(frac=1, random_state=SEED).reset_index(drop=True)
df_train = df_all.head(N_TRAIN)
df_val = df_all.iloc[N_TRAIN:N_TRAIN + N_VAL]
print(f"✅ Train: {len(df_train)} | Val: {len(df_val)}")
_sl = df_train["summary"].str.len()   # ★ 학습 정답 길이 확인
print(f"학습 정답 요약 길이(자): 중앙값 {_sl.median():.0f} | 평균 {_sl.mean():.1f} | p90 {_sl.quantile(0.9):.0f} | "
      f"평균 쉼표 수 {df_train['summary'].str.count(',').mean():.2f}")

# ==========================================
# [3] 프롬프트 (GPT/Claude/Gemini baseline과 동일 문구) + Instruct 채팅 템플릿
# ==========================================
SYSTEM_PROMPT = (
    "너는 전문 뉴스 에디터야. "
    "주어진 뉴스 기사의 핵심을 파악하여 단 하나의 간결한 문장으로 요약해 줘"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
EOT_ID = tokenizer.convert_tokens_to_ids("<|eot_id|>")   # ★ 길이: EOS 가중치 대상 토큰

def build_prompt_ids(content: str, max_len: int):
    """채팅 템플릿 프롬프트 토큰 (assistant 헤더까지 포함).
    max_len 초과 시 본문 중간을 잘라내고 끝(요약: + assistant 헤더)은 보존."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"뉴스 기사:\n{content[:1500]}\n\n요약:"},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, date_string=DATE_STR
    )
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) > max_len:
        ids = ids[:max_len - KEEP_TAIL] + ids[-KEEP_TAIL:]
    return ids

def tokenize_fn(examples):
    out = {"input_ids": [], "attention_mask": [], "labels": []}
    for content, summary in zip(examples["content"], examples["summary"]):
        ans = tokenizer(summary + "<|eot_id|>", add_special_tokens=False)["input_ids"][:MAX_OUTPUT_LEN]
        prompt = build_prompt_ids(content, MAX_LEN - len(ans))
        ids = prompt + ans
        out["input_ids"].append(ids)
        out["attention_mask"].append([1] * len(ids))
        out["labels"].append([-100] * len(prompt) + ans)   # 프롬프트 loss 마스킹
    return out

def to_dataset(df):
    ds = Dataset.from_dict({
        "content": df["content"].tolist(),
        "summary": df["summary"].tolist(),
    })
    return ds.map(tokenize_fn, batched=True, remove_columns=["content", "summary"], num_proc=4)

print("▶ 토크나이징 진행 중...")
train_dataset = to_dataset(df_train)
eval_dataset = to_dataset(df_val.head(N_VAL_EVAL))
print(f"학습 {len(train_dataset)}개, 검증 {len(eval_dataset)}개")

# ==========================================
# [4] 모델 로드 (bf16 + SDPA) 및 LoRA (r=32, alpha=64)
# ==========================================
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
    attn_implementation="sdpa",
    device_map={"": 0},
    token=HF_TOKEN,
)
model.config.use_cache = False
model.enable_input_require_grads()   # gradient checkpointing + LoRA 필수

peft_config = LoraConfig(
    r=32,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# ==========================================
# [5] 학습 설정 (3 epoch + Early Stopping, Instruct용 학습률 2e-5)
# ==========================================
output_dir = f"{OUT_DIR}/{REPO_NAME}"

training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=3,
    per_device_train_batch_size=2,       # 4 → 2
    per_device_eval_batch_size=2,        # 4 → 2
    gradient_accumulation_steps=16,      # 8 → 16 (유효 배치 32 유지)
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    optim="adamw_torch",
    learning_rate=2e-5,
    lr_scheduler_type="cosine",
    warmup_steps=47,                     # ★ 1만 건: 전체 약 940스텝의 5% (원본 470 = 9,375스텝의 5%)
    bf16=True,
    fp16=False,
    logging_steps=50,
    eval_strategy="steps",
    eval_steps=100,                      # ★ 1만 건: 원본 500은 평가가 2번뿐이라 100으로
    save_strategy="steps",
    save_steps=100,                      # eval_steps와 같거나 배수여야 Early Stopping 동작
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    push_to_hub=True,
    hub_model_id=f"{HF_USERNAME}/{REPO_NAME}",
    hub_strategy="checkpoint",
    report_to="none",
    seed=SEED,
)

# ★ 길이: EOS 가중치 Trainer
#  - 학습: <|eot_id|> 토큰 loss만 EOS_WEIGHT배 (일찍 끝내도록 유도)
#  - 평가: 가중치 없는 평균 loss → 다른 실험과 eval_loss 직접 비교 가능
class EosWeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits[:, :-1]
        tgt = labels[:, 1:]
        mask = tgt != -100
        sel_logits = logits[mask].float()          # 정답 토큰 위치만 계산 (메모리 절약)
        sel_tgt = tgt[mask]
        tok_loss = F.cross_entropy(sel_logits, sel_tgt, reduction="none")

        w = torch.ones_like(tok_loss)
        if model.training:
            w[sel_tgt == EOT_ID] = EOS_WEIGHT

        if model.training and num_items_in_batch is not None and getattr(self, "model_accepts_loss_kwargs", True):
            loss = (tok_loss * w).sum() / num_items_in_batch   # gradient accumulation 전체 토큰 기준 정규화
        else:
            loss = (tok_loss * w).sum() / w.sum()
        return (loss, outputs) if return_outputs else loss

trainer = EosWeightedTrainer(            # ★ Trainer → EosWeightedTrainer
    model=model,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    args=training_args,
    data_collator=DataCollatorForSeq2Seq(
        tokenizer=tokenizer, pad_to_multiple_of=8, label_pad_token_id=-100, return_tensors="pt"
    ),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3, early_stopping_threshold=0.0005)],
)

last_ckpt = get_last_checkpoint(output_dir) if os.path.isdir(output_dir) else None
print(f"재개할 체크포인트: {last_ckpt}" if last_ckpt else "체크포인트 없음 -> 처음부터 학습 시작")
trainer.train(resume_from_checkpoint=last_ckpt)

# ==========================================
# [6] 저장 및 Hub 업로드
# ==========================================
final_dir = f"{OUT_DIR}/final_model_instruct_10k"   # ★ 1만 건
trainer.model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)
trainer.push_to_hub()
print(f"🎉 학습, 저장, 허브 업로드 완료 → https://huggingface.co/{HF_USERNAME}/{REPO_NAME}")
