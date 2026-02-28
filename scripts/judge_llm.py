#!/usr/bin/env python3
"""LLM-as-Judge pairwise comparison for base vs finetuned outputs.

Supports OpenAI-compatible endpoints with built-in provider presets:
- qwen
- deepseek
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any
from urllib import request
from urllib.error import HTTPError, URLError

import yaml
from tqdm import tqdm


PROVIDER_PRESETS = {
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "api_key_env": "DASHSCOPE_API_KEY",
        "model": "qwen-plus",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "api_key_env": "DEEPSEEK_API_KEY",
        "model": "deepseek-chat",
    },
}


def load_env_file_if_exists() -> None:
    # Prefer project root .env when script is run from `scripts/`.
    candidate_paths = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent.parent / ".env",
    ]
    for env_path in candidate_paths:
        if not env_path.exists():
            continue
        for raw in env_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")
            os.environ.setdefault(key, value)
        break


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run pairwise LLM-as-judge on two model outputs.")
    parser.add_argument("--config", type=Path, default=Path("configs/judge_llm.yaml"))
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid yaml: {path}")
    return cfg


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{i}") from exc
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["model_a", "model_b", "total", "a_win", "b_win", "tie", "a_win_rate", "b_win_rate", "tie_rate"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    match = re.search(r"\{.*\}", text, flags=re.S)
    if not match:
        raise ValueError(f"Judge response is not valid JSON: {text[:200]}")
    data = json.loads(match.group(0))
    if not isinstance(data, dict):
        raise ValueError("Judge response JSON is not an object.")
    return data


def get_api_settings(cfg: dict[str, Any]) -> tuple[str, str, str]:
    provider = str(cfg.get("provider", "qwen")).strip().lower()
    if provider not in PROVIDER_PRESETS:
        raise ValueError(f"Unsupported provider: {provider}")
    preset = PROVIDER_PRESETS[provider]

    base_url = str(cfg.get("base_url", preset["base_url"])).rstrip("/")
    model = str(cfg.get("model", preset["model"]))
    api_key_env = str(cfg.get("api_key_env", preset["api_key_env"]))
    api_key = os.getenv(api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Missing API key env: {api_key_env}. "
            f"Set it in shell or write it into project .env."
        )
    return base_url, model, api_key


def call_judge(
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout_sec: int,
) -> dict[str, Any]:
    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }
    req = request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=timeout_sec) as resp:
        body = resp.read().decode("utf-8")
    obj = json.loads(body)
    content = obj["choices"][0]["message"]["content"]
    return extract_json_object(content)


def call_judge_with_retry(
    *,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout_sec: int,
    max_retries: int,
    retry_backoff_sec: float,
) -> dict[str, Any]:
    last_err: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return call_judge(
                base_url=base_url,
                api_key=api_key,
                model=model,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_sec=timeout_sec,
            )
        except (HTTPError, URLError, TimeoutError, ValueError, json.JSONDecodeError) as err:
            last_err = err
            if attempt >= max_retries:
                break
            time.sleep(retry_backoff_sec * (2**attempt))
    raise RuntimeError(f"Judge request failed after retries: {last_err}") from last_err


def normalize_winner(raw: Any) -> str:
    s = str(raw).strip().upper()
    if s in {"A", "MODEL_A"}:
        return "A"
    if s in {"B", "MODEL_B"}:
        return "B"
    return "TIE"


def build_default_system_prompt() -> str:
    return (
        "You are a strict impartial LLM judge.\n"
        "Compare two candidate answers to the same prompt using these criteria:\n"
        "correctness, instruction_following, helpfulness, safety, and conciseness.\n"
        "Output JSON only with keys: winner, score_a, score_b, reason.\n"
        "winner must be one of: A, B, TIE.\n"
        "score_a/score_b are 0-10 numbers.\n"
        "reason should be <= 80 words."
    )


def map_winner_to_model(winner: str, a_is_model_a: bool) -> str:
    if winner == "TIE":
        return "TIE"
    if a_is_model_a:
        return "A" if winner == "A" else "B"
    return "B" if winner == "A" else "A"


def judge_single_sample(
    *,
    sid: str,
    ra: dict[str, Any],
    rb: dict[str, Any],
    id_field: str,
    prompt_field: str,
    response_field: str,
    model_a_name: str,
    model_b_name: str,
    run_bi_order: bool,
    base_url: str,
    api_key: str,
    model: str,
    system_prompt: str,
    temperature: float,
    max_tokens: int,
    timeout_sec: int,
    max_retries: int,
    retry_backoff_sec: float,
) -> tuple[dict[str, Any], str]:
    prompt = str(ra.get(prompt_field, ""))
    ans_a = str(ra.get(response_field, ""))
    ans_b = str(rb.get(response_field, ""))

    user1 = (
        f"[Prompt]\n{prompt}\n\n"
        f"[Answer A]\n{ans_a}\n\n"
        f"[Answer B]\n{ans_b}\n"
    )
    judge1 = call_judge_with_retry(
        base_url=base_url,
        api_key=api_key,
        model=model,
        system_prompt=system_prompt,
        user_prompt=user1,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout_sec=timeout_sec,
        max_retries=max_retries,
        retry_backoff_sec=retry_backoff_sec,
    )
    w1 = normalize_winner(judge1.get("winner", "TIE"))
    mapped1 = map_winner_to_model(w1, a_is_model_a=True)

    mapped2 = "TIE"
    judge2 = None
    if run_bi_order:
        user2 = (
            f"[Prompt]\n{prompt}\n\n"
            f"[Answer A]\n{ans_b}\n\n"
            f"[Answer B]\n{ans_a}\n"
        )
        judge2 = call_judge_with_retry(
            base_url=base_url,
            api_key=api_key,
            model=model,
            system_prompt=system_prompt,
            user_prompt=user2,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_sec=timeout_sec,
            max_retries=max_retries,
            retry_backoff_sec=retry_backoff_sec,
        )
        w2 = normalize_winner(judge2.get("winner", "TIE"))
        mapped2 = map_winner_to_model(w2, a_is_model_a=False)

    if not run_bi_order:
        final = mapped1
    elif mapped1 == mapped2:
        final = mapped1
    elif "TIE" in {mapped1, mapped2}:
        final = mapped1 if mapped2 == "TIE" else mapped2
    else:
        final = "TIE"

    return (
        {
            id_field: sid,
            "prompt": prompt,
            "model_a_name": model_a_name,
            "model_b_name": model_b_name,
            "winner_order1": mapped1,
            "winner_order2": mapped2,
            "final_winner": final,
            "judge_order1": judge1,
            "judge_order2": judge2,
        },
        final,
    )


def main() -> None:
    args = parse_args()
    load_env_file_if_exists()
    cfg = load_yaml(args.config)

    input_a = Path(cfg.get("input_a", "data/sft_eval/outputs_base.jsonl"))
    input_b = Path(cfg.get("input_b", "data/sft_eval/outputs_peft.jsonl"))
    output_file = Path(cfg.get("output_file", "data/sft_eval/judge_pairwise.jsonl"))
    summary_file = Path(cfg.get("summary_file", "data/sft_eval/judge_summary.csv"))

    model_a_name = str(cfg.get("model_a_name", "base"))
    model_b_name = str(cfg.get("model_b_name", "peft"))
    id_field = str(cfg.get("id_field", "id"))
    prompt_field = str(cfg.get("prompt_field", "prompt"))
    response_field = str(cfg.get("response_field", "generated"))
    run_bi_order = bool(cfg.get("run_bi_order", True))
    limit = cfg.get("limit")

    temperature = float(cfg.get("temperature", 0.0))
    max_tokens = int(cfg.get("max_tokens", 512))
    timeout_sec = int(cfg.get("timeout_sec", 120))
    num_workers = int(cfg.get("num_workers", 8))
    max_retries = int(cfg.get("max_retries", 2))
    retry_backoff_sec = float(cfg.get("retry_backoff_sec", 1.0))
    system_prompt = str(cfg.get("system_prompt", build_default_system_prompt()))

    rows_a = load_jsonl(input_a)
    rows_b = load_jsonl(input_b)
    map_a = {str(r[id_field]): r for r in rows_a if id_field in r}
    map_b = {str(r[id_field]): r for r in rows_b if id_field in r}
    common_ids = [k for k in map_a if k in map_b]
    common_ids.sort()
    if limit is not None:
        common_ids = common_ids[: int(limit)]
    if not common_ids:
        raise ValueError("No overlap by id_field between input_a and input_b.")

    base_url, model, api_key = get_api_settings(cfg)

    out_rows: list[dict[str, Any]] = []
    a_win = 0
    b_win = 0
    tie = 0

    futures = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for sid in common_ids:
            futures.append(
                executor.submit(
                    judge_single_sample,
                    sid=sid,
                    ra=map_a[sid],
                    rb=map_b[sid],
                    id_field=id_field,
                    prompt_field=prompt_field,
                    response_field=response_field,
                    model_a_name=model_a_name,
                    model_b_name=model_b_name,
                    run_bi_order=run_bi_order,
                    base_url=base_url,
                    api_key=api_key,
                    model=model,
                    system_prompt=system_prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout_sec=timeout_sec,
                    max_retries=max_retries,
                    retry_backoff_sec=retry_backoff_sec,
                )
            )

        for fut in tqdm(as_completed(futures), total=len(futures), desc="Judging (parallel)"):
            row, final = fut.result()
            if final == "A":
                a_win += 1
            elif final == "B":
                b_win += 1
            else:
                tie += 1
            out_rows.append(row)

    # Keep stable ordering for easier diff/inspection.
    out_rows.sort(key=lambda x: str(x.get(id_field, "")))

    total = len(out_rows)
    summary = {
        "model_a": model_a_name,
        "model_b": model_b_name,
        "total": total,
        "a_win": a_win,
        "b_win": b_win,
        "tie": tie,
        "a_win_rate": round(a_win / total, 6),
        "b_win_rate": round(b_win / total, 6),
        "tie_rate": round(tie / total, 6),
    }

    write_jsonl(output_file, out_rows)
    write_summary_csv(summary_file, [summary])
    print(f"Saved pairwise details: {output_file}")
    print(f"Saved summary: {summary_file}")
    print(summary)


if __name__ == "__main__":
    main()
