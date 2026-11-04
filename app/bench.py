"""
第1回 宿題（発展）: モデルと質問を変えて、応答時間とトークン数を比べる。

使い方:
    uv run python -m app.bench
"""

import csv
import statistics
import time

from google import genai
from google.genai import errors

from app.config import get_api_key

# (モデル名, 1質問あたりの回数, 呼び出しの間隔[秒])
PLAN = [
    ("gemini-3.5-flash-lite", 3, 5),   # 毎分15 → 4秒以上空ける
    ("gemini-3.8-flash", 1, 13),       # 毎分5  → 12秒以上空ける
]

QUESTIONS = [
    "LoRA とは何ですか？",
    "LoRA と QLoRA の違いを、仕組みとメモリ使用量の観点から詳しく説明してください。",
    "AIE2026の講座の宿題にある「3レベル」とは何ですか？",
]

FIELDS = [
    "model", "question", "latency_s",
    "input_tokens", "output_tokens", "total_tokens", "error",
]


def measure(client, model: str, question: str) -> dict:
    start = time.perf_counter()
    response = client.models.generate_content(model=model, contents=question)
    elapsed = time.perf_counter() - start

    usage = response.usage_metadata
    return {
        "model": model,
        "question": question,
        "latency_s": round(elapsed, 2),
        "input_tokens": usage.prompt_token_count,
        "output_tokens": usage.candidates_token_count,
        "total_tokens": usage.total_token_count,
    }


def run_all(client) -> list[dict]:
    rows = []
    for model, repeats, wait in PLAN:
        for question in QUESTIONS:
            for _ in range(repeats):
                try:
                    row = measure(client, model, question)
                except errors.APIError as e:
                    row = {"model": model, "question": question, "error": e.code}
                rows.append(row)
                print(row)
                time.sleep(wait)
    return rows

def save_csv(rows: list[dict], path: str = "lectures/01/results.csv") -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, restval="")
        writer.writeheader()
        writer.writerows(rows)

def summarize(rows: list[dict]) -> None:
    for model in dict.fromkeys(r["model"] for r in rows):
        ok = [r for r in rows if r["model"] == model and not r.get("error")]
        failed = sum(1 for r in rows if r["model"] == model and r.get("error"))
        if not ok:
            print(f"{model}: 成功なし（失敗{failed}件）")
            continue
        latency = statistics.mean(r["latency_s"] for r in ok)
        output = statistics.mean(r["output_tokens"] for r in ok)
        print(
            f"{model}: 成功{len(ok)}件 失敗{failed}件 "
            f"平均{latency:.2f}秒 平均出力{output:.0f}トークン"
        )


def main() -> None:
    client = genai.Client(api_key=get_api_key())
    rows = run_all(client)
    save_csv(rows)  
    summarize(rows) 


if __name__ == "__main__":
    main()