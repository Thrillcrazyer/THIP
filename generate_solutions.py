"""
DeepMath-103k 문제를 Local vLLM 서버로 보내 r1_solution을 생성하고 CSV로 저장하는 스크립트.

사용법:
  # vLLM 서버 실행 (별도 터미널)
  # vllm serve <model_name> --port 8000

  # 스크립트 실행
  python generate_solutions.py \
      --input DeepMath-103k_id.csv \
      --output deepmath_solutions.csv \
      --base-url http://localhost:8000/v1 \
      --model <model_name> \
      --max-rows 1000 \
      --batch-size 64 \
      --max-concurrent 32 \
      --max-tokens 16384
"""

import argparse
import asyncio
import csv
import os
import time

from openai import AsyncOpenAI


SYSTEM_PROMPT = (
    "You are a helpful math assistant. "
    "Solve the given math problem step by step, showing your reasoning clearly."
)


async def generate_one(
    client: AsyncOpenAI,
    model: str,
    question: str,
    row_id: int,
    semaphore: asyncio.Semaphore,
    max_tokens: int,
    temperature: float,
) -> dict:
    """단일 문제에 대해 LLM solution을 생성한다."""
    async with semaphore:
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            msg = response.choices[0].message
            # 전체 응답 텍스트를 그대로 저장
            solution = msg.content or ""
            if not solution.strip():
                print(f"[WARN] id={row_id}: empty response")
        except Exception as e:
            print(f"[ERROR] id={row_id}: {e}")
            solution = ""
        return {"id": row_id, "r1_solution": solution}


async def process_batch(
    client: AsyncOpenAI,
    model: str,
    batch: list[dict],
    semaphore: asyncio.Semaphore,
    max_tokens: int,
    temperature: float,
) -> list[dict]:
    """배치 단위로 비동기 요청을 동시에 보낸다."""
    tasks = [
        generate_one(client, model, row["question"], row["id"], semaphore, max_tokens, temperature)
        for row in batch
    ]
    return await asyncio.gather(*tasks)


def read_questions(input_path: str, max_rows: int | None = None) -> list[dict]:
    """CSV에서 id와 question만 읽어온다 (메모리 절약)."""
    rows = []
    with open(input_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows is not None and i >= max_rows:
                break
            rows.append({"id": int(row["id"]), "question": row["question"]})
    return rows


def load_existing_ids(output_path: str) -> set[int]:
    """이미 처리된 id를 읽어서 중복 방지 (이어하기 지원)."""
    done = set()
    if os.path.exists(output_path):
        with open(output_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                done.add(int(row["id"]))
    return done


async def main():
    parser = argparse.ArgumentParser(description="Generate r1_solutions via local vLLM")
    parser.add_argument("--input", type=str, default="DeepMath-103k_id.csv")
    parser.add_argument("--output", type=str, default="deepmath_solutions.csv")
    parser.add_argument("--base-url", type=str, default="http://localhost:8000/v1")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--api-key", type=str, default="EMPTY",
                        help="vLLM은 기본적으로 API key가 불필요하므로 EMPTY 사용")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="처리할 최대 행 수 (None이면 전체)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="한 번에 파일에 flush할 배치 크기")
    parser.add_argument("--max-concurrent", type=int, default=32,
                        help="동시 요청 수 (semaphore)")
    parser.add_argument("--max-tokens", type=int, default=4096,
                        help="max_model_len보다 작아야 함 (입력 토큰 공간 확보)")
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--resume", action="store_true",
                        help="기존 출력 파일이 있으면 이어서 처리")
    args = parser.parse_args()

    # 데이터 읽기
    print(f"Reading questions from {args.input} (max_rows={args.max_rows}) ...")
    questions = read_questions(args.input, args.max_rows)
    print(f"  Loaded {len(questions)} questions.")

    # 이어하기: 이미 처리된 항목 제외
    done_ids: set[int] = set()
    write_header = True
    if args.resume:
        done_ids = load_existing_ids(args.output)
        if done_ids:
            print(f"  Resuming — skipping {len(done_ids)} already-processed rows.")
            write_header = False
    questions = [q for q in questions if q["id"] not in done_ids]
    print(f"  {len(questions)} questions to process.")

    if not questions:
        print("Nothing to do.")
        return

    # 클라이언트 생성
    client = AsyncOpenAI(base_url=args.base_url, api_key=args.api_key)
    semaphore = asyncio.Semaphore(args.max_concurrent)

    # 출력 파일 준비 (QUOTE_ALL: solution 내 쉼표/줄바꿈 안전하게 처리)
    out_file = open(args.output, "a" if not write_header else "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(out_file, fieldnames=["id", "r1_solution"], quoting=csv.QUOTE_ALL)
    if write_header:
        writer.writeheader()

    total = len(questions)
    processed = 0
    t_start = time.time()

    # 배치 단위 처리
    for batch_start in range(0, total, args.batch_size):
        batch = questions[batch_start : batch_start + args.batch_size]
        results = await process_batch(
            client, args.model, batch, semaphore, args.max_tokens, args.temperature
        )

        # 결과를 즉시 파일에 기록
        for r in results:
            writer.writerow(r)
        out_file.flush()

        processed += len(batch)
        elapsed = time.time() - t_start
        speed = processed / elapsed if elapsed > 0 else 0
        print(
            f"  [{processed}/{total}] "
            f"{processed / total * 100:.1f}% | "
            f"{speed:.1f} rows/s | "
            f"elapsed {elapsed:.0f}s"
        )

    out_file.close()
    elapsed_total = time.time() - t_start
    print(f"\nDone! {processed} solutions saved to {args.output} in {elapsed_total:.1f}s")


if __name__ == "__main__":
    asyncio.run(main())
