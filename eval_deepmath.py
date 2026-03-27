"""
DeepMath-103k 평가 스크립트
- vLLM API 서버를 통해 비동기 병렬 생성 + 중간 저장 (generation cache jsonl)
- id별 정답 여부 (accuracy_reward_old 방식) + process_reward 계산 (Ray 병렬)
- Resume 기능: 생성/평가 모두 이미 처리된 id는 건너뜀

사용법:
  # 1) vLLM 서버 먼저 실행
  # vllm serve <model> --port 8000

  # 2) 스크립트 실행
  # python eval_deepmath.py --api_base http://localhost:8000/v1 --model <model>
"""

import os
import re
import json
import asyncio
import fire
import ray
import pandas as pd
from tqdm import tqdm
from openai import AsyncOpenAI
from datasets import load_dataset
from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify

from pm.miner import Miner
from pm.checker import Checker
import reward
import pm


def split_think_and_answer(text):
    parts = re.split(r'</think>', text, maxsplit=1, flags=re.DOTALL)
    if len(parts) == 2:
        think_part = parts[0]
        answer = parts[1].strip()
        think = re.sub(r'^.*?<think>', '', think_part, flags=re.DOTALL).strip()
        return think, answer
    else:
        return text.strip(), ""


def compute_accuracy(response: str, gold_answer: str) -> float:
    """accuracy_reward_old 방식: math_verify로 파싱 후 verify, 실패 시 text match"""
    _, answer = split_think_and_answer(response)

    try:
        gold_parsed = parse(gold_answer)
    except Exception:
        gold_parsed = []

    if len(gold_parsed) != 0:
        try:
            answer_parsed = parse(
                answer,
                extraction_config=[
                    LatexExtractionConfig(
                        normalization_config=NormalizationConfig(units=True),
                        boxed_match_priority=0,
                        try_extract_without_anchor=False,
                    )
                ],
                extraction_mode="first_match",
            )
            return float(verify(gold_parsed, answer_parsed))
        except Exception:
            return 0.0
    else:
        return float(answer.strip().lower() == gold_answer.strip().lower())


def compute_process_reward(response: str, case_id: int, truelogs_df: pd.DataFrame) -> float:
    """process_reward_func 방식: think → eventlog → petri net → conformance check"""
    think, _ = split_think_and_answer(response)
    if not think.strip():
        return 0.0

    try:
        think_log = reward.Answer2EventAgent().make_event_log(think)
        if think_log is None or think_log is False:
            return 0.0
        think_log.log['Case ID'] = str(case_id)
        reason_net = Miner(think_log).discover()

        true_log_df = truelogs_df[truelogs_df['Case ID'] == case_id].copy()
        if true_log_df.empty:
            return 0.0
        true_log_df['Case ID'] = str(case_id)
        true_eventlog = pm.EventLog(true_log_df)

        conf_df = Checker(true_eventlog, reason_net).check()
        return conf_df['F1 Score'].values[0]
    except Exception as e:
        print(f"  [ERROR] process_reward failed for id={case_id}: {e}")
        return 0.0


@ray.remote
def evaluate_single(
    case_id: int,
    question: str,
    gold: str,
    response: str,
    skip_process_reward: bool,
    truelogs_df: pd.DataFrame,
) -> dict:
    """Ray remote: 단일 샘플에 대해 accuracy + process_reward 계산"""
    acc = compute_accuracy(response, gold)

    proc_reward = 0.0
    if not skip_process_reward:
        proc_reward = compute_process_reward(response, case_id, truelogs_df)

    return {
        'id': case_id,
        'question': question,
        'gold_answer': gold,
        'response': response,
        'accuracy': acc,
        'process_reward': proc_reward,
    }


# ── vLLM API 비동기 생성 ──

async def generate_one(
    client: AsyncOpenAI,
    model: str,
    row: dict,
    semaphore: asyncio.Semaphore,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> dict:
    """단일 문제에 대해 vLLM API로 응답 생성"""
    async with semaphore:
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": row['question']})

            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
            )
            text = response.choices[0].message.content or ""
        except Exception as e:
            print(f"[ERROR] id={row['id']}: {e}")
            text = ""

        return {
            'id': int(row['id']),
            'question': row['question'],
            'gold_answer': str(row['final_answer']),
            'response': text,
        }


async def generate_batch_async(
    client: AsyncOpenAI,
    model: str,
    batch: list[dict],
    semaphore: asyncio.Semaphore,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> list[dict]:
    """배치의 모든 요청을 비동기 병렬로 실행"""
    tasks = [
        generate_one(client, model, row, semaphore, system_prompt, max_tokens, temperature, top_p)
        for row in batch
    ]
    return await asyncio.gather(*tasks)


def run_generation(
    need_gen_rows: list[dict],
    generation_cache: str,
    generated: dict,
    api_base: str,
    api_key: str,
    model: str,
    system_prompt: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    max_concurrent: int,
    gen_batch_size: int,
):
    """vLLM API를 통해 비동기 배치 생성 + 즉시 캐시 저장"""

    async def _run():
        client = AsyncOpenAI(base_url=api_base, api_key=api_key)
        semaphore = asyncio.Semaphore(max_concurrent)
        total = len(need_gen_rows)
        processed = 0
        for batch_start in range(0, total, gen_batch_size):
            batch = need_gen_rows[batch_start:batch_start + gen_batch_size]
            results = await generate_batch_async(
                client, model, batch, semaphore,
                system_prompt, max_tokens, temperature, top_p,
            )

            # 즉시 cache에 append
            with open(generation_cache, 'a', encoding='utf-8') as f:
                for entry in results:
                    f.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    generated[entry['id']] = entry

            processed += len(batch)
            print(f"  Generated {processed}/{total} ({processed/total*100:.1f}%)")

    asyncio.run(_run())


def main(
    # vLLM API
    api_base: str = "http://localhost:8000/v1",
    api_key: str = "EMPTY",
    model: str = None,
    system_prompt: str = "Please reason step by step, and put your final answer within \\boxed{}.",

    # generation
    max_tokens: int = 16384,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_concurrent: int = 32,
    gen_batch_size: int = 256,

    # data (CSV or Hugging Face dataset)
    csv_path: str = None,
    dataset_name: str = None,
    dataset_split: str = "train",
    question_col: str = "question",
    answer_col: str = "final_answer",
    id_col: str = "id",
    start_idx: int = 0,
    end_idx: int = 5000,
    max_questions: int = None,

    # output
    output_path: str = "eval_deepmath_results.csv",
    generation_cache: str = None,

    # options
    skip_process_reward: bool = False,
    batch_save_interval: int = 50,
    ray_num_cpus: int = None,
):
    assert model is not None, "--model is required (e.g. the model name served by vLLM)"
    assert csv_path is not None or dataset_name is not None, \
        "Either --csv_path or --dataset_name must be provided"

    # generation cache 경로 자동 설정
    if generation_cache is None:
        generation_cache = output_path.replace('.csv', '_generations.jsonl')

    # ── Load data ──
    if dataset_name is not None:
        ds = load_dataset(dataset_name, split=dataset_split)
        df = ds.to_pandas()
        print(f"Loaded dataset '{dataset_name}' (split={dataset_split}), {len(df)} rows")
    else:
        df = pd.read_csv(csv_path)

    # 컬럼명 통일: question, final_answer, id
    col_map = {}
    if question_col != "question":
        col_map[question_col] = "question"
    if answer_col != "final_answer":
        col_map[answer_col] = "final_answer"
    if id_col != "id":
        col_map[id_col] = "id"
    if col_map:
        df = df.rename(columns=col_map)

    # id 컬럼이 없으면 자동 생성
    if "id" not in df.columns:
        df["id"] = range(len(df))

    if start_idx is not None and end_idx is not None:
        df = df.iloc[start_idx:end_idx].reset_index(drop=True)
    if max_questions is not None:
        df = df.head(max_questions).reset_index(drop=True)
    print(f"Loaded {len(df)} problems")

    # ── Phase 1: vLLM API 비동기 생성 (resume 지원) ──
    generated = {}
    if os.path.exists(generation_cache):
        with open(generation_cache, 'r', encoding='utf-8') as f:
            for line in f:
                entry = json.loads(line)
                generated[entry['id']] = entry
        print(f"Generation cache: {len(generated)} responses loaded")

    need_gen_rows = df[~df['id'].isin(generated.keys())].to_dict('records')

    if need_gen_rows:
        print(f"Generating {len(need_gen_rows)} responses via vLLM API ({api_base})...")
        run_generation(
            need_gen_rows, generation_cache, generated,
            api_base, api_key, model, system_prompt,
            max_tokens, temperature, top_p,
            max_concurrent, gen_batch_size,
        )
    else:
        print("All responses already generated.")

    # ── Phase 2: Ray 병렬 평가 (resume 지원) ──
    done_ids = set()
    existing_results = []
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        existing_df = pd.read_csv(output_path)
        done_ids = set(existing_df['id'].tolist())
        existing_results = existing_df.to_dict('records')
        print(f"Eval resume: {len(done_ids)} ids already evaluated")

    to_eval = [g for g in generated.values() if g['id'] not in done_ids]
    if not to_eval:
        print("All ids already evaluated. Nothing to do.")
        result_df = pd.DataFrame(existing_results)
        if not result_df.empty:
            avg_acc = result_df['accuracy'].mean()
            avg_proc = result_df['process_reward'].mean()
            print(f"\n=== Results ===")
            print(f"Total: {len(result_df)}, Avg Accuracy: {avg_acc:.4f}, Avg Process Reward: {avg_proc:.4f}")
        return

    print(f"Evaluating {len(to_eval)} samples with Ray...")
    ray.init(ignore_reinit_error=True, num_cpus=ray_num_cpus, num_gpus=0)

    truelogs_df = pd.read_csv('eventlogs/DeepMath_eventlog.csv')
    truelogs_ref = ray.put(truelogs_df)

    results = list(existing_results)
    futures = []
    for entry in to_eval:
        fut = evaluate_single.remote(
            entry['id'], entry['question'], entry['gold_answer'], entry['response'],
            skip_process_reward, truelogs_ref,
        )
        futures.append(fut)

    pending = list(futures)
    pbar = tqdm(total=len(pending), desc="Evaluating (Ray)")

    while pending:
        done, pending = ray.wait(pending, num_returns=min(batch_save_interval, len(pending)))
        batch_results = ray.get(done)
        results.extend(batch_results)
        pbar.update(len(batch_results))

        result_df = pd.DataFrame(results)
        result_df.to_csv(output_path, index=False)
        pbar.set_postfix(saved=len(results))

    pbar.close()
    ray.shutdown()

    # ── Summary ──
    result_df = pd.DataFrame(results)
    result_df.to_csv(output_path, index=False)

    avg_acc = result_df['accuracy'].mean()
    avg_proc = result_df['process_reward'].mean()
    print(f"\n=== Results ===")
    print(f"Total: {len(result_df)}")
    print(f"Avg Accuracy:       {avg_acc:.4f}")
    print(f"Avg Process Reward: {avg_proc:.4f}")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    fire.Fire(main)
