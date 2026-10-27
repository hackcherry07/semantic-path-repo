import re
import json
import argparse
import os

try:
    from openai import OpenAI  # pyright: ignore[reportMissingImports]
except ImportError:  # [added] 没装 openai 库时的最小替代, 接口与下面用到的 client.chat.completions.create 一致
    import urllib.request
    from types import SimpleNamespace

    class OpenAI:
        def __init__(self, api_key=None, base_url=None):
            self.api_key, self.base_url = api_key, base_url.rstrip("/")
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

        def _create(self, model, messages, **kw):
            body = json.dumps({"model": model, "messages": messages, **kw}).encode("utf-8")
            req = urllib.request.Request(self.base_url + "/chat/completions", data=body,
                                         headers={"Content-Type": "application/json",
                                                  "Authorization": f"Bearer {self.api_key}"})
            with urllib.request.urlopen(req, timeout=900) as r:
                d = json.load(r)
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(
                content=d["choices"][0]["message"]["content"]))])

try:
    from tqdm import tqdm  # pyright: ignore[reportMissingModuleSource]
except ImportError:  # [added] 没装 tqdm 时的最小替代
    class tqdm:
        def __init__(self, total=None, desc=None, unit=None): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def update(self, n=1): pass

# [removed] control_flow / retrieval / pandas 的 import: 只做到 Semantic Path, 用不到

# [changed] 原: client = OpenAI(api_key="YOUR_API_KEY", base_url="https://api.poixe.com/v1")
#           现: 指向本地 Ollama 的 OpenAI 兼容接口 (Qwen 7B), 其余调用方式不变
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5:7b")
client = OpenAI(api_key=os.environ.get("LLM_API_KEY", "ollama"),
                base_url=os.environ.get("LLM_BASE_URL", "http://localhost:11434/v1"))

user_prompt1_base = (
    "You are a senior software engineer and static code analyst.\n"
    "Your task is to understand the functionality of the given source code.\n\n"

    "Instructions:\n"
    "1. Carefully analyze the code to determine its main functionality.\n"
    "2. Identify important operations such as input handling, memory operations, file operations, or system calls.\n"
    "3. Describe how data flows through the program when possible.\n"
    "4. Think through the logic internally step-by-step, but DO NOT output your reasoning process.\n"
    "5. Only describe behavior that can be directly inferred from the code.\n"
    "6. Do NOT speculate about functionality that is not visible in the code.\n"
    "7. Do NOT repeat or quote the code.\n"
    "8. Do NOT use markdown, bullet points, or special formatting.\n"
)

user_prompt1_concise = (
    user_prompt1_base +
    "9. Keep the summary concise and factual.\n\n"
    "If the code is incomplete or unclear, describe only the observable behavior rather than guessing the intended functionality.\n\n"
)

user_prompt1_normal = (
    user_prompt1_base +
    "\nIf the code is incomplete or unclear, describe only the observable behavior rather than guessing the intended functionality.\n\n"
)

user_prompt1_detailed = (
    user_prompt1_base +
    "9. A detailed, complete code description is required.\n\n"
    "If the code is incomplete or unclear, describe only the observable behavior rather than guessing the intended functionality.\n\n"
)

# [removed] user_prompt2_max2 / user_prompt2_max4 (Knowledge Path)
# [removed] vuln_detection_prompt_pair / vuln_detection_prompt_single (最终判定)

def call_gpt(prompt):
    try:
        completion = client.chat.completions.create(
            model=LLM_MODEL,  # [changed] 原: "gpt-4o-2024-11-20"
            messages=[
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            seed=42,
            max_tokens=1024,
        )

        return completion.choices[0].message.content

    except Exception as e:
        return f"调用失败：{str(e)}"


def read_jsonl_iteratively(file_path):
    """
    迭代读取 JSONL 文件，每次返回两条数据的 target 和 func 字段
    :param file_path: JSONL 文件路径
    :yield: 包含两个字典的元组，每个字典包含 target 和 func
    """
    buffer = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            # 跳过空行
            line = line.strip()
            if not line:
                continue

            try:
                # 解析单行 JSON
                data = json.loads(line)
                # 提取 target 和 func 字段（不存在则返回 None）
                result = {
                    "target": data.get("target"),
                    "func": data.get("func")
                }
                buffer.append(result)

                # 当缓冲区有两条数据时，一起产出
                if len(buffer) == 2:
                    yield buffer[0], buffer[1]
                    buffer = []

            except json.JSONDecodeError as e:
                # 捕获 JSON 解析错误，避免程序中断
                print(f"第 {line_num} 行 JSON 解析失败：{e}")
                continue

        # 处理最后剩余的单条数据（如果总数为奇数）
        if len(buffer) == 1:
            print(f"警告：文件末尾剩余 1 条数据，无法配对")


def count_lines_in_jsonl(file_path):
    """
    统计 JSONL 文件的有效行数（非空行）
    :param file_path: 文件路径
    :return: 有效行数
    """
    count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
    return count


# [removed] process_string_to_array / get_knowledge / extract_preds / calculate_metrics (后面的步骤)


def load_done(out_file):
    """[added] 断点续跑: 读出已经完成的 pair 序号"""
    done = set()
    if os.path.exists(out_file):
        with open(out_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                if r.get("understanding1") and r.get("understanding2"):
                    done.add(r["pair_idx"])
    return done


# ======================================================================
# [added] 失败日志 (独立的 txt 文件). 只追加记录, 不影响上面/下面任何原有逻辑;
#         日志写入出错也只会被忽略, 不会中断主流程.
# ======================================================================
FAILURE_LOG = os.environ.get("FAILURE_LOG", "semantic_failures.txt")

_ERR_RULES = [
    (("connection refused", "actively refused", "connection error", "failed to establish", "urlopen error"),
     "Ollama is not running (run: ollama serve) or wrong base_url"),
    (("timed out", "timeout"), "request timed out (long function / slow CPU)"),
    (("404", "not found"), "model name not found (LLM_MODEL must match `ollama list`)"),
    (("500", "runner", "out of memory", "oom", "cuda"), "server error / out of memory (function too long?)"),
    (("429", "rate limit"), "rate limited"),
    (("401", "403", "api key", "unauthorized"), "authentication problem"),
]


def _failure_reason(text):
    low = text.lower()
    for keys, why in _ERR_RULES:
        if any(k in low for k in keys):
            return why
    return "unknown error"


def log_failure(pair_idx, func_no, kind, detail="", code="", label=None):
    """追加一行到失败日志. kind: CALL_FAILED / EMPTY_REPLY / EMPTY_FUNC"""
    try:
        import time
        reason = _failure_reason(detail) if kind == "CALL_FAILED" else ""
        with open(FAILURE_LOG, "a", encoding="utf-8") as lf:
            lf.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} | pair {pair_idx} | {func_no} | {kind} | "
                     f"label={label} | model={LLM_MODEL}"
                     + (f" | reason: {reason}" if reason else "")
                     + (f" | raw: {detail[:300]!r}" if detail else "")
                     + f" | code: {code[:80].strip()!r}\n")
    except Exception:
        pass


def evaluate_jsonl_file_pair_wise(jsonl_file="primevul_train_paired_cur.jsonl", desc_level="concise",
                                  out_file="semantic_out.jsonl", max_pairs=0, start=0):
    """
    [changed] 原函数: 完整流程 (Agent1 语义描述 -> Agent2 漏洞查询 -> 知识检索 -> 图信息 -> 最终判定 -> 指标)
              现在: 只执行 Agent1 (Semantic Path), 结果写入 out_file
    Args:
        jsonl_file: JSONL文件路径
        max_pairs / start: [added] 只处理一部分样本对 (0 = 全部)
    """
    total_cases = count_lines_in_jsonl(jsonl_file)
    done = load_done(out_file)  # [added]

    results = []
    with tqdm(total=total_cases / 2, desc="处理样本对", unit="item") as pbar:
        with open(out_file, "a", encoding="utf-8") as fout:  # [added]
            for pair_idx, (item1, item2) in enumerate(read_jsonl_iteratively(jsonl_file)):
                if pair_idx < start:  # [added]
                    continue
                if max_pairs and pair_idx - start >= max_pairs:  # [added]
                    break
                if pair_idx in done:  # [added]
                    print(f"[pair {pair_idx}] already done, skip")
                    continue

                code_content1, label1 = item1['func'], item1['target']
                code_content2, label2 = item2['func'], item2['target']

                if code_content1 and code_content2:
                    # ---------- Semantic Path (原 Agent 1) ----------
                    if desc_level == "concise":
                        user_prompt1 = user_prompt1_concise
                    elif desc_level == "normal":
                        user_prompt1 = user_prompt1_normal
                    elif desc_level == "detailed":
                        user_prompt1 = user_prompt1_detailed

                    agent1_prompt1 = user_prompt1 + f"Target Code:\n```\n{code_content1}\n```"
                    agent1_prompt2 = user_prompt1 + f"Target Code:\n```\n{code_content2}\n```"

                    understanding1 = call_gpt(agent1_prompt1)
                    understanding2 = call_gpt(agent1_prompt2)
                    # ---------- 到此为止 ----------

                    # [added] call_gpt 失败时返回 "调用失败：..." 字符串, 这里识别出来, 不当成描述保存
                    failed = understanding1.startswith("调用失败") or understanding2.startswith("调用失败")
                    # [added] 记录失败原因到 txt (不影响上面的结果保存)
                    for _k, _u, _c, _l in (("func1", understanding1, code_content1, label1),
                                           ("func2", understanding2, code_content2, label2)):
                        if _u.startswith("调用失败"):
                            log_failure(pair_idx, _k, "CALL_FAILED", _u, _c, _l)
                        elif not _u.strip():
                            log_failure(pair_idx, _k, "EMPTY_REPLY", "", _c, _l)
                    record = {
                        "pair_idx": pair_idx, "desc_level": desc_level, "model": LLM_MODEL,
                        "target1": label1, "func1": code_content1,
                        "understanding1": None if understanding1.startswith("调用失败") else understanding1,
                        "target2": label2, "func2": code_content2,
                        "understanding2": None if understanding2.startswith("调用失败") else understanding2,
                    }
                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                    fout.flush()
                    results.append(record)
                    print(f"[pair {pair_idx}] {'FAILED' if failed else 'ok'}  t1={label1} t2={label2} | "
                          f"{(understanding1 or '')[:90]!r}")

                else:  # [added] 有函数为空, 原逻辑会静默跳过这一对, 这里记一笔
                    _k = "func1" if not code_content1 else "func2"
                    log_failure(pair_idx, _k, "EMPTY_FUNC", "", "", label1 if _k == "func1" else label2)

                pbar.update(1)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Code Vulnerability Detection - Semantic Path only")

    parser.add_argument("--input_file", type=str, required=True,
                        help="Input file path (PrimeVul paired JSONL)")
    parser.add_argument("--desc_level", type=str, default="concise", choices=["concise", "normal", "detailed"],
                        help="Description level: concise, normal, detailed")
    # [added] 下面三个参数
    parser.add_argument("--out_file", type=str, default="semantic_out.jsonl", help="Where to save the descriptions")
    parser.add_argument("--n", type=int, default=5, help="Number of PAIRS to process (0 = all)")
    parser.add_argument("--start", type=int, default=0, help="First pair index")

    args = parser.parse_args()

    evaluate_jsonl_file_pair_wise(
        jsonl_file=args.input_file,
        desc_level=args.desc_level,
        out_file=args.out_file,
        max_pairs=args.n,
        start=args.start,
    )
