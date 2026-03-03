import argparse
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple


TARGET_DATASETS = [
    "mmlu",
    "mmlu_pro",
    "cmmlu",
    "ceval",
    "supergpqa",
]

AGIEVAL_CLOZE_SETS = {"gaokao-mathcloze", "math"}
AGIEVAL_CHINESE_SETS = {
    "gaokao-chinese",
    "gaokao-english",
    "gaokao-geography",
    "gaokao-history",
    "gaokao-biology",
    "gaokao-chemistry",
    "gaokao-physics",
    "gaokao-mathqa",
    "logiqa-zh",
    "gaokao-mathcloze",
}
AGIEVAL_INTRO = {
    "gaokao-chinese": "以下是一道中国高考语文选择题，请选择正确的答案。",
    "gaokao-english": "以下是一道中国高考英语选择题，请选择正确的答案。",
    "gaokao-geography": "以下是一道中国高考地理选择题，请选择正确的答案。",
    "gaokao-history": "以下是一道中国高考历史选择题，请选择正确的答案。",
    "gaokao-biology": "以下是一道中国高考生物选择题，请选择正确的答案。",
    "gaokao-chemistry": "以下是一道中国高考化学选择题，请选择正确的答案。",
    "gaokao-physics": "以下是一道中国高考物理选择题，请选择正确的答案。",
    "gaokao-mathqa": "以下是一道中国高考数学选择题，请选择正确的答案。",
    "logiqa-zh": "以下是一道中国公务员考试题，请选择正确的答案。",
    "lsat-ar": "The following is a LSAT Analytical Reasoning question. Please select the correct answer.",
    "lsat-lr": "The following is a LSAT Logical Reasoning question. Please select the correct answer.",
    "lsat-rc": "The following is a LSAT Reading Comprehension question. Please select the correct answer.",
    "logiqa-en": "The following is a Logic Reasoning question. Please select the correct answer.",
    "sat-math": "The following is a SAT Math question. Please select the correct answer.",
    "sat-en": "The following is a SAT English question. Please select the correct answer.",
    "sat-en-without-passage": "The following is a SAT English question. Please select the correct answer.",
    "aqua-rat": "The following is a AQUA-RAT question. Please select the correct answer.",
    "jec-qa-kd": "以下是一道中国司法考试基础知识题，请选择正确的答案。",
    "jec-qa-ca": "以下是一道中国司法考试案例分析题，请选择正确的答案。",
    "gaokao-mathcloze": "以下是一道中国高考数学填空题，请填入正确的答案。",
    "math": "The following is a Math question. Please select the correct answer.",
}


def iter_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path} at line {line_no}: {exc}") from exc


def write_jsonl(path: Path, records: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def drop_internal_fields(item: Dict) -> Dict:
    return {k: v for k, v in item.items() if not k.startswith("__")}


def normalize_text(text: str) -> str:
    return " ".join(text.split()).strip().lower()


def extract_question_text(item: Dict) -> str:
    for field in ("question", "prompt", "problem", "input", "query", "instruction"):
        value = item.get(field)
        if isinstance(value, str) and value.strip():
            return value
    return json.dumps(item, ensure_ascii=False, sort_keys=True)


def strip_common_suffix(text: str) -> str:
    # Remove shared answer-format suffix templates used by multiple datasets.
    patterns = [
        r"\s*choose an answer in a,b,c,d\.\s*answer with\\boxed\{\{a\}\},\\boxed\{\{b\}\},\\boxed\{\{c\}\}, or\\boxed\{\{d\}\}\.?\s*$",
        r"\s*choose an answer,\s*answer with\\boxed\{\{your option\}\}\.?\s*$",
        r"\s*choose an answer\.\s*answer with\\boxed\{\{option\}\}\.?\s*$",
        r"\s*choose an answer.*?answer with\\boxed.*$",
    ]
    cleaned = text
    for pattern in patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def loose_clean(text: str) -> str:
    text = strip_common_suffix(text)
    # Remove common option/instruction suffix templates for more robust cross-dataset matching.
    patterns = [
        r"\boptions\s*:",
        r"\bchoose an answer\b",
        r"\banswer with\b",
        r"选项\s*[:：]",
        r"请选择",
        r"请从.*?中选择",
    ]
    cleaned = text
    for pattern in patterns:
        parts = re.split(pattern, cleaned, flags=re.IGNORECASE, maxsplit=1)
        cleaned = parts[0]
    return normalize_text(cleaned)


def question_key(item: Dict, mode: str) -> str:
    text = strip_common_suffix(extract_question_text(item))
    if mode == "strict":
        return normalize_text(text)
    if mode == "loose":
        return loose_clean(text)
    raise ValueError(f"Unsupported match mode: {mode}")


def normalize_option_text(option: str) -> str:
    text = option.strip()
    # Normalize "(A) xxx", "A) xxx", "A. xxx" into "A. xxx".
    matched = re.match(r"^\(?\s*([A-Z])\s*[\)\.\:]?\s*(.*)$", text)
    if matched:
        letter, content = matched.group(1), matched.group(2).strip()
        if content:
            return f"{letter}. {content}"
        return f"{letter}."
    return text


def build_question_with_options(item: Dict) -> str:
    dataset_name = str(item.get("__dataset_name", "")).strip()
    intro = AGIEVAL_INTRO.get(dataset_name, "")
    hint = "答案是： " if dataset_name in AGIEVAL_CHINESE_SETS else "The answer is "

    question = str(item.get("question", "")).strip() or extract_question_text(item).strip()
    passage = item.get("passage")
    if isinstance(passage, str) and passage.strip() and passage.strip().lower() != "none":
        question = f"{passage.strip()}\n{question}"
    question = strip_common_suffix(question)

    if dataset_name in AGIEVAL_CLOZE_SETS:
        return "\n".join([x for x in (intro, question, hint) if x])

    options = item.get("options")
    if not isinstance(options, list) or not options:
        return "\n".join([x for x in (intro, question, hint) if x])

    normalized_options: List[str] = []
    for idx, option in enumerate(options):
        if not isinstance(option, str):
            continue
        normalized = normalize_option_text(option)
        # Ensure fallback option letter if source option has no prefix.
        if not re.match(r"^[A-Z]\.\s*", normalized):
            normalized = f"{chr(ord('A') + idx)}. {normalized}"
        normalized_options.append(normalized)

    if not normalized_options:
        return "\n".join([x for x in (intro, question, hint) if x])
    return "\n".join([x for x in (intro, question, "\n".join(normalized_options), hint) if x])


def build_answer_text(item: Dict) -> str:
    for key in ("label", "answer"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def to_sharegpt_records(records: List[Dict]) -> List[Dict]:
    out: List[Dict] = []
    for idx, item in enumerate(records):
        human_text = build_question_with_options(item)
        answer_text = build_answer_text(item)
        out.append(
            {
                "id": f"v1_1_{idx}",
                "conversations": [
                    {"from": "human", "value": human_text},
                    {"from": "gpt", "value": answer_text},
                ],
            }
        )
    return out


def merge_v1_1_jsonl(v1_dir: Path, merged_output: Path) -> Tuple[List[Dict], List[Path]]:
    files = sorted(v1_dir.glob("*.jsonl"))
    merged_records: List[Dict] = []
    for file_path in files:
        dataset_name = file_path.stem
        for item in iter_jsonl(file_path):
            record = dict(item)
            record["__dataset_name"] = dataset_name
            merged_records.append(record)
    write_jsonl(merged_output, [drop_internal_fields(x) for x in merged_records])
    return merged_records, files


def collect_nano_eval_keys(
    nano_eval_dir: Path,
    match_mode: str,
) -> Tuple[Set[str], Dict[str, Dict[str, int]]]:
    all_keys: Set[str] = set()
    stats: Dict[str, Dict[str, int]] = {}

    for name in TARGET_DATASETS:
        input_path = nano_eval_dir / f"{name}.jsonl"
        if not input_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {input_path}")

        total = 0
        unique_in_dataset: Set[str] = set()
        for item in iter_jsonl(input_path):
            total += 1
            key = question_key(item, match_mode)
            unique_in_dataset.add(key)
            all_keys.add(key)

        stats[name] = {
            "total": total,
            "unique": len(unique_in_dataset),
            "duplicated_inside_dataset": total - len(unique_in_dataset),
        }

    return all_keys, stats


def dedup_v1_by_nano_keys(
    v1_records: List[Dict], nano_keys: Set[str], match_mode: str
) -> Tuple[List[Dict], int]:
    kept: List[Dict] = []
    removed = 0
    for item in v1_records:
        key = question_key(item, match_mode)
        if key in nano_keys:
            removed += 1
            continue
        kept.append(item)
    return kept, removed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Merge all JSONL files in v1_1, then remove records that overlap with five "
            "nano_eval datasets (mmlu, mmlu_pro, cmmlu, ceval, supergpqa)."
        )
    )
    parser.add_argument(
        "--v1-dir",
        type=Path,
        default=Path("outputs/v1_1"),
        help="Directory containing v1_1 jsonl files.",
    )
    parser.add_argument(
        "--nano-dir",
        type=Path,
        default=Path("outputs/nano_eval"),
        help="Directory containing nano_eval jsonl files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("outputs/dedup"),
        help="Output directory for merged and deduplicated files.",
    )
    parser.add_argument(
        "--merged-file",
        type=str,
        default="v1_1_merged.jsonl",
        help="Filename for merged v1_1 output in out-dir.",
    )
    parser.add_argument(
        "--new-v1-file",
        type=str,
        default="v1_1_dedup_by_nano_eval.jsonl",
        help="Filename for the new v1_1 after removing overlap with nano_eval 5 sets.",
    )
    parser.add_argument(
        "--match-mode",
        choices=["strict", "loose"],
        default="loose",
        help=(
            "Question matching mode: strict uses raw normalized text; "
            "loose also strips common options/instruction suffixes."
        ),
    )
    parser.add_argument(
        "--sharegpt-file",
        type=str,
        default="v1_1_dedup_by_nano_eval_sharegpt.jsonl",
        help="Filename for ShareGPT-formatted output in out-dir.",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    merged_output = args.out_dir / args.merged_file
    new_v1_output = args.out_dir / args.new_v1_file
    sharegpt_output = args.out_dir / args.sharegpt_file

    merged_records, merged_files = merge_v1_1_jsonl(args.v1_dir, merged_output)
    nano_keys, stats = collect_nano_eval_keys(args.nano_dir, args.match_mode)
    new_v1_records, removed = dedup_v1_by_nano_keys(
        merged_records, nano_keys, args.match_mode
    )
    write_jsonl(new_v1_output, [drop_internal_fields(x) for x in new_v1_records])
    sharegpt_records = to_sharegpt_records(new_v1_records)
    write_jsonl(sharegpt_output, sharegpt_records)

    print(f"Merged {len(merged_files)} files from {args.v1_dir} -> {merged_output}")
    print(f"Match mode: {args.match_mode}")
    print(f"Total merged records: {len(merged_records)}")
    print("Nano_eval 5-set key stats:")
    for dataset, item in stats.items():
        print(
            f"  {dataset}: total={item['total']}, unique={item['unique']}, "
            f"duplicated_inside_dataset={item['duplicated_inside_dataset']}"
        )
    print(f"Unique keys from all 5 sets: {len(nano_keys)}")
    print(f"Removed from v1_1 due to overlap: {removed}")
    print(f"New v1_1 records: {len(new_v1_records)}")
    print(f"New v1_1 output: {new_v1_output}")
    print(f"ShareGPT output: {sharegpt_output}")


if __name__ == "__main__":
    main()
