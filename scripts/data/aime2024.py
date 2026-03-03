import json
import argparse
from tqdm import tqdm
from datasets import load_dataset

def load():
    """
    加载SuperGPQA数据集并格式化为标准多项选择题格式。
    """
    dataset = load_dataset("/mnt/llm-train/users/explore-train/qingyu/.cache/SuperGPQA", split="train")
    for idx, row in enumerate(tqdm(dataset, desc="Loading")):
        question = row.get("question", "")
        options = row.get("options", [])
        if options:
            # 选项按照A., B., C., ...格式拼接
            options_text = "\n".join(
                f"{chr(ord('A')+i)}. {opt}" for i, opt in enumerate(options)
            )
            # 指示只用选项字母作答
            prompt = (
                f"{question}\n\nOptions:\n{options_text}\n\n"
                "Choose an answer. Answer with \\boxed{{OPTION}}"
            )
        else:
            prompt = f"{question}\nChoose an answer. Answer with \\boxed{{OPTION}}"

        yield {
            "prompt": prompt,
            # label保持为answer_letter，例如"A"
            "label": row.get("answer_letter", ""),
        }

def main():
    """
    解析命令行参数，将SuperGPQA数据集导出为JSONL文件。
    """
    parser = argparse.ArgumentParser(description="Export SuperGPQA dataset to JSONL.")
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="输出JSONL文件路径"
    )
    args = parser.parse_args()

    records = list(load())
    with open(args.output, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"已写入{len(records)}行至{args.output}")

if __name__ == "__main__":
    main()