import pandas as pd
from pathlib import Path
from datasets import Dataset
import json
import os
import random, json
from typing import List, Dict, Any, Union

AG_LABEL_MAP = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech"
}

import json
import re
from typing import Any, Dict, List

import re
from typing import List, Optional

def extract_json_codeblocks(text: str, first_only: bool = True) -> Optional[str] | List[str]:
    """
    提取 ```json ... ``` 内的内容；若没有 `json` 标签，则回退到任意 ``` ... ```。
    - first_only=True  返回第一个匹配的字符串或 None
    - first_only=False 返回所有匹配的列表（可能为空列表）
    """
    # 优先匹配 ```json ... ```
    pattern_json = re.compile(r"```json\s*([\s\S]*?)```", re.IGNORECASE)
    matches = pattern_json.findall(text)
    if not matches:
        # 回退到普通 ``` ... ```
        pattern_any = re.compile(r"```(?:[\w+-]+)?\s*([\s\S]*?)```", re.IGNORECASE)
        matches = pattern_any.findall(text)

    if first_only:
        return matches[0].strip() if matches else None
    return [m.strip() for m in matches]



def extract_all_json_dicts(text: str) -> List[Dict[str, Any]]:
    """提取所有代码块并转成 dict 列表。"""
    blocks = extract_json_codeblocks(text, first_only=False)
    dicts = []
    for b in blocks:
        b = b.replace("{{", "{").replace("}}", "}")
        dicts.append(json.loads(b))
    return dicts


def preprocess_agnews_to_sft(train_csv: str,
                             test_csv: str,
                             instruction_tmpl: str = "Based on the following news title and brief, determine which category it belongs to. The available categories are: {label_list}. Output only the category name. {input}",
                             output_tmpl: str = "{label}",
                             rename_to_prompt_response: bool = False):
    """
    将 AG News CSV 转为 HF SFT json（每行一个 dict）。
    生成的字段：instruction / input / output
    可选：重命名为 prompt / response（prepare_dataset_for_sft 如果需要）。

    Parameters
    ----------
    train_csv : str
    test_csv  : str
    train_json_out : str
    test_json_out  : str
    instruction_tmpl : str
    output_tmpl      : str
    rename_to_prompt_response : bool
        若 True，则把 instruction->prompt, output->response, input 保留或丢弃按需处理
    """

    def _build_record(title, desc, label_id):
        label = AG_LABEL_MAP[int(label_id)]
        inp = f"\n Title: {title}\Description: {desc}"
        instruction = instruction_tmpl.format(label_list=", ".join(AG_LABEL_MAP.values()), input=inp)
        out = output_tmpl.format(label=label)
        rec = {
            "instruction": instruction+inp,
            "output": out
        }
        if rename_to_prompt_response:
            rec = {
                "prompt": instruction + "\n" + inp,  # 如果 prepare_dataset_for_sft 只吃一个字段，可合并
                "response": out
            }
        return rec

    def _csv_to_records(csv_path):
        df = pd.read_csv(csv_path)
        if "Class Index" in df.columns:  # Kaggle官方CSV格式
            df = df.rename(columns={
                "Class Index": "label",
                "Title": "title",
                "Description": "description"
            })
        else:  # 没有表头的版本（比如HF上常见的）
            df.columns = ["label", "title", "description"]
        df["label"] = df["label"].astype(int)
        return [_build_record(r.title, r.description, r.label) for r in df.itertuples(index=False)]

    # 生成并保存
    train_records = _csv_to_records(train_csv)
    test_records  = _csv_to_records(test_csv)

    out_dir = Path(train_csv).parent
    with open(out_dir / "train.json", "w", encoding="utf-8") as f:
        json.dump(train_records, f, ensure_ascii=False, indent=4)
    with open(out_dir / "test.json", "w", encoding="utf-8") as f:
        json.dump(test_records, f, ensure_ascii=False, indent=4)

    # 如需直接得到 HF Dataset 也可返回
    train_ds = Dataset.from_list(train_records)
    test_ds  = Dataset.from_list(test_records)
    return train_ds, test_ds


def generate_new_data(model,
                      prompt: str,
                      dataset: str,
                      examples: List[Dict[str, str]],
                      max_new_tokens: int = 1024,
                      temperature: float = 0.8,
                      num_samples: int = 10000) -> List[Dict[str, Any]]:
    label_list = ""
    if dataset == "ag_news":
        label_list = ", ".join(AG_LABEL_MAP.values())
    
    samples = []
    for i in range(num_samples):
        input = prompt.format(
            label_list=label_list,
            examples=select_examples(examples, num_examples=2, seed=42),
        )
        inputs = model.tokenizer(input, return_tensors="pt").to(model.model.device)

        outputs = model.model.generate(
            inputs.input_ids,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.8,
            top_p=0.95,
        )
        
        generated_text = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
        new_samples = extract_all_json_dicts(generated_text.strip())
        # samples add
        for sample in new_samples:
            if isinstance(sample, dict):
                # 确保每个样本都是 dict
                if "instruction" in sample and "output" in sample:
                    samples.append(sample)
                else:
                    # 如果没有 instruction/output 字段，可能是格式不对，跳过
                    continue
            else:
                # 如果不是 dict，可能是格式不对，跳过
                continue
        

    return samples
    
    
def select_examples(dataset: Union[List[Dict[str, Any]], Dataset],
                    num_examples: int = 5,
                    seed: int = 42,
                    escape_braces: bool = True,
                    join: bool = True) -> Union[str, List[str]]:
    """
    随机抽样若干条，并把每条 dict 转成 JSON 字符串。
    - escape_braces=True 时，把 { 和 } 变成 {{ 和 }}，用于后续 str.format 安全填充。
    - join=True 时返回一个多行字符串，否则返回字符串列表。
    """

    rng = random.Random(seed)

    # 处理 HF Dataset / list
    if  isinstance(dataset, Dataset):
        idxs = rng.sample(range(len(dataset)), min(num_examples, len(dataset)))
        picked = [dataset[i] for i in idxs]
    else:
        picked = rng.sample(dataset, min(num_examples, len(dataset)))

    def to_str(d: Dict[str, Any]) -> str:
        s = json.dumps(d, ensure_ascii=False, separators=(",", ":"))
        if escape_braces:
            s = s.replace("{", "{{").replace("}", "}}")
        return s

    strs = [to_str(ex) for ex in picked]
    return "\n".join(strs) + ("\n" if join else "") if join else strs