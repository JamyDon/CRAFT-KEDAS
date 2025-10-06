import json


def convert_dataset(data):
    results = []
    count = 0

    for item in data:
        context = item[0][0]
        questions = item[1]

        for q in questions:
            question = q["question"]
            choices = q["choice"]
            answer_text = q["answer"]

            # 答案位置 -> 字母
            try:
                answer_index = choices.index(answer_text)
            except ValueError:
                raise ValueError(f"答案 `{answer_text}` 不在选项 {choices} 中")

            answer_letter = chr(ord('A') + answer_index)

            # 构造选项字符串
            choice_str = ""
            for i, c in enumerate(choices):
                choice_str += f"{chr(ord('A')+i)}. {c}\n"

            prompt = (
                "请根据以下材料回答问题，并且只输出选项对应的字母。\n\n"
                f"材料：{context}\n\n"
                f"问题：{question}\n"
                f"选项：\n{choice_str}\n"
                "请直接给出正确答案的字母。"
            )

            results.append({
                "prompt": prompt,
                "answer": answer_letter
            })
            count += 1

    return results, count


if __name__ == "__main__":
    input_files = ["c3-m-train.json", "c3-m-dev.json", "c3-m-test.json"]
    all_converted = []
    total_num = 0

    for file in input_files:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)

        converted, num = convert_dataset(data)  # 这里用你已有的函数
        all_converted.extend(converted)
        total_num += num
        print(f"📂 {file} 已转换 {num} 条")

    with open("converted_c3_data.json", "w", encoding="utf-8") as f:
        json.dump(all_converted, f, ensure_ascii=False, indent=2)

    print(f"✅ 共生成 {total_num} 条更新的问题，已保存到 converted_c3_data.json")