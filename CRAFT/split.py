import json

def split_json(input_file, output_file1, output_file2, num_samples=4000):
    # 读取 c3.json
    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    # 切分数据
    subset1 = data[:num_samples]
    subset2 = data[num_samples:]

    # 保存前 num_samples 条
    with open(output_file1, "w", encoding="utf-8") as f:
        json.dump(subset1, f, ensure_ascii=False, indent=2)
    
    # 保存剩余条目
    with open(output_file2, "w", encoding="utf-8") as f:
        json.dump(subset2, f, ensure_ascii=False, indent=2)

    print(f"✅ 已保存 {len(subset1)} 条到 {output_file1}")
    print(f"✅ 已保存 {len(subset2)} 条到 {output_file2}")

if __name__ == "__main__":
    # split_c3_json("converted_c3_data.json", "c3_first5000.json", "c3_rest5000.json", num_samples=5000)
    # split_json("RTSF-Statistical.json", "RTSF-Statistical-test500.json", "RTSF-Statistical-train.json", num_samples=500)
    split_json("RTSF-Financial.json", "RTSF-Financial-test500.json", "RTSF-Financial-train.json", num_samples=500)
    