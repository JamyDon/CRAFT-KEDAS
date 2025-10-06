import json

def convert_to_knowedit_format(original_data, c3_data):
    knowedit_data = []
    
    # c3.json 按顺序取，两条一组
    c3_iter = iter(c3_data)

    for i, item in enumerate(original_data, 1):
        subject = item["entity"] + "的" + item["indicator"]

        # case 合并
        case = {
            "case_id": f"{i}",
            "subject": [subject, subject],
            "prompt": [item["prompt"][0], item["prompt"][1]],
            "target_new": [str(item["target_new"][0]), str(item["target_new"][1])],
            "portability": {
                "Subject_Aliasing": [
                    {
                        "prompt": item["portability"]["alias_portability_1"]["prompt"][0],
                        "ground_truth": [str(item["portability"]["alias_portability_1"]["ground_truth"][0])]
                    },
                    {
                        "prompt": item["portability"]["alias_portability_1"]["prompt"][1],
                        "ground_truth": [str(item["portability"]["alias_portability_1"]["ground_truth"][1])]
                    }
                ],
                "Reasoning": [
                    {
                        "prompt": item["portability"]["composite_portability"]["prompt"],
                        "ground_truth": [str(item["portability"]["composite_portability"]["ground_truth"])]
                    }
                ]
            },
            "locality": {
                "Temporal": [
                    {
                        "prompt": item["locality"]["retention"]["prompt"][0],
                        "ground_truth": [str(item["locality"]["retention"]["ground_truth"][0])]
                    },
                    {
                        "prompt": item["locality"]["retention"]["prompt"][1],
                        "ground_truth": [str(item["locality"]["retention"]["ground_truth"][1])]
                    }
                ],
                "common_sense": []
            }
        }

        # 从 c3.json 里取两条，加入 common_sense
        try:
            q1 = next(c3_iter)
            q2 = next(c3_iter)
            case["locality"]["common_sense"].extend([
                {"prompt": q1["prompt"], "ground_truth": [q1["answer"]]},
                {"prompt": q2["prompt"], "ground_truth": [q2["answer"]]}
            ])
        except StopIteration:
            print(f"⚠️ 警告：c3.json 不够，case {i} 未能分配两条 common_sense")

        knowedit_data.append(case)
    
    return knowedit_data


if __name__ == "__main__":
    # 输入文件
    files_to_process = 'cnstat_samples.json'
    c3_file = 'c3dataset/c3_first5000.json'  
    output_filename = 'RTSF-Statistical.json'  # 输出文件名

    files_to_process='company_samples.json' 
    c3_file = 'c3dataset/c3_rest5000.json' 
    output_filename = 'RTSF-Financial.json' # 输出文件名

    # 加载原始数据和 c3 数据
    with open(files_to_process, "r", encoding="utf-8") as f:
        original_data = json.load(f)

    with open(c3_file, "r", encoding="utf-8") as f:
        c3_data = json.load(f)

    # 转换
    result = convert_to_knowedit_format(original_data, c3_data)

    # 打印前 2 条示例
    print("\n前2条数据示例:")
    for i, item in enumerate(result[:2]):
        print(f"\n--- 第{i+1}条 ---")
        print(json.dumps(item, ensure_ascii=False, indent=2))

    # 保存结果
    with open(output_filename, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 已保存 {len(result)} 条数据到 {output_filename}")
