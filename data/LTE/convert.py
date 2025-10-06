import json

def lte2llama():
    lte_fn = 'CRAFT_alignment_data.json'
    llama_fn = 'CRAFT_alignment_data_llama.json'
    
    with open(lte_fn, 'r') as f:
        lte_data = json.load(f)

    with open(llama_fn, 'w') as f:
        all_data = []
        for item in lte_data:
            if 'conversations' not in item:
                continue
            conversations = item['conversations']

            if len(conversations) != 2:
                continue
            if conversations[0]['from'] != 'human' or conversations[1]['from'] != 'gpt':
                continue

            messages = {
                "instruction": conversations[0]['value'],
                "input": "",
                "output": conversations[1]['value']
            }

            all_data.append(messages)

    with open(llama_fn, 'w') as f:
        json.dump(all_data, f, indent=4, ensure_ascii=False)

if __name__ == '__main__':
    lte2llama()