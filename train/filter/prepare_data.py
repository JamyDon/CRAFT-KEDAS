import json
import os
from pathlib import Path
from typing import List, Dict, Any

def process_instance(instance: Dict[str, Any]) -> List[Dict[str, Any]]:
    result = []
    max_length = 0
    max_knowledge_length = 0
    
    # Get the knowledge
    prompts = instance.get("prompt", [])
    target_news = instance.get("target_new", [])
    qa_pairs = [prompt + " " + target_new for prompt, target_new in zip(prompts, target_news)]
    if not qa_pairs:
        return result, max_length, max_knowledge_length
    if len(qa_pairs) == 0:
        return result, max_length, max_knowledge_length
    
    for i, qa_pair in enumerate(qa_pairs):
        # Process main request
        # main_request = instance.get("request", "")
        # if main_request:
        main_request = prompts[i]
        result.append({
            "request": main_request,
            "knowledge": qa_pair,
            "relevance": 1
        })
        max_length = max(max_length, len(main_request))
        max_knowledge_length = max(max_knowledge_length, len(qa_pair))

        # Process locality requests
        locality_requests = instance.get("locality", {})
        for key in locality_requests.keys():
            for req in locality_requests[key]:
                if req:  # Only process non-empty requests
                    result.append({
                        "request": req["prompt"],
                        "knowledge": qa_pair,
                        "relevance": 0
                    })
                max_length = max(max_length, len(req["prompt"]))
                max_knowledge_length = max(max_knowledge_length, len(qa_pair))

        # Process portability requests
        portability_requests = instance.get("portability", {})
        for key in portability_requests.keys():
            for req in portability_requests[key]:
                if req:  # Only process non-empty requests
                    result.append({
                        "request": req["prompt"],
                        "knowledge": qa_pair,
                        "relevance": 1
                    })
                    max_length = max(max_length, len(req["prompt"]))
                    max_knowledge_length = max(max_knowledge_length, len(qa_pair))
    
    return result, max_length, max_knowledge_length

def process_json_file(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    processed_instances = []
    max_length = 0
    max_knowledge_length = 0
    for instance in data:
        processed_instances_instance, max_length_instance, max_knowledge_length_instance = process_instance(instance)
        processed_instances.extend(processed_instances_instance)
        max_length = max(max_length, max_length_instance)
        max_knowledge_length = max(max_knowledge_length, max_knowledge_length_instance)

    print(f"Max length for {file_path}: {max_length}")
    print(f"Max knowledge length for {file_path}: {max_knowledge_length}")
    return processed_instances

def main():
    # Define the input and output directories
    data_dir = Path("../../data/CRAFT")
    output_dir = Path("../../data/filter")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process all JSON files in the data directory and its subdirectories
    all_processed_instances = []
    for root, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith("-train.json"):
                file_path = os.path.join(root, file)
                print(f"Processing {file_path}")
                processed_instances = process_json_file(file_path)
                all_processed_instances.extend(processed_instances)
    
    # Save the processed data
    output_file = output_dir / "CRAFT.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_processed_instances, f, ensure_ascii=False, indent=2)
    
    print(f"Processed {len(all_processed_instances)} instances")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    main()
