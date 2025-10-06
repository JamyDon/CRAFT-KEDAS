import sys
sys.path.append('..')
import json
import os
from tqdm import tqdm
from openai import OpenAI


client = OpenAI(
    base_url="https://api.openai-proxy.org/v1",
    api_key=os.getenv("OPENAI_API_KEY")
)


def api_inference(prompt):
    while True:
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
            )
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Retrying...")
            continue
    
    try:
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error: {e}")
        return ""


def get_qa(prompt, target_new):
    return f"{prompt} {target_new}"


def get_declaration(query, target_new):
    prompt = f"You are a helpful assistant. \
    You are given a query and a target new. \
    Please generate the declaration form of the query and target new. \
    Here is an example: \
    Query: 2024年华大智造的总资产（亿元）是多少？ \
    Target new: 103.15 \
    Declaration: 2024年华大智造的总资产（亿元）是103.15。 \
    Please generate the declaration form of the query and target new below: \
    Query: {query} \
    Target new: {target_new} \
    Declaration: "
    return api_inference(prompt)


def get_alias(query, target_new, declaration):
    prompt = f"You are a helpful assistant. \
    You are given a query, a target new, and a declaration. \
    Please generate a paraphrased sentence of the declaration with the central term translated to English. \
    Here is an example: \
    Query: 2024年华大智造的总资产（亿元）是多少？ \
    Target new: 103.15 \
    Declaration: 2024年华大智造的总资产（亿元）是103.15。 \
    Paraphrased sentence: 2024年华大智造的Total Assets(100 million yuan)是103.15。 \
    Here is another example: \
    Query: 2024年9月中国的订销报纸份数当期值(万份)是多少？\
    Target new: 137885.2 \
    Declaration: 2024年9月中国的订销报纸份数当期值(万份)是137885.2。 \
    Paraphrased sentence: 2024年9月中国的Issue of Newspapers, Current Period Value(10000 pieces)是137885.2。 \
    Please generate a paraphrased sentence of the declaration below: \
    Query: {query} \
    Target new: {target_new} \
    Declaration: {declaration} \
    Paraphrased sentence: "
    response = api_inference(prompt)
    return response


new_data_dir = "../../data/preprocessed_CRAFT"
os.makedirs(new_data_dir, exist_ok=True)
print(f"Processing CRAFT...")

filenames = ["CRAFT-Statistical-test500", "CRAFT-Statistical-train", "CRAFT-Financial-test500", "CRAFT-Financial-train"]
for filename in filenames:
    preprocessed_data = []
    requests = json.load(open(f"../../data/CRAFT/{filename}.json", "r"))

    for request in tqdm(requests, total=len(requests), desc=f"Processing {filename}"):
        prompts = request["prompt"]
        target_news = request["target_new"]
        locality_requests, portability_requests = [], []
        for key in request['locality']:
            for req in request['locality'][key]:
                locality_requests.append(req['prompt'])
        for key in request['portability']:
            for req in request['portability'][key]:
                portability_requests.append(req['prompt'])
        
        for prompt, target_new in zip(prompts, target_news):
            # get the Q-A form
            qa_form = f"{prompt} {target_new}"
            # get the declaration form
            declaration = get_declaration(prompt, target_new)
            # get the paraphrased forms
            paraphrased = get_alias(prompt, target_new, declaration)

            if "train" in filename:
                preprocessed_data.append({
                    "request": prompt,
                    "target_new": target_new,
                    "Q-A": qa_form,
                    "Declaration": declaration,
                    "Paraphrased": paraphrased,
                    "locality_requests": locality_requests,
                    "portability_requests": portability_requests
                })
            else:
                preprocessed_data.append({
                    "request": prompt,
                    "target_new": target_new,
                    "Q-A": qa_form,
                    "Declaration": declaration,
                    "Paraphrased": paraphrased
                })

    with open(os.path.join(new_data_dir, f"{filename}.json"), "w") as f:
        json.dump(preprocessed_data, f, indent=4, ensure_ascii=False)
