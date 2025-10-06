import pickle
import json
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# sentence_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2').to('cuda')
sentence_model = SentenceTransformer("./hugging_cache/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2").to('cuda')

# data_path="../../data/CRAFT/CRAFT-Statistical-train.json"
# with open(data_path, 'r', encoding='utf-8') as input_file:
#     input_data = json.load(input_file)
# sentences = []
# subjects = []
# for i, train_data in enumerate(input_data):
#     for prompt, target_new, subject in zip(train_data['prompt'], train_data['target_new'], train_data['subject']):
#         new_fact = prompt + ' ' + target_new
#         sentences.append(new_fact)
#         subjects.append(subject)
# embeddings = sentence_model.encode(sentences)

# with open(data_path.split('.')[0] + '_embeddings.pkl', "wb") as fOut:
#     pickle.dump({'sentences': sentences, 'subjects': subjects, 'embeddings': embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

# data_path="../../data/CRAFT/CRAFT-Financial-train.json"
# with open(data_path, 'r', encoding='utf-8') as input_file:
#     input_data = json.load(input_file)
# sentences = []
# subjects = []
# for i, train_data in enumerate(input_data):
#     for prompt, target_new, subject in zip(train_data['prompt'], train_data['target_new'], train_data['subject']):
#         new_fact = prompt + ' ' + target_new
#         sentences.append(new_fact)
#         subjects.append(subject)
# embeddings = sentence_model.encode(sentences)

# with open(data_path.split('.')[0] + '_embeddings.pkl', "wb") as fOut:
#     pickle.dump({'sentences': sentences, 'subjects': subjects, 'embeddings': embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

import random
import json
import torch
import pickle
from sentence_transformers import util


def knowledge_edit_template(new_facts, question):
    return "Please acknowledge the updated information provided below and respond to the subsequent question.\n\n[Updated Information]:\n" \
        + new_facts + "\n\n[Question]:\n" + question


def sentence_completion_prompt(question):
    return f"Please complete the sentence below. You should ONLY output the completed part.\n\n{question}"


def text_completion_prompt(question):
    return f"Please complete the text below. You should ONLY output the completed part.\n\n{question}"


def question_answering_prompt(question):
    return f"Please answer the question below. You should ONLY output the answer.\n\n{question}"


def data_append(data, source, idx, new_facts, question, answer):
    if new_facts:
        data.append({
            "id": "identity_{0}_{1}".format(str(idx), source),
            "conversations": [
            {
                "from": 'human',
                "value": knowledge_edit_template(new_facts, question)
            },
            {
                "from": 'gpt',
                "value": answer
            },
            ]
        })
    else:
        data.append({
            "id": "identity_{0}_{1}".format(str(idx), source),
            "conversations": [
            {
                "from": 'human',
                "value": question
            },
            {
                "from": 'gpt',
                "value": answer
            },
            ]
        })
    idx += 1
    return idx, data


def retrieve_new_facts(stored_data, sentence_model, query_sentences, query_subjects, num):
    stored_sentences = stored_data['sentences']
    stored_subjects = stored_data['subjects']
    stored_embeddings = stored_data['embeddings']

    stored_embeddings = torch.tensor(stored_embeddings).to('cuda')
    stored_embeddings = util.normalize_embeddings(stored_embeddings)

    retrieved_sent = []
    for query_sentence, query_subject in zip(query_sentences, query_subjects):
        query_embedding = util.normalize_embeddings(torch.tensor(sentence_model.encode(
            query_sentence, show_progress_bar=False)).unsqueeze(0).to('cuda'))

        hits = util.semantic_search(query_embedding, stored_embeddings, score_function=util.dot_score, top_k=5)
        assert len(hits) == 1
        hit = hits[0]
        retrieved_sentences = [stored_sentences[hit[k]["corpus_id"]] for k in range(len(hit))]
        retrieved_subjects = [stored_subjects[hit[k]["corpus_id"]] for k in range(len(hit))]

        for i in range(len(retrieved_sentences)):
            if retrieved_subjects[i] != query_subject and retrieved_sentences[i] != query_sentence:
                retrieved_sent.append(retrieved_sentences[i])

    retrieved_sent = list(set(retrieved_sent))

    try:        
        retrieved_sent = random.sample(retrieved_sent, num)
        new_facts_list = query_sentences + retrieved_sent
        new_facts = create_new_facts(new_facts_list)
    except:
        new_facts = create_new_facts(query_sentences)
        
    return new_facts


def create_new_facts(new_facts_list):
    new_facts_list = [f"{i + 1}. " + new_facts_list[i] for i in range(len(new_facts_list))]
    return "\n".join(new_facts_list)


out_of_scope_questions = []
data = []
need_gpt4 = []

idx = 0

filenames = ["../../data/CRAFT/CRAFT-Statistical-train", "../../data/CRAFT/CRAFT-Financial-train"]
for filename in filenames:
    with open(filename + ".json", 'r', encoding='utf-8') as input_file:
        input_data = json.load(input_file)

    # input_data = random.sample(input_data, 1000)

    sentences = []
    subjects = []
    for i, train_data in enumerate(input_data):
        for prompt, target_new, subject in zip(train_data['prompt'], train_data['target_new'], train_data['subject']):
            new_fact = prompt + ' ' + target_new
            sentences.append(new_fact)
            subjects.append(subject)
    embeddings = sentence_model.encode(sentences)
    stored_data = {
        'sentences': sentences,
        'subjects': subjects,
        'embeddings': embeddings
    }

    for i in tqdm(range(len(input_data))):
        prompts = input_data[i]['prompt']
        target_news = input_data[i]['target_new']
        subjects = input_data[i]['subject']
        new_facts_list = [prompt + " " + target_new for prompt, target_new in zip(prompts, target_news)]

        rand_num = random.random()
        if rand_num < 0.5:
            new_facts = create_new_facts(new_facts_list)
        elif rand_num < 0.75:
            new_facts = retrieve_new_facts(stored_data, sentence_model, new_facts_list, subjects, num=1)
        elif rand_num < 0.95:
            new_facts = retrieve_new_facts(stored_data, sentence_model, new_facts_list, subjects, num=2)
        else:
            new_facts = retrieve_new_facts(stored_data, sentence_model, new_facts_list, subjects, num=3)

        for question, answer in zip(prompts, target_news):
            idx, data = data_append(data, filename.split('/')[-1], idx, new_facts, question, answer)
        
        for attribution in ['portability']:
            if attribution in input_data[i]:
                for k in input_data[i][attribution].keys():
                    for j in range(len(input_data[i][attribution][k])):
                        question = input_data[i][attribution][k][j]['prompt']
                        answer = input_data[i][attribution][k][j]['ground_truth'][0]
                        if question != "" and answer != "":
                            idx, data = data_append(data, filename.split('/')[-1], idx, new_facts, question, answer)
                            # idx, data = data_append(data, filename.split('/')[-1], idx, None, question, None)
                            # need_gpt4.append({'source': filename.split('/')[-1], 'new_facts': None, 'question': question, 'prompt_new': sentence_completion_prompt(question), 'answer': None})

        for attribution in ['locality']:
            if attribution in input_data[i]:
                for k in input_data[i][attribution].keys():
                    for j in range(len(input_data[i][attribution][k])):
                        question = input_data[i][attribution][k][j]['prompt']
                        answer = input_data[i][attribution][k][j]['ground_truth'][0]
                        if question != "" and answer != "":
                            idx, data = data_append(data, filename.split('/')[-1], idx, new_facts, question, answer)
                            idx, data = data_append(data, filename.split('/')[-1], idx, None, question, answer)

### save data
data = random.sample(data, 20000)
json_str = json.dumps(data, indent=4, ensure_ascii=False)
with open("../../data/LTE/CRAFT_alignment_data.json", mode='w', encoding='utf-8') as output_file:
    output_file.write(json_str)
print(len(data))

# with open("../../data/LTE/CRAFT_alignment_data_need_gpt4.jsonl", 'w', encoding='utf-8') as output_file:
#     for i in range(len(need_gpt4)):
#         output_file.write(json.dumps(need_gpt4[i], ensure_ascii=False)+ "\n")

# exit(0)

# import time
# import os
# from openai import OpenAI
# from tqdm import tqdm

# print("Creating GPT4 answers...")
# client = OpenAI(
#     api_key=os.getenv("OPENAI_API_KEY"),
#     base_url="https://api.openai-proxy.org/v1"
# )
# for datum in tqdm(need_gpt4):
#     question = datum['question']
#     prompt_new = datum['prompt_new']
#     answer = datum['answer']
#     source = datum['source']
#     new_facts = datum['new_facts']
#     if answer is None:
#         fail_cnt = 0
#         while True:
#             try:
#                 response = client.chat.completions.create(
#                     model="gpt-5",
#                     messages=[{"role": "user", "content": prompt_new}],
#                     temperature=0.0,
#                     timeout=30,
#                     max_completion_tokens=2048,
#                 )
#                 if response.choices[0].message.content is None or response.choices[0].message.content == "":
#                     raise Exception("Empty response")
#                 datum['answer'] = response.choices[0].message.content
#                 idx, data = data_append(data, source, idx, new_facts, question, datum['answer'])
#                 if 'both' in datum.keys() and datum['both']:
#                     idx, data = data_append(data, source, idx, None, question, datum['answer'])
#                 break
#             except Exception as e:
#                 print(f'Error: {e}')
#                 time.sleep(1)
#                 fail_cnt += 1
#                 if fail_cnt > 10:
#                     print(f'Failed to get answer for {datum["source"]} {question}')
#                     break

# json_str = json.dumps(data, indent=4, ensure_ascii=False)
# with open("../../data/LTE/CRAFT_alignment_data.json", mode='w', encoding='utf-8') as output_file:
#     output_file.write(json_str)
# print(f'Final data: {len(data)}')

# json_str = json.dumps(need_gpt4, indent=4)
# with open("../../data/LTE/CRAFT_alignment_data_gpt4_answers.jsonl", 'w', encoding='utf-8') as output_file:
#     for i in range(len(need_gpt4)):
#         output_file.write(json.dumps(need_gpt4[i], ensure_ascii=False)+ "\n")
