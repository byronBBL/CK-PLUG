import re
import ast
import string
import json
import re
import argparse
from tqdm import tqdm
import os
import torch
from transformers import AutoTokenizer, AutoModel, StoppingCriteria, StoppingCriteriaList, AutoModelForCausalLM
import logging
os.environ["MKL_THREADING_LAYER"] = "GNU"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from ck import CK

class LLamaQaStoppingCriteria(StoppingCriteria):
    def __init__(self, list_token_ids_sequence: list = []):
        self.token_ids_sequences = []
        self.lengths = []
        for token_ids_sequence in list_token_ids_sequence:
            self.token_ids_sequences.append(torch.tensor(token_ids_sequence, dtype=torch.long))
            self.lengths.append(len(token_ids_sequence))
        
    # @add_start_docstrings(STOPPING_CRITERIA_INPUTS_DOCSTRING)
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # check the final {self.length} tokens
        stop = False
        for token_ids_sequence, length in zip(self.token_ids_sequences, self.lengths):
            if input_ids.shape[-1] < length:
                continue
            else:
                if bool(torch.all(input_ids[0, -length:] == token_ids_sequence.to(input_ids.device))):
                    stop = True
                    break
        return stop

def set_stop_words(tokenizer, stop):
    stop_words = stop
    list_stop_word_ids = []
    for stop_word in stop_words:
            stop_word_ids = tokenizer.encode('\n' + stop_word)[3:]
            list_stop_word_ids.append(stop_word_ids)
            print("Added stop word: ", stop_word, 'with the ids', stop_word_ids, flush=True)
    stopping_criteria = StoppingCriteriaList()
    stopping_criteria.append(LLamaQaStoppingCriteria(list_stop_word_ids))
    return stopping_criteria
            

def call_ck(model, base_prompts, context_prompts, stop, params_dict):
    
    sequences = model.generate(base_prompts, context_prompts, **params_dict)

    for stop_word in stop:
        length_to_remove = len(stop_word)
        if sequences[-length_to_remove:] == stop_word:
            sequences = sequences[:-length_to_remove]
    output_str = sequences.strip()
    return output_str

negation_words = [
    "no", "not", "never", "none", "cannot", "nobody", "nothing", "nowhere", 
    "neither", "nor", "without", "hardly"
]

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def white_space_fix(text):
        return ' '.join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))    

def recall_score(prediction, ground_truth):
    prediction = normalize_answer(prediction)
    ground_truth = normalize_answer(ground_truth)
    return (ground_truth in prediction)

def get_score(preds, golds):
    em, recall = 0, 0
    for pred, gold in zip(preds, golds):
        # contains_negation = any(word in pred.split() for word in negation_words)
        # if contains_negation: 
        #     continue
        if isinstance(gold, list):
            _em, _recall = 0, 0
            for g in gold:
                _em = max(exact_match_score(pred, g), _em)
                _recall = max(recall_score(pred, g), _recall)
            em += _em
            recall += _recall
        else:
            em += exact_match_score(pred, gold)
            recall += recall_score(pred, gold)
    em = em * 100 / (len(preds) + 1e-5)
    recall = recall * 100 / (len(preds) + 1e-5)
    return em, recall

def qa_to_prompt_baseline(query, context, schema):
    def get_prompt(query, context, schema, answer=''):
        if schema == 'base':
            prompt = '{}\nQ:{}\nA:{}'.format(context, query, answer)
        elif schema == 'opin':
            context = context.replace('"', "")
            prompt = 'Bob said "{}"\nQ: {} in Bob\'s opinion?\nA:{}'.format(context, query[:-1], answer)
        elif schema == 'instr+opin':
            context = context.replace('"', "")
            prompt = 'Bob said "{}"\nQ: {} in Bob\'s opinion?\nA:{}'.format(context, query[:-1], answer)
        elif schema == 'attr':
            prompt = '{}\nQ:{} based on the given tex?\nA:{}'.format(context, query[:-1], answer)
        elif schema == 'instr':
            prompt = '{}\nQ:{}\nA:{}'.format(context, query, answer)
        return prompt
    prompt = ''
    if schema in ('instr', 'instr+opin'):
        prompt = 'Instruction: read the given information and answer the corresponding question.\n\n'
    prompt = prompt + get_prompt(query, context, schema=schema)
    return prompt

    
def eval(pred_answers, orig_answers, gold_answers):
    em, ps = get_score(pred_answers, gold_answers)
    _, po = get_score(pred_answers, orig_answers)
    mr = po / (ps + po + 1e-10) * 100
    print('ps {}, po {}, mr {}, em {}.'.format(ps, po, mr, em))
    return po > 0
    
def create_log_path(log_path):
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write('') 
        logging.info(f"Log file {log_path} created.")
    else:
        logging.info(f"Log file {log_path} already exists.")

def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default="Models/Llama-3-8B-Instruct", type=str)
    parser.add_argument("--data_path", default="work7/faith/ConFiQA-QA.json", type=str)
    parser.add_argument("--schema", default="base", type=str, help="Choose from the following prompting templates: base, attr, instr, opin, instr+opin.")
    parser.add_argument("--num-gpus", type=str, default="1")
    parser.add_argument("--max_gpu_memory", type=int, default=27)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument('--mode', type=str, default='ck', 
                    help='ck, base_rag, base_no_rag')
    args = parser.parse_args()
    
    with open(args.data_path, 'r') as fh:
        data = json.load(fh)

    model_name = args.model_name
    num_gpus = args.num_gpus
    device = args.device
    model = CK(model_name, device, num_gpus, max_gpu_memory=args.max_gpu_memory)
    stop = ['Q:']
    model.set_stop_words(stop)

    params_dict = {
            "repetition_penalty": 1.0,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 100,
            "max_new_tokens": 64,
            "logprobs": None,
            "mode": args.mode,
            "alpha": 0.0,
            "select_top": 10,
        }
    
    step = 0
    gold_answers, pred_answers, orig_answers = [], [], []
    for _id, d in enumerate(tqdm(data)):
        step += 1
        query = d['question']
        context = d['cf_context']
        cf_answer = d['cf_answer']
        orig_answer = d['orig_answer']
        
        query = 'Q: {}\nA: '.format(query)
        prompt = qa_to_prompt_baseline(query, context, schema=args.schema)
        pred = call_ck(model, query, prompt, stop, params_dict)
        pred_answers.append(pred)
        if len(d['cf_alias']) != 0:
            cf_answer = [cf_answer] + d['cf_alias']
        if len(d['orig_alias']) != 0:
            orig_answer = [orig_answer] + d['orig_alias']
        gold_answers.append(cf_answer)
        orig_answers.append(orig_answer)

    eval(pred_answers, orig_answers, gold_answers)
    print("The parameter configuration is as follows:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    

if __name__ == '__main__':
    main()