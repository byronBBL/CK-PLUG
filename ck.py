from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria
import torch
from transformers.generation.stopping_criteria import StoppingCriteriaList

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
    
class CK:

    def __init__(self, model_name, device, num_gpus, max_gpu_memory=27):
        self.model_name = model_name
        self.device = device
        self.num_gpus = num_gpus
        self.stopping_criteria = None
        self.max_gpu_memory = max_gpu_memory

        self.model, self.tokenizer = self.load_model(model_name)

    def load_model(self, model_name):
        if self.device == "cuda":
            kwargs = {"torch_dtype": torch.float16, "offload_folder": f"{model_name}/offload"}
            if self.num_gpus == "auto":
                kwargs["device_map"] = "auto"
            else:
                self.num_gpus = int(self.num_gpus)
                if self.num_gpus != 1:
                    kwargs.update({
                        "device_map": "auto",
                        "max_memory": {i: f"{self.max_gpu_memory}GiB" for i in range(self.num_gpus)},
                    })
        elif self.device == "cpu":
            kwargs = {}
        else:
            raise ValueError(f"Invalid device: {self.device}")
        
        tokenizer = AutoTokenizer.from_pretrained(model_name if not 'vicuna' in model_name else 'huggyllama/llama-7b')
        model = AutoModelForCausalLM.from_pretrained(model_name,
            low_cpu_mem_usage=True, **kwargs)

        if self.device == "cuda" and self.num_gpus == 1:
            model.cuda()
        
        return model, tokenizer
    
    def set_stop_words(self, stop_words):
        self.stop_words = stop_words
        self.stopping_criteria = StoppingCriteriaList()
        list_stop_word_ids = []
        for stop_word in self.stop_words:
            stop_word_ids = self.tokenizer.encode('\n' + stop_word)[3:]
            list_stop_word_ids.append(stop_word_ids)
            print("Added stop word: ", stop_word, 'with the ids', stop_word_ids, flush=True)
        self.stopping_criteria.append(LLamaQaStoppingCriteria(list_stop_word_ids))

    def generate(self, base_prompt, context_prompt, alpha=0.0, select_top=10, max_new_tokens=32, top_p=1, top_k=1, temperature=1.0, mode='base_no_rag', verbose=False, remove_stop_words=False, relative_top=0.1, **kwargs):
        with torch.no_grad():
            
            if mode == 'base_no_rag':
                assert base_prompt is not None, "base_prompt must be specified"
                base_ids = self.tokenizer(base_prompt, return_tensors="pt").input_ids.to(self.device)
                max_len = base_ids.shape[-1] + max_new_tokens
                outputs = self.model.generate(base_ids, max_length=max_len, num_return_sequences=1,
                                    output_scores=True, return_dict_in_generate=True, ck_decoding=False,
                                    top_p=top_p, top_k=top_k, temperature=temperature, stopping_criteria=self.stopping_criteria, **kwargs)

            elif mode == 'base_rag':
                assert context_prompt is not None, "base_prompt must be specified"
                context_ids = self.tokenizer(context_prompt, return_tensors="pt").input_ids.to(self.device)
                max_len = context_ids.shape[-1] + max_new_tokens
                outputs = self.model.generate(context_ids, max_length=max_len, num_return_sequences=1,
                                    output_scores=True, return_dict_in_generate=True, ck_decoding=False,
                                    top_p=top_p, top_k=top_k, temperature=temperature, stopping_criteria=self.stopping_criteria, **kwargs)
            
            elif mode == 'ck':
                assert base_prompt is not None, "base_prompt must be specified"
                assert context_prompt is not None, "context_prompt must be specified"
                base_ids = self.tokenizer(base_prompt, return_tensors="pt").input_ids.to(self.device)
                max_len_base = base_ids.shape[-1] + max_new_tokens
                context_ids = self.tokenizer(context_prompt, return_tensors="pt").input_ids.to(self.device)
                max_len_context = context_ids.shape[-1] + max_new_tokens
                max_len = max(max_len_base, max_len_context)
                outputs = self.model.generate(context_ids, base_ids, alpha = alpha, max_length=max_len, num_return_sequences=1,
                                        output_scores=True, return_dict_in_generate=True, ck_decoding=True, select_top=select_top,
                                        top_p=top_p, top_k=top_k, temperature=temperature, stopping_criteria=self.stopping_criteria, **kwargs,)


            sequences = outputs.sequences
            if mode == 'base_no_rag':
                gen_sequences = sequences[:, base_ids.shape[-1]:][0, :]
            else: gen_sequences = sequences[:, context_ids.shape[-1]:][0, :]
            gen_arr = gen_sequences.cpu().numpy()

            output_str = self.tokenizer.decode(gen_sequences, skip_special_tokens=True)

            if verbose:
                print('MODEL OUTPUT: \n{0}'.format(output_str))

            if remove_stop_words:
                for stop_word in self.stop_words:
                    length_to_remove = len(stop_word)
                    if output_str[-length_to_remove:] == stop_word:
                        output_str = output_str[:-length_to_remove]
                output_str = output_str.strip()

        if self.device:
            torch.cuda.empty_cache()

        return output_str
