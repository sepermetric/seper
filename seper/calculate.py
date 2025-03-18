import numpy as np
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader,DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from datetime import timedelta
import torch
import math
import random

def check_entail(text1, text2, model, entailment_option='bi'):
    # text1, text2: b, n
    implication_1, logits_1 = model.check_implication(text1, text2)
    implication_2, logits_2 = model.check_implication(text2, text1)  # pylint: disable=arguments-out-of-order
    assert (implication_1[0].item() in [0, 1, 2]) and (implication_2[0].item() in [0, 1, 2])

    if entailment_option == 'bi':
        semantically_equivalent = (implication_1 == 2) & (implication_2 == 2)
    elif entailment_option == 'a_entails_b': # text 1 entails text 2
        semantically_equivalent = (implication_1 == 2)
    elif entailment_option == 'b_entails_a': # text 2 entails text 1
        semantically_equivalent = (implication_2 == 2)
    elif entailment_option == 'loose':
        # Check if none of the implications are 0 (contradiction) and not both of them are neutral.
        no_contradiction = (implication_1 != 0) & (implication_2 != 0)
        # 检查 implication_1 和 implication_2 是否都为 1 (neutral)
        not_both_neutral = ~((implication_1 == 1) & (implication_2 == 1))
        semantically_equivalent = no_contradiction & not_both_neutral

    entail_logits = {
        'a_entails_b': logits_1,
        'b_entails_a': logits_2,
    }
    return semantically_equivalent, entail_logits

def calculate_uncertainty_soft_batch(
        example, 
        entailment_model,
        computation_chunk_size=128
        ):
    bsz = len(example['question'])
    n_responses = len(example["response_text"][0])

    batch_questions = example['question'] # b
    batch_response_texts = example["response_text"] # b, 10
    batch_gt_answers = example['answers'] # b, n?
    likelihood = example['likelihood'] # b, 10 

    all_responses, all_answer_condition_on_question = [], []
    bid_counter = [0]
    for bi in range(bsz):
        n_gt_ans = len(example['answers'][bi])
        
        assert n_gt_ans > 0, f"n_gt_ans: {n_gt_ans}, question: {example['question'][bi]}"

        all_responses += [[f'{batch_questions[bi]} {r}' for r in batch_response_texts[bi]]] * n_gt_ans
        all_answer_condition_on_question += [[f'{batch_questions[bi]} {answer}'] * n_responses for answer in batch_gt_answers[bi]] 
        bid_counter.append(bid_counter[-1] + n_gt_ans * n_responses)

    all_responses = np.array(all_responses).flatten()
    all_answer_condition_on_question = np.array(all_answer_condition_on_question).flatten()

 
    all_reponse_entails_answer, all_answer_entails_response = [], []
    for i in range(0, len(all_responses), computation_chunk_size):
        chunk_responses = all_responses[i:i+computation_chunk_size]
        chunk_answer_on_questions = all_answer_condition_on_question[i:i+computation_chunk_size]
        _, entail_logits = check_entail(chunk_responses.tolist(), chunk_answer_on_questions.tolist(), entailment_model, 'loose')
        r_e_a =  F.softmax(entail_logits['a_entails_b'], dim=-1)[...,2].cpu().numpy().tolist()
        a_e_r = F.softmax(entail_logits['b_entails_a'], dim=-1)[...,2].cpu().numpy().tolist()
        all_reponse_entails_answer.extend(r_e_a)
        all_answer_entails_response.extend(a_e_r)
    
    # response entails answer: 满分
    # answer entails response: 最多算一半
    # TODO: check whether this setting is optimal
    sepers = []
    for bi in range(bsz):
        reponse_entails_answer = np.array(all_reponse_entails_answer[bid_counter[bi]: bid_counter[bi+1]]).reshape(n_responses,-1)
        answer_entails_response = np.array(all_answer_entails_response[bid_counter[bi]: bid_counter[bi+1]]).reshape(n_responses, -1)
        '''
        # setting1
        entail_mass = np.stack([likelihood[bi][..., np.newaxis] * reponse_entails_answer, 
                                0.5 * likelihood[bi][..., np.newaxis] * (reponse_entails_answer + answer_entails_response)], axis=-1) # 10, n, 2
        sp = np.max(np.sum(np.max(entail_mass, axis=-1), axis=0))
        '''
        # setting2
        entail_mass = np.stack([likelihood[bi][..., np.newaxis] * reponse_entails_answer, 
                                likelihood[bi][..., np.newaxis] * answer_entails_response], axis=-1) # 10, n, 2
        try:
            sp = np.max(np.sum(np.max(entail_mass, axis=-1), axis=0))
        except:
            print(f"reponse_entails_answer: {reponse_entails_answer.shape}")
            print(f"answer_entails_response: {answer_entails_response.shape}")
            print(f"bi: {bi}")
            print(f"bid_counter: {bid_counter}")
            print(bid_counter[bi], bid_counter[bi+1])
            raise ValueError
        sepers.append(sp)
    return sepers

def calculate_uncertainty_hard_batch(
        example, 
        entailment_model,
        strict_entailment=True):
    # only support batch size = 1 because of clustering
    assert len(example['question']) == 1
    question = example['question'][0]
    responses = example['response_text'][0]
    gt_answers = example['answers'][0]
    log_liks_agg = example['log_liks_agg'][0]
    n_gt_ans = len(gt_answers)
    n_responses = len(responses)
    assert n_gt_ans > 0, f"n_gt_ans: {n_gt_ans}, question: {example['question'][0]}"
   
    responses = [f'{question} {r}' for r in responses]
    # Initialise all ids with -1.
    semantic_set_ids = [-1] * len(responses)
    
    # Keep track of current id.
    next_id = 0
    for i, string1 in enumerate(responses):
        if semantic_set_ids[i] == -1: # unassigned
            semantic_set_ids[i] = next_id
            if i == len(responses) - 1:
                next_id += 1
                break
            # 创建一个列表，其中包含当前字符串string1，重复多次以匹配剩余字符串列表的长度
            current_string_list = [string1] * (len(responses) - i - 1)
            # 获取未分配id的字符串的列表
            remaining_strings = responses[i+1:]
            # 执行批量等价性检查
            strict_type = 'bi' if strict_entailment else 'loose'
            equivalence_results, _ = check_entail(current_string_list, remaining_strings, entailment_model, strict_type)
            # 根据等价性结果更新semantic_set_ids
            for j, equivalent in enumerate(equivalence_results):
                if equivalent:
                    semantic_set_ids[i + 1 + j] = next_id
            next_id += 1
            
    # Compute semantic entropy.
    log_likelihood_per_semantic_id = logsumexp_by_id(semantic_set_ids, log_liks_agg, agg='sum_normalized')


    n_cluster = next_id
    responses_per_cluster = [[] for _ in range(n_cluster)]
    for i, response in enumerate(responses):
        responses_per_cluster[semantic_set_ids[i]].append(response)
    cluster_texts = [resp[0] for resp in responses_per_cluster]


    all_clusters = cluster_texts * n_gt_ans
    all_answer_condition_on_question = [[f"{question} {answer}"] * n_cluster for answer in gt_answers] 
    all_answer_condition_on_question = np.array(all_answer_condition_on_question).flatten().tolist()
    ans_equivalence_results, _ = check_entail(all_clusters, all_answer_condition_on_question, entailment_model, 'loose')
    ans_equivalence_results =ans_equivalence_results.reshape(n_cluster, -1).detach().cpu().numpy() # n_cluster, n_gt_ans
    # fill the score which are less than 1 with 0
    ans_equivalence_results = np.max(ans_equivalence_results, axis=1) # n_cluster
    seper_hard = 0.
    for cluster_id in range(n_cluster):
        if ans_equivalence_results[cluster_id]:
            seper_hard += np.exp(log_likelihood_per_semantic_id[cluster_id])

    return seper_hard


def setup(rank, world_size):
    dist.init_process_group("nccl", 
                            rank=rank,
                            world_size=world_size,
                            timeout=timedelta(minutes=120)
                            )
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def create_collate_fn(keys):
    def my_collate_fn(batch):
        result = {}
        for k in keys:
            result[k] = [b[k] for b in batch]
        return result
    return my_collate_fn

### for generation

BRIEF_PROMPTS = {
    'default': "Answer the following question as briefly as possible.\n",
    'chat': 'Answer the following question in a single brief but complete sentence.\n',
    'long-form': "Answer a question. When the question is ambiguous, include all correct answers. \
Write your answer concisely, accurately and unbiasedly along with a brief explanation.\n"}

def truncate_context_by_words(context, max_words):
    tokens = context.split()
    if len(tokens) <= max_words:
        return context
    else:
        return ' '.join(tokens[:max_words])

def make_prompt(question,
                instruction=BRIEF_PROMPTS['default'], 
                answer=None,
                context=None,
                max_words=512):
    prompt = instruction
    if context:
        truncated_context = truncate_context_by_words(context, max_words)
        prompt += f"Context: {truncated_context}\n"

    prompt += f"Question: {question}\n"
    # for perplexity calculation
    if answer:
        prompt += f"Answer: {answer}"
    else:
        prompt += 'Answer:'
    return prompt

def gen_answers_batch(example, 
                model,
                temperature,
                num_generations,
                sub_batch_size=1,
                max_new_tokens=None,
                prompt_type='default',
                device='cuda',
                max_context_words=512):
    question, context = example["question"], example['context']
    current_input = make_prompt(question, context=context, instruction=BRIEF_PROMPTS[prompt_type], max_words=max_context_words)
    # local_prompt = prompt + current_input
    local_prompt = current_input # for now, no few-shot prompt
    full_responses = []

    model.model.eval()
    with torch.no_grad():
        predicted_answer = []
        token_log_likelihoods = []
        # every time generate a batch of sub_batch_size generations, total num_generations generations
        num_iter = math.ceil(num_generations / sub_batch_size)
        for idx in range(num_iter):
            sub_num_gen = min(sub_batch_size, num_generations - idx * sub_batch_size)

            sub_predicted_answer, sub_token_log_likelihoods, sub_embedding = model.batch_predict(
                local_prompt, 
                num_generations=sub_num_gen, 
                temperature=temperature, 
                device=device, max_new_tokens=max_new_tokens)
            predicted_answer.extend(sub_predicted_answer)
            token_log_likelihoods.extend(sub_token_log_likelihoods)
        
    full_responses = [*zip(predicted_answer, token_log_likelihoods)]
    example['responses'] = full_responses

    return example


def my_collate_fn(batch):
    return batch

def process_item_for_seper(item, subsample=-1, n_repeat=0):
    if subsample > 0:
        random.seed(n_repeat)
        item['responses'] = random.sample(item['responses'], subsample) # for compute variance 
    example = {}
    example['question'] = item['question'] + '?' if not item['question'].endswith('?') else item['question']
    example['answers'] = item['answers']
    example['context'] = item['context']
    example['context_label'] = item['context_label'] if 'context_label' in item else None
    example['response_text'] = [r[0] for r in item['responses']]
    example['response_log_likelihood'] = [r[1] for r in item['responses']]
    log_liks_agg = [np.mean(log_lik) for log_lik in example['response_log_likelihood']]
    example['log_liks_agg'] = log_liks_agg
    log_z = np.log(np.sum(np.exp(log_liks_agg)))
    example['likelihood'] = np.exp(log_liks_agg - log_z)

    return example