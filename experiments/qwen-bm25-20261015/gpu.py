"""Qwen 8B embedding and 4B reranking over a frozen small corpus, without labels."""
import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

TASK = 'Given a question, retrieve relevant passages from enterprise documents that help answer it.'
PREFIX = '<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be "yes" or "no".<|im_end|>\n<|im_start|>user\n'
SUFFIX = '<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n'
BATCH_TOKENS = 32768


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def load(root):
    chunks = [json.loads(s) for s in (root / 'input/chunks.jsonl').read_text().splitlines()]
    queries = json.loads((root / 'input/queries.json').read_text())
    bm25 = json.loads((root / 'input/bm25.json').read_text())
    return chunks, queries, bm25


def check_deadline(deadline):
    if time.time() >= deadline:
        raise TimeoutError('GPU experiment deadline reached; partial outputs retained')


def encoder(path):
    tokenizer = AutoTokenizer.from_pretrained(path, padding_side='left', local_files_only=True)
    model = AutoModel.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation='sdpa',
                                      local_files_only=True).cuda().eval()
    return tokenizer, model


def batches(inputs, batch_size):
    order = sorted(range(len(inputs)), key=lambda i: len(inputs[i]))
    start = 0
    while start < len(order):
        end = start + 1
        while end < min(start + batch_size, len(order)):
            if (end - start + 1) * len(inputs[order[end]]) > BATCH_TOKENS:
                break
            end += 1
        yield order[start:end]
        start = end


@torch.inference_mode()
def encode(texts, tokenizer, model, batch_size, deadline, progress=None):
    inputs = tokenizer(texts, padding=False, truncation=False)['input_ids']
    if max(map(len, inputs)) > 8192:
        raise ValueError('Input exceeds frozen 8192-token limit; do not silently truncate')
    vectors, positions = [], []
    for indices in batches(inputs, batch_size):
        check_deadline(deadline)
        batch = tokenizer.pad({'input_ids': [inputs[i] for i in indices]}, padding=True,
                              return_tensors='pt').to('cuda')
        hidden = model(**batch, use_cache=False).last_hidden_state[:, -1]
        vectors.append(torch.nn.functional.normalize(hidden.float(), p=2, dim=1).cpu().numpy())
        positions.extend(indices)
        if progress:
            save(progress, {'stage': 'encoding', 'completed': len(positions), 'total': len(texts)})
    torch.cuda.synchronize()
    return np.concatenate(vectors)[np.argsort(positions)], sum(map(len, inputs))


def distinct_chunks(order, chunks, count=100):
    seen, selected = set(), []
    for i in order:
        doc = chunks[int(i)]['canonical_id']
        if doc not in seen:
            seen.add(doc)
            selected.append(int(i))
        if len(selected) == count:
            break
    return selected


def embed(args):
    root = Path(args.root)
    chunks, queries, bm25 = load(root)
    out = root / 'output'
    out.mkdir(exist_ok=True)
    text = [c['title'] + '\n' + c['text'] for c in chunks]
    started = time.monotonic()
    tokenizer, model = encoder(args.embedding)
    loaded = time.monotonic() - started
    if args.probe:
        indices = np.random.default_rng(20261015).choice(len(chunks), min(512, len(chunks)), replace=False)
        chosen = [text[int(i)] for i in indices]
    else:
        chosen = text
    begun = time.monotonic()
    vectors, tokens = encode(chosen, tokenizer, model, args.batch_size, args.deadline,
                             None if args.probe else out / 'progress.json')
    seconds = time.monotonic() - begun
    info = {'stage': 'embedding_probe' if args.probe else 'embedding', 'chunks': len(chosen), 'tokens': tokens,
            'seconds': seconds, 'load_seconds': loaded, 'chunks_per_second': len(chosen)/seconds,
            'estimated_all_chunks_seconds': len(chunks)*seconds/len(chosen),
            'peak_cuda_bytes': torch.cuda.max_memory_allocated(), 'dtype': 'bfloat16', 'attention': 'sdpa',
            'batch_size': args.batch_size, 'batch_tokens': BATCH_TOKENS, 'batching': 'length_sorted'}
    save(out / ('embedding-probe.json' if args.probe else 'embedding-stats.json'), info)
    if args.probe:
        print(json.dumps(info), flush=True)
        return
    query_vectors, _ = encode(['Instruct: '+TASK+'\nQuery: '+q['question'] for q in queries],
                               tokenizer, model, args.batch_size, args.deadline)
    np.save(out / 'vectors.npy', vectors)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    matrix = torch.from_numpy(vectors).cuda()
    by_id = {c['id']: i for i, c in enumerate(chunks)}
    rankings = []
    for q, v, lexical in zip(queries, query_vectors, bm25, strict=True):
        check_deadline(args.deadline)
        began = time.monotonic()
        scores = (matrix @ torch.from_numpy(v).cuda()).cpu().numpy()
        order = np.lexsort((np.arange(len(scores)), -scores))
        dense = distinct_chunks(order, chunks)
        lex = [by_id[h['chunk_id']] for h in lexical['hits']]
        # Rank at document level. Keep at most one passage from each path per document.
        doc_scores = {}
        for route in (lex, dense):
            for rank, i in enumerate(route, 1):
                doc = chunks[i]['canonical_id']
                doc_scores[doc] = doc_scores.get(doc, 0.) + 1./(60+rank)
        candidate = sorted(set(lex+dense), key=lambda i: (-doc_scores[chunks[i]['canonical_id']], chunks[i]['id']))
        rankings.append({'id': q['id'], 'dense': [chunks[i]['id'] for i in dense],
                         'candidates': [chunks[i]['id'] for i in candidate],
                         'fused_documents': sorted(doc_scores, key=lambda d: (-doc_scores[d], d)),
                         'dense_search_seconds': time.monotonic()-began})
    save(out / 'candidates.json', rankings)
    print(json.dumps(info), flush=True)


def reranker(path):
    tokenizer = AutoTokenizer.from_pretrained(path, padding_side='left', local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation='sdpa',
                                                local_files_only=True).cuda().eval()
    return tokenizer, model


@torch.inference_mode()
def rank(pairs, tokenizer, model, batch_size, deadline):
    prefix = tokenizer.encode(PREFIX, add_special_tokens=False)
    suffix = tokenizer.encode(SUFFIX, add_special_tokens=False)
    yes, no = tokenizer.convert_tokens_to_ids(['yes', 'no'])
    texts = [f'<Instruct>: {TASK}\n<Query>: {q}\n<Document>: {d}' for q,d in pairs]
    raw = tokenizer(texts, add_special_tokens=False, padding=False, truncation=False)['input_ids']
    inputs = [prefix+x+suffix for x in raw]
    if max(map(len, inputs)) > 8192:
        raise ValueError('Reranker input exceeds frozen 8192-token limit')
    results = [None] * len(pairs)
    for indices in batches(inputs, batch_size):
        check_deadline(deadline)
        batch = tokenizer.pad({'input_ids': [inputs[i] for i in indices]}, padding=True,
                              return_tensors='pt').to('cuda')
        logits = model(**batch, use_cache=False, logits_to_keep=1).logits[:, -1]
        for i, score in zip(indices, (logits[:, yes]-logits[:, no]).float().cpu().tolist()):
            results[i] = score
    torch.cuda.synchronize()
    return results, sum(map(len, inputs))


def rerank(args):
    root = Path(args.root)
    chunks, queries, bm25 = load(root)
    by_id = {c['id']: c for c in chunks}
    out = root / 'output'
    started = time.monotonic()
    tokenizer, model = reranker(args.reranker)
    loaded = time.monotonic()-started
    if args.probe:
        schedule = []
        for q, row in zip(queries, bm25, strict=True):
            for h in row['hits'][:2]:
                c = by_id[h['chunk_id']]
                schedule.append((q['question'], c['title']+'\n'+c['text']))
        started = time.monotonic()
        _, tokens = rank(schedule, tokenizer, model, args.batch_size, args.deadline)
        seconds = time.monotonic()-started
        info = {'pairs': len(schedule), 'tokens': tokens, 'seconds': seconds, 'load_seconds': loaded,
                'pairs_per_second': len(schedule)/seconds, 'estimated_max_20000_pairs_seconds': 20000*seconds/len(schedule),
                'peak_cuda_bytes': torch.cuda.max_memory_allocated(), 'dtype': 'bfloat16',
                'batch_size': args.batch_size, 'batch_tokens': BATCH_TOKENS, 'batching': 'length_sorted'}
        save(out / 'reranker-probe.json', info)
        print(json.dumps(info), flush=True)
        return
    rows = json.loads((out / 'candidates.json').read_text())
    results, total_pairs, total_tokens = [], 0, 0
    started = time.monotonic()
    for q, row in zip(queries, rows, strict=True):
        case_started = time.monotonic()
        ids = row['candidates']
        pairs = [(q['question'], by_id[i]['title']+'\n'+by_id[i]['text']) for i in ids]
        scores, tokens = rank(pairs, tokenizer, model, args.batch_size, args.deadline)
        order = sorted(zip(ids,scores), key=lambda x:(-x[1], x[0]))
        found, seen = [], set()
        for i, score in order:
            c = by_id[i]
            if c['canonical_id'] not in seen:
                seen.add(c['canonical_id'])
                found.append({'chunk_id': i, 'document_id': c['canonical_id'], 'score': score})
        total_pairs += len(pairs)
        total_tokens += tokens
        results.append({'id': q['id'], 'hits': found[:20], 'pairs': len(pairs), 'seconds': time.monotonic()-case_started})
        save(out / 'qwen.json', results)
        save(out / 'progress.json', {'stage':'reranking','completed':len(results),'total':len(queries)})
    info = {'questions':len(results),'pairs':total_pairs,'tokens':total_tokens,'seconds':time.monotonic()-started,
            'load_seconds':loaded,'peak_cuda_bytes':torch.cuda.max_memory_allocated(),'dtype':'bfloat16',
            'batch_size':args.batch_size,'batch_tokens':BATCH_TOKENS,'batching':'length_sorted'}
    save(out / 'reranker-stats.json', info)
    print(json.dumps(info), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('stage', choices=['embed','rerank'])
    p.add_argument('--root', default='.')
    p.add_argument('--embedding', default='models/embedding')
    p.add_argument('--reranker', default='models/reranker')
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--probe', action='store_true')
    p.add_argument('--deadline', type=float, required=True, help='Absolute Unix deadline shared by all stages')
    a = p.parse_args()
    torch.manual_seed(20261015)
    if not torch.cuda.is_available():
        raise RuntimeError('A CUDA GPU is required; no silent CPU fallback')
    (embed if a.stage=='embed' else rerank)(a)
