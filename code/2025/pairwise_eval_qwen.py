import json
from pathlib import Path
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import torch
import tqdm
import os
from typing import List, Dict
from pyserini.search.lucene import LuceneSearcher
import datasets
import copy

def create_final_prompt(
    answer: str,
    statement: str,
    citation: str,
) -> List[Dict[str, str]]:
    """
    Creates a prompt for nugget assignment (support, partial_support, or not_support).
    """
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant in verying whether the statement can by supported by its corresponding citation."
        },
        {
            "role": "user",
            "content": get_prompt(answer, statement, citation),
        },
    ]

def get_prompt(
    answer: str,
    statement: str,
    citation: str
) -> str:
    SUPPORT_EVAL_PROMPT = """
    In this task, you will evaluate whether each statement is supported by its corresponding citation document. 
    Note that the system responses may appear very fluent and well-formed, but contain slight inaccuracies that are not easy to discern at first glance. 
    Pay close attention to the text. Read it carefully as you would when proofreading. 
    
    It may be helpful to ask yourself whether it is accurate to say "according to the citation document" with a
    statement following this phrase. Be sure to check all of the information in the statement. You will be given three options:

    - "Full Support": All of the information in the statement is supported in the citation document.
    - "Partial Support": Only some of the information is supported in the citation document, but other parts of the information are missing from the citation document.
    - "No Support": This citation document does not support any part of the statement.

    Think step-by-step under the <think> </think> tokens and please provide your response based on the information in the citation document. If you are unsure, use your best judgment. 
    Respond as either "Full Support", "Partial Support", or "No Support" under the <support>...</support> XML tag. 
    Provide your reasoning under the <reasoning>...</reasoning> XML tag and highlight the most likely sentence in the citation document your decision under the <evidence>...</evidence> XML tag. 
    If the label is "No Support", please respond with <evidence>No evidence found</evidence>.

    Response (for reference): {answer}
    Statement (from the response): {statement}
    Citation document: {citation}

    Response:
    """
    return SUPPORT_EVAL_PROMPT.format(answer=answer, statement=statement, citation=citation).strip()

def main(
        model_path: str, 
        input_filepath: str, 
        output_jsonl_path: str, 
        temperature: float, 
        top_k: int, 
        top_p: float,
        min_p: float,
        lucene_index: str,
        no_thinking: bool,
        max_citations: int = 3,
        gpu_memory_utilization: float = 0.6):
    
    # pyserini load index

    seg_index = LuceneSearcher(lucene_index)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=8192,  # Adjusted for Qwen3-8B max length
        top_p=top_p,
        top_k=top_k,
        min_p=min_p
    )

    num_gpus = torch.cuda.device_count()
    llm = LLM(
        model=model_path,
        tensor_parallel_size=num_gpus,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=32768,
        distributed_executor_backend="mp",
    )

    output_path = Path(output_jsonl_path)
    writer = output_path.open("a", encoding="utf-8")

    all_texts, all_rows = [], []

    print(f"Loading the input results: {input_filepath}")
    with open(input_filepath, "r", encoding="utf-8") as f:
        ds = [json.loads(line) for line in f]

    # load into a HF dataset for easy filtering
    ds = datasets.Dataset.from_list(ds)

    for row in tqdm.tqdm(ds, total=len(ds), desc="Processing entries"):
        # Determine unique ID based on available field
        document_ids = row["references"]
        
        if "answer" in row:
            answer_key = "answer"
        elif "responses" in row:
            answer_key = "responses"

        ### go through each row of answer and cited documents
        if len(row[answer_key]) == 0:
            continue

        full_answer = " ".join([line["text"] for line in row[answer_key]]).strip()

        for idx, line in enumerate(row[answer_key]):
            citations = []
            statement = line["text"]
            
            if len(line["citations"]) == 0:
                continue

            # if citations start msmarco_ then we do not need to map them to the actual document ids 
            citations = copy.deepcopy(line["citations"])

            if type(citations) is dict:
                # sort based on highest to lowest value
                citations = [c[0] for c in sorted(citations.items(), key=lambda x: x[1], reverse=True)]
            
            if len(citations) > 0:
                if type(citations[0]) is int:
                    citation_ids = [document_ids[citation] for citation in citations]
                else:
                    citation_ids = citations

            if len(citation_ids) > max_citations:
                citation_ids = citation_ids[:max_citations]  # only consider top 3 citations
            
            print(f"Processing answer {idx} with {len(citations)} citations....")

            for document_id in citation_ids:
                try:
                    seg = json.loads(seg_index.doc(document_id).raw())
                    corpus_text = seg['title'] + ': ' + seg['segment']   
                    prompt = create_final_prompt(full_answer, statement, corpus_text)
                    text = tokenizer.apply_chat_template(
                        prompt, tokenize=False, add_generation_prompt=True, 
                        enable_thinking=not no_thinking
                    )
                    all_texts.append(text)
                    all_rows.append({
                        "narrative_id": row["metadata"]["narrative_id"],
                        "narrative": row["metadata"]["narrative"],
                        "answer_index": idx,
                        "statement": statement,
                        "citation_id": document_id,
                        "citation_doc": corpus_text,
                    })

                except:
                    print(f"Document ID {document_id} not found in the index. Skipping.")
                    continue

    outputs = llm.generate(all_texts, sampling_params)

    for idx, output in enumerate(outputs):
        row = all_rows[idx]
        row["support_output"] = output.outputs[0].text
        writer.write(json.dumps(row, ensure_ascii=False) + "\n")
        writer.flush()

    writer.close()
    print(f"Wrote {len(ds)} records to {output_jsonl_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate nugget scores using Qwen3-8B")

    parser.add_argument("--input_filepath", type=str, required=True)
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--output_file_save", type=str, required=False, default=None)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--min_p", type=float, default=0.0)
    parser.add_argument("--max_completion_tokens", type=int, default=512)
    parser.add_argument("--results_json", type=str, required=False, default=None)
    parser.add_argument("--lucene_index", type=str, required=False, default="msmarco-v2.1-doc-segmented")
    parser.add_argument("--max_citations", required=False, type=int, default=3)
    parser.add_argument("--no_thinking", action="store_true")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.6)
    args = parser.parse_args()

    ## Ensure the output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    output_jsonl_path = os.path.join(args.output_dir, f"{args.output_file}")

    main(
        args.model_name_or_path, 
        args.input_filepath,
        output_jsonl_path, 
        args.temperature, 
        args.top_k,
        args.top_p,
        args.min_p,
        args.lucene_index, 
        args.no_thinking, 
        args.max_citations,
        args.gpu_memory_utilization)