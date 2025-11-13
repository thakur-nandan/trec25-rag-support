"""
python create_xlsx_files_for_annotation_w_pred.py \
       --runfile_dirs "/mnt/users/n3thakur/2025/projects/2025-trecrag/trec25-rag/runs/anon/gen" \
       --support_dirs "/mnt/users/n3thakur/2025/projects/2025-trecrag/support/results/gen/qwen3-8b-closed-book" \
       --topics_filepath "/mnt/users/n3thakur/2025/projects/2025-trecrag/trec25-rag/topics/trec25_narratives_final.json" \
       --output_filedir "/mnt/users/n3thakur/2025/projects/2025-trecrag/support/human_annotation/with_qwen3_8b_predictions"
"""

import argparse
import json
import os
import re
import csv
from tqdm.autonotebook import tqdm
from pyserini.search.lucene import LuceneSearcher
import pandas as pd


def clean_text(text):
    paragraphs = re.split(r'\n\s*\n', text)

    clean_paragraphs = []
    for para in paragraphs:
        # Step 2: Replace newlines and multiple spaces within the paragraph with a single space
        clean_para = re.sub(r'\s+', ' ', para).strip()
        if clean_para:  # skip empty paragraphs
            clean_paragraphs.append(clean_para)

    # Step 3: Join paragraphs with a single newline between them
    text = '\n\n'.join(clean_paragraphs)
    return text


# Create a Pandas Excel writer using XlsxWriter as the engine
def tsv_to_excel(tsv_file, excel_file, answer_col=1, document_col=0, context_col=6):
    # Read TSV
    df = pd.read_csv(tsv_file, sep='\t', encoding='utf-8', quoting=csv.QUOTE_MINIMAL)

    writer = pd.ExcelWriter(excel_file, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1', index=False, freeze_panes=(1, 0))

    workbook = writer.book
    worksheet = writer.sheets['Sheet1']

    font_format = workbook.add_format({'text_wrap': True})
    COLUMN_WIDTHS = [125, 50, 7, 7, 7, 12, 125, 12]
    for idx, width in enumerate(COLUMN_WIDTHS):
        worksheet.set_column(idx, idx, width, font_format)

    # Hide extra columns
    for col in range(len(COLUMN_WIDTHS), df.shape[1]):
        worksheet.set_column(col, col, None, None, {'hidden': True})

    # Formats
    bold_format = workbook.add_format({'bold': True, 'text_wrap': True, 'bottom': 1})
    yellow_format = workbook.add_format({'bg_color': '#FFFF00', 'text_wrap': True, 'bottom': 1})
    normal_format = workbook.add_format({'text_wrap': True, 'bottom': 1})

    # Preprocess doc column for duplicate check
    cleaned_docs = df.iloc[:, document_col].astype(str).apply(
    lambda x: re.sub(r'<<EVIDENCE:(.*?)>>', r'\1', x)
    )

    for row in range(len(df)):
        cell_value = str(df.iloc[row, document_col])
        parts = []
        last_idx = 0

        # Split into bold evidence and normal text
        for match in re.finditer(r'<<EVIDENCE:(.*?)>>', cell_value):
            start, end = match.span()
            evidence_text = match.group(1)

            if start > last_idx:
                parts.append(cell_value[last_idx:start])

            parts.append(bold_format)
            parts.append(evidence_text)
            last_idx = end

        if last_idx < len(cell_value):
            parts.append(cell_value[last_idx:])

        # If there was any evidence, use write_rich_string
        if any(isinstance(p, type(bold_format)) for p in parts):
            worksheet.write_rich_string(row + 1, document_col, *parts)
        else:
            worksheet.write(row + 1, document_col, cell_value, normal_format)

    # Highlight repeated doc ids in yellow
    for row in range(1, len(df)):
        if cleaned_docs[row] != cleaned_docs[row - 1]:
            # Only highlight if first occurrence of a new doc
            worksheet.write(row, document_col, cleaned_docs[row], yellow_format)

    # Formula to check if any of the columns C, D, E has "x"
    for row in range(1, len(df) + 1):
        worksheet.write_formula(f'H{row + 1}', f'=IF(COUNTIF(C{row + 1}:E{row + 1}, "x") = 1, "Yes", "No")')

    # Bold border after every row
    # border_format = workbook.add_format({'bottom': 1, 'text_wrap': True})
    # for row_num in range(0, df.shape[0] + 1):
    #     worksheet.set_row(row_num, None, border_format)

    writer._save()





def load_topics(topics_file):
    queries = {}
    with open(topics_file, 'r') as f:
        data = json.load(f)
    for id, row in enumerate(data):
        query_id = str(row['id'])
        query = row['narrative']
        queries[query_id] = query
    return queries


def extract_support(solution_str):
    answer_pattern = r'<support>(.*?)</support>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    try:
        return matches[0].group(1).strip()
    except IndexError:
        return None


def extract_evidence(solution_str):
    answer_pattern = r'<evidence>(.*?)</evidence>'
    match = re.finditer(answer_pattern, solution_str, re.DOTALL)
    matches = list(match)
    try:
        return matches[0].group(1).strip()
    except IndexError:
        return None


def postprocess_support(response):
    citation = None
    if response:
        if "full support" in response.lower():
            citation = "FS"
        elif "partial support" in response.lower():
            citation = "PS"
        elif "no support" in response.lower():
            citation = "NS"
    return citation


def load_support_results(support_filepath):
    support_results = {}
    try:
        with open(support_filepath, 'r') as fin:
            for line in fin:
                data = json.loads(line)
                topic_id = str(data["narrative_id"])
                if topic_id not in support_results:
                    support_results[topic_id] = {}

                citation_id = str(data["citation_id"])
                answer_id = data["answer_index"]
                support = postprocess_support(extract_support(data["support_output"]))
                evidence = extract_evidence(data["support_output"])

                if answer_id not in support_results[topic_id]:
                    support_results[topic_id][answer_id] = {}

                support_results[topic_id][answer_id][citation_id] = {
                    "support": support,
                    "evidence": evidence,
                }
    except FileNotFoundError:
        print(f"Support file {support_filepath} not found. Skipping...")
    return support_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--runfile_dirs", required=True, type=str, nargs="+", help="Path to the runfiles directory")
    parser.add_argument("--support_dirs", required=False, type=str, nargs="+", help="Path to the support files directory")
    parser.add_argument("--topics_filepath", required=True, type=str, help="Path to the qrels file")
    parser.add_argument("--output_filedir", required=True, type=str, help="Path to the qrels file")
    parser.add_argument("--temperature", required=False, type=float, default=0.1)
    parser.add_argument("--num_sequence", required=False, type=int, default=1)
    parser.add_argument("--lucene_index", required=False, type=str, default="/mnt/users/n3thakur/cache/indexes/lucene-inverted.msmarco-v2.1-doc-segmented.20240418.4f9675")
    parser.add_argument("--max_citations", required=False, type=int, default=1)

    args = parser.parse_args()

    seg_index = LuceneSearcher(args.lucene_index)
    topics = load_topics(args.topics_filepath)
    print(f"Loaded {len(topics)} topics.")

    results = {}

    for runfile_dir, support_dir in zip(args.runfile_dirs, args.support_dirs):
        input_filepaths = os.listdir(runfile_dir)
        os.makedirs(runfile_dir, exist_ok=True)
        task = runfile_dir.split("/")[-1]

        input_filepaths = [input_filepath for input_filepath in input_filepaths if input_filepath != "know-author"]
        print(f"Loaded {len(input_filepaths)} runfiles for {task}.")

        for input_filepath in tqdm(input_filepaths, total=len(input_filepaths), desc="All Submissions: "):
            print(f"Processing {input_filepath} for {task}...")
            support_predictions = load_support_results(os.path.join(support_dir, f"{input_filepath}.v0.prompt.temp.0.6.jsonl"))

            task = runfile_dir.split("/")[-1]

            with open(os.path.join(runfile_dir, input_filepath), 'r') as fin:
                runfiles = {}
                for line in tqdm(fin, desc="All Topics:"):
                    data = json.loads(line)
                    runfiles[str(data["metadata"]["narrative_id"])] = data

            for topic_id in topics:
                if topic_id not in results:
                    results[topic_id] = {}

                if topic_id not in runfiles:
                    continue

                document_ids = runfiles[topic_id]['references']
                for doc_id in document_ids:
                    if doc_id not in results[topic_id]:
                        results[topic_id][doc_id] = []

                prev_answer, whole_answer = "", ""

                answer_key = None
                if "answer" in runfiles[topic_id]:
                    answer_key = "answer"
                elif "responses" in runfiles[topic_id]:
                    answer_key = "responses"

                for answer in runfiles[topic_id][answer_key]:
                    whole_answer += answer["text"] + " "
                whole_answer = whole_answer.strip()

                for idy, answer in enumerate(runfiles[topic_id][answer_key]):
                    answer_sentence = answer["text"]
                    citations = answer["citations"]
                    predictions = support_predictions[topic_id][idy] if topic_id in support_predictions and idy in support_predictions[topic_id] else {}

                    if type(citations) is dict:
                        citations = [c[0] for c in sorted(citations.items(), key=lambda x: x[1], reverse=True)]

                    if len(citations) > 0:
                        if type(citations[0]) is int:
                            citation_ids = [document_ids[citation] for citation in citations]
                        else:
                            citation_ids = citations

                        for idx in range(0, args.max_citations):
                            if idx >= len(citation_ids):
                                break
                            citation_id = citation_ids[idx]
                            results[topic_id][citation_id].append({
                                "task": task,
                                "filepath": input_filepath,
                                "answer": answer_sentence.strip(),
                                "prediction": predictions[citation_id]["support"] if citation_id in predictions else "",
                                "evidence": predictions[citation_id]["evidence"] if citation_id in predictions else "",
                                "answer_id": idy,
                                "prev_answer": prev_answer.strip(),
                                "whole_answer": whole_answer.strip()
                            })

                    try:
                        prev_answer += answer_sentence + " [" + ", ".join([str(document_ids.index(citation_id)) for citation_id in citation_ids]) + "] "
                    except:
                        prev_answer += answer_sentence + " [] "

    os.makedirs(os.path.join(args.output_filedir, 'tsv'), exist_ok=True)
    for topic_id in results:
        filename = f"{topic_id}: {topics[topic_id][:100]}.tsv"
        with open(os.path.join(args.output_filedir, 'tsv', filename), "w") as fout:
            writer = csv.writer(fout, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
            writer.writerow(["CITED PASSAGE", "SENTENCE", "FULL", "PARTIAL", "NONE", "PREDICTION", "SENTENCE CONTEXT", "COMPLETED?", "TASK", "RUNID", "DOCID", "ANSWERID"])
            print(f"Writing {topic_id}: {topics[topic_id]}")

            for doc_id in results[topic_id]:
                if len(results[topic_id][doc_id]) == 0:
                    continue
                for row in results[topic_id][doc_id]:
                    try:
                        seg = json.loads(seg_index.doc(doc_id).raw())
                        doc_text = (seg['title'] + ': ' + seg['segment'])
                        evidence = row["evidence"] if "evidence" in row else ""
                        if len(evidence) > 0 and evidence in doc_text:
                            doc_text = doc_text.replace(evidence, f"<<EVIDENCE:{evidence}>>")

                        writer.writerow([
                            "\n" + clean_text(doc_text.replace("\t", " ")),
                            row["answer"].replace("\t", " "),
                            "",
                            "",
                            "",
                            row["prediction"],
                            "\n" + row["whole_answer"].replace("\t", " "),
                            "",
                            row["task"],
                            row["filepath"],
                            doc_id,
                            row["answer_id"]
                        ])
                    except Exception as e:
                        print(f"Error processing doc_id {doc_id}: {e}")
                        continue

    for topic_id in results:
        os.makedirs(os.path.join(args.output_filedir, 'xlsx'), exist_ok=True)
        filename_tsv = f"{topic_id}: {topics[topic_id][:100]}.tsv"
        filename_xlsx = f"{topic_id}: {topics[topic_id][:100]}.xlsx"
        tsv_file = os.path.join(args.output_filedir, 'tsv', filename_tsv)
        excel_file = os.path.join(args.output_filedir, 'xlsx', filename_xlsx)
        tsv_to_excel(tsv_file, excel_file)
