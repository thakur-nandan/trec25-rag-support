"""
python create_xlsx_files_for_annotation_wo_pred_bugfix.py \
       --runfile_dirs "/store/scratch/n3thakur/trec-rag-2024/trec2024-rag/support_eval/2025/trec25-rag/runs/anon/gen" "/store/scratch/n3thakur/trec-rag-2024/trec2024-rag/support_eval/2025/trec25-rag/runs/anon/auggen" \
       --runfile_ids 'past-fifth' 'status-produce' 'shortly-seek' 'arab-exciting' 'activity-manufacturer' 'worry-repeat' 'badly-salary' 'anger-interpretation' 'closer-submit' 'cook-desperate' 'that-darkness' 'know-author' 'contemporary-dispute' 'truth-muslim' 'entrance-population' 'transportation-error' 'distinguish-fitness' 'plant-indeed' 'pure-lawyer' 'arrange-final' 'commercial-childhood' 'child-poison' 'keep-select' 'institution-painting' 'gently-disagree' \
       --topics_filepath "/store/scratch/n3thakur/trec-rag-2024/trec2024-rag/support_eval/2025/trec25-rag/topics/trec25_narratives_final.json" \
       --topic_id "144" \
       --output_filedir "/store/scratch/n3thakur/trec25-rag-support/wo_prediction" \
       --lucene_index "/store/scratch/rpradeep/nuggetizer/data/indexes/lucene-inverted.msmarco-v2.1-doc-segmented"
"""

import argparse
import json
import os
import csv
from tqdm.autonotebook import tqdm
from pyserini.search.lucene import LuceneSearcher
import pandas as pd

# ---------------------------------------------------------
# Existing function preserved
# ---------------------------------------------------------
def tsv_to_excel(tsv_file, excel_file, answer_col=1, document_col=0, context_col=5):
    df = pd.read_csv(tsv_file, sep='\t', encoding='utf-8', quoting=csv.QUOTE_MINIMAL)
    writer = pd.ExcelWriter(excel_file, engine='xlsxwriter')
    df.to_excel(writer, sheet_name='Sheet1', index=False, freeze_panes=(1, 0))

    workbook = writer.book
    worksheet = writer.sheets['Sheet1']
    font_format = workbook.add_format({'text_wrap': True})

    COLUMN_WIDTHS = [125, 50, 7, 7, 7, 125, 12]
    for idx, width in enumerate(COLUMN_WIDTHS):
        worksheet.set_column(idx, idx, width, font_format)

    for col in range(len(COLUMN_WIDTHS), df.shape[1]):
        worksheet.set_column(col, col, None, None, {'hidden': True})

    # Bold highlight for answer span matches
    for row in range(len(df)):
        try:
            answer_text = df.iloc[row, answer_col]
            cell_value = df.iloc[row, context_col]
            if answer_text in cell_value:
                bold = workbook.add_format({'bold': True})
                start = cell_value.find(answer_text)
                end = start + len(answer_text)

                if start >= 0:
                    worksheet.write_rich_string(
                        f'F{row+2}',
                        cell_value[:start],
                        bold, cell_value[start:end],
                        cell_value[end:]
                    )
        except:
            pass

    writer._save()


def load_topics(topics_file):
    queries = {}
    with open(topics_file, 'r') as f:
        data = json.load(f)
    for row in data:
        queries[str(row['id'])] = row['narrative']
    return queries

# ---------------------------------------------------------
# Modified Main
# ---------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--runfile_dirs", required=True, nargs="+")
    parser.add_argument("--runfile_ids", required=True, nargs="+",
                        help="List of specific runfile filenames to load")
    parser.add_argument("--topic_id", required=True, type=str,
                        help="Single topic ID to process")
    parser.add_argument("--topics_filepath", required=True)
    parser.add_argument("--output_filedir", required=True)
    parser.add_argument("--lucene_index", required=False, type=str,
                        default="/mnt/users/n3thakur/cache/indexes/lucene-inverted.msmarco-v2.1-doc-segmented.20240418.4f9675")
    parser.add_argument("--max_citations", required=False, type=int, default=1)

    args = parser.parse_args()

    seg_index = LuceneSearcher(args.lucene_index)
    topics = load_topics(args.topics_filepath)

    if args.topic_id not in topics:
        raise ValueError(f"Topic {args.topic_id} not found in topics file.")

    topic_id = args.topic_id
    topic_text = topics[topic_id]

    results = {topic_id: {}}
    run_ids = set(args.runfile_ids)

    # ---------------------------------------------------------
    # Load only runfiles requested
    # ---------------------------------------------------------
    for runfile_dir in args.runfile_dirs:
        task = os.path.basename(runfile_dir)

        available_files = os.listdir(runfile_dir)
        selected_files = [f for f in available_files if any(rid in f for rid in run_ids)]

        print(f"[{task}] Found {selected_files} matching runfiles.")

        for fname in tqdm(selected_files, desc=f"{task}: Runfiles"):
            print(f"Processing {fname}...")

            # Load runfile
            with open(os.path.join(runfile_dir, fname), 'r') as fin:
                runfiles = {}
                for line in fin:
                    data = json.loads(line)
                    rid = str(data["metadata"]["narrative_id"])
                    runfiles[rid] = data

            # Skip runfile if topic missing
            if topic_id not in runfiles:
                print(f"Topic {topic_id} not found in {fname}, skipping...")
                continue

            data = runfiles[topic_id]
            document_ids = data['references']

            for doc_id in document_ids:
                if doc_id not in results[topic_id]:
                    results[topic_id][doc_id] = []

            # Determine answer field
            answer_key = "answer" if "answer" in data else "responses"
            answers = data[answer_key]

            if len(answers) == 0:
                print(f"Answer is empty found for topic {topic_id} in {fname}, skipping...")
                continue

            whole_answer = " ".join([a["text"] for a in answers]).strip()
            prev_answer = ""

            for idy, answer in enumerate(answers):
                answer_text = answer["text"]
                citations = answer["citations"]

                if isinstance(citations, dict):
                    citations = [c[0] for c in sorted(citations.items(), key=lambda x: x[1], reverse=True)]

                # Map integer citation indices → document_ids
                if len(citations) > 0:
                    if isinstance(citations[0], int):
                        citation_ids = [document_ids[c] for c in citations]
                    else:
                        citation_ids = citations
                else:
                    citation_ids = []

                for idx in range(min(len(citation_ids), args.max_citations)):
                    cid = citation_ids[idx]
                    results[topic_id][cid].append({
                        "task": task,
                        "filepath": fname,
                        "answer": answer_text.strip(),
                        "answer_id": idy,
                        "prev_answer": prev_answer.strip(),
                        "whole_answer": whole_answer
                    })

                prev_answer += answer_text + f" {[str(document_ids.index(cid)) if cid in document_ids else '' ]} "

    # ---------------------------------------------------------
    # Write TSV for this single topic
    # ---------------------------------------------------------
    os.makedirs(os.path.join(args.output_filedir, "tsv"), exist_ok=True)

    out_tsv = os.path.join(
        args.output_filedir,
        "tsv",
        f"{topic_id}: {topic_text[:100]}.tsv"
    )

    print(f"Writing TSV → {out_tsv}")

    with open(out_tsv, "w") as fout:
        writer = csv.writer(fout, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["CITED PASSAGE", "SENTENCE", "FULL", "PARTIAL", "NONE",
                         "SENTENCE CONTEXT", "COMPLETED?", "TASK", "RUNID", "DOCID", "ANSWERID"])

        for doc_id in results[topic_id]:
            for row in results[topic_id][doc_id]:
                try:
                    seg = json.loads(seg_index.doc(doc_id).raw())
                    text = (seg['title'] + ': ' + seg['segment']).replace("\t", " ")

                    writer.writerow([
                        text,
                        row["answer"],
                        "",
                        "",
                        "",
                        row["whole_answer"],
                        "",
                        row["task"],
                        row["filepath"],
                        doc_id,
                        row["answer_id"]
                    ])
                except Exception as e:
                    print(f"Error doc {doc_id}: {e}")

    # ---------------------------------------------------------
    # Convert TSV → XLSX
    # ---------------------------------------------------------
    os.makedirs(os.path.join(args.output_filedir, "xlsx"), exist_ok=True)

    out_xlsx = os.path.join(
        args.output_filedir,
        "xlsx",
        f"{topic_id}: {topic_text[:100]}.xlsx"
    )

    tsv_to_excel(out_tsv, out_xlsx)
    print(f"Created: {out_xlsx}")
