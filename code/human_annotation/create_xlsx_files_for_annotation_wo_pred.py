"""
python create_xlsx_files_for_annotation_wo_pred.py \
       --runfile_dirs "/store/scratch/n3thakur/trec-rag-2024/trec2024-rag/support_eval/2025/trec25-rag/runs/anon/gen" "/store/scratch/n3thakur/trec-rag-2024/trec2024-rag/support_eval/2025/trec25-rag/runs/anon/auggen" \
       --topics_filepath "/store/scratch/n3thakur/trec-rag-2024/trec2024-rag/support_eval/2025/trec25-rag/topics/trec25_narratives_final.json" \
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

# Create a Pandas Excel writer using XlsxWriter as the engine
def tsv_to_excel(tsv_file, excel_file, answer_col=1, document_col=0, context_col=5):
    
    # Read the TSV file into a DataFrame
    df = pd.read_csv(tsv_file, sep='\t', encoding='utf-8', quoting=csv.QUOTE_MINIMAL)

    df.style.set_properties(**{
    'text-align': 'left',
    'font-size': '20pt'})
    
    writer = pd.ExcelWriter(excel_file, engine='xlsxwriter')

    # Convert the dataframe to an XlsxWriter Excel object
    df.to_excel(writer, sheet_name='Sheet1', index=False, freeze_panes=(1, 0))

    # Access the XlsxWriter workbook and worksheet objects
    workbook = writer.book
    worksheet = writer.sheets['Sheet1']

    font_format = workbook.add_format({'text_wrap': True})  # Adjust font size as needed

    # Set the column widths and font size for the entire worksheet
    COLUMN_WIDTHS = [125, 50, 7, 7, 7, 125, 12]
    for idx, width in enumerate(COLUMN_WIDTHS):
        worksheet.set_column(idx, idx, width, font_format)

    # set other columns as hidden
    for col in range(len(COLUMN_WIDTHS), df.shape[1]):
        worksheet.set_column(col, col, None, None, {'hidden': True})

    # Loop through cells where individual text rows need to be highlighted
    for row in range(len(df)):
        answer_text = df.iloc[row, answer_col]  # Get the cell value for the answer column
        cell_value = df.iloc[row, context_col]  # Get the cell value for the context column
        try:
            if answer_text in cell_value:
                bold = workbook.add_format({'bold': True})
                # highlight part in bold
                start_pos = cell_value.find(answer_text)
                end_pos = start_pos + len(answer_text)

                if start_pos == 0 and end_pos == len(cell_value):
                    worksheet.write_rich_string(f'F{row + 2}', " ", bold, cell_value, " ")
                
                elif start_pos == 0:
                    worksheet.write_rich_string(f'F{row + 2}',
                            " ", bold, cell_value[start_pos:end_pos],
                            cell_value[end_pos:])
                
                elif end_pos == len(cell_value):
                    worksheet.write_rich_string(f'F{row + 2}',
                            cell_value[:start_pos],
                            bold, cell_value[start_pos:end_pos])
                else:
                    worksheet.write_rich_string(f'F{row + 2}',
                                cell_value[:start_pos],
                                bold, cell_value[start_pos:end_pos],
                                cell_value[end_pos:])
        except:
            pass
    
    # highlight a row & column if it is exactly the same as the previous row & column    
    yellow_format = workbook.add_format({'bg_color': 'yellow', 'text_wrap': True, 'bottom': 1})
    resize_format = workbook.add_format({'text_wrap': True, 'bottom': 1})
    
    # Alternate the colors in the highlight formats: https://stackoverflow.com/a/63136692
    for row in range(1, len(df)):
        if df.iloc[row, document_col] == df.iloc[row - 1, document_col]:
            worksheet.write(row, document_col, df.iloc[row, document_col], resize_format)
        else:
            worksheet.write(row, document_col, df.iloc[row, document_col], yellow_format)
    
    # formula to check if any of the columns C, D, E has an "x" value
    for row in range(1, len(df) + 1):
        worksheet.write_formula(f'G{row + 1}', f'=IF(COUNTIF(C{row + 1}:E{row + 1}, "x") = 1, "Yes", "No")')

    border_format = workbook.add_format({'bottom': 1, 'text_wrap': True})

    # Iterate over rows and add the border
    for row_num in range(0, df.shape[0] + 1):
        worksheet.set_row(row_num, None, border_format)
    
    # Close the Pandas Excel writer and output the Excel file
    writer._save()

def load_topics(topics_file):
    queries = {}
    
    # json load
    with open(topics_file, 'r') as f:
        data = json.load(f)
    
    for id, row in enumerate(data):
        query_id = str(row['id'])
        query = row['narrative']
        queries[query_id] = query
    
    return queries


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    parser.add_argument("--runfile_dirs", required=True, type=str, nargs="+", help="Path to the runfiles directory")
    parser.add_argument("--topics_filepath", required=True, type=str, help="Path to the qrels file")
    parser.add_argument("--output_filedir", required=True, type=str, help="Path to the qrels file")
    parser.add_argument("--lucene_index", required=False, type=str, default="/mnt/users/n3thakur/cache/indexes/lucene-inverted.msmarco-v2.1-doc-segmented.20240418.4f9675")
    parser.add_argument("--max_citations", required=False, type=int, default=1)

    args = parser.parse_args()

    # pyserini load index
    seg_index = LuceneSearcher(args.lucene_index)

    # Load qrels filepath
    topics = load_topics(args.topics_filepath)
    print(f"Loaded {len(topics)} topics.")

    results = {}

    # walk through the results file
    for runfile_dir in args.runfile_dirs:

        input_filepaths = os.listdir(runfile_dir)
        os.makedirs(runfile_dir, exist_ok=True)
        task = runfile_dir.split("/")[-1]

        # filter the runfiles
        input_filepaths = [input_filepath for input_filepath in input_filepaths]
        print(f"Loaded {len(input_filepaths)} runfiles for {task}.")

        for input_filepath in tqdm(input_filepaths, total=len(input_filepaths), desc="All Submissions: "):
            print(f"Processing {input_filepath} for {task}...")
            
            # load all the support files
            task = runfile_dir.split("/")[-1]
            
            # load all the runfiles
            with open(os.path.join(runfile_dir, input_filepath), 'r') as fin:
                runfiles = {}
                for line in tqdm(fin, desc="All Topics:"):
                    data = json.loads(line)
                    runfiles[str(data["metadata"]["narrative_id"])] = data

            # construct the final results -> topic_id -> doc_id: ["answer", "support", "citations"]
            for topic_id in topics:
                
                if topic_id not in results:
                    results[topic_id] = {}
                
                if topic_id not in runfiles:
                    continue

                # get the document ids for each topic_id from the runfiles
                document_ids = runfiles[topic_id]['references']
                
                for doc_id in document_ids:
                    if doc_id not in results[topic_id]:
                        results[topic_id][doc_id] = []
                
                # get the support for each answer
                prev_answer, whole_answer = "", ""

                ### check which key has the responses
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
                    
                    if type(citations) is dict:
                        # sort based on highest to lowest value
                        citations = [c[0] for c in sorted(citations.items(), key=lambda x: x[1], reverse=True)]
                    
                    #### check if citation ids 0,1,2 or are document_ids
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
                                "answer_id": idy,
                                "prev_answer": prev_answer.strip(),
                                "whole_answer": whole_answer.strip()
                            })
                    
                    # last update the previous answer
                    try:
                        prev_answer += answer_sentence + " [" + ", ".join([str(document_ids.index(citation_id)) for citation_id in citation_ids]) + "] "
                    except:
                        prev_answer += answer_sentence + " [] "

    # write the results to the output file
    os.makedirs(os.path.join(args.output_filedir, 'tsv'), exist_ok=True)
    for topic_id in results:
        filename = f"{topic_id}: {topics[topic_id][:100]}.tsv"
        with open(os.path.join(args.output_filedir, 'tsv', filename), "w") as fout:
            writer = csv.writer(fout, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
            prev_doc_text = ""
            writer.writerow(["CITED PASSAGE", "SENTENCE", "FULL", "PARTIAL", "NONE", "SENTENCE CONTEXT", "COMPLETED?", "TASK", "RUNID", "DOCID", "ANSWERID"])
            print(f"Writing {topic_id}: {topics[topic_id]}")
            
            for doc_id in results[topic_id]:
                if len(results[topic_id][doc_id]) == 0:
                    continue
                for row in results[topic_id][doc_id]:
                    try:
                        seg = json.loads(seg_index.doc(doc_id).raw())
                        doc_text = (seg['title'] + ': ' + seg['segment'])
                        writer.writerow([
                            doc_text.replace("\t", " "), # the document text
                            row["answer"].replace("\t", " "), # the answer sentence
                            "", # keep empty
                            "", # keep empty
                            "", # keep empty
                            row["whole_answer"].replace("\t", " "), # the full answer
                            "", # keep empty
                            row["task"], # the task
                            row["filepath"], # the runid
                            doc_id, # the docid
                            row["answer_id"] # the answer id
                        ])
                    except Exception as e:
                        print(f"Error processing doc_id {doc_id}: {e}")
                        continue
    
    # Convert the TSV files to Excel files
    for topic_id in results:
        os.makedirs(os.path.join(args.output_filedir, 'xlsx'), exist_ok=True)
        filename_tsv = f"{topic_id}: {topics[topic_id][:100]}.tsv"
        filename_xlsx = f"{topic_id}: {topics[topic_id][:100]}.xlsx"
        tsv_file = os.path.join(args.output_filedir, 'tsv', filename_tsv)
        excel_file = os.path.join(args.output_filedir, 'xlsx', filename_xlsx)
        tsv_to_excel(tsv_file, excel_file)