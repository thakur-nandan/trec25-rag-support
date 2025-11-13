export CUDA_VISIBLE_DEVICES=2

export OMP_NUM_THREADS=1
export HF_HOME=/mnt/users/n3thakur/cache
export DATASETS_HF_HOME=/mnt/users/n3thakur/cache
export PYSERINI_CACHE=/mnt/users/n3thakur/cache
export VLLM_ATTENTION_BACKEND=TRITON_ATTN_VLLM_V1
export RESULTS_DIR=/mnt/users/n3thakur/2025/projects/2025-trecrag/trec25-rag/runs/anon
export OUTPUT_DIR=/mnt/users/n3thakur/2025/projects/2025-trecrag/support/results/

# ALL_RUNS=(
#     arrange-final cook-desperate figure-civilian keep-select relevant-testimony store-ethics viewer-whisper bite-stir criteria-african gently-disagree neither-fitness review-basic that-darkness both-rate editor-elsewhere image-climate peer-necessarily sake-frame transportation-error closer-submit entrance-population internal-withdraw radical-twice shock-description vary-occasion
# )

ALL_RUNS=(
    keep-select relevant-testimony store-ethics viewer-whisper bite-stir criteria-african gently-disagree neither-fitness review-basic that-darkness both-rate editor-elsewhere image-climate peer-necessarily sake-frame transportation-error closer-submit entrance-population internal-withdraw radical-twice shock-description vary-occasion
)

for MODEL_NAME in Qwen/Qwen3-8B; do
    for RUN_NAME in ${ALL_RUNS[@]}; do
        python -m pairwise_eval_qwen_v2 \
            --model_name_or_path ${MODEL_NAME} \
            --input_filepath ${RESULTS_DIR}/auggen/${RUN_NAME} \
            --output_dir ${OUTPUT_DIR}/auggen/qwen3-8b-closed-book/ \
            --output_file ${RUN_NAME}.v0.prompt.temp.0.6.jsonl \
            --lucene_index /mnt/users/n3thakur/cache/indexes/lucene-inverted.msmarco-v2.1-doc-segmented.20240418.4f9675 \
            --max_completion_tokens 4096 \
            --temperature 0.6 \
            --top_p 0.95 \
            --top_k 20 \
            --min_p 0 
    done
done

# ALL_RUNS=(
#     "know-author"
# )

# for MODEL_NAME in Qwen/Qwen3-8B; do
#     for RUN_NAME in ${ALL_RUNS[@]}; do
#         python -m pairwise_eval_qwen \
#             --model_name_or_path ${MODEL_NAME} \
#             --input_filepath ${RESULTS_DIR}/gen/${RUN_NAME} \
#             --output_dir ${OUTPUT_DIR}/gen/qwen3-8b-closed-book/ \
#             --output_file ${RUN_NAME}.v0.prompt.temp.0.6.jsonl \
#             --lucene_index /mnt/users/n3thakur/cache/indexes/lucene-inverted.msmarco-v2.1-doc-segmented.20240418.4f9675 \
#             --max_completion_tokens 4096 \
#             --temperature 0.6 \
#             --top_p 0.95 \
#             --top_k 20 \
#             --min_p 0 
#     done
# done