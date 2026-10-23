#!/usr/bin/env bash
# Build the teacher datasets used for training (run from the repository root).
#
#   MODEL=llada scripts/build_teachers.sh   # LLaDA-8B-Instruct, all four teacher cells
#   MODEL=dream scripts/build_teachers.sh   # Dream-v0-Instruct-7B, student/rejoin cell only
#
# GPU steps (trace collection, teacher-terminal collection) loop over shards
# sequentially here; on a cluster, run the shards as separate jobs.
set -euo pipefail
MODEL=${MODEL:-llada}
SHARDS=${SHARDS:-8}            # trace shards per dataset
TERMINAL_SHARDS=${TERMINAL_SHARDS:-128}
case "$MODEL" in
  llada) model_args=(--model-path GSAI-ML/LLaDA-8B-Instruct) ;;
  dream) model_args=(--model-path Dream-org/Dream-v0-Instruct-7B
                     --model-revision 05334cb9faaf763692dcf9d8737c642be2b2a6ae
                     --model-type Dream --mask-id 151666) ;;
  *) echo "MODEL must be llada or dream" >&2; exit 2 ;;
esac
traces=traces/$MODEL
work=work/$MODEL
teachers=data/teachers/$MODEL
mkdir -p "$traces" "$work"

# 1. Deterministic Fast-dLLM trajectories (threshold 0.9, length 256, block 32)
#    on the GSM8K and MATH training prompts (GPU).
for dataset in gsm8k math; do
  for ((shard = 0; shard < SHARDS; shard++)); do
    output="$traces/${dataset}_train_full_shard${shard}.json.gz"
    [[ -e "$output" ]] || python -m teacher.collect_traces "${model_args[@]}" \
      --dataset "$dataset" --num-shards "$SHARDS" --shard-index "$shard" \
      --threshold 0.9 --generation-length 256 --block-length 32 \
      --max-prompt-length 256 --output "$output"
  done
done

# 2. Trajectories with a correct final answer (CPU).
python -m teacher.convert "$traces"/*_train_full_shard*.json* --output-dir "$work/full"
python -m teacher.filter_correct --raw-dir "$traces" --input-dir "$work/full" \
  --output-dir "$work/correct"

# 3. Teacher cells: {student, teacher} states x {rejoin, terminal} oracle (CPU).
python -m teacher.build_matrix --raw-dir "$traces" --output "$teachers" \
  --teacher-shards "$TERMINAL_SHARDS"

if [[ "$MODEL" == llada ]]; then
  # 4. Teacher-state / terminal-oracle cell (GPU).
  for ((shard = 0; shard < TERMINAL_SHARDS; shard++)); do
    output="$teachers/teacher_terminal/shard${shard}.parquet"
    [[ -e "$output" ]] || python -m teacher.collect_teacher_terminal \
      --input "$teachers/teacher_terminal_inputs/shard${shard}.jsonl" --output "$output"
  done
  python -m teacher.finalize_matrix --root "$teachers" --correct-source "$work/correct"
else
  python -m teacher.finalize_matrix --root "$teachers" --correct-source "$work/correct" \
    --variants student_rejoin
fi
echo "teacher datasets: $teachers/*_correct"
