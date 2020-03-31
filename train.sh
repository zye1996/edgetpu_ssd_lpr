WORK_DIRECTORY=/home/yzy/Documents/lp_detect
CUR_DIRECTORY=$(pwd)
cd ${WORK_DIRECTORY} || exit

PIPELINE_CONFIG_PATH=${WORK_DIRECTORY}/pretrained/pipeline.config
NUM_TRAIN_STEPS=150000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
MODEL_DIR=${WORK_DIRECTORY}/training

python ~/Documents/TensorFlow/models/research/object_detection/model_main.py \
  --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
  --num_train_steps=${NUM_TRAIN_STEPS} \
  --model_dir=${MODEL_DIR} \
  --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
  --alsologtostderr

cd ${CUR_DIRECTORY} || exit