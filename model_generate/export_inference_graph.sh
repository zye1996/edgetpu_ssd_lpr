WORK_DIRECTORY=/home/yzy/Documents/lp_detect
CUR_DIRECTORY=$(pwd)
cd ${WORK_DIRECTORY} || exit

python ~/Documents/TensorFlow/models/research/object_detection/export_inference_graph.py \
  --input_type image_tensor \
  --pipeline_config_path pretrained/pipeline.config \
  --trained_checkpoint_prefix training/model.ckpt-150000 \
  --output_directory /tmp/pycharm_project_158/export_graph/detection_model_ssd_300

cd ${CUR_DIRECTORY} || exit