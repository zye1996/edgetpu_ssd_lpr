WORK_DIRECTORY=/home/yzy/Documents/lp_detect
CUR_DIRECTORY=$(pwd)
cd ${WORK_DIRECTORY} || exit

python ~/Documents/TensorFlow/models/research/object_detection/export_tflite_ssd_graph.py \
    -pipeline_config_path=pretrained/pipeline.config \
    -trained_checkpoint_prefix=training/model.ckpt-150000 \
    -output_directory=/tmp/pycharm_project_158/export_graph/ \
    -add_postprocessing_op=true

cd ${CUR_DIRECTORY} || exit