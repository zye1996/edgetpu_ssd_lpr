import os
import io
import cv2
import pandas as pd
import numpy as np
import tensorflow as tf
import contextlib2
from PIL import Image
from collections import namedtuple, OrderedDict
from object_detection.utils import dataset_util
from object_detection.dataset_tools import tf_record_creation_util

dir = "/home/yzy/Documents/ccpd_dataset/ccpd_base"

flags = tf.app.flags
flags.DEFINE_string('csv_input', '', 'Path to the CSV input')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
flags.DEFINE_string('label', '', 'Name of class label')
flags.DEFINE_string('img_path', '', 'Path to images')
FLAGS = flags.FLAGS


def save_csv(root_dir=dir, test_split=0.05):

    # print all files
    count = 0
    dat = []
    # sort to csv file
    for f in os.listdir(root_dir):
        bounding_box = f.split('-')[2]
        x_min, y_min = tuple(bounding_box.split('_')[0].split('&'))
        x_max, y_max = tuple(bounding_box.split('_')[1].split('&'))
        dat.append([f, x_min, y_min, x_max, y_max, "lp"])
        count+=1
        if count % 1000 == 0:
            print(count)
    df = pd.DataFrame(dat, columns=['filename', "xmin", "ymin", "xmax", "ymax", "class"])
    train_mask = np.random.rand(len(df)) > test_split
    df[train_mask].to_csv("ccpd_dataset_train.csv", index=False)
    df[~train_mask].to_csv("ccpd_dataset_test.csv", index=False)


def class_text_to_int(row_label):
    if row_label == FLAGS.label:  # 'ship':
        return 1
    # comment upper if statement and uncomment these statements for multiple labelling
    # if row_label == FLAGS.label0:
    #   return 1
    # elif row_label == FLAGS.label1:
    #   return 0
    else:
        return None

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

# creat tfrecord for one image
def create_tf_example(group, path):

    with tf.io.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    # check if the image format is matching with your images.
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    # width, height = image.size
    width, height = 720, 720

    for index, row in group.object.iterrows():


        # random crop
        top = max(0, row['ymax']-720)
        bottom = min(1163-720, row['ymin'])
        crop_top = np.random.randint(top, bottom)

        ymax = row['ymax']-crop_top
        ymin = row['ymin']-crop_top
        image = image.crop((0, crop_top, 720, crop_top+720))
        buf = io.BytesIO()
        image.save(buf, format='JPEG')
        encoded_jpg = buf.getvalue()

        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(ymin / height) #row['ymin'] / height)
        ymaxs.append(ymax / height) #row['ymax'] / height)
        #ymins.append(row['ymin'] / height)
        #ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def main(_):
    #writer = tf.io.TFRecordWriter(FLAGS.output_path)
    path =  FLAGS.img_path #os.path.join(os.getcwd(), FLAGS.img_path)
    examples = pd.read_csv(FLAGS.csv_input)
    grouped = split(examples, 'filename')
    num_shards= int(np.ceil(len(examples) // 1000))
    count = 0
    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
            tf_record_close_stack, FLAGS.output_path, num_shards)
        for index, group in enumerate(grouped):
            tf_example = create_tf_example(group, path)
            output_shard_index = index % num_shards
            output_tfrecords[output_shard_index].write(tf_example.SerializeToString())
            # writer.write(tf_example.SerializeToString())
            count += 1
            if count % 10000 == 0:
                print(count)

    # writer.close()
    output_path = os.path.join(os.getcwd(), FLAGS.output_path)
    print('Successfully created the TFRecords: {}'.format(output_path))


if __name__ == '__main__':
    # save_csv()
    tf.compat.v1.app.run()


