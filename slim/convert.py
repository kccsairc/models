from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import random
import os
import sys
import time

def filepath_label(path):
	data = []
	files = os.listdir(path)
        files_dir = [f for f in files if os.path.isdir(os.path.join(path, f))]
        for dir_name in files_dir:
            files = os.listdir(os.path.join(path, dir_name))
            for file_name in files:
                file_path = os.path.join(path, os.path.join(dir_name,file_name))
		data.append([file_path, dir_name])
	random.shuffle(data)

	filepaths = [path for inner in data for path in inner[::2]]
	labels = [label for inner in data for label in inner[1::2]]
	return filepaths, labels

def label_to_integer(labels):
	label_to_id = [] 
	for i in xrange(len(labels)):
		if labels[i] not in label_to_id:
			label_to_id.append(labels[i])
		labels[i] = label_to_id.index(labels[i])
	return labels, label_to_id
	
def convert_tfrecord_and_write(dataset_name,filepaths, labels, units, validations, tfrecord_dir):
	trains_end = len(filepaths) // units * units
	train_filepaths = zip(*[iter(filepaths[validations:trains_end])]*units)
	train_labels = zip(*[iter(labels[validations:trains_end])]*units)
	validation_filepaths = zip(*[iter(filepaths[0:validations])]*units)
	validation_labels = zip(*[iter(labels[0:validations])]*units)
	
	write_tfrecord(dataset_name,'train', train_filepaths, train_labels, tfrecord_dir)
	write_tfrecord(dataset_name,'validation', validation_filepaths, validation_labels, tfrecord_dir)

def _int64_feature(values):
	if not isinstance(values, (tuple, list)):
		values = [values]
	return tf.train.Feature(int64_list=tf.train.Int64List(value=values))

def _bytes_feature(values):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def image_to_tfexample(image_data, image_format, height, width, class_id):
	return tf.train.Example(features=tf.train.Features(feature={
		'image/encoded': _bytes_feature(image_data),
		'image/format': _bytes_feature(image_format),
		'image/class/label': _int64_feature(class_id),
		'image/height': _int64_feature(height),
		'image/width': _int64_feature(width)}))

def write_tfrecord(dataset_name,split_name, filepath_lists, label_lists, tfrecord_dir):
	jpeg_path = tf.placeholder(dtype=tf.string)
	jpeg_data = tf.read_file(jpeg_path)
	decode_jpeg = tf.image.decode_jpeg(jpeg_data, channels=3)

	with tf.Session() as sess:
		for i, filepath_list in enumerate(filepath_lists):
			output_filename = '%s_%s_%05d-of-%05d.tfrecord'%(dataset_name, split_name, i, len(filepath_lists))
			with tf.python_io.TFRecordWriter(os.path.join(tfrecord_dir,output_filename)) as writer:
				for j,filepath in enumerate(filepath_list):
					sys.stdout.write('\r>> Converting image %d/%d'%(j+1, len(filepath_list)))
					sys.stdout.flush()
					image_data, image = sess.run([jpeg_data, decode_jpeg], feed_dict={jpeg_path:filepath})
					example = image_to_tfexample(image_data, 'jpg', image.shape[0], image.shape[1], label_lists[i][j])
					writer.write(example.SerializeToString())
			print(' Finished: %s'%(output_filename))

def write_label_map(label_list, output_filename):
	with tf.gfile.Open(output_filename, 'w') as f:
		for i, label in enumerate(label_list):
			f.write('%d:%s\n'%(i,label))
	
def createTfrecord(input_dir="/img/testdata/",tfrecord_dir=".",num_data=10,validations=100,dataset_name="labellio"):
	if not input_dir:
		raise ValueError('You must supply the image directory with --input_dir')
	if not validations % num_data == 0 :
		raise ValueError('<validations> must be divisible by <num_data>')

	filepaths, labels = filepath_label(input_dir)
	labels, label_to_id = label_to_integer(labels)

	convert_tfrecord_and_write(dataset_name, filepaths, labels, num_data, validations, tfrecord_dir)
	write_label_map(label_to_id, os.path.join(tfrecord_dir,"label.txt"))
