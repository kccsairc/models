# -*- coding: utf-8 -*-
from convert import createTfrecord

if __name__ == '__main__':
    #create tfrecord
    createTfrecord(input_dir="/img/testdata/", tfrecord_dir="/img/tfrecord/", num_data=5, validations=30, dataset_name="labellio")
