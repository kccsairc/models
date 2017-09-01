# -*- coding: utf-8 -*-
from convert import createTfrecord

if __name__ == '__main__':
    #create tfrecord
    createTfrecord(input_dir="/img/testdata/", 
                   tfrecord_dir="/img/tfrecord/", 
                   num_data=100, 
                   validations=300, 
                   dataset_name="labellio")
