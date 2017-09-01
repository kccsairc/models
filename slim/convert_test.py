# -*- coding: utf-8 -*-
from convert import createTfrecord

if __name__ == '__main__':
    #create tfrecord
    createTfrecord(input_dir="/img/testdata/", 
                   output_dir="/img/tfrecord/", 
                   num_data=20, 
                   num_val=20, 
                   dataset_name="labellio")
