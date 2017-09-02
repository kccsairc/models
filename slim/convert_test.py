# -*- coding: utf-8 -*-
from convert import createTfrecord

if __name__ == '__main__':
    #create tfrecord
    createTfrecord(input_dir="/img/testdata2/", 
                   output_dir="/img/tfrecord2/", 
                   num_data=40, 
                   num_val=40, 
                   dataset_name="labellio")
