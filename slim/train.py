# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Generic training script that trains a model using a given dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from datasets import dataset_factory_v1
from deployment import model_deploy
from nets import nets_factory
from preprocessing import preprocessing_factory
import ConfigParser
import os

slim = tf.contrib.slim
STATUS = "status"
default_config = {
    'status' : '',
}
config = ConfigParser.SafeConfigParser(default_config)

def main(train_dir="/tmp/model/",dataset_name="labellio",dataset_dir=".",num_train=None,num_val=None,num_classes=None,model_name="mobilenet_v1",max_number_of_steps=1000,batch_size=10,learning_rate=0.01,learning_rate_decay_type="fixed",optimizer_type="rmsprop",model_every_n_steps=10,end_learning_rate=None,learning_rate_decay_factor=None,decay_steps=10,checkpoint_path=None):

  if not dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  tf.logging.set_verbosity(tf.logging.INFO)
  with tf.Graph().as_default():
    #######################
    # Config model_deploy #
    #######################
    deploy_config = model_deploy.DeploymentConfig(
        num_clones=1,
        clone_on_cpu=False,
        replica_id=0,
        num_replicas=1,
        num_ps_tasks=0)

    # Create global_step
    with tf.device(deploy_config.variables_device()):
      global_step = slim.create_global_step()

    ######################
    # Select the dataset #
    ######################
    train_dataset = dataset_factory_v1.get_dataset(
        dataset_name, "train", dataset_dir, num_train, num_classes)
    valid_dataset = dataset_factory_v1.get_dataset(
        dataset_name, "validation", dataset_dir, num_val, num_classes)

    ######################
    # Select the network #
    ######################
    network_fn = nets_factory.get_network_fn(
        model_name,
        num_classes=train_dataset.num_classes,
        weight_decay=0.00004,
        is_training=True)

    #####################################
    # Select the preprocessing function #
    #####################################
    preprocessing_name = model_name
    image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=True)

    ##############################################################
    # Create a dataset provider that loads data from the dataset #
    ##############################################################
    with tf.device(deploy_config.inputs_device()):
      train_provider = slim.dataset_data_provider.DatasetDataProvider(
          train_dataset,
          num_readers=4,
          common_queue_capacity=20 * batch_size,
          common_queue_min=10 * batch_size)
      valid_provider = slim.dataset_data_provider.DatasetDataProvider(
          valid_dataset,
          num_readers=4,
          common_queue_capacity=20 * batch_size,
          common_queue_min=10 * batch_size)
        
      [image, label] = train_provider.get(['image', 'label'])
      [val_image, val_label] = valid_provider.get(['image', 'label'])

      train_image_size = None or network_fn.default_image_size

      image = image_preprocessing_fn(image, train_image_size, train_image_size)
      val_image = image_preprocessing_fn(val_image, train_image_size, train_image_size)

      images, labels = tf.train.batch(
          [image, label],
          batch_size=batch_size,
          num_threads=4,
          capacity=5 * batch_size)
      val_images, val_labels = tf.train.batch(
          [val_image, val_label],
          batch_size=batch_size,
          num_threads=4,
          capacity=5 * batch_size)
        
      labels = slim.one_hot_encoding(
          labels, train_dataset.num_classes)
      val_labels = slim.one_hot_encoding(
          val_labels, valid_dataset.num_classes)
        
      # batch queues for training
      batch_queue = slim.prefetch_queue.prefetch_queue(
          [images, labels], capacity=2 * deploy_config.num_clones)

    ########################
    # Config Learning Rate #
    ########################
    def _configure_learning_rate(num_samples_per_epoch, global_step):
      """Configures the learning rate.
      Args:
        num_samples_per_epoch: The number of samples in each epoch of training.
        global_step: The global_step tensor.
      Returns:
        A `Tensor` representing the learning rate.
      Raises:
        ValueError: if
      """
      decay_steps = int(num_samples_per_epoch / batch_size *
                        2.0)

      if learning_rate_decay_type == 'exponential':
        return tf.train.exponential_decay(learning_rate,
                                          global_step,
                                          decay_steps,
                                          learning_rate_decay_factor,
                                          staircase=True,
                                          name='exponential_decay_learning_rate')
      elif learning_rate_decay_type == 'fixed':
        return tf.constant(learning_rate, name='fixed_learning_rate')
      elif learning_rate_decay_type == 'polynomial':
        return tf.train.polynomial_decay(learning_rate,
                                         global_step,
                                         decay_steps,
                                         end_learning_rate,
                                         power=1.0,
                                         cycle=False,
                                         name='polynomial_decay_learning_rate')
      else:
        raise ValueError('learning_rate_decay_type [%s] was not recognized',
                         FLAGS.learning_rate_decay_type)

    ####################
    # Config Optimizer #
    ####################
    def _configure_optimizer(learning_rate):
      """Configures the optimizer used for training.
      Args:
        learning_rate: A scalar or `Tensor` learning rate.
      Returns:
        An instance of an optimizer.
      Raises:
        ValueError: if FLAGS.optimizer is not recognized.
      """
      if optimizer_type == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate,
            rho=0.95,
            epsilon=1.0)
      elif optimizer_type == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(
            learning_rate,
            initial_accumulator_value=0.1)
      elif optimizer_type == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate,
            beta1=0.9,
            beta2=0.999,
            epsilon=1.0)
      elif optimizer_type == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(
            learning_rate,
            learning_rate_power=-0.5,
            initial_accumulator_value=0.1,
            l1_regularization_strength=0.0,
            l2_regularization_strength=0.0)
      elif optimizer_type == 'momentum':
        optimizer = tf.train.MomentumOptimizer(
            learning_rate,
            momentum=0.9,
            name='Momentum')
      elif optimizer_type == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=0.9,
            momentum=0.9,
            epsilon=1.0)
      elif optimizer_type == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
      else:
        raise ValueError('Optimizer [%s] was not recognized', optimizer)
      return optimizer

    def _get_init_fn():
      """Returns a function run by the chief worker to warm-start the training.
      Note that the init_fn is only run when initializing the model during the very
      first global step.
      Returns:
        An init function run by the supervisor.
      """
      if checkpoint_path is None:
        return None

      checkpoint_exclude_scopes = None
      if model_name == "mobilenet_v1":
        checkpoint_exclude_scopes="MobilenetV1/Logits,MobilenetV1/AuxLogits"
      elif model_name == "inception_v4":
        checkpoint_exclude_scopes="InceptionV4/Logits,InceptionV4/AuxLogits"
      elif model_name == "resnet_v1_152":
        checkpoint_exclude_scopes="resnet_v2_152/logits"
      # Warn the user if a checkpoint exists in the train_dir. Then we'll be
      # ignoring the checkpoint anyway.
      if tf.train.latest_checkpoint(train_dir):
        tf.logging.info(
            'Ignoring --checkpoint_path because a checkpoint already exists in %s'
            % train_dir)
        return None

      exclusions = []
      if checkpoint_exclude_scopes:
        exclusions = [scope.strip()
                      for scope in checkpoint_exclude_scopes.split(',')]

      # TODO(sguada) variables.filter_variables()
      variables_to_restore = []
      for var in slim.get_model_variables():
        excluded = False
        for exclusion in exclusions:
          if var.op.name.startswith(exclusion):
            excluded = True
            break
        if not excluded:
          variables_to_restore.append(var)

      if tf.gfile.IsDirectory(checkpoint_path):
        checkpoint_path_fn = tf.train.latest_checkpoint(checkpoint_path)
      else:
        checkpoint_path_fn = checkpoint_path

      tf.logging.info('Fine-tuning from %s' % checkpoint_path)

      return slim.assign_from_checkpoint_fn(
          checkpoint_path_fn,
          variables_to_restore,
          ignore_missing_vars=False)

    def _get_variables_to_train():
      """Returns a list of variables to train.
      Returns:
        A list of variables to train by the optimizer.
      """
      trainable_scopes = None
      if trainable_scopes is None:
        return tf.trainable_variables()
      else:
        scopes = [scope.strip() for scope in trainable_scopes.split(',')]

      variables_to_train = []
      for scope in scopes:
        variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
        variables_to_train.extend(variables)
      return variables_to_train

        
    ####################
    # Define the model #
    ####################
    def clone_fn(batch_queue):
      """Allows data parallelism by creating multiple clones of network_fn."""
      images, labels = batch_queue.dequeue()
      logits, end_points = network_fn(images)

      #############################
      # Specify the loss function #
      #############################
      if 'AuxLogits' in end_points:
        tf.losses.softmax_cross_entropy(
            logits=end_points['AuxLogits'], onehot_labels=labels,
            label_smoothing=0.0, weights=0.4, scope='aux_loss')
      tf.losses.softmax_cross_entropy(
          logits=logits, onehot_labels=labels,
          label_smoothing=0.0, weights=1.0)
      return end_points

    # Gather initial summaries.
    summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))

    clones = model_deploy.create_clones(deploy_config, clone_fn, [batch_queue])
    first_clone_scope = deploy_config.clone_scope(0)
    # Gather update_ops from the first clone. These contain, for example,
    # the updates for the batch_norm variables created by network_fn.
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, first_clone_scope)

    # Add summaries for end_points.
    end_points = clones[0].outputs
    for end_point in end_points:
      x = end_points[end_point]
      summaries.add(tf.summary.histogram('activations/' + end_point, x))
      summaries.add(tf.summary.scalar('sparsity/' + end_point,
                                      tf.nn.zero_fraction(x)))

    # Add summaries for losses.
    for loss in tf.get_collection(tf.GraphKeys.LOSSES, first_clone_scope):
      summaries.add(tf.summary.scalar('losses/%s' % loss.op.name, loss))

    # Add summaries for variables.
    for variable in slim.get_model_variables():
      summaries.add(tf.summary.histogram(variable.op.name, variable))

    #################################
    # Configure the moving averages #
    #################################
    moving_average_variables, variable_averages = None, None

    #########################################
    # Configure the optimization procedure. #
    #########################################
    with tf.device(deploy_config.optimizer_device()):
      learning_rate = _configure_learning_rate(train_dataset.num_samples, global_step)
      optimizer = _configure_optimizer(learning_rate)
      summaries.add(tf.summary.scalar('learning_rate', learning_rate))

    # Variables to train.
    variables_to_train = _get_variables_to_train()

    #  and returns a train_tensor and summary_op
    total_loss, clones_gradients = model_deploy.optimize_clones(
        clones,
        optimizer,
        var_list=variables_to_train)
    # Add total_loss to summary.
    summaries.add(tf.summary.scalar('total_loss', total_loss))

    # Create gradient updates.
    grad_updates = optimizer.apply_gradients(clones_gradients,
                                             global_step=global_step)
    update_ops.append(grad_updates)

    update_op = tf.group(*update_ops)
    with tf.control_dependencies([update_op]):
      train_tensor = tf.identity(total_loss, name='train_op')

    # Add the summaries from the first clone. These contain the summaries
    # created by model_fn and either optimize_clones() or _gather_clone_loss().
    summaries |= set(tf.get_collection(tf.GraphKeys.SUMMARIES,
                                       first_clone_scope))

    # Merge all summaries together.
    summary_op = tf.summary.merge(list(summaries), name='summary_op')
    
    ###########################
    # Variables for network   #
    ###########################
    print(" train_image_classifier: pred for train and validation")
    with tf.variable_scope('', reuse=True):
  	predictions = network_fn(images)
    	predictions_validation, end_points = network_fn(val_images)
    	accuracy_validation = slim.metrics.accuracy(tf.to_int32(tf.argmax(predictions_validation, 1)), tf.to_int32(tf.argmax(labels, 1)))
	    
    ###########################
    # Train step fn         . #
    ###########################
    import os
    from tensorflow.contrib.slim.python.slim.learning import train_step
    
    # save
    saver = tf.train.Saver(max_to_keep=None) # saver
    save_directory = train_dir

    # validation
    val_scores = 0

    if not os.path.exists(save_directory):
      os.mkdir(save_directory)

    # train step fn
    def train_step_fn(session, *args, **kwargs):
      step = int(session.run(args[1]))
      total_loss, should_stop = train_step(session, *args, **kwargs)
      print(" train_image_classifier: train_step_fn: step", step)
      conf_file = os.path.join(train_dir,"config.conf")
 
      #try:
      config.read(conf_file)
      status = config.get("0",STATUS)
      print(status)
      if status == "stop":
        import sys
        sys.exit(0)
      #except:
        #print("Error occured")
        #return False
      print(step)
      mode = step % model_every_n_steps
      if mode == 0 or mode == 1:
        accuracy = session.run(accuracy_validation)
        print('_________________varidation score_____________________')
        print('Step %s - Loss: %.2f Accuracy: %.2f%%' % (str(step).rjust(6, '0'), total_loss, accuracy * 100))
        print('_________________save ckpt models_____________________')
        print(" train_image_classifier: train_step_fn: save ckpt", step)
        saver.save(session, save_directory+"model__step"+str(step)+".ckpt")        
      
      total_loss, should_stop = train_step(session, *args, **kwargs)
      return [total_loss, should_stop] 


    ###########################
    # Kicks off the training. #
    ###########################
    slim.learning.train(
        train_tensor,
        logdir=train_dir,
        master="",
        is_chief=(0 == 0),
        init_fn=_get_init_fn(),
        summary_op=summary_op,
        number_of_steps=max_number_of_steps,
        log_every_n_steps=0,
        train_step_fn=train_step_fn,
        save_summaries_secs=0,
        save_interval_secs=0,
        sync_optimizer=optimizer if False else None)


def createModel(train_dir="/tmp/model/",dataset_name="labellio",dataset_dir=".",num_train=None,num_val=None,num_classes=None,model_name="mobilenet_v1",max_number_of_steps=1000,batch_size=10,learning_rate=0.01,learning_rate_decay_type="fixed",optimizer="rmsprop",model_every_n_steps=10,learning_rate_decay_factor=None,decay_steps=10,utilization_per_gpu=1.0,gpu_number="0",checkpoint_path="default"):

    conf_file = os.path.join(train_dir,"config.conf")

    #try:
    config.add_section("0")
    config.set("0",STATUS, "training")
    config.write(open(conf_file, 'w'))
    #except Exception, e:
    #    print(e)
    #    return False

    gpuConfig = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=utilization_per_gpu,visible_device_list=gpu_number)) 
    with tf.Session(config=gpuConfig) as session:
        main(train_dir=train_dir,dataset_name=dataset_name,dataset_dir=dataset_dir,num_train=num_train,num_val=num_val,num_classes=num_classes,model_name=model_name,max_number_of_steps=max_number_of_steps,batch_size=batch_size,learning_rate=learning_rate,learning_rate_decay_type=learning_rate_decay_type,optimizer_type=optimizer,model_every_n_steps=model_every_n_steps,learning_rate_decay_factor=learning_rate_decay_factor,decay_steps=decay_steps,checkpoint_path=checkpoint_path)

def stop(train_dir=None):
    conf_file = os.path.join(train_dir,"config.conf")
    #try:
    config.add_section("0")
    config.set("0",STATUS, "stop")
    config.write(open(conf_file, 'w'))
    return False

def restart(train_dir="/tmp/model/",dataset_name="labellio",dataset_dir=".",num_train=None,num_val=None,num_classes=None,model_name="mobilenet_v1",max_number_of_steps=1000,batch_size=10,learning_rate=0.01,learning_rate_decay_type="fixed",optimizer="rmsprop",model_every_n_steps=10,learning_rate_decay_factor=None,decay_steps=10,utilization_per_gpu=1.0,gpu_number="0"):
    conf_file = os.path.join(train_dir,"config.conf")
    #try:
    config.add_section("0")
    config.set("0",STATUS, "training")
    config.write(open(conf_file, 'w'))
    main(train_dir=train_dir,dataset_name=dataset_name,dataset_dir=dataset_dir,num_train=num_train,num_val=num_val,num_classes=num_classes,model_name=model_name,max_number_of_steps=max_number_of_steps,batch_size=batch_size,learning_rate=learning_rate,learning_rate_decay_type=learning_rate_decay_type,optimizer_type=optimizer,model_every_n_steps=model_every_n_steps,learning_rate_decay_factor=learning_rate_decay_factor,decay_steps=decay_steps)
    return False

def relearn(train_dir="/tmp/model/",dataset_name="labellio",dataset_dir=".",num_train=None,num_val=None,num_classes=None,model_name="mobilenet_v1",max_number_of_steps=1000,batch_size=10,learning_rate=0.01,learning_rate_decay_type="fixed",optimizer="rmsprop",model_every_n_steps=10,learning_rate_decay_factor=None,decay_steps=10,utilization_per_gpu=1.0,gpu_number="0"):

    conf_file = os.path.join(train_dir,"config.conf")

    #try:
    config.add_section("0")
    config.set("0",STATUS, "training")
    config.write(open(conf_file, 'w'))
    #except Exception, e:
    #    print(e)
    #    return False

    main(train_dir=train_dir,dataset_name=dataset_name,dataset_dir=dataset_dir,num_train=num_train,num_val=num_val,num_classes=num_classes,model_name=model_name,max_number_of_steps=max_number_of_steps,batch_size=batch_size,learning_rate=learning_rate,learning_rate_decay_type=learning_rate_decay_type,optimizer_type=optimizer,model_every_n_steps=model_every_n_steps,learning_rate_decay_factor=learning_rate_decay_factor,decay_steps=decay_steps)
    return False
