from . import dataset
import tensorflow as tf
import json
import io
import os
import pandas as pd
import numpy as np
from google.cloud import bigquery
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
from sklearn.utils import shuffle

private_key= '''[Service Account Key Here]'''

#Prepare input data
#-------------------
img_size_x = 1250
img_size_y = 250
num_channels=1 # Grayscale only
dpi=1000
imageType="PlaceSubTypeImage" #Either PlaceSubTypeImage or PlaceTypeImage
EVAL_INTERVAL = 60

def read_data_bq(hparams):

    # Load all data including binary images into a dataframe
    # Each row represents a single vehicle for one year's worth of data, its associated industry and the image representations
    # of the the frequency of stops to different place types (i.e. restaurant) and sub-type (i.e. fast food restaurant)
    
    allData = '''
        SELECT
          *,
        (case 
              when (govvocation ='MUNICIPAL' AND industry ='MISCELLANEOUS') OR (govvocation ='STATE' AND industry ='MISCELLANEOUS') OR
                   (govvocation ='MUNICIPAL' AND industry ='GOVERNMENT/MISCELLANEOUS') OR (govvocation ='STATE' AND industry ='GOVERNMENT/MISCELLANEOUS') OR
                   (govvocation ='FEDERAL' AND industry ='GOVERNMENT/MISCELLANEOUS')
              THEN 'GOVERNMENT-MISCELLANEOUS'
              WHEN (govvocation ='MUNICIPAL' AND industry ='SERVICES') OR (govvocation ='STATE' AND industry ='SERVICES') OR
                   (govvocation ='FEDERAL' AND industry ='SERVICES')  
              THEN 'GOVERNMENT-SERVICES'
              WHEN industry='BUS TRANSPORTATION' THEN 'BUS TRANSPORTATION'
              WHEN industry='ROAD/HIGHWAY MAINTENANCE' THEN 'ROAD/HIGHWAY MAINTENANCE'
              WHEN industry='UTILITY SERVICES' THEN 'UTILITY SERVICES'              
              WHEN govvocation <> 'None' THEN concat(govvocation,'-',industry)
              ELSE industry end) as NormalizedIndustry,          
          concat(VehicleType,'-',WeightClass) as NormalizedVehicleType
        FROM
          `industryclassification.ImageFeatures.VehicleIndustryPlaceImages`
        WHERE
          INDUSTRY NOT IN ('DEALER','SERVICES','INDIVIDUAL','LEASE/RENTAL','LEASE/FINANCE','MISCELLANEOUS','LEASE/MANUFACTURER SPONSORED') and
          numplacesubtypeelements > 100
        ORDER BY NumPlaceTypeElements DESC
        LIMIT
    ''' + str(hparams['num_records'])
    
    dfAll = pd.read_gbq(allData, project_id='industryclassification', private_key=private_key, dialect='standard')
    
    # Get all Industry Classifications
    
    industryClasses = '''
      SELECT
          (case 
              when (govvocation ='MUNICIPAL' AND industry ='MISCELLANEOUS') OR (govvocation ='STATE' AND industry ='MISCELLANEOUS') OR
                   (govvocation ='MUNICIPAL' AND industry ='GOVERNMENT/MISCELLANEOUS') OR (govvocation ='STATE' AND industry ='GOVERNMENT/MISCELLANEOUS') OR
                   (govvocation ='FEDERAL' AND industry ='GOVERNMENT/MISCELLANEOUS')
              THEN 'GOVERNMENT-MISCELLANEOUS'
              WHEN (govvocation ='MUNICIPAL' AND industry ='SERVICES') OR (govvocation ='STATE' AND industry ='SERVICES') OR
                   (govvocation ='FEDERAL' AND industry ='SERVICES')  
              THEN 'GOVERNMENT-SERVICES'
              WHEN industry='BUS TRANSPORTATION' THEN 'BUS TRANSPORTATION'
              WHEN industry='ROAD/HIGHWAY MAINTENANCE' THEN 'ROAD/HIGHWAY MAINTENANCE'
              WHEN industry='UTILITY SERVICES' THEN 'UTILITY SERVICES'              
              WHEN govvocation <> 'None' THEN concat(govvocation,'-',industry)
              ELSE industry end) as NormalizedIndustry
        FROM
          `industryclassification.ImageFeatures.VehicleIndustryPlaceImages`
        WHERE INDUSTRY NOT IN ('DEALER','SERVICES','INDIVIDUAL','LEASE/RENTAL','LEASE/FINANCE','MISCELLANEOUS','LEASE/MANUFACTURER SPONSORED')
        GROUP BY 1
    '''
    dfIndustries = pd.read_gbq(industryClasses, project_id='industryclassification', private_key=private_key, dialect='standard')
    
    # Get all Vehicle Types
    
    vehicleTypes = '''
        SELECT
          concat(VehicleType,'-',WeightClass) as NormalizedVehicleType
        FROM
          `industryclassification.ImageFeatures.VehicleIndustryPlaceImages`
        GROUP BY 1
    '''
    dfVehicleTypes = pd.read_gbq(vehicleTypes, project_id='industryclassification', private_key=private_key, dialect='standard')
    
    #Prepare input data
    classes = dfIndustries["NormalizedIndustry"].tolist()
    vehicleTypes = dfVehicleTypes["NormalizedVehicleType"].tolist()

    return dfAll, classes, vehicleTypes


#-------------------

# Specifies architecture of each NN layer

def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
 
def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))

# Specifies the architecture of each convolutional layer (this is called 3X for each of the layers)
def create_convolutional_layer(input, num_input_channels, conv_filter_size, num_filters, conv_stride, max_pool_ksize, max_pool_stride):
    
    # Define the weights that will be trained using create_weights function.
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    # Create biases using the create_biases function. These are also trained.
    biases = create_biases(num_filters)
 
    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, conv_stride, conv_stride, 1], padding='SAME')
    layer += biases
    
    # Set max pooling layer with ReLu activation function
    layer = tf.nn.max_pool(value=layer, ksize=[1, max_pool_ksize, max_pool_ksize, 1], strides=[1, max_pool_stride, max_pool_stride, 1], padding='SAME')
    layer = tf.nn.relu(layer)
    return layer

# Specifies the architecture for the flattened layer
def create_flatten_layer(layer):
  
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()
    layer = tf.reshape(layer, [-1, num_features])
    return layer

# Specifies the architecture for the last fully-connected layer with dropout layer
def create_fc_layer(input, num_inputs, num_outputs, dropout_rate, mode, use_relu=True):
    
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)
 
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)
    layer = tf.layers.dropout(inputs=layer, rate=dropout_rate, training = (mode == tf.estimator.ModeKeys.TRAIN))
    return layer

# Specifies the arhchitecture for the first fully-connected layer which employs late fusion with the
# vehicle type feature (called new_feature here).  This also employs a dropout layer
def create_fc_new_feature_layer(input, new_feature, num_inputs, num_outputs, dropout_rate, mode, use_relu=True):
    
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)
    
    layer = tf.matmul(tf.concat([input, new_feature], 1), weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)
    layer = tf.layers.dropout(inputs=layer, rate=dropout_rate, training = (mode == tf.estimator.ModeKeys.TRAIN))
    return layer

# Creates our customer classifier that calls each of the NN layer functions above
def industry_classifier(features, labels, mode, params):
    
    # Specifies the NN layer inputs and labels
    x = features['x']
    v = features['v']
    y_true = labels['y_true']
    y_true_cls = labels['y_true_cls']
    
    # First convolution layer
    layer_conv1 = create_convolutional_layer(input=x,
                   num_input_channels=num_channels,
                   conv_filter_size=params['ksize1'],
                   num_filters=params['nfil1'],
                   conv_stride=params['conv_stride'],
                   max_pool_ksize=params['max_pool_ksize'],
                   max_pool_stride=params['max_pool_stride'])

    # Second convolution layer
    layer_conv2 = create_convolutional_layer(input=layer_conv1,
                   num_input_channels=params['nfil1'],
                   conv_filter_size=params['ksize2'],
                   num_filters=params['nfil2'],
                   conv_stride=params['conv_stride'],
                   max_pool_ksize=params['max_pool_ksize'],
                   max_pool_stride=params['max_pool_stride'])   

    # Third convolution layer  
    layer_conv3= create_convolutional_layer(input=layer_conv2,
                   num_input_channels=params['nfil2'],
                   conv_filter_size=params['ksize3'],
                   num_filters=params['nfil3'],
                   conv_stride=params['conv_stride'],
                   max_pool_ksize=params['max_pool_ksize'],
                   max_pool_stride=params['max_pool_stride'])

    # Flattening Layer                             
    layer_flat = create_flatten_layer(layer_conv3)
    
    # First Fully-Connected Layer with late fusion of vehicle type vector
    layer_fc1 = create_fc_new_feature_layer(input=layer_flat,
                         new_feature=v,
                         num_inputs=layer_flat.get_shape()[1:4].num_elements() + params['num_vehicleTypes'],
                         num_outputs=params['fc_layer_size'],
                         dropout_rate=params['dropout_rate'],
                         mode=mode,
                         use_relu=True)

    # Second fully-connected layer
    layer_fc2 = create_fc_layer(input=layer_fc1,
                         num_inputs=params['fc_layer_size'],
                         num_outputs=params['num_classes'],
                         dropout_rate=params['dropout_rate'],
                         mode=mode,
                         use_relu=False)
    
    # Softmax and output
    y_pred = tf.nn.softmax(layer_fc2,name='y_pred') 
    y_pred_cls = tf.argmax(y_pred, dimension=1)
    
    # Training and Evaluation Mode
    if mode == tf.estimator.ModeKeys.TRAIN or mode == tf.estimator.ModeKeys.EVAL:
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_fc2, labels=y_true)
        cost = tf.reduce_mean(cross_entropy)
        correct_prediction = tf.equal(y_pred_cls, y_true_cls)

        # Specifies evaluation metrics for output in TensorBoard
        accuracy = tf.metrics.accuracy(y_pred_cls, tf.argmax(y_true, 1), name='eval_accuracy')
        evalmetrics = {'eval_accuracy': accuracy}
        tf.summary.scalar('train_accuracy', accuracy[1])

        # Training Mode parameters
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=params['learning_rate'])
            train_op = optimizer.minimize(cost, global_step=tf.train.get_global_step())
            evalmetrics = None
            predictions = None
        else:
            # Evaluation Mode Parameters
            train_op = None
            predictions = None
    else:
        # Prediction Mode Parameters
        cost = None
        train_op = None
        evalmetrics = None
        predictions = {"probabilities": y_pred, "classes": y_pred_cls}
    
    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=predictions,
        loss=cost,
        train_op=train_op,
        eval_metric_ops=evalmetrics
    )

# Defines custom input functions
def input_fn(data, train, hparams):
    
    if train == True:
        x_batch, y_true_batch, cls_batch, vehicleType_batch = data.train.next_batch(hparams['train_batch_size'])
    else:
        x_batch, y_true_batch, cls_batch, vehicleType_batch  = data.valid.next_batch(hparams['train_batch_size'])

    y_true_cls = tf.argmax(y_true_batch, dimension=1)    
    features = {'x' : x_batch, 'v': vehicleType_batch }
    labels = {'y_true' : y_true_batch, 'y_true_cls': y_true_cls }
    
    return features, labels

# Defines main training and evaluation function
def train_and_evaluate(output_dir, hparams):
    
    dfAll, classes, vehicleTypes = read_data_bq(hparams)
    hparams['num_classes'] = len(classes)
    hparams['num_vehicleTypes'] = len(vehicleTypes)
    hparams['output_dir'] = output_dir
    
    # Imports all data
    data = dataset.read_train_sets(dfAll, imageType, img_size_x, img_size_y, classes, vehicleTypes, validation_size=hparams['validation_size'], dpi=dpi)

    # Creates custom estimator leveraging our custom industry classifier
    estimator = tf.estimator.Estimator(model_fn = industry_classifier,
                                     params = hparams,
                                     model_dir = output_dir)

    # Specifies training and evaluation parameters
    train_spec = tf.estimator.TrainSpec(input_fn = lambda: input_fn(data=data, train=True, hparams=hparams),
                                    max_steps = hparams['train_steps'])
    eval_spec = tf.estimator.EvalSpec(input_fn = lambda: input_fn(data=data, train=False, hparams=hparams),
                                  steps = 10000,
                                  throttle_secs = EVAL_INTERVAL)

    # Calls function to train and evaluate
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)