import tensorflow as tf
import io
import cv2
import pandas as pd
import numpy as np
import base64
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from google.cloud import bigquery
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
from sklearn.utils import shuffle
import gc

#Prepare input data
batch_size = 100
validation_size = 0.2
img_size_x = 1250
img_size_y = 250
num_channels=1 # Grayscale only
dpi=1000
imageType="PlaceSubTypeImage" #Either PlaceSubTypeImage or PlaceTypeImage
numRecords = 2000

##Network graph parameters

# Incoming 500x100
filter_size_conv1 = 2 
num_filters_conv1 = 32 

# Incoming 250x50
filter_size_conv2 = 2
num_filters_conv2 = 32

#Incoming 125x25
filter_size_conv3 = 2
num_filters_conv3 = 64

#125x25x64 <-- flattening layer (auto-calculated)
#Incoming 125x25
fc_layer_size = 128 


# Load all data including binary images into a dataframe
# Each row represents a single vehicle for one year's worth of data, its associated industry and the image representations
# of the the frequency of stops to different place types (i.e. restaurant) and sub-type (i.e. fast food restaurant)

allData = '''
SELECT
  *,
  (case when govvocation not in ('STATE', 'FEDERAL','MUNICIPAL') THEN Industry ELSE concat(govvocation,'-',industry) end) as NormalizedIndustry,
  concat(VehicleType,'-',WeightClass) as NormalizedVehicleType 
FROM 
  `industryclassification.ImageFeatures.VehicleIndustryPlaceImages` 
WHERE
  industry not in ('DEALER', 'SERVICES','INDIVIDUAL','LEASE/RENTAL') and 
  numplacesubtypeelements > 100
ORDER BY NumPlaceTypeElements DESC 
LIMIT 
''' + str(numRecords)

dfAll = pd.read_gbq(allData, project_id='industryclassification', dialect='standard')

print ("Core Data Imported")

# Get all Industry Classifications

industryClasses = '''

SELECT 
  (case when govvocation not in ('STATE', 'FEDERAL','MUNICIPAL') THEN Industry ELSE concat(govvocation,'-',industry) end) as NormalizedIndustry 
FROM 
  `industryclassification.ImageFeatures.VehicleIndustryPlaceImages`
GROUP BY 1

'''
dfIndustries = pd.read_gbq(industryClasses, project_id='industryclassification', dialect='standard')

print ("Industry Classifications Imported")

# Get all Vehicle Types

vehicleTypes = '''

SELECT 
  concat(VehicleType,'-',WeightClass) as NormalizedVehicleType 
FROM 
  `industryclassification.ImageFeatures.VehicleIndustryPlaceImages`
GROUP BY 1

'''
dfVehicleTypes = pd.read_gbq(vehicleTypes, project_id='industryclassification', dialect='standard')

print ("Vehicle Types Imported")

#Prepare input data
classes = dfIndustries["NormalizedIndustry"].tolist()
vehicleTypes = dfVehicleTypes["NormalizedVehicleType"].tolist()
num_classes = len(classes)
num_vehicleTypes = len(vehicleTypes)

import dataset


# We shall load all the training and validation images and labels into memory using openCV and use that during training
data = dataset.read_train_sets(dfAll, imageType, img_size_x, img_size_y, classes, vehicleTypes, validation_size=validation_size, dpi=dpi)

print("Complete reading input data. Will Now print a snippet of it")
print("Number of files in Training-set:\t\t{}".format(len(data.train.labels)))
print("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))

session = tf.Session()
x = tf.placeholder(tf.float32, shape=[None, img_size_y,img_size_x, num_channels], name='x')
## Vehicle Type Feature
v = tf.placeholder(tf.float32, shape=[None, num_vehicleTypes], name='v')


## labels
y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
y_true_cls = tf.argmax(y_true, dimension=1)

gc.collect()

print ("Data Fully Loaded")

# Specifies architecture of each NN layer

def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))
 
def create_biases(size):
    return tf.Variable(tf.constant(0.05, shape=[size]))
  
def create_convolutional_layer(input, num_input_channels, conv_filter_size, num_filters):  
    
    ## We shall define the weights that will be trained using create_weights function.
    weights = create_weights(shape=[conv_filter_size, conv_filter_size, num_input_channels, num_filters])
    ## We create biases using the create_biases function. These are also trained.
    biases = create_biases(num_filters)
 
    ## Creating the convolutional layer
    layer = tf.nn.conv2d(input=input, filter=weights, strides=[1, 1, 1, 1], padding='SAME')
 
    layer += biases
 
    ## We shall be using max-pooling.  
    layer = tf.nn.max_pool(value=layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    ## Output of pooling is fed to Relu which is the activation function for us.
    layer = tf.nn.leaky_relu(layer,0.1)
    return layer
 
def create_flatten_layer(layer):
  
    layer_shape = layer.get_shape()
    num_features = layer_shape[1:4].num_elements()    
    layer = tf.reshape(layer, [-1, num_features]) 
    return layer

def create_fc_layer(input, num_inputs, num_outputs, use_relu=True):
    
    #Let's define trainable weights and biases.
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)
 
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.leaky_relu(layer, 0.1)
    return layer

def create_fc_new_feature_layer(input, new_feature, num_inputs, num_outputs, use_relu=True):
    
    #Let's define trainable weights and biases.
    weights = create_weights(shape=[num_inputs, num_outputs])
    biases = create_biases(num_outputs)
    
    layer = tf.matmul(tf.concat([input, new_feature], 1), weights) + biases
    if use_relu:
        layer = tf.nn.leaky_relu(layer,0.1)
    return layer

# Specifies the NN layer inputs and interaction

layer_conv1 = create_convolutional_layer(input=x,
               num_input_channels=num_channels,
               conv_filter_size=filter_size_conv1,
               num_filters=num_filters_conv1)
layer_conv2 = create_convolutional_layer(input=layer_conv1,
               num_input_channels=num_filters_conv1,
               conv_filter_size=filter_size_conv2,
               num_filters=num_filters_conv2)

layer_conv3= create_convolutional_layer(input=layer_conv2,
               num_input_channels=num_filters_conv2,
               conv_filter_size=filter_size_conv3,
               num_filters=num_filters_conv3)
          
layer_flat = create_flatten_layer(layer_conv3)

layer_fc1 = create_fc_new_feature_layer(input=layer_flat,
                     new_feature=v,
                     num_inputs=layer_flat.get_shape()[1:4].num_elements() + num_vehicleTypes,
                     num_outputs=fc_layer_size,
                     use_relu=True)

layer_fc2 = create_fc_layer(input=layer_fc1,
                     num_inputs=fc_layer_size,
                     num_outputs=num_classes,
                     use_relu=False) 

y_pred = tf.nn.softmax(layer_fc2,name='y_pred')

y_pred_cls = tf.argmax(y_pred, dimension=1)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(logits=layer_fc2, labels=y_true)
cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-5).minimize(cost)
session.run(tf.global_variables_initializer())
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

print ("Model Created")


def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss))

total_iterations = 0

saver = tf.train.Saver()

def train(num_iteration):
    global total_iterations
    
    for i in range(total_iterations, total_iterations + num_iteration):

        x_batch, y_true_batch, cls_batch, vehicleType_batch = data.train.next_batch(batch_size)
        x_valid_batch, y_valid_batch, valid_cls_batch, valid_vehicleType_batch = data.valid.next_batch(batch_size)

        
        feed_dict_tr = {x: x_batch, y_true: y_true_batch, v: vehicleType_batch}
        feed_dict_val = {x: x_valid_batch, y_true: y_valid_batch, v: valid_vehicleType_batch}

        session.run(optimizer, feed_dict=feed_dict_tr)

        if i % int(data.train.num_examples/batch_size) == 0: 
            val_loss = session.run(cost, feed_dict=feed_dict_val)
            epoch = int(i / int(data.train.num_examples/batch_size))    
            
            show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss)
            #saver.save(session, './vehicle-industry-model') 

    total_iterations += num_iteration

train(num_iteration=3000)
