import tensorflow as tf
from tensorflow.data import Dataset
from tensorflow.keras.layers import Dense

def RNInit(dims):
    return tf.random_normal(dims, mean=0.0, stddev=0.01)

"""
This architecture is simply a more annotated and refactored
version of smallcorgi's architecture which can be found here
https://github.com/smallcorgi/Anticipating-Accidents
We aim to understand their architecture better and improve on it
"""
def create_model(params):
    # Parameters
    n_img_hidden = params['n_img_hidden']
    n_att_hidden = params['n_att_hidden']
    n_frames = params['n_frames']
    n_detection = params['n_detection']

    # Weights
    weights = {
        'att_w': tf.Variable(tf.random_normal([n_att_hidden, 1], mean=0.0, stddev=0.01)),
        'att_wa': tf.Variable(tf.random_normal([n_hidden, n_att_hidden], mean=0.0, stddev=0.01)),
        'att_ua': tf.Variable(tf.random_normal([n_att_hidden, n_att_hidden], mean=0.0, stddev=0.01)),
        'out': tf.Variable(tf.random_normal([n_hidden, n_classes], mean=0.0, stddev=0.01))
    }
    biases = {
        'att_ba': tf.Variable(tf.zeros([n_att_hidden])),
        'out': tf.Variable(tf.random_normal([n_classes], mean=0.0, stddev=0.01))
    }  

    # Placeholders
    """
    @TODO: Replace this with tf.data.Dataset
    """
    # X is a [batch_size, n_frames, n_detection, n_input] tensor
    X = tf.placeholder("float", [None, n_frames, n_detection, n_input])
    # Y is a [batch_size, n_classes] tensor
    Y = tf.placeholder("float", [None, n_classes])
    keep = tf.placeholder("float", [None])

    # Weights & biases
    ImgFCW = tf.Variable(RNInit([n_input, n_img_hidden]), name='ImgFC-Weights')
    ImgFCB = tf.Variable(RNInit([n_img_hidden]), name='ImgFC-Biases')
    ObjFCW = tf.Variable(RNInit([n_input, n_att_hidden]), name='ObjFC-Weights')
    ObjFCB = tf.Variable(RNInit([n_att_hidden]), name='ObjFC-Biases')
    AttWW = tf.Variable(RNInit([n_att_hidden, 1]), name='ImgFC-Weights')
    AttWB = tf.Variable(RNInit([n_att_hidden]), name='ImgFC-Biases')

    """
    In our batch, the first element of the 2nd dimension is the image,
    and the remaining n_detection ones are the detected objects.
    Here we split them into two tensors
    """
    # images is a [n_frames, batch_size, n_input] tensor
    images = tf.transpose(X[:, :, 0, :], [1, 0, 2])
    # objects is a [(n_detection-1), n_frames, batch_size, n_input] tensor
    objects = tf.transpose(X[:, :, 1:n_detection, :], [1, 2, 0, 3])

    """
    Tensor of 1's and 0's
    1: There is at least one detected object in frame n
    0: There are no detected objects
    """
    # zeros_object is a [n_frames, batch_size, n_input] tensor
    zeros_object = tf.reduce_sum(objects, 2)
    zeros_object = tf.to_float(tf.not_equal(zeros_object, 0))

    for i in range(0, n_frames): 
        """
        Our image goes through a fully connected (FC) layer
        """
        image = images[i, :, :]
        # fc_image is a [batch_size, n_img_hidden] tensor
        fc_image = tf.matmul(image, ImgFCW) + ImgFCB
        
        """
        Our object goes through a fully connected (FC) layer
        and then we mask it
        """
        # obj is a [(n_detection-1) * batch_size, n_input] tensor
        obj = tf.reshape(objects[i, :, :], [-1, n_input])
        # fc_obj is a [(n_detection-1) * batch_size, n_att_hidden] tensor
        fc_obj = tf.matmul(obj, ObjFCW) + ObjFCB
        # We reshape the output tensor back to 
        # [n_detection - 1, batch_size, n_att_hidden]
        fc_obj = tf.reshape(fc_obj, [n_detection - 1, batch_size, n_att_hidden])
        # mask is a [batch_size, n_input, 1] tensor
        mask = tf.expand_dims(zeros_object[i], 2)
        # fc_obj is a [n_detection -1, batch_size, n_att_hidden] tensor
        fc_obj = tf.multiply(fc_obj, mask)


