import tensorflow as tf

# Model
#################################################################################

# Parametric ReLU
def PReLU(x):
    return tf.maximum(x*.2, x)


def doubleConv2D(C, n_filters):
   # Double Conv2D with Parametric ReLU
   C = tf.keras.layers.Conv2D(n_filters, (3, 3), padding = "same", activation = PReLU, kernel_initializer = "he_normal")(C)
   C = tf.keras.layers.Conv2D(n_filters, (3, 3), padding = "same", activation = PReLU, kernel_initializer = "he_normal")(C)
   return C

def downsample_block(C, n_filters):
   fc = doubleConv2D(C, n_filters)
   skip =  tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(fc)
   return fc, skip

def upsample_and_concat(C, conv_size, n_filters):
    C = tf.keras.layers.UpSampling2D(size=(2, 2))(C)
    C = tf.keras.layers.concatenate([C, conv_size])
    C = doubleConv2D(C, n_filters)

    return C

def UNet():
 # inputs
   inputs = tf.keras.Input(shape=(256, 256, 3))

   # encoder block - downsample

   fc1, skip1 = downsample_block(inputs, 64)

   fc2, skip2 = downsample_block(skip1, 128)

   fc3, skip3 = downsample_block(skip2, 256)

   fc4, skip4 = downsample_block(skip3, 512)

   # bottleneck layer
   bottleneck = doubleConv2D(skip4, 1024)

   # decoder block - upsample
   up6 = upsample_and_concat(bottleneck, fc4, 512)

   up7 = upsample_and_concat(up6, fc3, 256)

   up8 = upsample_and_concat(up7, fc2, 128)

   up9 = upsample_and_concat(up8, fc1, 64)

   # outputs
   output = tf.keras.layers.Conv2D(3, 1, activation = 'relu')(up9)

   # unet model with Keras Functional API
   unet_model = tf.keras.Model(inputs, output, name="U-Net")

   return unet_model

# Metrics
#################################################################################

# Signal to Reconstruction Error ratio (SRE)
def SRE(y_true, y_pred):
    # Square of mean of the true image
    mu_squared_true = tf.square(tf.reduce_mean(y_true))
    denominator = tf.norm((y_pred-y_true))/256 #256 - size of the image
    
    sre = mu_squared_true / denominator

    sre = 10.0 * tf.math.log(sre) / tf.math.log(10.)

    return sre

# Structural Similarity (SSIM)
def SSIM(y_true, y_pred):
    K1 = 0.01
    K2 = 0.03
    L = 65535.0

    # Mean of the images
    mu_true = tf.reduce_mean(y_true)
    mu_pred = tf.reduce_mean(y_pred)

    # Variance of the images
    var_true = tf.reduce_mean(tf.square(y_true - mu_true))
    var_pred = tf.reduce_mean(tf.square(y_pred - mu_pred))

    # Covariance between the images
    covar = tf.reduce_mean((y_true - mu_true) * (y_pred - mu_pred))

    # Calculate SSIM
    c1 = (K1 * L) ** 2
    c2 = (K2 * L) ** 2
    numerator = (2 * mu_true * mu_pred + c1) * (2 * covar + c2)
    denominator = (mu_true ** 2 + mu_pred ** 2 + c1) * (var_true + var_pred + c2)

    ssim = numerator / denominator

    return ssim

# Image Processing
#################################################################################

