import tensorflow.keras.preprocessing.image as process_im
from tensorflow.python.keras import models
import matplotlib.pyplot as plt
import tensorflow as tf
import IPython.display
from PIL import Image
import numpy as np

def load_file(image_path, size_wrap):
    image = Image.open(image_path)
    new_size = size_wrap
    img_size_base = image.size[1]/image.size[0]

    if size_wrap == 1080:
        img_size_x = image.size[0]
        img_size_y = image.size[1]
    else:
        img_size_x=int(new_size)
        img_size_y=int(new_size*img_size_base)

    image_resized = image.resize((img_size_x, img_size_y))

    im_array = process_im.img_to_array(image_resized)
    im_array = np.expand_dims(im_array, axis=0)
    return im_array


def show_im(img, title=None):
    img = np.squeeze(img, axis=0)
    plt.imshow(np.uint8(img))
    if title is None:
        pass
    else:
        plt.title(title)
    plt.imshow(np.uint8(img))


def img_preprocess(img_path, size_wrap):
    image = load_file(img_path, size_wrap)
    img = tf.keras.applications.vgg19.preprocess_input(image)
    return img


def deprocess_img(processed_img):
    x = processed_img.copy()
    if len(x.shape) == 4:
        x = np.squeeze(x, 0)
    assert len(x.shape) == 3
    x[:, :, 0] += 105
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


content_layers = ['block1_conv2',
                  'block2_conv2',
                  'block3_conv2']
style_layers = ['block1_conv1',
                'block2_conv1',
                'block3_conv1',
                'block4_conv1',
                'block5_conv1']
number_content = len(content_layers)
number_style = len(style_layers)


def get_model():
    vgg = tf.keras.applications.vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    content_output = [vgg.get_layer(layer).output for layer in content_layers]
    style_output = [vgg.get_layer(layer).output for layer in style_layers]
    model_output = style_output + content_output
    return models.Model(vgg.input, model_output)


def get_content_loss(noise, target):
    loss = tf.reduce_mean(tf.square(noise - target))
    return loss


def gram_matrix(tensor):
    channels = int(tensor.shape[-1])
    vector = tf.reshape(tensor, [-1, channels])
    n = tf.shape(vector)[0]
    matrix = tf.matmul(vector, vector, transpose_a=True)
    return matrix / tf.cast(n, tf.float32)


def get_style_loss(noise, target):
    gram_noise = gram_matrix(noise)
    loss = tf.reduce_mean(tf.square(target - gram_noise))
    return loss


def get_features(model, content_path, style_path, size_wrap):
    content_img = img_preprocess(content_path, size_wrap)
    style_image = img_preprocess(style_path, size_wrap)
    content_output = model(content_img)
    style_output = model(style_image)
    content_feature = [layer[0] for layer in content_output[number_style:]]
    style_feature = [layer[0] for layer in style_output[:number_style]]
    return content_feature, style_feature


def compute_loss(model, loss_weights, image, gram_style_features, content_features):
    style_weight, content_weight = loss_weights
    output = model(image)
    content_loss = 0
    style_loss = 0
    noise_style_features = output[:number_style]
    noise_content_feature = output[number_style:]
    weight_per_layer = 1.0 / float(number_style)
    for a, b in zip(gram_style_features, noise_style_features):
        style_loss += weight_per_layer * get_style_loss(b[0], a)
    weight_per_layer = 1.0 / float(number_content)
    for a, b in zip(noise_content_feature, content_features):
        content_loss += weight_per_layer * get_content_loss(a[0], b)
    style_loss *= style_weight
    content_loss *= content_weight
    total_loss = content_loss + style_loss
    return total_loss, style_loss, content_loss


def compute_grads(dictionary):
    with tf.GradientTape() as tape:
        all_loss = compute_loss(**dictionary)
    total_loss = all_loss[0]
    return tape.gradient(total_loss, dictionary['image']), all_loss

import time
def run_style_transfer(size_wrap, content_path, style_path, epochs, content_weight=1e5, style_weight=1):
    start = time.time()
    model = get_model()
    for layer in model.layers:
        layer.trainable = False
    content_feature, style_feature = get_features(model, content_path, style_path, size_wrap)
    style_gram_matrix = [gram_matrix(feature) for feature in style_feature]
    noise = img_preprocess(content_path, size_wrap)
    noise = tf.Variable(noise, dtype=tf.float32)
    optimizer = tf.keras.optimizers.Adam(learning_rate=5, beta_1=0.99, epsilon=1e-1)
    loss_weights = (style_weight, content_weight)
    dictionary = {'model': model,
                  'loss_weights': loss_weights,
                  'image': noise,
                  'gram_style_features': style_gram_matrix,
                  'content_features': content_feature}
    norm_means = np.array([105, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means

    for epoch in range(epochs):
        grad, all_loss = compute_grads(dictionary)
        total_loss, style_loss, content_loss = all_loss
        optimizer.apply_gradients([(grad, noise)])
        clipped = tf.clip_by_value(noise, min_vals, max_vals)
        noise.assign(clipped)
        plot_img = noise.numpy()
        plot_img = deprocess_img(plot_img)
        trans_img = Image.fromarray(plot_img)
        print(f"Epoch:{epoch+1}; Total loss:{total_loss}; Style loss:{style_loss}; Content loss:{content_loss}")

        IPython.display.clear_output(wait=True)
    end = time.time()
    print(f"Total time: {end - start}")
    print(f'{Image.open(content_path).size[0]}x{Image.open(content_path).size[1]}')

    return trans_img
