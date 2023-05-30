import tensorflow.keras.preprocessing.image as process_im
from tensorflow.python.keras import models
import matplotlib.pyplot as plt
import tensorflow as tf
import IPython.display
from tqdm import tqdm
from PIL import Image
import numpy as np


def load_file(image_path):
    image = Image.open(image_path)
    # max_dim = 1080
    max_dim = 512
    factor = max_dim / max(image.size)
    image = image.resize((round(image.size[0] * factor), round(image.size[1] * factor)), Image.ANTIALIAS)
    # if type(image_path)!=str: image = np.flipud(image) #Bandymas
    im_array = process_im.img_to_array(image)
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


def img_preprocess(img_path):
    image = load_file(img_path)
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


def get_features(model, content_path, style_path):
    content_img = img_preprocess(content_path)
    style_image = img_preprocess(style_path)
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


def run_style_transfer(content_path, style_path, epochs=500, content_weight=1e3, style_weight=1e1):
    model = get_model()
    for layer in model.layers:
        layer.trainable = False
    content_feature, style_feature = get_features(model, content_path, style_path)
    style_gram_matrix = [gram_matrix(feature) for feature in style_feature]
    noise = img_preprocess(content_path)
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

    progress_bar = tqdm(range(epochs), desc="Style Transfer Progress")
    # progress_bar = tqdm(range(epochs), desc="Style Transfer Progress", leave=False, position=0)

    for i in progress_bar:
        grad, all_loss = compute_grads(dictionary)
        total_loss, style_loss, content_loss = all_loss
        optimizer.apply_gradients([(grad, noise)])
        clipped = tf.clip_by_value(noise, min_vals, max_vals)
        noise.assign(clipped)
        plot_img = noise.numpy()
        plot_img = deprocess_img(plot_img)
        trans_img = Image.fromarray(plot_img)
        progress_bar.set_postfix({"Total Loss": total_loss.numpy(), "Style Loss": style_loss.numpy(),
                                  "Content Loss": content_loss.numpy()})

    IPython.display.clear_output(wait=True)
    return trans_img
