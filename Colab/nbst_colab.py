import numpy as np
import keras
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.python.keras import models
from google.colab import files
import os
from io import BytesIO
import time
import tarfile
import tempfile
import imageio
from six.moves import urllib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class IO():
    """
    IO Class with file paths
    """
    def __init__(self):
        self.transition_output_path = "Output/Transition_Images/"
        self.transition_output_file_name = "nbst_"
        self.output_image_path = "Output/Final_Image/"
        self.output_gif_path = "Output/Animation/"
        self.output_file_name = "Style_Transfer.jpg"
        self.intermediate_images_file_name = "Intermediate_Images.jpg"
        self.gif_file_name = "nbst_animation.gif"
        self.mask_image_path = "util_images/mask.jpg"
        self._make_dirs()

    def _make_dirs(self):
        tf.gfile.MkDir("util_images")
        tf.gfile.MkDir("Output")
        tf.gfile.MkDir(self.transition_output_path)
        tf.gfile.MkDir(self.output_image_path)
        tf.gfile.MkDir(self.output_gif_path)

class SemanticSegmentation():
    """
    Semantic Segmentation
    Parameters
    ----------
    model_name : String, Optional (default  = mobilenetv2_coco_voctrainaug)
    The base model for runninng DeepLab. The code works with the following
    models
    - mobilenetv2_coco_voctrainaug
    - mobilenetv2_coco_voctrainval
    - xception_coco_voctrainaug
    - xception_coco_voctrainval

    verbose : boolean, Optional (default=False)
    Controls verbosity of output:
    - False: No Output
    - True: Displays the updates.
    """
    def __init__(self, model_name, verbose):
        self.model_name = model_name
        self.verbose = verbose
        self.path = IO()

    def generate_mask(self, input_file):
        """
        Generates the mask.

        Parameters
        ----------
        input_file : String
            Path to input file.
        """
        self.input_file = input_file
        model = self._fetch_DeepLab_model()
        self._generate_mask(model)
        if(os.path.exists(self.path.mask_image_path)):
            if(self.verbose):
                print("Mask Created!")

    def _generate_semantic_segmentation(self, seg_map, original_image):
        """
        Converts the Segmentation map retreived from the DeepLab Model into
        a binary mask.
        In the segmentation map, there are multiple classes and each class is
        given a specific number. Naive Background Style Transfer will
        focus only on people.
        """
        seg_map = np.expand_dims(seg_map, axis = 2)
        for i in range(0, 2):
            seg_map = np.concatenate((seg_map, seg_map), axis=2)
        mask_image = np.where(seg_map == 15, [255, 255, 255, 255], [0, 0, 0, 255]).astype(np.uint8)
        img = Image.fromarray(mask_image)
        rgb_img = img.convert('RGB')
        rgb_img = rgb_img.resize((original_image.size[0], original_image.size[1]), Image.ANTIALIAS)
        rgb_img.save(self.path.mask_image_path)

    def _generate_mask(self, model):
        """
        Creates the image from the input file, runs the retreived DeepLab model
        on the image to generate a segmentation map, and passes the segmentation
        map to generate a binary mask.
        """
        img = Image.open(self.input_file)
        if img.mode != 'RGB':
         img = img.convert('RGB')
        if(self.verbose):
            print("Running deeplab on image %s ..." % self.input_file)
        seg_map = model.run(img)
        self._generate_semantic_segmentation(seg_map, img)

    def _fetch_DeepLab_model(self):
        """
        Fetches the DeepLab Model from the DeepLabModel class.
        """
        download_url_prefix = 'http://download.tensorflow.org/models/'
        model_urls = {
            'mobilenetv2_coco_voctrainaug':
                'deeplabv3_mnv2_pascal_train_aug_2018_01_29.tar.gz',
            'mobilenetv2_coco_voctrainval':
                'deeplabv3_mnv2_pascal_trainval_2018_01_29.tar.gz',
            'xception_coco_voctrainaug':
                'deeplabv3_pascal_train_aug_2018_01_04.tar.gz',
            'xception_coco_voctrainval':
                'deeplabv3_pascal_trainval_2018_01_04.tar.gz',
        }
        tarball_name = 'deeplab_model.tar.gz'
        model_dir = tempfile.mkdtemp()
        os.makedirs(model_dir, exist_ok=True)
        download_path = os.path.join(model_dir, tarball_name)
        model_dir = tempfile.mkdtemp()
        if(self.verbose):
            print("Downloading model ...")
        urllib.request.urlretrieve(download_url_prefix + model_urls[self.model_name],
                           download_path)
        if(self.verbose):
            print("Download Complete! Loading DeepLabModel...")
        model = DeepLabModel(download_path)
        if(self.verbose):
            print("Model Loaded Successfully!")
        return model

"""
DeepLabv3+ code by Google, TensorFlow

Code Link: https://colab.research.google.com/github/tensorflow/models/blob/master/research/deeplab/deeplab_demo.ipynb#scrollTo=aUbVoHScTJYe
Algorithm: deeplabv3plus2018
Title: Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation
Author: Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, and Hartwig Adam
Book Title: ECCV
Year: 2018
"""

class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 513
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, tarball_path):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()

    graph_def = None
    # Extract frozen graph from tar archive.
    tar_file = tarfile.open(tarball_path)
    for tar_info in tar_file.getmembers():
      if self.FROZEN_GRAPH_NAME in os.path.basename(tar_info.name):
        file_handle = tar_file.extractfile(tar_info)
        graph_def = tf.GraphDef.FromString(file_handle.read())
        break

    tar_file.close()

    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.Session(graph=self.graph)

  def run(self, image):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    width, height = image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    return seg_map


class NaiveBackgroundStyleTransfer():
    """
    Naive Background Style Transfer

    Parameters
    ----------
    number_of_epochs : int, Optional (default = 1000)
    Number of iterations for training.

    content_weight : int, Optional (default  = 1e3)
    Affects the content loss. Should be comparatively much higher.

    style_weight : int, Optional (default=1e-2)
    Affects the style loss. Should be comparatively much lower.
    
    model_name : String, Optional (default = mobilenetv2_coco_voctrainaug)
    The base model for runninng DeepLab. The code works with the following
    models
    - mobilenetv2_coco_voctrainaug
    - mobilenetv2_coco_voctrainval
    - xception_coco_voctrainaug
    - xception_coco_voctrainval
    
    enable_gpu: Boolean, Optional (default = False)
    -Trains the model using the GPU if set to True
    -Trains the model using the CPU if set to False

    verbose : boolean, Optional (default=False)
    Controls verbosity of output:
    - False: No Output
    - True: Displays the completed epoch.
    """

    def __init__(self, number_of_epochs=1000, content_weight=1e3, style_weight=1e-2, model_name="mobilenetv2_coco_voctrainaug", enable_gpu=False, verbose=False):
        self.number_of_epochs = number_of_epochs
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.content_layers = ["block5_conv2"]
        self.style_layers = ["block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1", "block5_conv1"]
        self.model_name = model_name
        self.verbose = verbose
        self.enable_gpu = enable_gpu
        self.path = IO()
        self._set_device()
        tf.enable_eager_execution()
        if(self.verbose == True):
            print ("Eager Execution: {}".format(tf.executing_eagerly()))
        

    def perform(self, content_file, style_file):
        """
        Performs Naive Background Style Transfer

        Parameters
        ----------
        content_file : image
        Content Image.
        
        style_file : image
        Style Image.

            Returns
        -------
        final_image : image
        Returns the final output of naive background style transfer.
        """
        return self._perform(content_file, style_file)

    def show_image(self, image):
        """
        Displays and stores the image.

        Parameters
        ----------
        image : image
            Final Image.
        """
        return self._show_image(image)

    def generate_gif(self, speed):
        """
        Generates a gif image.
        """
        return self._generate_gif(speed)
    
    def _set_device(self):
        """
        Sets device as CPU or GPU
        """
        if(self.enable_gpu):
            device_name = tf.test.gpu_device_name()
            if device_name != '/device:GPU:0':
              raise SystemError('GPU device not found')
            if(self.verbose):
                print('Found GPU at: {}'.format(device_name))
            self.device_name = "/gpu:0"
        else:
            self.device_name = "/cpu:0"

    def _load_image(self, filename):
        """
        Loads the image and corrects its dimensions.
        """
        img = Image.open(filename)
        # Downsample Image using Image.ANTIALIAS
        img = img.resize((img.size[0], img.size[1]), Image.ANTIALIAS)
        img = np.array(img, dtype=np.float32)
        # Remove Alpha Channel if present.
        if(img.shape[2] == 4):
            img = img[:, :, :3]
        # Convert Image Shape to (1, X, X, X)
        img = np.expand_dims(img, axis=0)
        return img

    def _generate_mask(self, input_file):
        """
        Generates a mask for the content image needed for Naive background
        Style Transfer.
        """
        seg = SemanticSegmentation(model_name=self.model_name, verbose=self.verbose)
        seg.generate_mask(input_file)
        return
          
    def _load_mask(self, input_file):
        """
        Load and normalize the mask.
        """
        self._generate_mask(input_file)
        mask = Image.open(self.path.mask_image_path).convert('L')
        # Downsample mask using Image.ANTIALIAS
        mask = mask.resize((mask.size[0], mask.size[1]), Image.ANTIALIAS)
        mask = np.array(mask, dtype=np.float32)
        # Normalize the mask.
        mask = mask / 255.
        # Convert Image Shape to (1, X, X, X)
        mask = np.expand_dims(mask, axis=0)
        return mask

    def _preprocess_image(self, filename):
        """
        Preprocesses the image into the necessary format.
        """
        img = self._load_image(filename)
        # Preprocess Image to VGG19 Input format (Subtracts Input by VGG Mean and
        # uses cv2 to open the image which uses BGR format)
        #im = cv2.resize(cv2.imread('cat.jpg'), (224, 224)).astype(np.float32)
        #im[:,:,0] -= 103.939
        #im[:,:,1] -= 116.779
        #im[:,:,2] -= 123.68
        img = tf.keras.applications.vgg19.preprocess_input(img)
        return img

    def _deprocess_image(self, processed_image):
        """
        Deprocess the processed image.
        """
        x = processed_image.copy()
        if(len(x.shape) == 4):
            x = np.squeeze(x)
        assert(len(x.shape) == 3)

        # Deprocess Image by adding back the VGG Mean.
        x[:, :, 0] += 103.939
        x[:, :, 1] += 116.779
        x[:, :, 2] += 123.680

        # Convert from BGR to RGB.
        x = x[:, :, ::-1]
        # Make sure minimum value of a pixel is 0 and maximum value of a pixel is 255.
        x = np.clip(x, 0, 255).astype('uint8')
        return x

    def _get_model(self):
        """
        Fetches the appropriate layers of the VGG19 Model and activates only
        those layers.
        """
        vgg_model = tf.keras.applications.vgg19.VGG19(include_top = False, weights='imagenet')
        vgg_model.trainable = False
        if(self.verbose == True):
            print(vgg_model.summary())

        # Get Outputs of Content Layers and Style Layers
        content_outputs = [vgg_model.get_layer(layer).output for layer in self.content_layers]
        style_outputs = [vgg_model.get_layer(layer).output for layer in self.style_layers]

        model_outputs = style_outputs + content_outputs
        return models.Model(vgg_model.inputs, model_outputs)

    def _get_activations(self, model, content_file, style_file):
        """
        Get activations of the selected layers.
        """
        content_image = self._preprocess_image(content_file)
        style_image = self._preprocess_image(style_file)

        content_outputs = model(content_image)
        style_outputs = model(style_image)

        # Get activations of respective layers. content_layer[0] is done to convert
        # list shape of (1, X, X, X) to (X, X, X)
        content_image_activations = [content_layer[0] for content_layer in content_outputs[len(self.style_layers):]]
        style_image_activations = [style_layer[0] for style_layer in style_outputs[:len(self.style_layers)]]

        return content_image_activations, style_image_activations

    def _content_loss_computation(self, content_image_activations, generated_image_activations):
        """
        Computes the Content Loss.
        """
        # Content Loss Computation
        return tf.reduce_mean(tf.square(content_image_activations - generated_image_activations))

    def _gram_matrix_computation(self, input_activations):
        """
        Computes the Gram Matrix.
        """
        # Gram Matrix Computation
        channels = input_activations.shape[-1]
        activations = tf.reshape(input_activations, [-1, channels])
        number_of_activations = activations.shape[0]
        gram_matrix = tf.matmul(activations, activations, transpose_a=True)
        gram_matrix = gram_matrix / tf.cast(number_of_activations, tf.float32)
        return gram_matrix

    def _style_loss_computation(self, gram_matrix_style_image, gram_matrix_generated_image):
        """
        Computes the Style Loss.
        """
        # Style Loss Computation
        return tf.reduce_mean(tf.square(gram_matrix_style_image - gram_matrix_generated_image))

    def _compute_overall_loss(self, model, loss_weights, generated_image, content_image_activations, style_image_activations):
        """
        Computes the Overall Loss (Content Loss + Style Loss).
        """
        # While training, first the style loss decreases rapidly to some small value.
        #After the content loss starts decreasing, the style loss either decreases
        #much more slowly or fluctuates.
        #Weights should be initialized such that the content loss is significantly
        #greater than the style loss at this point. Otherwise network will not learn
        #any of the content.
        content_weight, style_weight = loss_weights

        model_outputs = model(generated_image)

        # Get respective activations of generated image
        generated_image_activations_content = [content_layer[0] for content_layer in model_outputs[len(self.style_layers):]]
        generated_image_activations_style = [style_layer[0] for style_layer in model_outputs[:len(self.style_layers)]]
        # Convert style image and generated image into their respective gram matrices
        gram_matrices_style_image = [self._gram_matrix_computation(activation) for activation in style_image_activations]
        gram_matrices_generated_image = [self._gram_matrix_computation(activation) for activation in generated_image_activations_style]

        style_loss = 0
        content_loss = 0
        # Compute the content loss
        weight_per_content_layer = 1.0 / float(len(self.content_layers))
        for a, b in zip(content_image_activations, generated_image_activations_content):
            content_loss = content_loss + (weight_per_content_layer * self._content_loss_computation(a, b))

        # Compute the style loss
        weight_per_style_layer = 1.0 / float(len(self.style_layers))
        for a, b in zip(gram_matrices_style_image, gram_matrices_generated_image):
            style_loss = style_loss + (weight_per_style_layer * self._style_loss_computation(a, b))

        content_loss = content_loss * content_weight
        style_loss = style_loss * style_weight
        overall_loss = content_loss + style_loss

        return overall_loss, content_loss, style_loss

    def _compute_gradients(self, parameters):
        """
        Determines the gradients for optimization.
        """
        with tf.GradientTape() as g:
            losses = self._compute_overall_loss(**parameters)
        overall_loss = losses[0]
        # Compute the Gradient dJ(G) / d(G) using tf.GradientTape()
        return g.gradient(overall_loss, parameters['generated_image']), losses

    def _perform(self, content_file, style_file):
        """
        Performs Naive Background Style Transfer and saves the intermediate stages of
        the style transfer process.
        """
        with tf.device(self.device_name):
          model = self._get_model()
          for layer in model.layers:
              layer.trainable = False
          mask_image = self._load_mask(content_file)
          content_image_activations, style_image_activations = self._get_activations(model, content_file, style_file)
          content_image = self._preprocess_image(content_file)
          generated_image = self._preprocess_image(content_file)
          generated_image = tf.Variable(generated_image, dtype=tf.float32)
          optimizer = tf.train.AdamOptimizer(learning_rate=5, beta1=0.99, epsilon=1e-1)
          loss_weights = (self.content_weight, self.style_weight)
          final_loss, final_image = np.inf, None
          parameters = {
          "model" : model,
          "loss_weights": loss_weights,
          "generated_image": generated_image,
          "content_image_activations": content_image_activations,
          "style_image_activations": style_image_activations
          }
          gif_interval = [0, 1, 2, 3, 4, 5, 10, 15, 20, 25, 40, 60, 80, 100, 120, 140,
                          160, 180, 200, 220, 250, 280, 300, 330, 360, 390, 420, 450, 480, 510, 540, 580,
                          620, 660, 700, 725, 750, 775, 800, 825, 875, 900, 950, 999]
          num_rows = 2
          num_columns = 5
          display_interval = self.number_of_epochs / (num_rows*num_columns)

          norm_means = np.array([103.939, 116.779, 123.680])
          min_value = -norm_means
          max_value = 255 - norm_means
          intermediate_images = []
          for epoch in range(self.number_of_epochs):
              gradients, losses = self._compute_gradients(parameters)
              overall_loss, content_loss, style_loss = losses
              optimizer.apply_gradients([(gradients, generated_image)])
              clipped_image = tf.clip_by_value(generated_image, min_value, max_value)
              generated_image.assign(clipped_image)

              if overall_loss < final_loss:
                final_loss = overall_loss
                final_image = generated_image.numpy()
                if(epoch == self.number_of_epochs-1):
                  for i in range(0, mask_image.shape[1]):
                    for j in range(0, mask_image.shape[2]):
                      if(mask_image[:, i, j] == 1.):
                        final_image[:, i, j, :] = content_image[:, i, j, :]
                final_image = self._deprocess_image(final_image)
              if(self.verbose == True):
                  print("Epoch Number %d : Completed" % (epoch+1))


              # This part of the code is for smaller intervals(Every 4th iteration),
              #inorder to make the .gif file.
              
              if epoch in gif_interval:
                  transition_image = generated_image.numpy()
                  for i in range(0, mask_image.shape[1]):
                    for j in range(0, mask_image.shape[2]):
                      if(mask_image[:, i, j] == 1.):
                        transition_image[:, i, j, :] = content_image[:, i, j, :]
                  transition_image = self._deprocess_image(transition_image)
                  img = Image.fromarray(transition_image)
                  img.save(self.path.transition_output_path+self.path.transition_output_file_name+str(100000+epoch+1)+".jpg")
                  #files.download(self.path.transition_output_path+self.path.transition_output_file_name+str(epoch+1)+".jpg")
                  plt.close()

              if epoch % display_interval == 0:
                  intermediate_image = generated_image.numpy()
                  intermediate_image = self._deprocess_image(intermediate_image)
                  intermediate_images.append(intermediate_image)

          # Show Intermediate Images
          plt.figure(figsize=(14,4))
          for i,intermediate_image in enumerate(intermediate_images):
              plt.subplot(num_rows,num_columns,i+1)
              plt.imshow(intermediate_image)
              plt.xticks([])
              plt.yticks([])
          plt.savefig(self.path.output_image_path+self.path.intermediate_images_file_name)
          #files.download(self.path.output_image_path+self.path.intermediate_images_file_name)
          plt.close()
        return final_image

    def _show_image(self, image):
        """
        Displays the final image.
        """
        # Display the final image.
        img = Image.fromarray(image)
        img.save(self.path.output_image_path + self.path.output_file_name)
        #files.download(self.path.output_image_path + self.path.output_file_name)
        return

    def _generate_gif(self, speed):
        """
        Generate gif Image.
        """
        assert (os.path.exists(self.path.transition_output_path)), "Naive Background Style Transfer not performed. No Transition Images available."
        filenames = [os.path.join(self.path.transition_output_path, f) for f in os.listdir(self.path.transition_output_path) if f.endswith(".jpg")]
        filenames = sorted(filenames)
        images = []
        for filename in filenames:
            images.append(imageio.imread(filename))
        if not os.path.exists(self.path.output_gif_path):
            os.makedirs(self.path.output_gif_path)
        imageio.mimsave(self.path.output_gif_path+self.path.gif_file_name, images, duration=speed)

content_file = "portrait.jpg"
style_file = "scream.jpg"
start_time = time.time()
nbst = NaiveBackgroundStyleTransfer(number_of_epochs=1000, enable_gpu=True, verbose=True)
image = nbst.perform(content_file, style_file)
end_time = time.time()
print("Time Elapsed: %ds" %(end_time - start_time))
nbst.show_image(image)
nbst.generate_gif(speed=0.8)
