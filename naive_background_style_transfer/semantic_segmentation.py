import tarfile
import tempfile
from six.moves import urllib
import os
import numpy as np
from PIL import Image
from deep_lab import DeepLabModel
from utils import IO
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class SemanticSegmentation():
    """
        Semantic Segmentation
        Parameters
        ----------
        model_name : String, (default  = mobilenetv2_coco_voctrainaug)
        The base model for runninng DeepLab. The code works with the following
        models
        - mobilenetv2_coco_voctrainaug
        - mobilenetv2_coco_voctrainval
        - xception_coco_voctrainaug
        - xception_coco_voctrainval

        verbose : boolean, (default=False)
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
                print "Mask Created!"

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
            print "Running deeplab on image "+str(self.input_file)+" ..."
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
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        download_path = os.path.join(model_dir, tarball_name)
        if(self.verbose):
            print "Downloading model ..."
        urllib.request.urlretrieve(download_url_prefix + model_urls[self.model_name],
                           download_path)
        if(self.verbose):
            print "Download Complete! Loading DeepLabModel..."
        model = DeepLabModel(download_path)
        if(self.verbose):
            print "Model Loaded Successfully!"
        return model
