from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence

from numbers import Number

import uuid
import logging
import random
import math
import numpy as np

import cv2
from PIL import Image, ImageEnhance, ImageOps


class BboxError(ValueError):
    pass


class ImageError(ValueError):
    pass


class BaseOperator(object):
    def __init__(self, name=None):
        if name is None:
            name = self.__class__.__name__
        self._id = name + '_' + str(uuid.uuid4())[-6:]

    def __call__(self, sample, context=None):
        """ Process a sample.
        Args:
            sample (dict): a dict of sample, eg: {'image':xx, 'label': xxx}
            context (dict): info about this sample processing
        Returns:
            result (dict): a processed sample
        """
        return sample

    def __str__(self):
        return str(self._id)

class Permute(BaseOperator):
    def __init__(self, channel_first=True):
        super(Permute, self).__init__()
        self.channel_first = channel_first

    def __call__(self, sample, context=None):
        samples = sample
        batch_input = True
        if not isinstance(samples, Sequence):
            batch_input = False
            samples = [samples]
        for sample in samples:
            assert 'image' in sample, "image data not found"
            if self.channel_first:
                im = sample['image']
                im = np.swapaxes(im, 1, 2)
                im = np.swapaxes(im, 1, 0)
                sample['image'] = im
        if not batch_input:
            samples = samples[0]
        return samples

class DecodeImage(BaseOperator):
    def __init__(self, to_rgb=True, with_mixup=False):
        """ Transform the image data to numpy format.

        Args:
            to_rgb (bool): whether to convert BGR to RGB
            with_mixup (bool): whether or not to mixup image and gt_bbbox/gt_score
        """

        super(DecodeImage, self).__init__()
        self.to_rgb = to_rgb
        self.with_mixup = with_mixup
        if not isinstance(self.to_rgb, bool):
            raise TypeError("{}: input type is invalid.".format(self))
        if not isinstance(self.with_mixup, bool):
            raise TypeError("{}: input type is invalid.".format(self))

    def __call__(self, sample, context=None):
        """ load image if 'im_file' field is not empty but 'image' is"""
        if 'image' not in sample:
            with open(sample['im_file'], 'rb') as f:
                sample['image'] = f.read()

        im = sample['image']
        data = np.frombuffer(im, dtype='uint8')
        im = cv2.imdecode(data, 1)  # BGR mode, but need RGB mode
        if self.to_rgb:
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        sample['image'] = im

        if 'h' not in sample:
            sample['h'] = im.shape[0]
        if 'w' not in sample:
            sample['w'] = im.shape[1]
        # make default im_info with [h, w, 1]
        sample['im_info'] = np.array(
            [im.shape[0], im.shape[1], 1.], dtype=np.float32)
        # decode mixup image
        if self.with_mixup and 'mixup' in sample:
             self.__call__(sample['mixup'], context)
        return sample

class NormalizeImage(BaseOperator):
    def __init__(self,
                 mean=[0.485, 0.456, 0.406],
                 std=[0.229, 0.224, 0.225],
                 is_scale=True,
                 is_channel_first=False):
        """
        Args:
            mean (list): the pixel mean
            std (list): the pixel variance
        """
        super(NormalizeImage, self).__init__()
        self.mean = mean
        self.std = std
        self.is_scale = is_scale
        self.is_channel_first = is_channel_first
        if not (isinstance(self.mean, list) and isinstance(self.std, list) and
                isinstance(self.is_scale, bool)):
            raise TypeError("{}: input type is invalid.".format(self))
        from functools import reduce
        if reduce(lambda x, y: x * y, self.std) == 0:
            raise ValueError('{}: std is invalid!'.format(self))

    def __call__(self, sample, context=None):
        """Normalize the image.
        Operators:
            1.(optional) Scale the image to [0,1]
            2. Each pixel minus mean and is divided by std
        """
        samples = sample
        batch_input = True
        if not isinstance(samples, Sequence):
            batch_input = False
            samples = [samples]
        for sample in samples:
            for k in sample.keys():
                # hard code
                if k.startswith('image'):
                    im = sample[k]
                    im = im.astype(np.float32, copy=False)
                    if self.is_channel_first:
                        mean = np.array(self.mean)[:, np.newaxis, np.newaxis]
                        std = np.array(self.std)[:, np.newaxis, np.newaxis]
                    else:
                        mean = np.array(self.mean)[np.newaxis, np.newaxis, :]
                        std = np.array(self.std)[np.newaxis, np.newaxis, :]
                    if self.is_scale:
                        im = im / 255.0
                    im -= mean
                    im /= std
                    sample[k] = im
        if not batch_input:
            samples = samples[0]
        return samples

    
class ResizeImage(BaseOperator):
    def __init__(self, height=384, width=384, interp=cv2.INTER_LINEAR, use_cv2=True):
        super(ResizeImage, self).__init__()
        self.height = height
        self.width = width
        self.interp = int(interp)
        self.use_cv2 = use_cv2


    def __call__(self, sample, context=None):
        im = sample['image']
        h,w,c = im.shape
        if h == self.height and w == self.width:
            return sample
        im_scale_x = float(self.width) / float(w)
        im_scale_y = float(self.height) / float(h)
        im = cv2.resize( im, None, None, fx=im_scale_x, fy=im_scale_y, interpolation=self.interp)
        #im = cv2.resize( im, (self.width, self.height), interpolation=self.interp)

        #im = im.astype('uint8')
        #im = Image.fromarray(im)
        #im = im.resize((int(self.width), int(self.height)), Image.BILINEAR)
        #im = np.array(im)
        sample['image'] = im
        return sample    
    
    
class RandomResizeImage(BaseOperator):
    def __init__(self, interp=cv2.INTER_LINEAR, use_cv2=True):
        super(RandomResizeImage, self).__init__()

        self.interp = int(interp)
        self.use_cv2 = use_cv2

    def __call__(self, sample, context=None):
            
        resize_w = sample["target_size"]
        resize_h = sample["target_size"]
        
        im = sample['image']
        h,w,c = im.shape
        #print(im.shape)
        if h == resize_h and w == resize_w:
            return sample
        im_scale_x = float(resize_w) / float(w)
        im_scale_y = float(resize_h) / float(h)
        im = cv2.resize(im, None, None, fx=im_scale_x, fy=im_scale_y, interpolation=self.interp)
        #im = cv2.resize( im, (self.width, self.height), interpolation=self.interp)

        #im = im.astype('uint8')
        #im = Image.fromarray(im)
        #im = im.resize((int(self.width), int(self.height)), Image.BILINEAR)
        #im = np.array(im)
        sample['image'] = im
        return sample


class ResizeRandomCrop(BaseOperator):
    def __init__(self, big_height=288, big_width=384, target_height=288, target_width=384, interp=cv2.INTER_LINEAR, use_cv2=True):
        
        self.big_height = big_height
        self.big_width = big_width
        self.target_height = target_height
        self.target_width = target_width
        assert self.big_height >= self.target_height and self.big_width >= self.target_width
        self.interp = int(interp)
        self.use_cv2 = use_cv2


    def __call__(self, sample, context=None):
        im = sample['image']
        h,w,c = im.shape
        if not isinstance(im, np.ndarray):
            raise TypeError("{}: image type is not numpy.".format(self))
        if len(im.shape) != 3:
            raise ImageError('{}: image is not 3-dimensional.'.format(self))

        # Resize to big size
        im_scale_x = float(self.big_width) / float(w)
        im_scale_y = float(self.big_height) / float(h)
        im = cv2.resize( im, None, None, fx=im_scale_x, fy=im_scale_y, interpolation=self.interp)


        # random crop
        crop_w = self.big_width -  self.target_width
        crop_h = self.big_height - self.target_height
        if crop_w == 0 and crop_h == 0:
            sample['image'] = im
            return sample

        x1 = random.randint(0, crop_w)
        y1 = random.randint(0, crop_h)
        im = im[y1:(y1+self.target_height), x1:(x1+self.target_width)]
        sample['image'] = im
    
        return sample


class RandomHorizontalFlip(BaseOperator):
    def __init__(self, prob=0.5):
        """
        Args:
            prob (float): the probability of flipping image
            is_normalized (bool): whether the bbox scale to [0,1]
            is_mask_flip (bool): whether flip the segmentation
        """
        super(RandomHorizontalFlip, self).__init__()
        self.prob = prob
        
        if not isinstance(self.prob, float):
            raise TypeError("{}: input type is invalid.".format(self))

    def __call__(self, sample, context=None):
        """Filp the image and bounding box.
        Operators:
            1. Flip the image numpy.
            2. Transform the bboxes' x coordinates.
              (Must judge whether the coordinates are normalized!)
            3. Transform the segmentations' x coordinates.
              (Must judge whether the coordinates are normalized!)
        Output:
            sample: the image, bounding box and segmentation part
                    in sample are flipped.
        """

        samples = sample
        batch_input = True
        if not isinstance(samples, Sequence):
            batch_input = False
            samples = [samples]
        for sample in samples:
            im = sample['image']
            if not isinstance(im, np.ndarray):
                raise TypeError("{}: image is not a numpy array.".format(self))
            if len(im.shape) != 3:
                raise ImageError("{}: image is not 3-dimensional.".format(self))
            height, width, _ = im.shape
            if np.random.uniform(0, 1) < self.prob:
                im = im[:, ::-1, :]
                sample['flipped'] = True
                sample['image'] = im
        sample = samples if batch_input else samples[0]
        return sample

class HorizontalFlip(BaseOperator):
    def __init__(self):
        super(HorizontalFlip, self).__init__()
        
    def __call__(self, sample, context=None):
        """Filp the image and bounding box.
        Operators:
            1. Flip the image numpy.
            2. Transform the bboxes' x coordinates.
              (Must judge whether the coordinates are normalized!)
            3. Transform the segmentations' x coordinates.
              (Must judge whether the coordinates are normalized!)
        Output:
            sample: the image, bounding box and segmentation part
                    in sample are flipped.
        """

        samples = sample
        batch_input = True
        if not isinstance(samples, Sequence):
            batch_input = False
            samples = [samples]
        for sample in samples:
            im = sample['image']
            if not isinstance(im, np.ndarray):
                raise TypeError("{}: image is not a numpy array.".format(self))
            if len(im.shape) != 3:
                raise ImageError("{}: image is not 3-dimensional.".format(self))
            height, width, _ = im.shape
            im = im[:, ::-1, :]
            sample['flipped'] = True
            sample['image'] = im
        sample = samples if batch_input else samples[0]
        return sample


class RandomErasing(BaseOperator):
    ### has verify


    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
        probability: The probability that the Random Erasing operation will be performed.
        sl: Minimum proportion of erased area against input image.
        sh: Maximum proportion of erased area against input image.
        r1: Minimum aspect ratio of erased area.
        mean: Erasing value. 
    """

    def __init__(self, probability = 0.5, sl = 0.02, sh = 0.4, r1 = 0.3, mean=[0.485, 0.456, 0.406]):
        super(RandomErasing, self).__init__()
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, sample, context=None):
        if random.uniform(0, 1) > self.probability:
            return sample
        img = sample['image']
        for attempt in range(100):
            ori_h = img.shape[0]
            ori_w = img.shape[1]
            area = ori_h * ori_w

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < ori_w and h < ori_h:
                x1 = random.randint(0, ori_h - h)
                y1 = random.randint(0, ori_w - w)
                if img.shape[2] == 3:
                    #img[0, x1:x1+h, y1:y1+w] = self.mean[0]
                    #img[1, x1:x1+h, y1:y1+w] = self.mean[1]
                    #img[2, x1:x1+h, y1:y1+w] = self.mean[2]
                    #img[x1:x1 + h, y1:y1 + w, :] = self.mean
                    img[x1:x1+h, y1:y1+w,0] = self.mean[0]
                    img[x1:x1+h, y1:y1+w,1] = self.mean[1]
                    img[x1:x1+h, y1:y1+w,2] = self.mean[2]
                else:
                    img[x1:x1+h, y1:y1+w, 0] = self.mean[0]
                sample['image'] = img
                return sample
        sample['image'] = im
        return sample

class MixupImage(BaseOperator):
    def __init__(self, alpha=1.5, beta=1.5):
        """ Mixup image and gt_bbbox/gt_score
        Args:
            alpha (float): alpha parameter of beta distribute
            beta (float): beta parameter of beta distribute
        """
        super(MixupImage, self).__init__()
        self.alpha = alpha
        self.beta = beta
        if self.alpha <= 0.0:
            raise ValueError("alpha shold be positive in {}".format(self))
        if self.beta <= 0.0:
            raise ValueError("beta shold be positive in {}".format(self))

    def _mixup_img(self, img1, img2, factor):
        h = max(img1.shape[0], img2.shape[0])
        w = max(img1.shape[1], img2.shape[1])
        img = np.zeros((h, w, img1.shape[2]), 'float32')
        img[:img1.shape[0], :img1.shape[1], :] = \
            img1.astype('float32') * factor
        img[:img2.shape[0], :img2.shape[1], :] += \
            img2.astype('float32') * (1.0 - factor)
        return img.astype('uint8')

    def __call__(self, sample, context=None):
        if 'mixup' not in sample:
            return sample
        factor = np.random.beta(self.alpha, self.beta)
        factor = max(0.0, min(1.0, factor))
        if factor >= 1.0:
            sample['factor'] = 1.0
            sample['mixup_factor'] = 0.0
            sample['mixup_pid'] = sample['mixup']['pid']
            sample['mixup_colorid'] = sample['mixup']['colorid']
            sample['mixup_typeid'] = sample['mixup']['typeid']
            sample.pop('mixup')
            return sample
        if factor <= 0.0:
            sample['image'] = sample['mixup']['image']
            sample['factor'] = 0.0
            sample['mixup_factor'] = 1.0
            sample['mixup_pid'] = sample['mixup']['pid']
            sample['mixup_colorid'] = sample['mixup']['colorid']
            sample['mixup_typeid'] = sample['mixup']['typeid']
            sample.pop('mixup')
            return sample

        im = self._mixup_img(sample['image'], sample['mixup']['image'], factor)
        sample['image'] = im
        sample['factor'] = factor
        sample['mixup_factor'] = 1.0 - factor
        sample['mixup_pid'] = sample['mixup']['pid']
        sample['mixup_colorid'] = sample['mixup']['colorid']
        sample['mixup_typeid'] = sample['mixup']['typeid']
        sample.pop('mixup')
        return sample


class ImageNetPolicy(BaseOperator):

    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.4, "posterize", 8, 0.6, "rotate", 9, fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor),
            SubPolicy(0.6, "posterize", 7, 0.6, "posterize", 6, fillcolor),
            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),

            SubPolicy(0.4, "equalize", 4, 0.8, "rotate", 8, fillcolor),
            SubPolicy(0.6, "solarize", 3, 0.6, "equalize", 7, fillcolor),
            SubPolicy(0.8, "posterize", 5, 1.0, "equalize", 2, fillcolor),
            SubPolicy(0.2, "rotate", 3, 0.6, "solarize", 8, fillcolor),
            SubPolicy(0.6, "equalize", 8, 0.4, "posterize", 6, fillcolor),

            SubPolicy(0.8, "rotate", 8, 0.4, "color", 0, fillcolor),
            SubPolicy(0.4, "rotate", 9, 0.6, "equalize", 2, fillcolor),
            SubPolicy(0.0, "equalize", 7, 0.8, "equalize", 8, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),

            SubPolicy(0.8, "rotate", 8, 1.0, "color", 2, fillcolor),
            SubPolicy(0.8, "color", 8, 0.8, "solarize", 7, fillcolor),
            SubPolicy(0.4, "sharpness", 7, 0.6, "invert", 8, fillcolor),
            SubPolicy(0.6, "shearX", 5, 1.0, "equalize", 9, fillcolor),
            SubPolicy(0.4, "color", 0, 0.6, "equalize", 3, fillcolor),

            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor)
        ]
        self.num_policy_minus = len(self.policies) -1

    def __call__(self, sample, context=None):
        policy_idx = random.randint(0, self.num_policy_minus)

        img = sample['image']
        img = np.ascontiguousarray(img)
        img = Image.fromarray(img)

        img = self.policies[policy_idx](img)
        img = np.asarray(img)
        sample['image'] = img
        return sample


class SubPolicy(object):
    def __init__(self, p1, operation1, magnitude_idx1, p2, operation2, magnitude_idx2, fillcolor=(128, 128, 128)):
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(np.int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10
        }

        # from https://stackoverflow.com/questions/5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(rot, Image.new("RGBA", rot.size, (128,) * 4), rot).convert(img.mode)

        func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                Image.BICUBIC, fillcolor=fillcolor),
            "translateX": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor),
            "translateY": lambda img, magnitude: img.transform(
                img.size, Image.AFFINE, (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
                fillcolor=fillcolor),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            # "rotate": lambda img, magnitude: img.rotate(magnitude * random.choice([-1, 1])),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: ImageOps.equalize(img),
            "invert": lambda img, magnitude: ImageOps.invert(img)
        }

        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.p2 = p2
        self.operation2 = func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]

    def __call__(self, img,context=None):
        if random.random() < self.p1: img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2: img = self.operation2(img, self.magnitude2)
        return img

    
class Rotate(BaseOperator):    
    def __init__(self, magnitude=0):
        super(Rotate, self).__init__()

        self.angle = int(random.randint(-10,10) * magnitude * 0.1)

        def rotate_with_fill (img, angle):
            rot = img.convert("RGBA").rotate(angle)
            return Image.composite(rot,Image.new("RGBA", rot.size, (128, ) * 4),rot).convert(img.mode)

        self.func = {"rotate": lambda img, angle: rotate_with_fill(img, angle)}
        
    def __call__(self,  sample, context=None):
        img = sample['image']
        img = np.ascontiguousarray(img)
        img = Image.fromarray(img)

        img = self.func["rotate"](img, self.angle)
        img = np.asarray(img)
        sample['image'] = img
        return sample
    
    
    
class RandAugment(BaseOperator):
    def __init__(self, num_layers=2, magnitude=5, fillcolor=(128, 128, 128)):
        super(RandAugment, self).__init__()
        self.num_layers = num_layers
        self.magnitude = magnitude
        self.max_level = 10

        abso_level = self.magnitude / self.max_level
        self.level_map = {
            #"shearX": 0.2 * abso_level,
            #"shearY": 0.2 * abso_level,
            #"translateX": 150.0 / 331 * abso_level,
            #"translateY": 150.0 / 331 * abso_level,
#             "rotate": 90 * abso_level,
            "rotate": int(random.randint(-10,10) * 45 * 0.1),
            #"color": 0.9 * abso_level,
            #"posterize": int(4.0 * abso_level),
            #"solarize": 256.0 * abso_level,
            "contrast": int(random.randint(0,10)* 0.1) * abso_level,
            "sharpness": int(random.randint(0,10)* 0.1) * abso_level,
            "brightness": int(random.randint(0,10)* 0.1) * abso_level,
            "autocontrast": 0,
            "equalize": 0
            #"invert": 0,
            #"motionblur":random.randint(5,10)
            }

        # from https://stackoverflow.com/questions/5252170/
        # specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(rot,
                                   Image.new("RGBA", rot.size, (128, ) * 4),
                                   rot).convert(img.mode)

        rnd_ch_op = random.choice

        self.func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, magnitude * rnd_ch_op([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC,
                fillcolor=fillcolor),
            "shearY": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, 0, magnitude * rnd_ch_op([-1, 1]), 1, 0),
                Image.BICUBIC,
                fillcolor=fillcolor),
            "translateX": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, magnitude * img.size[0] * rnd_ch_op([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor),
            "translateY": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, 0, 0, 1, magnitude * img.size[1] * rnd_ch_op([-1, 1])),
                fillcolor=fillcolor),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(
                1 + magnitude * rnd_ch_op([-1, 1])),
            "posterize": lambda img, magnitude:
                ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude:
                ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude:
                ImageEnhance.Contrast(img).enhance(
                    1 + magnitude * rnd_ch_op([-1, 1])),
            "sharpness": lambda img, magnitude:
                ImageEnhance.Sharpness(img).enhance(
                    1 + magnitude * rnd_ch_op([-1, 1])),
            "brightness": lambda img, magnitude:
                ImageEnhance.Brightness(img).enhance(
                    1 + magnitude * rnd_ch_op([-1, 1])),
            "autocontrast": lambda img, magnitude:
                ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: ImageOps.equalize(img),
            "invert": lambda img, magnitude: ImageOps.invert(img),
            #"motionblur": lambda img, magnitude:Image.fromarray(iaa.MotionBlur(magnitude)(images=[np.array(img)])[0])
        }

    def __call__(self,  sample, context=None):
        img = sample['image']
        img = np.ascontiguousarray(img)
        img = Image.fromarray(img)
        avaiable_op_names = list(self.level_map.keys())
        for layer_num in range(self.num_layers):
            op_name = np.random.choice(avaiable_op_names)
            img = self.func[op_name](img, self.level_map[op_name])
        img = np.asarray(img)
        sample['image'] = img
        return sample
    
    
class RandomResizeTwiceImage(BaseOperator):
    def __init__(self, prob=0.5):
        super(RandomResizeTwiceImage, self).__init__()
        self.height = [i for i in range(50,71)]
        self.width = [i for i in range(90,111)]
        self.interps = [
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_AREA,
            cv2.INTER_CUBIC,
            cv2.INTER_LANCZOS4,
        ]
        self.prob = prob

    def __call__(self, sample, context=None):
        im = sample['image']
        h,w,c = im.shape
        resize_width = np.random.choice(a=self.width, size=1, replace=False, p=None)[0]
        resize_height = np.random.choice(a=self.height, size=1, replace=False, p=None)[0]
        if h == resize_height and w == resize_width:
            return sample
        if np.random.uniform(0, 1) >= self.prob:
            return sample

        # downsample
        resize_type = np.random.choice(a=self.interps, size=1, replace=False, p=None)[0]
        im_scale_x = float(resize_width) / float(w)
        im_scale_y = float(resize_height) / float(h)
        im = cv2.resize( im, None, None, fx=im_scale_x, fy=im_scale_y, interpolation=resize_type)

        # upsample
        resize_type2 = np.random.choice(a=self.interps, size=1, replace=False, p=None)[0]
        im_scale_x = float(w) / float(resize_width)
        im_scale_y = float(h) / float(resize_height)
        im = cv2.resize( im, None, None, fx=im_scale_x, fy=im_scale_y, interpolation=resize_type2)
        sample['image'] = im
        return sample