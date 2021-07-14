# -*- coding: utf-8 -*-
""" gcrlm/processors """

import copy
import os
import random
import re
import shutil
from collections import defaultdict
from math import floor, ceil
from unittest.mock import patch, MagicMock

import cv2
import numpy as np
from gtorch_utils.constants import DB
from gutils.datasets.utils import TrainValTestSplit
from gutils.image_augmentation import ImageAugmentationProcessor
from gutils.mock import notqdm
from logzero import logger
from tabulate import tabulate
from tqdm import tqdm

from gcrlm.constants import Label, CRLMProcessingTypes
from gcrlm.core.exceptions.crlm import SampleFolderAlreadyExists
from gcrlm.core.models import BoundingBox, AugmentDict
from gcrlm.managers import CRLM


class CRLMDatasetProcessor:
    """
    Procesess all the NDPI and NDPA files and creates crops and masks

    Usage:
        from Data.dataset_processors.constants import CRLMProcessingTypes

        CRLMDatasetProcessor(
            level=4,
            images_path='/media/giussepi/2_0_TB_Hard_Disk/improved_CRLM_dataset/Annotated - Training set/'
        )(
            min_mask_area=.000001, processing_type=CRLMProcessingTypes.ALL_WITH_ROIS,
            multiple_ann=True, multiple_classes=True
        )
    """

    def __init__(self, **kwargs):
        """
        Initialized the object instance

        Kwargs:
            level            <int>: magnification level (dim/2**level)
            images_path      <str>: path to the folder for containing the NDPI files
            annotations_path <str>: path to folder for containing the annotations. If not provided,
                                    the images_path is used instead
            filename_tpl     <str>: filename template to create the file name using the index
                                    e.g.: for CRLM_001.npi you should use CRLM_{:03d}.ndpi
                                    Default r'CRLM {}.ndpi'
            filename_reg     <str>: regular expression to get the index id from the filename
                                    Default r'CRLM 0*(?P<index>\d+).ndpi'
        """
        self.level = kwargs.get('level')
        self.images_path = kwargs.get('images_path')
        self.annotations_path = kwargs.get('annotations_path', '')
        self.filename_tpl = kwargs.get('filename_tpl', r'CRLM {}.ndpi')
        self.filename_reg = kwargs.get('filename_reg', r'CRLM 0*(?P<index>\d+).ndpi')

        assert isinstance(self.level, int)
        assert self.level >= 0
        assert isinstance(self.images_path, str)
        assert os.path.isdir(self.images_path)
        assert isinstance(self.annotations_path, str)
        assert isinstance(self.filename_tpl, str)
        assert isinstance(self.filename_reg, str)

        if self.annotations_path:
            assert os.path.isdir(self.annotations_path)

    def __call__(self, **kwargs):
        """ functor call """
        return self.process_files(**kwargs)

    @patch('gcrlm.managers.logger.info', MagicMock())
    @patch('gcrlm.managers.tqdm', notqdm)
    def process_files(self, **kwargs):
        """
        Creates crops and masks based on the provided parameters

        Kwargs:
            patch_size              <int>: size of squared patch. Default 640
            saving_path             <str>: folder path to save the extracted annotations,
                                           Default f'{self.images_path}/../annotations_masks'
            min_mask_area         <float>: min mask area in a annotation patches. Avoids creating
                                           patches with tiny or no mask on them.
            apply_image_transforms <bool>: whether or not apply image transforms
            processing_type <CRLMProcessingTypes>: Type of processing to be performend over CRLM
            central_area_offset   <float>: percentage to ignore before considering the central area
                            e.g.: if dim0=100, dim1=200 and central_area_offset=.2, the central area
                            will be image[20:80, 40:160]
            background_threshold  <float>: value between 0 and 1 used to identify background pixels
                                           (the image is turned into gray scale before the comparison)
            img_format              <str>: Saving image extension. Default 'tiff'
            mask_format             <str>: Saving mask extension. Default 'png'
            multiple_ann           <bool>: Whether or not plot multiple annotations with the same label
                                           per crop. Default False
            multiple_classes       <bool>: Whether or not plot annotations from multiple classes per crop.
                                           Requires multiple_ann = True. Default False
        """
        if 'processing_type' in kwargs.keys():
            processing_type = kwargs.pop('processing_type')
            CRLMProcessingTypes.validate_option(processing_type)
        else:
            processing_type = CRLMProcessingTypes.ALL_WITH_ROIS

        kwargs = copy.deepcopy(kwargs)
        kwargs['level'] = self.level
        ndpis_list = list(filter(lambda x: x.endswith('.ndpi'), os.listdir(self.images_path)))
        pattern = re.compile(self.filename_reg)

        logger.info("Processing ndpi files")

        for ndpi_file in tqdm(ndpis_list):
            try:
                ndpi_index = pattern.fullmatch(ndpi_file).groupdict()['index']
            except AttributeError as err:
                logger.error(err)
                continue
            else:
                ndpi_index = int(ndpi_index)

            slide = CRLM(ndpi_index, self.level, self.images_path, self.annotations_path, self.filename_tpl)

            if processing_type == CRLMProcessingTypes.ALL_WITHOUT_ROIS:
                kwargs['roi_clusters'] = None
                kwargs['inside_roi'] = False
            elif processing_type == CRLMProcessingTypes.ALL_WITH_ROIS:
                kwargs['roi_clusters'] = slide.cluster_annotations_by_roi(all_annotations=True)
                kwargs['inside_roi'] = True
            else:
                # processing_type == CRLMProcessingTypes.ONLY_ROIS
                kwargs['roi_clusters'] = slide.cluster_annotations_by_roi()
                kwargs['inside_roi'] = True

            slide.extract_annotations_and_masks(**kwargs)


class CRLMDatasetAnalyzer:
    """
    Analyzes all the NDPI and NDPA files and prints relevant information

    Usage:
        CRLMDatasetAnalyzer(
            images_path='/media/giussepi/2_0_TB_Hard_Disk/improved_CRLM_dataset/Annotated - Training set/'
        )()
    """

    def __init__(self, **kwargs):
        """
        Initialized the object instance

        Kwargs:
            images_path      <str>: path to the folder for containing the NDPI files
            annotations_path <str>: path to folder for containing the annotations. If not provided,
                                    the images_path is used instead
            filename_tpl     <str>: filename template to create the file name using the index
                                    e.g.: for CRLM_001.npi you should use CRLM_{:03d}.ndpi
            filename_reg     <str>: regular expression to get the index id from the filename
        """
        self.images_path = kwargs.get('images_path')
        self.annotations_path = kwargs.get('annotations_path', '')
        self.filename_tpl = kwargs.get('filename_tpl', 'CRLM {}.ndpi')
        self.filename_reg = kwargs.get('filename_reg', r'CRLM 0*(?P<index>\d+).ndpi')

        assert isinstance(self.images_path, str)
        assert os.path.isdir(self.images_path)
        assert isinstance(self.annotations_path, str)
        assert isinstance(self.filename_tpl, str)
        assert isinstance(self.filename_reg, str)

        if self.annotations_path:
            assert os.path.isdir(self.annotations_path)

    def __call__(self, **kwargs):
        """ functor call """
        return self.process_files(**kwargs)

    @patch('gcrlm.managers.logger.info', MagicMock())
    @patch('gcrlm.managers.tqdm', notqdm)
    def process_files(self):
        """
        Analyzes all the NDPI and NDPA files, calculates some related information and prints
        the results
        """
        ndpis_list = list(filter(lambda x: x.endswith('.ndpi'), os.listdir(self.images_path)))
        pattern = re.compile(self.filename_reg)

        without_rois = 0
        ann_rois_data = []
        label_data = defaultdict(lambda: dict(counter=0, shapes=[]))
        raw_label_data = defaultdict(lambda: dict(counter=0, shapes=[]))

        print("Analizing NDPI files...")

        valid_labels = list(copy.deepcopy(Label.file_labels))
        _ = valid_labels.pop(valid_labels.index(Label.background.file_label))
        valid_labels = set(valid_labels)
        ndpis_all_labels = [["Image", "Labels"]]
        label_imgs = defaultdict(set)

        for ndpi_file in tqdm(ndpis_list):
            try:
                ndpi_index = pattern.fullmatch(ndpi_file).groupdict()['index']
            except AttributeError as err:
                logger.error(err)
                continue
            else:
                ndpi_index = int(ndpi_index)

            slide = CRLM(ndpi_index, fileroot=self.images_path,
                         annotation_root=self.annotations_path, filename_tpl=self.filename_tpl)
            total_rois = len(slide.all_rois)

            if total_rois == 0:
                without_rois += 1
            else:
                ann_rois_data.append((ndpi_index, total_rois, slide.cluster_annotations_by_roi()))

            slide_labels = set()

            for ann_index in range(slide.root_len):
                ann_name = slide.get_aname(ann_index)

                try:
                    ann_bbox = BoundingBox(*slide.get_annotation_bbox(ann_index))
                except Exception as err:
                    # print(ndpi_file, ann_index, ann_name, err)
                    pass
                else:
                    # raw annotations names. Some of them
                    # have the same letter as the cleaned labels, so it could
                    # confusing
                    raw_label_data[ann_name]['counter'] += 1
                    raw_label_data[ann_name]['shapes'].append(ann_bbox.shape)
                    label = slide.get_cleaned_label(ann_name)

                    if label:
                        label_data[label]['counter'] += 1
                        label_data[label]['shapes'].append(ann_bbox.shape)
                        slide_labels.add(label)
                        label_imgs[label].add(ndpi_index)

            if valid_labels.intersection(slide_labels) == valid_labels:
                ndpis_all_labels.append([self.filename_tpl.format(ndpi_index), slide_labels])

        print("\n NDPIs with all annotations")
        print(tabulate(ndpis_all_labels, headers="firstrow", showindex=True, tablefmt="orgtbl"))

        for k, v in label_imgs.items():
            label_imgs[k] = list(v)
            label_imgs[k].sort(reverse=False)

        label_imgs_counter = [(k, len(v), v) for k, v in label_imgs.items()]
        label_imgs_counter.sort(key=lambda x: x[1], reverse=False)

        table_data = [["Label", "Num NDPIs", "NDPI ids"]]

        for label, length, ids in label_imgs_counter:
            id_chunks = []
            for i in range(0, len(ids), 10):
                id_chunks.append(', '.join(str(i) for i in ids[i:i+10]))
            multiline_ids = '\n'.join(id_chunks)
            table_data.append([label, length, multiline_ids])

        print("\nNDPIs grouped by label")
        print(tabulate(table_data, headers="firstrow", showindex=True, tablefmt="orgtbl"))

        print("\n Total number of NDPI files without any ROI: {}".format(without_rois))
        print("\n Files with ROIs:")

        total_ann_inside_rois = 0
        table_data = [["Image", "ROIs", "Annotations inside ROIS"]]

        for img_data in ann_rois_data:
            ann_inside_rois = sum([len(roiann.ann_indexes) for roiann in img_data[2]])
            table_data.append([self.filename_tpl.format(img_data[0]), img_data[1], ann_inside_rois])
            total_ann_inside_rois += ann_inside_rois

        print(tabulate(table_data, headers="firstrow", showindex=True, tablefmt="orgtbl"))

        print(
            "\n Total annotations inside ROIs excluding labels {}: {}"
            .format(CRLM.excluded_ann, total_ann_inside_rois)
        )
        print("\n Quantified annotations:")

        total_ann = 0
        table_data = [["Label", "Annotations", "Mean bbox width & height"]]

        for label, ann_dict in label_data.items():
            table_data.append([
                label,
                ann_dict['counter'],
                '{:010.2f}, {:010.2f}'.format(*np.array(ann_dict['shapes']).mean(axis=0))
            ])
            if label not in CRLM.excluded_ann:
                total_ann += ann_dict['counter']

        print(tabulate(table_data, headers="firstrow", showindex=True, tablefmt="orgtbl"))
        print("\n Total annotations excluding labels {}: {}".format(CRLM.excluded_ann, total_ann))
        print("\n Quantified raw annotations:")

        total_ann = 0
        table_data = [["Raw Label", "Annotations", "Mean bbox width & height"]]

        for label, ann_dict in raw_label_data.items():
            table_data.append([
                label,
                ann_dict['counter'],
                '{:010.2f}, {:010.2f}'.format(*np.array(ann_dict['shapes']).mean(axis=0))
            ])
            if label not in CRLM.excluded_ann:
                total_ann += ann_dict['counter']

        print(tabulate(table_data, headers="firstrow", showindex=True, tablefmt="orgtbl"))
        print("\n Total annotations excluding labels {}: {}".format(CRLM.excluded_ann, total_ann))


class CRLMCropsAnalyzer:
    """
    Quantifies the crop per label, prints the results and returns a defaultdict with labels as keys
    and number of crops as values

    Usage:
        ann_counter = CRLMCropsAnalyzer(
            crops_masks_path='/media/giussepi/2_0_TB_Hard_Disk/improved_CRLM_dataset/annotations_masks_allwithrois/'
        )()
    """

    def __init__(self, **kwargs):
        """
        Initialized the object instance

        Kwargs:
            crops_masks_path   <str>: path to the folder for containing the NDPI files
            filename_reg       <str>: regular expression to get the index id from the crop filename.
                                      Default r'(?P<label>[A-Z])_f(?P<file>\d+)_r(?P<roi>\-?\d+)_a(?P<annotation>\d+)_c(?P<counter>\d+)_?(?P<transforms>[A-Z\-]*).ann.tiff'
            filename_extension <str>: crop extension. Default '.ann.tiff'
            verbose           <bool>: Whether or not print the results. Default True
        """
        self.crops_masks_path = kwargs.get('crops_masks_path')
        self.filename_reg = kwargs.get(
            'filename_reg',
            r'(?P<label>[A-Z])_f(?P<file>\d+)_r(?P<roi>\-?\d+)_a(?P<annotation>\d+)_c(?P<counter>\d+)_?(?P<transforms>[A-Z\-]*).ann.tiff',
        )
        self.filename_extension = kwargs.get('filename_extension', '.ann.tiff')
        self.verbose = kwargs.get('verbose', True)

        assert isinstance(self.crops_masks_path, str), type(self.crops_masks_path)
        assert os.path.isdir(self.crops_masks_path), self.crops_masks_path
        assert isinstance(self.filename_reg, str), type(self.filename_reg)
        assert isinstance(self.filename_extension, str), type(self.filename_extension)
        assert isinstance(self.verbose, bool), type(self.verbose)

    def __call__(self, **kwargs):
        """ functor call """
        return self.process_files(**kwargs)

    def process_files(self):
        """
        Quantifies the crops per label and returns a defaultdict with labels as keys
        and number of crops as values

        Returns:
            ann_counter <defaultdict>
        """
        crops = list(filter(
            lambda x: x.endswith(self.filename_extension), os.listdir(self.crops_masks_path)))
        pattern = re.compile(self.filename_reg)

        ann_counter = defaultdict(lambda: 0)

        if self.verbose:
            print("Quantifying crops per label...")

        for file_ in tqdm(crops):
            try:
                ann_label = pattern.fullmatch(file_).groupdict()['label']
            except AttributeError as err:
                logger.error(err)
                continue

            ann_counter[ann_label] += 1

        if self.verbose:
            print("\n Number of crops created per label:")

        table_data = [["Index", "Label", "Crops"]]

        for label, counter in ann_counter.items():
            table_data.append((label, counter))

        if self.verbose:
            print(tabulate(table_data, headers="firstrow", showindex=True, tablefmt="orgtbl"))

        return ann_counter


class CRLMAugmentationProcessor:
    """
    Uses data augmentation to multiply the number of images/masks per label/class. The multipliers
    must be integers >= 0.

    If the instance is initialized with mask_name_reg != '', then the same transformations
    applied to images will be applied to their corresponding masks. This will happen
    only if the images and masks have the same name with different extension, e.g.:
    <image_name>.ann.tiff and <image_name>.mask.png. The extension must be provided with
    the argument image_extension and mask_extension.

    Usage:
        from Data.dataset_processors.models import AugmentDict

        # using custom class multipliers
        $ the following code will only duplicate the number of crops and masks from the classs foreign_body
        CRLMAugmentationProcessor(
            images_masks_path='/media/giussepi/2_0_TB_Hard_Disk/improved_CRLM_dataset/annotations_masks/',
            augment_dict=AugmentDict(
                hepatocyte=1,
                necrosis=1,
                fibrosis=1,
                tumour=1,
                inflamation=1,
                mucin=1,
                blood=1,
                foreign_body=2,
                macrophages=1,
                bile_duct=1
            )
        )()

        # calculating class multipliers automatically
        CRLMAugmentationProcessor(
            images_masks_path='/media/giussepi/2_0_TB_Hard_Disk/improved_CRLM_dataset/annotations_masks/',
        )()

        # multiplying the whole dataset times 2
        CRLMAugmentationProcessor(
            images_masks_path='/media/giussepi/2_0_TB_Hard_Disk/improved_CRLM_dataset/annotations_masks/',
            class_multiplier=2,
        )()

    """

    def __init__(self, **kwargs):
        """
        Initializes the object instance

        Kwargs:
            images_masks_path <str>: path to the folder for containing images and masks
            augment_dict <AugmentDict>: class multiplier dictionary. The keys are the
                                     classes and the values are the multipliers
                                     (how many times you want to multiply the images/masks
                                     from an especific class). If provided it supersedes
                                     any class_multiplier value.
                                     Default None
            class_multiplier  <int>: if > 0, it is used as the multipler for all the classes;
                                     else, the class multipliers are calculated automatically
                                     to have number of crops per class close to the class with
                                     the biggest number of crops.
                                     Default 0
            min_transforms    <int>: minimum number of transforms to apply. Default 1
            original         <bool>: whether or not include the original image. Default True
            image_reg         <str>: regular expression to get the index id from the filename.
                                     Default r'(?P<label>[A-Z])_f(?P<file>\d+)_r(?P<roi>\-?\d+)_a(?P<annotation>\d+)_c(?P<counter>\d+).ann.tiff'
            image_name_reg    <str>: regular expression to extract the name from the image name.
                                     Default r'(?P<filename>.+).ann.tiff'
            mask_name_reg     <str>: regular expression to extract the name from the mask name.
                                     Set it to '' if you don't want to work with masks.
                                     Default r'(?P<filename>.+).mask.png'
            image_extension   <str>: image extension. Default '.ann.tiff'
            mask_extension    <str>: mask extension. Default '.mask.png'
            saving_path       <str>: path to folder to save the augmented dataset.
                                     Default 'augmented_dataset'
            img_format        <str>: Saving image extension. Default 'tiff'
            mask_format       <str>: Saving mask format. Default 'png'
        """
        self.images_masks_path = kwargs.get('images_masks_path')
        self.augment_dict = kwargs.get('augment_dict', None)
        self.class_multiplier = kwargs.get('class_multiplier', 0)
        self.min_transforms = kwargs.get('min_transforms', 1)
        self.original = kwargs.get('original', True)
        self.image_reg = kwargs.get(
            'image_reg',
            r'(?P<label>[A-Z])_f(?P<file>\d+)_r(?P<roi>\-?\d+)_a(?P<annotation>\d+)_c(?P<counter>\d+).ann.tiff'
        )
        self.image_name_reg = kwargs.get('image_name_reg', r'(?P<filename>.+).ann.tiff')
        self.mask_name_reg = kwargs.get('mask_name_reg', r'(?P<filename>.+).mask.png')
        self.image_extension = kwargs.get('image_extension', '.ann.tiff')
        self.mask_extension = kwargs.get('mask_extension', '.mask.png')
        self.saving_path = kwargs.get('saving_path', 'augmented_dataset')
        self.img_format = kwargs.get('img_format', 'tiff')
        self.mask_format = kwargs.get('mask_format', 'png')

        assert isinstance(self.images_masks_path, str), type(self.images_masks_path)
        assert os.path.isdir(self.images_masks_path), self.images_masks_path

        if self.augment_dict is not None:
            assert isinstance(self.augment_dict, AugmentDict), type(self.augment_dict)
            self.augment_dict = self.augment_dict()
        else:
            if self.class_multiplier > 0:
                self.augment_dict = AugmentDict.get_initialized_instance(self.class_multiplier)()
            else:
                ann_counter = CRLMCropsAnalyzer(
                    crops_masks_path=self.images_masks_path,
                    filename_reg=self.image_reg,
                    filename_extension=self.image_extension,
                    verbose=False
                )()
                max_crops_per_class = max(ann_counter.values())

                # calculating the right multiplier per class to obtain, by data augmentation,
                # number of crops close to max_crops_per_class
                levelling_dict = dict()

                for label, num_crops in ann_counter.items():
                    ratio = max_crops_per_class/num_crops
                    levelling_dict[label] = ceil(ratio) if ratio - floor(ratio) > 0.5 else floor(ratio)

                # Creating the right augment_dict using an AugmentDict instance
                # initialized with ones as base
                self.augment_dict = AugmentDict.get_initialized_instance(1)()
                self.augment_dict.update(levelling_dict)

        assert isinstance(self.min_transforms, int), type(self.min_transforms)
        assert self.min_transforms >= 1, self.min_transforms
        assert isinstance(self.original, bool), type(self.original)
        assert isinstance(self.image_reg, str), type(self.image_reg)
        assert isinstance(self.image_name_reg, str), type(self.image_name_reg)
        assert isinstance(self.mask_name_reg, str), type(self.mask_name_reg)
        assert isinstance(self.image_extension, str), type(self.image_extension)
        assert isinstance(self.mask_extension, str), type(self.mask_extension)
        assert isinstance(self.saving_path, str), type(self.saving_path)
        assert isinstance(self.img_format, str), type(self.img_format)
        assert isinstance(self.mask_format, str), type(self.mask_format)

    def __call__(self):
        """ functor call """
        return self.process_images()

    def read_image(self, img_name):
        """
        Reads the image and returns a RGBA np.array

        Returns:
            RGBA image <np.array>
        """
        assert isinstance(img_name, str), type(img_name)

        img_path = os.path.join(self.images_masks_path, img_name)

        assert os.path.isfile(img_path), img_path

        img = cv2.imread(img_path)

        return cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)

    @patch('gutils.image_augmentation.logger.info', MagicMock())
    @patch('gutils.image_augmentation.tqdm', notqdm)
    def process_images(self):
        """
        Iterates over the images to create and save the augmentations. If the instance
        was initialized with mask_name_reg != '', then the same augmentations will be applied
        to their corresponding masks.
        """
        if not os.path.isdir(self.saving_path):
            os.makedirs(self.saving_path)

        images = list(
            filter(
                lambda x: x.endswith(self.image_extension),
                os.listdir(self.images_masks_path)
            )
        )
        img_pattern = re.compile(self.image_reg)
        augmentation = ImageAugmentationProcessor()

        print("Performing data augmentation over the dataset")

        for image_name in tqdm(images):
            try:
                label = img_pattern.fullmatch(image_name).groupdict()['label']
            except AttributeError as err:
                logger.error(err)
                continue

            try:
                multiplier = self.augment_dict[label]
            except KeyError as err:
                logger.error(err)
                continue

            image = self.read_image(image_name)

            if self.mask_name_reg != '':
                mask_name = image_name.replace(self.image_extension, self.mask_extension)
                mask = self.read_image(mask_name)
            else:
                mask_name = ''
                mask = None

            augmentation(
                n=multiplier, min_transforms=self.min_transforms, original=self.original,
                image=image, image_name=image_name, image_reg=self.image_name_reg,
                mask=mask, mask_name=mask_name, mask_reg=self.mask_name_reg,
                saving_path=self.saving_path, img_format=self.img_format,
                mask_format=self.mask_format
            )


class CRLMSplitDataset:
    """
    Randomly splits the CRLM dataset into train, validation and test subdatasets.

    Note: It doesn't care if the splits contains all the classes or not

    Usage:
        CRLMSplitDataset(
            dataset_path='/media/giussepi/2_0_TB_Hard_Disk/improved_CRLM_dataset/Annotated - Training set',
            val_size=.1, test_size=.2,
        )()
    """

    def __init__(self, **kwargs):
        """
        Initializes the object instance

        Kwargs:
            dataset_path  <str>: path to the folder for containing the NDPI and NDPA files
            saving_path   <str>: path to folder to save the splits.
                                 Default dataset_path
            ndpi_reg      <str>: regular expression to get the index id from the ndpi filename.
                                 Default r'CRLM 0*(?P<index>\d+).ndpi'.
            val_size    <float>: validation dataset size in range [0, 1]. Default .1
            test_size   <float>: test dataset size in range [0, 1]. Default .1
            move_files   <bool>: whether or not move files or create copies. Default True
        """
        self.dataset_path = kwargs.get('dataset_path')
        self.saving_path = kwargs.get('saving_path', self.dataset_path)
        self.ndpi_reg = kwargs.get('ndpi_reg', r'CRLM 0*(?P<index>\d+).ndpi')
        self.val_size = kwargs.get('val_size', .1)
        self.test_size = kwargs.get('test_size', .1)
        self.move_files = kwargs.get('move_files', True)

        assert os.path.isdir(self.dataset_path), self.dataset_path
        assert isinstance(self.ndpi_reg, str), self.ndpi_reg
        assert self.ndpi_reg != ''
        assert 0 < self.val_size < 1, self.val_size
        assert 0 < self.test_size < 1, self.test_size
        assert isinstance(self.move_files, bool), type(self.move_files)

    def __call__(self):
        """ functor call """
        return self.process()

    def move_to_folder(self, split_name, data):
        """
        Moves or copies the images and masks to the correct directory

        Args:
            split_name <str>: sub-dataset name
            data  <np.ndarray>: sub-dataset numpy array
        """
        assert split_name in DB.SUB_DATASETS, split_name
        assert isinstance(data, np.ndarray), type(data)

        operation = shutil.move if self.move_files else shutil.copy
        folder_path = os.path.join(self.saving_path, split_name)

        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)

        for file_ in data:
            operation(
                os.path.join(self.dataset_path, file_),
                os.path.join(folder_path, file_)
            )
            operation(
                os.path.join(self.dataset_path, f'{file_}.ndpa'),
                os.path.join(folder_path, f'{file_}.ndpa')
            )

    def process(self):
        """ Splits the dataset into train, validation and test subdatasets """
        ndpi_files = np.array(list(filter(lambda x: x.endswith('.ndpi'), os.listdir(self.dataset_path))))

        x_train, x_val, x_test, _, _, _ = TrainValTestSplit(
            ndpi_files, np.zeros_like(ndpi_files), self.val_size, self.test_size)()

        logger.info("Creating dataset splits...")

        for split, files in tqdm(list(zip(DB.SUB_DATASETS, (x_train, x_test, x_val)))):
            self.move_to_folder(split, files)


class CRLMSmartSplitDataset(CRLMSplitDataset):
    """
    Randomly splits the CRLM dataset into train, validation and test subdatasets.

    Note: It makes sure the classes are decently distributed through the subdatasets

    Usage:
        CRLMSmartSplitDataset(
            dataset_path='/media/giussepi/2_0_TB_Hard_Disk/improved_CRLM_dataset/Annotated - Training set',
            val_size=.1, test_size=.2,
        )()
    """

    def __init__(self, **kwargs):
        """
        Initialized the object instance

        Args:
            dataset_path  <str>: path to the folder for containing the NDPI and NDPA files
            saving_path   <str>: path to folder to save the splits.
                                 Default dataset_path
            ndpi_reg      <str>: regular expression to get the index id from the ndpi filename.
                                 Default r'CRLM 0*(?P<index>\d+).ndpi'.
            val_size    <float>: validation dataset size in range [0, 1]. Default .1
            test_size   <float>: test dataset size in range [0, 1]. Default .1
            move_files   <bool>: whether or not move files or create copies. Default True
            filename_tpl     <str>: filename template to create the file name using the index
                                 e.g.: for CRLM_001.npi you should use CRLM_{:03d}.ndpi
                                 Default 'CRLM {}.ndpi'
        """
        super().__init__(**kwargs)
        self.filename_tpl = kwargs.get('filename_tpl', 'CRLM {}.ndpi')

        assert isinstance(self.filename_tpl, str)

    @staticmethod
    def add_to_subset(
        subset, subset_counter, subset_size, available_imgs, imgs_counter,
        img_reps, active_labels, label_imgs
    ):
        """
        Tries to add a seleted image into the subset, if it succeed, then returns 1; otherwhise,
        returns 0

        Args:
            subset            <list>: subset to add the elements
            subset_counter     <int>: current number of elements in the subset
            subset_size        <int>: maximum number of elements allowed in the subset
            available_imgs    <list>: candidate images that could be insterted in the subset
            imgs_counter      <list>: total number of repetitions of each available image through
                                 all the labels
            img_reps   <defaultdict>: dictionary with image ids as keys and their total
                                 number of repetitions through all the labels as values.
                                 e.g. {img1: 10, img2: 5, ...}
            active_labels     <list>: current active labels. i.e. list of labels we are currently
                                 working with
            label_imgs <defaultdict>: dictionary with labels as keys and lists of images ids
                                 as values. e.g. {label1:[img1, img2, ...], label2:[img3], ...}

        Returns:
            <int>
        """
        assert isinstance(subset, list), type(subset)
        assert isinstance(subset_counter, int), type(subset_counter)
        assert isinstance(subset_size, int), type(subset_size)
        assert isinstance(available_imgs, list), type(available_imgs)
        assert isinstance(imgs_counter, list), type(imgs_counter)
        assert isinstance(img_reps, defaultdict), type(img_reps)
        assert isinstance(active_labels, list), type(active_labels)
        assert isinstance(label_imgs, defaultdict), type(label_imgs)

        if subset_counter >= subset_size:
            return 0

        if len(available_imgs) == 0:
            return 0

        selected_img = available_imgs[np.argmin(imgs_counter)]
        subset.append(selected_img)
        # removing selected image from general img counter
        _ = img_reps.pop(selected_img)

        # removing selected_image from all the label_imgs lists
        for label in list(label_imgs.keys()):
            try:
                selected_img_idx = label_imgs[label].index(selected_img)
                _ = label_imgs[label].pop(selected_img_idx)
            except (ValueError, KeyError) as err:
                # selected img not present in current label list
                pass
            else:
                if len(label_imgs[label]) == 0:
                    label_imgs.pop(label)
                    try:
                        active_labels.pop(active_labels.index(label))
                    except (ValueError, KeyError) as err:
                        # label already eliminated
                        pass
        return 1

    def process(self):
        """
        Splits the dataset into train, validation and test subdatasets

        All the algorithm is based in two basic rules:

        1. Always start selecting a label with the less number of associated image ids
        2. Always select the image id with the less number of repetitions

        These rules make sure that in each subdataset the less prepresented labels
        will be allocated with images first; thus, we increase the probabilities of
        having a decent (likely not perfect) distribution of the labels per subdataset.
        I.E. the subdataset will be as stratified as possible. To some extend, the outcomes
        are similar to results computed with train_test_split from sklearn using the
        stratify option
        """

        ndpi_list = list(filter(lambda x: x.endswith('.ndpi'), os.listdir(self.dataset_path)))
        pattern = re.compile(self.ndpi_reg)

        label_imgs = defaultdict(list)
        img_reps = defaultdict(lambda: 0)

        logger.info("Analizing dataset before splitting it")
        for ndpi_file in tqdm(ndpi_list):
            try:
                ndpi_index = pattern.fullmatch(ndpi_file).groupdict()['index']
            except AttributeError as err:
                logger.error(err)
                continue
            else:
                ndpi_index = int(ndpi_index)

            slide = CRLM(ndpi_index, fileroot=self.dataset_path,
                         annotation_root=self.dataset_path, filename_tpl=self.filename_tpl)

            for ann_index in range(slide.root_len):
                ann_name = slide.get_aname(ann_index)

                try:
                    ann_bbox = BoundingBox(*slide.get_annotation_bbox(ann_index))
                except Exception as err:
                    # print(ndpi_file, ann_index, ann_name, err)
                    pass
                else:
                    label = slide.get_cleaned_label(ann_name)

                    if label and (ndpi_index not in label_imgs[label]):
                        label_imgs[label].append(ndpi_index)
                        img_reps[ndpi_index] += 1

        db_size = len(ndpi_list)
        test_size = floor(self.test_size * db_size)
        val_size = floor(self.val_size * db_size)
        train_size = db_size - test_size - val_size
        train_counter = val_counter = test_counter = 0
        train_subset = []
        val_subset = []
        test_subset = []

        logger.info("Splitting dataset")

        while True:
            active_labels = list(label_imgs.keys())
            # logger.info(sum([len(v) for v in label_imgs.values()]))

            # if all the labels have been consumed, then stop the loop
            if not active_labels:
                break

            # selecting label
            while active_labels:
                label_img_counter = [len(label_imgs[label]) for label in active_labels]
                selected_label = active_labels[np.argmin(label_img_counter)]
                available_imgs = label_imgs[selected_label]

                # selecting images and adding them to the subdatsets
                imgs_counter = [img_reps[i] for i in available_imgs]

                if self.add_to_subset(
                        train_subset, train_counter, train_size, available_imgs, imgs_counter, img_reps,
                        active_labels, label_imgs
                ):
                    train_counter += 1
                    imgs_counter = [img_reps[i] for i in available_imgs]

                if self.add_to_subset(
                    val_subset, val_counter, val_size, available_imgs, imgs_counter, img_reps,
                    active_labels, label_imgs
                ):
                    val_counter += 1
                    imgs_counter = [img_reps[i] for i in available_imgs]

                if self.add_to_subset(
                    test_subset, test_counter, test_size, available_imgs, imgs_counter, img_reps,
                    active_labels, label_imgs
                ):
                    test_counter += 1
                    imgs_counter = [img_reps[i] for i in available_imgs]

                # if the selected label hasn't been removed yet, then remove it
                if selected_label in active_labels:
                    _ = active_labels.pop(np.argmin(label_img_counter))

        logger.info("Creating dataset splits...")

        for split, files in tqdm(list(zip(
                (DB.TRAIN, DB.VALIDATION, DB.TEST),
                (
                    np.array([self.filename_tpl.format(i) for i in train_subset]),
                    np.array([self.filename_tpl.format(i) for i in val_subset]),
                    np.array([self.filename_tpl.format(i) for i in test_subset]),
                )
        ))):
            self.move_to_folder(split, files)


class CRLMRandCropSplit(CRLMSplitDataset):
    """
    Randomly splits a dataset of crops into train, val andn test subdatasets

    Usage:
        CRLMRandCropSplit(dataset_path='<path_to_my_crops_dataset>', val_size=.1, test_size=.1)()
    """

    def __init__(self, **kwargs):
        """
        Initializes the object instance

        Kwargs:
            dataset_path    <str>: path to the folder for containing the NDPI and NDPA files
            saving_path     <str>: path to folder to save the splits.
                                   Default dataset_path
            image_reg       <str>: regular expression to get the index id from the crop filename.
                                   Default r'(?P<label>[A-Z])_f(?P<file>\d+)_r(?P<roi>\-?\d+)_a(?P<annotation>\d+)_c(?P<counter>\d+)_?(?P<transforms>[A-Z\-]*).ann.tiff'
            image_extension <str>: crop extension. Default '.ann.tiff'
            mask_extension  <str>: mask extension. Default '.mask.png'
            val_size      <float>: validation dataset size in range [0, 1]. Default .1
            test_size     <float>: test dataset size in range [0, 1]. Default .1
            move_files     <bool>: whether or not move files or create copies. Default True
        """
        super().__init__(**kwargs)
        self.image_reg = kwargs.get(
            'image_reg', r'(?P<label>[A-Z])_f(?P<file>\d+)_r(?P<roi>\-?\d+)_a(?P<annotation>\d+)_c(?P<counter>\d+)_?(?P<transforms>[A-Z\-]*).ann.tiff')
        self.image_extension = kwargs.get('image_extension', '.ann.tiff')
        self.mask_extension = kwargs.get('mask_extension', '.mask.png')

        assert isinstance(self.image_reg, str), type()
        assert isinstance(self.image_extension, str), type()
        assert isinstance(self.mask_extension, str), type()

    def move_to_folder(self, split_name, data):
        """
        Moves or copies the images and masks to the correct directory

        Args:
            split_name <str>: sub-dataset name
            data  <np.ndarray>: sub-dataset numpy array
        """
        assert split_name in DB.SUB_DATASETS, split_name
        assert isinstance(data, np.ndarray), type(data)

        operation = shutil.move if self.move_files else shutil.copy
        folder_path = os.path.join(self.saving_path, split_name)

        if not os.path.isdir(folder_path):
            os.makedirs(folder_path)

        for file_ in data:
            operation(
                os.path.join(self.dataset_path, file_),
                os.path.join(folder_path, file_)
            )
            mask = file_.replace(self.image_extension, self.mask_extension)
            operation(
                os.path.join(self.dataset_path, mask),
                os.path.join(folder_path, mask)
            )

    def process(self):
        """ Splits the dataset into train, validation and test subdatasets """
        images = np.array(list(filter(
            lambda x: x.endswith(self.image_extension), os.listdir(self.dataset_path))))

        img_pattern = re.compile(self.image_reg)

        labels = []
        for image_name in images:
            try:
                label = img_pattern.fullmatch(image_name).groupdict()['label']
            except AttributeError as err:
                logger.error(err)
                continue
            else:
                labels.append(label)

        labels = np.array(labels)

        x_train, x_val, x_test, _, _, _ = TrainValTestSplit(
            images, labels, self.val_size, self.test_size, stratify=labels)()

        logger.info("Creating dataset splits...")

        for split, files in tqdm(list(zip(DB.SUB_DATASETS, (x_train, x_test, x_val)))):
            self.move_to_folder(split, files)


class CRLMCropRandSample:
    """
    Samples a dataset using a user-defined number of crops/images & masks per class.

    Usage:
        CRLMCropRandSample(samples_per_label=1000, dataset_path='<my_dataset_path>')()
    """

    def __init__(self, **kwargs):
        """
        Initializes the object instance

        Kwargs:
            samples_per_label <int>: number of crops & masks you want per label/class without
                                     considering the masks to be moved. I.E. if you want to
                                     move 10 crops/images and 10 masks per label, then
                                     just set samples_per_label to 10. The mask will be moved
                                     automatically too.
            dataset_path      <str>: path to the folder for containing images and masks
            filename_reg      <str>: regular expression to get the label from the crop filename.
                                     Default r'(?P<label>[A-Z])_f(?P<file>\d+)_r(?P<roi>\-?\d+)_a(?P<annotation>\d+)_c(?P<counter>\d+)_?(?P<transforms>[A-Z\-]*).ann.tiff'
            image_extension   <str>: image extension. Default '.ann.tiff'
            mask_extension    <str>: mask extension. Default '.mask.png'
            saving_path       <str>: path to folder to save the seleted samples.
                                     Default f'{dataset_path}_sample'
            move_files       <bool>: whether or not move files or create copies. Default True
        """
        self.samples_per_label = kwargs.get('samples_per_label')
        self.dataset_path = kwargs.get('dataset_path')
        self.filename_reg = kwargs.get(
            'filename_reg',
            r'(?P<label>[A-Z])_f(?P<file>\d+)_r(?P<roi>\-?\d+)_a(?P<annotation>\d+)_c(?P<counter>\d+)_?(?P<transforms>[A-Z\-]*).ann.tiff',
        )
        self.image_extension = kwargs.get('image_extension', '.ann.tiff')
        self.mask_extension = kwargs.get('mask_extension', '.mask.png')
        self.saving_path = kwargs.get('saving_path', '')
        self.move_files = kwargs.get('move_files', True)

        assert isinstance(self.samples_per_label, int), type(self.samples_per_label)
        assert self.samples_per_label > 0
        assert isinstance(self.dataset_path, str), type(self.dataset_path)
        assert os.path.isdir(self.dataset_path), self.dataset_path

        if self.dataset_path.endswith(os.path.sep):
            self.dataset_path = os.path.dirname(self.dataset_path)

        assert isinstance(self.filename_reg, str), type(self.filename_reg)
        assert isinstance(self.image_extension, str), type(self.image_extension)
        assert isinstance(self.mask_extension, str), type(self.mask_extension)
        assert isinstance(self.saving_path, str), type(self.saving_path)

        if not self.saving_path:
            self.saving_path = f'{self.dataset_path}_sample'

        if os.path.exists(self.saving_path):
            raise SampleFolderAlreadyExists

        assert isinstance(self.move_files, bool), type(self.move_files)

    def __call__(self):
        """functor call"""
        self.process()

    def move_to_folder(self, data):
        """
        Moves or copies the images and masks to the correct directory

        Args:
            data <list>: list of images path to move
        """
        assert isinstance(data, list), type(data)

        operation = shutil.move if self.move_files else shutil.copy

        if not os.path.isdir(self.saving_path):
            os.makedirs(self.saving_path)

        for file_path in data:
            operation(
                os.path.join(self.dataset_path, file_path),
                os.path.join(self.saving_path, file_path)
            )
            mask = file_path.replace(self.image_extension, self.mask_extension)
            operation(
                os.path.join(self.dataset_path, mask),
                os.path.join(self.saving_path, mask)
            )

    def process(self):
        """
        Randomly moves a defined number of files ('samples_per_label') to the provided 'saving_path'.
        If 'samples_per_label' >= number of crops belonging to a label, then all of them are processed.
        """
        pattern = re.compile(self.filename_reg)
        images = tuple(filter(lambda x: x.endswith(self.image_extension), os.listdir(self.dataset_path)))
        cluster_labels = defaultdict(list)

        logger.info("Clustering crops by label")

        for img in tqdm(images):
            try:
                ann_label = pattern.fullmatch(img).groupdict()['label']
            except AttributeError as err:
                logger.error(err)
                continue

            cluster_labels[ann_label].append(img)

        verb = "Moving" if self.move_files else "Copying"

        logger.info(f"{verb} crops/images & masks")

        for label in tqdm(cluster_labels):
            num_images = len(cluster_labels[label])

            if self.samples_per_label >= num_images:
                self.move_to_folder(cluster_labels[label])
            else:
                self.move_to_folder(random.sample(cluster_labels[label], self.samples_per_label))


class CRLMSplitProcessAugment:
    """
    Splits the CRLM dataset into train, val and test. Creates the annotations crops and their
    respective masks at train_processed, val_processed and test_processed folders. Lastly,
    if an augment_dict is provided, then performs the data augmentation only on the training
    dataset and saves the results at the train_processed_augmented folder.

    The dataset slipt can be one randomly using splitter=CRLMSplitDataset, or in a
    stratified way using splitter=CRLMSmartSplitDataset.

    If rm_non_augmented is set to True, the train_processed folder will be removed after
    augmenting the data.

    Kwargs:
        # for CRLMSplitDataset ################################################
        splitter    <class>: Class to perform the datast split. The implemented splitter for
                             the CRLM dataset are CRLMSplitDataset and CRLMSmartSplitDataset.
                             Default CRLMSmartSplitDataset
        dataset_path  <str>: path to the folder for containing the NDPI and NDPA files
        ndpi_reg      <str>: regular expression to get the index id from the ndpi filename.
                             Default r'CRLM 0*(?P<index>\d+).ndpi'.
        val_size    <float>: validation dataset size in range [0, 1]. Default .1
        test_size   <float>: test dataset size in range [0, 1]. Default .1
        filename_tpl  <str>: filename template to create the file name using the index
                             e.g.: for CRLM_001.npi you should use CRLM_{:03d}.ndpi
                             Default 'CRLM {}.ndpi'

        # for CRLMDatasetProcessor ############################################
        level            <int>: magnification level (dim/2**level)
        filename_tpl     <str>: Same as above
        filename_reg     <str>: regular expression to get the index id from the filename
                                Default r'CRLM 0*(?P<index>\d+).ndpi'
        patch_size              <int>: size of squared patch
        min_mask_area         <float>: min mask area in a annotation patches. Avoids creating
                                       patches with tiny or no mask on them.
        processing_type <CRLMProcessingTypes>: Type of processing to be performend over CRLM
        central_area_offset   <float>: percentage to ignore before considering the central area
                        e.g.: if dim0=100, dim1=200 and central_area_offset=.2, the central area
                        will be image[20:80, 40:160]
        background_threshold  <float>: value between 0 and 1 used to identify background pixels
                                       (the image is turned into gray scale before the comparison)
        img_format              <str>: Saving image extension. Default 'tiff'
        mask_format             <str>: Saving mask extension. Default 'png'
        multiple_ann           <bool>: Whether or not plot multiple annotations with the same label
                                       per crop. Default False
        multiple_classes       <bool>: Whether or not plot annotations from multiple classes per crop.
                                       Requires multiple_ann = True. Default False

        # for CRLMAugmentationProcessor #######################################
        images_masks_path <str>: path to the folder for containing images and masks
        augment_dict <AugmentDict>: class multiplier dictionary. The keys are the
                                 classes and the values are the multipliers
                                 (how many times you want to multiply the images/masks
                                 from an especific class). If provided it supersedes
                                 any class_multiplier value.
                                 Default None
        class_multiplier  <int>: if > 0, it is used as the multipler for all the classes;
                                 else, the class multipliers are calculated automatically
                                 to have number of crops per class close to the class with
                                 the biggest number of crops.
                                 Default 0
        min_transforms    <int>: minimum number of transforms to apply. Default 1
        original         <bool>: whether or not include the original image. Default True
        image_reg         <str>: regular expression to get the index id from the filename.
                                 Default r'(?P<label>[A-Z])_f(?P<file>\d+)_r(?P<roi>\-?\d+)_a(?P<annotation>\d+)_c(?P<counter>\d+).ann.tiff'
        image_name_reg    <str>: regular expression to extract the name from the image name.
                                 Default r'(?P<filename>.+).ann.tiff'
        mask_name_reg     <str>: regular expression to extract the name from the mask name.
                                 Set it to '' if you don't want to work with masks.
                                 Default r'(?P<filename>.+).mask.png'
        image_extension   <str>: image extension. Default '.ann.tiff'
        mask_extension    <str>: mask extension. Default '.mask.png'
        img_format        <str>: Saving image extension. Default 'tiff'
        mask_format       <str>: Saving mask format. Default 'png'

        rm_non_augmented <bool>: whether or not remove the train_processed dataset after applying
                                 the augmentation. Default False

    Usage:
        from Data.dataset_processors.models import AugmentDict

        # using custom class multipliers
        CRLMSplitProcessAugment(
            dataset_path='/media/giussepi/2_0_TB_Hard_Disk/improved_CRLM_dataset/Annotated - Training set (copy)',
            val_size=.1, test_size=.2, level=4, min_mask_area=.000001, multiple_ann=True, multiple_classes=True,
            augment_dict=AugmentDict(
                hepatocyte=1,
                necrosis=2,
                fibrosis=1,
                tumour=1,
                inflamation=2,
                mucin=5,
                blood=7,
                foreign_body=17,
                macrophages=6,
                bile_duct=2
            )
        )()

        # calculating class multipliers automatically
        CRLMSplitProcessAugment(
            dataset_path='/media/giussepi/2_0_TB_Hard_Disk/improved_CRLM_dataset/Annotated - Training set (copy)',
            val_size=.1, test_size=.2, level=4, min_mask_area=.000001, multiple_ann=True, multiple_classes=True
        )()
    """

    def __init__(self, **kwargs):
        """ Initializes the object instance """
        # CRLMSplitDataset ####################################################
        self.splitter = kwargs.get('splitter', CRLMSmartSplitDataset)
        self.dataset_path = kwargs.get('dataset_path')
        self.ndpi_reg = kwargs.get('ndpi_reg', r'CRLM 0*(?P<index>\d+).ndpi')
        self.val_size = kwargs.get('val_size', .1)
        self.test_size = kwargs.get('test_size', .1)
        self.filename_tpl = kwargs.get('filename_tpl', 'CRLM {}.ndpi')
        # CRLMDatasetProcessor ################################################
        self.level = kwargs.get('level')
        # self.filename_tpl from above is reused for this class
        self.patch_size = kwargs.get('patch_size', 640)
        self.min_mask_area = kwargs.get('min_mask_area', .2)
        self.processing_type = kwargs.get('processing_type', CRLMProcessingTypes.ALL_WITH_ROIS)
        self.central_area_offset = kwargs.get('central_area_offset', .215)
        self.background_threshold = kwargs.get('background_threshold', .88)
        self.img_format = kwargs.get('img_format', 'tiff')
        self.mask_format = kwargs.get('mask_format', 'png')
        self.multiple_ann = kwargs.get('multiple_ann', False)
        self.multiple_classes = kwargs.get('multiple_classes', False)

        # CRLMAugmentationProcessor ###########################################
        self.augment_dict = kwargs.get('augment_dict', None)
        self.class_multiplier = kwargs.get('class_multiplier', 0)
        self.min_transforms = kwargs.get('min_transforms', 1)
        self.original = kwargs.get('original', True)
        self.image_reg = kwargs.get(
            'image_reg',
            r'(?P<label>[A-Z])_f(?P<file>\d+)_r(?P<roi>\-?\d+)_a(?P<annotation>\d+)_c(?P<counter>\d+).ann.tiff'
        )
        self.image_name_reg = kwargs.get('image_name_reg', r'(?P<filename>.+).ann.tiff')
        self.mask_name_reg = kwargs.get('mask_name_reg', r'(?P<filename>.+).mask.png')
        self.image_extension = kwargs.get('image_extension', '.ann.tiff')
        self.mask_extension = kwargs.get('mask_extension', '.mask.png')

        # other ###############################################################
        self.rm_non_augmented = kwargs.get('rm_non_augmented', False)

    def __call__(self):
        """ functor call """
        self.process_dataset()

    def process_dataset(self):
        """
        * Splits the dataset into train, test and val.
        * Creates the crops for each subdataset
        * Optionally, performs data augmentation over the processed train subdataset
        """

        self.splitter(
            dataset_path=self.dataset_path, ndpi_reg=self.ndpi_reg,
            val_size=self.val_size, test_size=self.test_size, filename_tpl=self.filename_tpl
        )()

        logger.info("Extracting crops")

        for split in DB.SUB_DATASETS:
            logger.info(f"Working on {split} sub-dataset")

            CRLMDatasetProcessor(
                level=self.level, images_path=os.path.join(self.dataset_path, split),
                filename_tpl=self.filename_tpl, filename_reg=self.ndpi_reg
            )(
                patch_size=self.patch_size,
                saving_path=os.path.join(self.dataset_path, f'{split}_processed'),
                min_mask_area=self.min_mask_area, processing_type=self.processing_type,
                central_area_offset=self.central_area_offset,
                background_threshold=self.background_threshold,
                img_format=self.img_format,
                mask_format=self.mask_format,
                multiple_ann=self.multiple_ann,
                multiple_classes=self.multiple_classes,
            )

        logger.info("Performing data augmentation on train split")

        train_processed_path = os.path.join(self.dataset_path, f'{DB.TRAIN}_processed')

        CRLMAugmentationProcessor(
            images_masks_path=train_processed_path,
            augment_dict=self.augment_dict, class_multiplier=self.class_multiplier,
            min_transforms=self.min_transforms,
            original=self.original, image_reg=self.image_reg, image_name_reg=self.image_name_reg,
            mask_name_reg=self.mask_name_reg, image_extension=self.image_extension,
            mask_extension=self.mask_extension,
            saving_path=os.path.join(self.dataset_path, f'{DB.TRAIN}_processed_augmented'),
            img_format=self.img_format, mask_format=self.mask_format
        )()

        if self.rm_non_augmented:
            logger.info(f"Removing {train_processed_path}")
            os.rmdir(train_processed_path)
