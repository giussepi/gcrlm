# -*- coding: utf-8 -*-
""" gcrlm/core/models """

from collections import namedtuple, defaultdict
from copy import deepcopy

import numpy as np

from gcrlm.constants import Label


# holds roi and annotaion indexes data
RoiAnn = namedtuple('RoiAnn', ['roi_index', 'roi_bbox', 'ann_indexes'])


class BoundingBox:
    """ Holds annotation bounding boxes """

    def __init__(self, xi, xa, yi, ya):
        """
        Initializes the object instance

        Args:
            xi <int, np.int64>: min x value
            xa <int, np.int64>: max x value
            yi <int, np.int64>: min y value
            ya <int, np.int64>: max y value
        """
        assert isinstance(xi, (int, np.int64))
        assert isinstance(xa, (int, np.int64))
        assert isinstance(yi, (int, np.int64))
        assert isinstance(ya, (int, np.int64))
        assert xi < xa
        assert yi < ya

        self.xi = xi
        self.xa = xa
        self.yi = yi
        self.ya = ya

    def __str__(self):
        """ Returns unicode representation """
        return "(({}, {}), ({}, {}))".format(self.xi, self.yi, self.xa, self.ya)

    @property
    def shape(self):
        """ Returns the width and height """
        return self.xa - self.xi, self.ya - self.yi

    def is_inside(self, bbox, expand_border=50):
        """
        Evaluates if the current bbox lies inside the provided bbox instance

        Args:
            bbox  <BoundingBox>: bbox to be verified to contain the current bbox instance
            expand_border <int>: how many pixels to expand the borders of the provided bbox
                                 before evaluating if the current bbox instance is inside
                                 the provided bbox. This is necessary for CRLM because not all
                                 annotations lies perfectly inside their ROIS

        Returns:
            <bool>
        """
        assert isinstance(bbox, self.__class__)

        return self.xi >= bbox.xi-expand_border and self.xa <= bbox.xa+expand_border and \
            self.yi >= bbox.yi-expand_border and self.ya <= bbox.ya+expand_border

    def intersects(self, bbox):
        """
        Evaluates if the current bbox intersects the provided bbox instance

        Args:
            bbox  <BoundingBox>: bbox to be verified to be intersected by the current bbox instance

        Returns:
            <bool>
        """
        assert isinstance(bbox, self.__class__)

        # top-left corner inside bbox
        if bbox.xi <= self.xi <= bbox.xa and bbox.yi <= self.yi <= bbox.ya:
            return True

        # top-right corner inside bbox
        if bbox.xi <= self.xa <= bbox.xa and bbox.yi <= self.yi <= bbox.ya:
            return True

        # bottom-left corner inside bbox
        if bbox.xi <= self.xi <= bbox.xa and bbox.yi <= self.ya <= bbox.ya:
            return True

        # bottom-right corner inside bbox
        if bbox.xi <= self.xa <= bbox.xa and bbox.yi <= self.ya <= bbox.ya:
            return True

        return False

    @property
    def values(self):
        """
        Returns:
            xi, xa, yi, ya <tuple>
        """
        return self.xi, self.xa, self.yi, self.ya


class AnnotationMGR:
    """
    Stores all the annotations and allows querying for anntotation ids which
    have an specific label and intersercts the provided bounding box

    Usage:
        ann_mgr = AnnotationMGR()
        # add your data
        for ann_index, label, bbox in my_iterable:
            ann_mgr.add(ann_index, label, bbox)
        # query for annotations
        intersections = ann_mgr.get_intersections(<my label>, <my BoundingBox>)
    """

    def __init__(self):
        """ Initializes the object instance """
        self.data = defaultdict(list)

    def add(self, ann_index, label, bbox, poly_array):
        """
        Adds data to the object intance

        Args:
            ann_index         <int>: index of the annotation
            label             <str>: annotation label
            bbox      <BoundingBox>: annotation BoundingBox
            poly_array <np.ndarray>: numpy array containing the polygon points
        """
        assert isinstance(ann_index, int), type(ann_index)
        assert isinstance(label, str), type(label)
        assert isinstance(bbox, BoundingBox), type(bbox)
        assert isinstance(poly_array, np.ndarray), type(poly_array)

        self.data[label].append((ann_index, bbox, poly_array))

    def get_intersections(self, label, bbox):
        """
        Filters the annotations by the provided label and returns a tuple with
        the annotation indexes, bounding boxes and polygon arrays belonging to the
        annotations whose bouding boxes intersec the provided bbox

        Args:
            label        <str>: annotation label
            bbox <BoundingBox>: annotation BoundingBox

        Returns:
            intersections <tuple>
        """
        assert isinstance(label, str), type(label)
        assert isinstance(bbox, BoundingBox), type(bbox)

        return tuple(deepcopy(i) for i in self.data[label] if i[1].intersects(bbox))


class AugmentDict:
    """ Holds the classes and their respective multipliers (to perform data augmentation) """

    def __init__(self, **kwargs):
        """
        Initializes the objec instance

        Kwargs:
            hepatocyte   <int>: class/label multiplier
            necrosis     <int>: class/label multiplier
            fibrosis     <int>: class/label multiplier
            tumour       <int>: class/label multiplier
            inflamation  <int>: class/label multiplier
            mucin        <int>: class/label multiplier
            blood        <int>: class/label multiplier
            foreign_body <int>: class/label multiplier
            macrophages  <int>: class/label multiplier
            bile_duct    <int>: class/label multiplier
        """
        self.hepatocyte = kwargs.get('hepatocyte')
        self.necrosis = kwargs.get('necrosis')
        self.fibrosis = kwargs.get('fibrosis')
        self.tumour = kwargs.get('tumour')
        self.inflamation = kwargs.get('inflamation')
        self.mucin = kwargs.get('mucin')
        self.blood = kwargs.get('blood')
        self.foreign_body = kwargs.get('foreign_body')
        self.macrophages = kwargs.get('macrophages')
        self.bile_duct = kwargs.get('bile_duct')

        assert isinstance(self.hepatocyte, int), type(self.hepatocyte)
        assert isinstance(self.necrosis, int), type(self.necrosis)
        assert isinstance(self.fibrosis, int), type(self.fibrosis)
        assert isinstance(self.tumour, int), type(self.tumour)
        assert isinstance(self.inflamation, int), type(self.inflamation)
        assert isinstance(self.mucin, int), type(self.mucin)
        assert isinstance(self.blood, int), type(self.blood)
        assert isinstance(self.foreign_body, int), type(self.foreign_body)
        assert isinstance(self.macrophages, int), type(self.macrophages)
        assert isinstance(self.bile_duct, int), type(self.bile_duct)

    def __str__(self):
        return f'{self.__class__.__name__}: {self.values}'

    def __call__(self):
        """ functor call """
        return self.get_label_augment_dict()

    @property
    def values(self):
        """
        Returns:
           attribute values <tuple>
        """
        return self.hepatocyte, self.necrosis, self.fibrosis, self.tumour, self.inflamation,\
            self.mucin, self.blood, self.foreign_body, self.macrophages, self.bile_duct

    @classmethod
    def get_initialized_instance(cls, value):
        """
        Creates an instances initialized with the provided value

        Args:
            value <int>: initial value for all classes

        Returns:
            instance <AugmentDict>
        """
        assert isinstance(value, int), type(value)
        assert value >= 0, value

        return cls(
            hepatocyte=value,
            necrosis=value,
            fibrosis=value,
            tumour=value,
            inflamation=value,
            mucin=value,
            blood=value,
            foreign_body=value,
            macrophages=value,
            bile_duct=value,
        )

    def get_label_augment_dict(self):
        """
        Translates the class names into labels

        Returns:
            label_augment_dict <dict>
        """
        return {
            Label.labels[0].file_label: self.hepatocyte,
            Label.labels[1].file_label: self.necrosis,
            Label.labels[2].file_label: self.fibrosis,
            Label.labels[3].file_label: self.tumour,
            Label.labels[4].file_label: self.inflamation,
            Label.labels[5].file_label: self.mucin,
            Label.labels[6].file_label: self.blood,
            Label.labels[7].file_label: self.foreign_body,
            Label.labels[8].file_label: self.macrophages,
            # we're skipping Label.labels[9] (background) on purpose
            Label.labels[10].file_label: self.bile_duct,
        }
