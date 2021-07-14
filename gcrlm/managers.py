# -*- coding: utf-8 -*-
""" gcrlm/managers """

import os
from collections import defaultdict
from math import ceil
from xml.etree import ElementTree

import cv2
import matplotlib.pyplot as plt
import numpy as np
import openslide as ops

from gutils.image_processing import get_slices_coords
from imutils import rotate
from logzero import logger
from skimage.color import rgba2rgb, rgb2gray
from skimage.draw import polygon2mask
from tqdm import tqdm

from gcrlm.constants import Label
from gcrlm.core.exceptions.crlm import NoRoiClusters, MultipleClassesButNoAnnotaionMGR, \
    MultipleClassesButNoMultipleAnnotationsMGR
from gcrlm.core.models import BoundingBox, RoiAnn, AnnotationMGR


class CRLM:
    """
    Class to review the npdi files, extract annotations, bounding boxes, plot annotations,
    plot slides, extract masks, create annotations images with some data augmentation, etc

    Usage:
        tc = CRLM(index=1)
        tc = CRLM(index=1, fileroot='/media/giussepi/2_0_TB_Hard_Disk/CRLM/Original/')
    """

    aname_list = ['F', 'H', 'T', 'N', 'M', 'I', 'B', 'FB', 'MF', 'BD']  # Annotation name list
    excluded_ann = ('ROI', 'O')

    def __init__(
            self, index, level=5,
            fileroot=os.path.join(os.path.expanduser('~'), 'DATA_CRLM', 'CRLM', 'Original'),
            annotation_root='', filename_tpl='CRLM {}.ndpi'
    ):
        """
        Initialized the object instance

        Args:
            index           <int>: index of ndpi image
            level           <int>: magnification level (dim/2**level)
            fileroot        <str>: path to the folder for containing the files
            annotation_root <str>: path to folder for containing the annotations
            filename_tpl    <str>: filename template to create the file name using the index
                                   e.g.: for CRLM_001.npi you should use CRLM_{:03d}.ndpi
        """
        assert isinstance(index, int)
        assert isinstance(level, int)
        assert level >= 0
        assert isinstance(fileroot, str)
        assert os.path.isdir(fileroot)
        assert isinstance(annotation_root, str)
        assert isinstance(filename_tpl, str)

        if annotation_root:
            assert os.path.isdir(annotation_root)

        self.index = index
        self.level = level
        self.fileroot = fileroot
        self.filename = filename_tpl.format(self.index)
        self.img = ops.open_slide(os.path.join(self.fileroot, self.filename))
        self.root = self.extract_root(rootname=annotation_root)
        self.root_len = self.root.__len__()

    def extract_root(self, rootname=''):
        """
        Args:
            rootname <str>: path to folder for containing the annotations
        """
        assert isinstance(rootname, str)

        if not rootname:
            rootname2 = os.path.join(self.fileroot, self.filename+'.ndpa')
        else:
            rootname2 = os.path.join(rootname, self.filename+'.ndpa')

        with open(rootname2) as f:
            tree = ElementTree.parse(f)

        return tree.getroot()

    def convert_xy(self, ax, by):
        xmpp = float(self.img.properties['openslide.mpp-x'])
        xoff = float(self.img.properties['hamamatsu.XOffsetFromSlideCentre'])
        ympp = float(self.img.properties['openslide.mpp-y'])
        yoff = float(self.img.properties['hamamatsu.YOffsetFromSlideCentre'])
        ld = self.img.dimensions
        nax = int((ax - xoff)/(xmpp*1000.0)+ld[0]/2.0)
        nby = int((by - yoff)/(ympp*1000.0)+ld[1]/2.0)

        return nax, nby

    def get_aname(self, index):
        """
        Returns the annotaion name

        Args:
            index <int>: annotation index

        Returns:
            annotation name
        """
        if self.root[index].__len__() == 0:
            return 'O'

        if self.root[index][0].text is not None and self.root[index][0].text.lower() != 'roi':
            if self.root[index][0].text.__len__() == 1:
                return self.root[index][0].text.upper()

            tindexname = self.root[index][0].text.strip()

            if tindexname.upper() in self.aname_list:
                return tindexname.upper()

            return tindexname[0].upper()

        if self.root[index][0].text is None:
            return 'O'

        if self.root[index][0].text.lower() == 'roi':
            return 'ROI'

        return 'O'

    @property
    def all_rois(self):
        """ Returns a list with all the indexes of the ROI annotations """
        roi_list = []

        for ann_index in range(self.root_len):
            if self.get_aname(ann_index) == 'ROI':
                roi_list.append(ann_index)

        return roi_list

    def get_annotations_inside(self, index):
        """
        Returns all annotations that lies inside the annotiation with the provided index

        Args:
            index <int>: annotation index

        Returns:
            annotations <list>
        """
        assert isinstance(index, int)

        container_bbox = BoundingBox(*self.get_annotation_bbox(index))

        annotations = list()

        for ann_index in range(self.root_len):
            if ann_index == index or self.get_aname(ann_index) in self.excluded_ann:
                continue

            if BoundingBox(*self.get_annotation_bbox(ann_index)).is_inside(container_bbox):
                annotations.append(ann_index)

        return annotations

    def cluster_annotations_by_roi(self, all_annotations=False):
        """
        Clusters the annotations by roi

        Args:
            all_annotations <bool>: Whether to include all annotations of just those
                                    lying inside ROIs

        Returns:
            clusters <list>
        """
        rois = {-1: None} if all_annotations else dict()
        clusters = defaultdict(list)

        for roi_index in self.all_rois:
            rois[roi_index] = BoundingBox(*self.get_annotation_bbox(roi_index))

        logger.info("Clustering annotations by roi")

        for ann_index in tqdm(range(self.root_len)):
            if self.get_aname(ann_index) in self.excluded_ann:
                continue

            ann_bbox = BoundingBox(*self.get_annotation_bbox(ann_index))

            for roi_index, roi_bbox in rois.items():
                if roi_bbox is not None and ann_bbox.is_inside(roi_bbox):
                    clusters[roi_index].append(ann_index)
                    break
            else:
                if all_annotations:
                    clusters[-1].append(ann_index)

        full_roi_clusters = [
            RoiAnn(roi_index, rois[roi_index], ann_indexes)
            for roi_index, ann_indexes in clusters.items()
        ]

        return full_roi_clusters

    def get_annotation_dots(self, index):
        '''
        Transforms the dots in the pixel coordinate

        Args:
            index <int>: annotation index

        Returns:
            annotations <np.ndarray>
        '''
        if self.root[index].__len__() == 0:
            return np.array([[0, 0], [0, 0]])

        if self.root[index][-1][-1].tag == 'pointlist':
            tplist = self.root[index][-1][-1]
        elif self.root[index][-1][-1].tag == 'specialtype':
            if self.root[index][-1][-3].tag == 'pointlist':
                tplist = self.root[index][-1][-3]
            elif self.root[index][-1][-2].tag == 'pointlist':
                tplist = self.root[index][-1][-2]
            else:
                return np.array([])
        else:
            return np.array([])

        plist1 = [
            self.convert_xy(int(tplist[i][0].text), int(tplist[i][1].text)) for i in range(tplist.__len__())]

        return np.array(plist1)

    def get_annotation_bbox(self, index, return_polygon=False):
        """
        Calculates and returns the annotation bounding box

        Args:
            index           <int>: annotation index
            return_polygon <bool>: whether or not return the polygon points

        Returns:
            xmin, xmax, ymin, ymax, annotation dots [optional]
        """
        pl_arr = self.get_annotation_dots(index)
        pl_arr = np.vstack((pl_arr, pl_arr[0]))
        xi = pl_arr[:, 0].min()
        xa = pl_arr[:, 0].max()
        yi = pl_arr[:, 1].min()
        ya = pl_arr[:, 1].max()

        if return_polygon:
            return xi, xa, yi, ya, pl_arr

        return xi, xa, yi, ya

    def extract_wsi(self, level=-1, offsetx=128, offsety=128):
        """
        Returns the image at the given level (dim/2**level)

        Args:
            level   <int>: magnification level and region size (dim/2**level)
            offsetx <int>: ofsset on x axis
            offsety <int>: offset on y axis

        Returns:
            scaled RGBA image <np.ndarray>
        """
        assert isinstance(level, int)
        assert isinstance(offsetx, int)
        assert isinstance(offsety, int)
        assert offsetx >= 0
        assert offsety >= 0

        level = level if level >= 0 else self.level
        img = self.img.read_region(
            (0-offsetx, 0-offsety),
            level,
            (int(self.img.dimensions[0] / (2**level)), int(self.img.dimensions[1] / (2**level)))
        )

        return np.array(img)

    def extract_image_block(self, startx=0, starty=0, sizex=100, sizey=100, level=-1):
        """
        Read the image at the given level (dim/2**level), and returns only the specified region

        Args:
            startx <int>: top left x pixel
            starty <int>: top left y pixel
            sizex  <int>: x size of the region
            sizey  <int>: y size of the region
            level  <int>: magnification level

        Returns:
            RGB image region <np.ndarray>
        """
        assert isinstance(startx, int)
        assert isinstance(starty, int)
        assert isinstance(sizex, int)
        assert isinstance(sizey, int)
        assert isinstance(level, int)
        assert startx >= 0
        assert starty >= 0
        assert sizex >= 0
        assert sizey >= 0

        level = level if level >= 0 else self.level
        im = self.img.read_region((startx, starty), level, (sizex, sizey))

        return np.array(im)[:, :, :3]

    def calculate_annotation_region(
            self, xi, xa, yi, ya, offsetxl, offsetyt, offsetxr=-1, offsetyb=-1, level=-1):
        """
        Calculates and returns the region to be read from the ndpi file using the offset and
        level provided

        Args:
            xi,xa, yi, ya <int>: Annotation bounding box. Just call self.get_annotation_bbox(index)
                                 to get them
            offsetxl      <int>: Extra margin to add to the left side of the annotation
            offsetyt      <int>: Extra margin to add to the top side of the annotation
            offsetxr      <int>: Extra margin to add to the right side of the annotation
            offsetyb      <int>: Extra margin to add to the bottom side of the annotation
            level         <int>: magnification level (dim/2**level)

        Returns:
            region size (width, height)
        """
        level = level if level >= 0 else self.level
        offsetxr = offsetxl if offsetxr < 0 else offsetxr
        offsetyb = offsetyt if offsetyb < 0 else offsetyb

        return int((xa-xi+offsetxl+offsetxr)/2**level), int((ya-yi+offsetyt+offsetyb)/2**level)

    def plot_annotation(self, index, offsetx=128, offsety=128, level=-1):
        """
        Plots the annotation bounding box

        Args:
            index  <int>: annotation index
            offsetx <int>: Extra margin to avoid annotation borders touching the end of the patch
            offsety <int>: Extra margin to avoid annotation borders touching the end of the patch
            level  <int>: magnification level (dim/2**level)
        """
        assert isinstance(index, int)
        assert isinstance(level, int)
        assert index >= 0
        assert offsetx >= 0
        assert offsety >= 0

        level = level if level >= 0 else self.level
        name = self.get_aname(index)
        xi, xa, yi, ya = self.get_annotation_bbox(index)
        tmpim = self.img.read_region(
            (xi-offsetx, yi-offsety),
            level,
            self.calculate_annotation_region(xi, xa, yi, ya, offsetx, offsety, level=level)
        )
        plt.title(name)
        plt.imshow(tmpim)
        plt.show()

    def plot_roi(self, index, offsetx=0, offsety=0, annotations=None, level=5, **kwargs):
        """
        Plots the ROI with the provided index along all the annotations that lies inside

        Args:
            index        <int>: annotation index
            offsetx      <int>: Extra margin to avoid annotation borders touching the end of the patch
            offsety      <int>: Extra margin to avoid annotation borders touching the end of the patch
            annotations <list>: List of annotations inside the ROI to plot. Set it to None to plot them all
            level        <int>: magnification level (dim/2**level)

        Kwargs:
            line_width <int, float>: line width to be used when plotting the polygons
        """
        assert isinstance(index, int)
        assert isinstance(offsetx, int)
        assert isinstance(offsety, int)
        assert isinstance(level, int)
        assert index >= 0
        assert offsetx >= 0
        assert offsety >= 0

        if annotations is not None:
            assert isinstance(annotations, list)

        line_width = kwargs.get('line_width', 3)
        assert isinstance(line_width, (float, int))

        level = level if level > 0 else 0
        roi_name = self.get_aname(index)

        if roi_name != 'ROI':
            logger.error('The annotation {} is not a ROI'.format(index))
            return

        roi_bbox = BoundingBox(*self.get_annotation_bbox(index))

        tmpim = self.img.read_region(
            (roi_bbox.xi-offsetx, roi_bbox.yi-offsety),
            level,
            self.calculate_annotation_region(*roi_bbox.values, offsetx, offsety, offsetx, offsety, level)
        )
        plt.imshow(tmpim)

        annotations = self.get_annotations_inside(index) if annotations is None else annotations

        for ann_index in annotations:
            # TODO: review what's the problem with this thing...
            ann_bbox = BoundingBox(*self.get_annotation_bbox(ann_index))

            if not ann_bbox.is_inside(roi_bbox):
                logger.warning("The annotation {} is not inside the ROI {}".format(ann_index, index))
                continue

            points = self.get_annotation_bbox(ann_index, return_polygon=True)[-1]
            points[:, 0] = points[:, 0] - roi_bbox.xi + offsetx
            points[:, 1] = points[:, 1] - roi_bbox.yi + offsety
            resized_points = points / 2**level
            plt.plot(resized_points[:, 0], resized_points[:, 1], 'y-', linewidth=line_width)

        plt.show()

    def plot_slide(self, level=-1):
        """
        Plots the whole slide

        Args:
            level  <int>: magnification level (dim/2**level)
        """
        assert isinstance(level, int)

        level = level if level >= 0 else self.level
        image = self.img.read_region(
            (int(self.img.dimensions[0] / (2**level)), int(self.img.dimensions[1] / (2**level))),
            level,
            (int(self.img.dimensions[0] / (2**level)), int(self.img.dimensions[1] / (2**level)))
        )
        im_arr = np.array(image)
        plt.imshow(im_arr[:, :, 0:3])
        plt.show()

    @staticmethod
    def get_mask_color(label, ann_mgr, multiple_classes):
        """
        Returns the color to be used by the label in the grayscale mask image. The value lies in
        the interval [0, 255]

        Args:
            label             <str>: Annotation label
            ann_mgr <AnnotationMGR>: AnnotationMGR instance. Default None
            multiple_classes <bool>: Whether or not plot annotations from multiple classes.
                                     Requires ann_mgr != None. Default False

        Returns
            colour <int>
        """
        if ann_mgr is not None and multiple_classes:
            return Label.label_mask_color_dict[label]

        return 255

    def extract_annotation_image(
            self, index, offsetxl=128, offsetyt=128, offsetxr=-1, offsetyb=-1, level=0, plot=False,
            ann_mgr=None, multiple_classes=False
    ):
        """
        Processes the annotation with the specified index and returns its name, image region
        and mask region

        Args:
            index             <int>: index of the annotation
            offsetxl          <int>: Extra margin to add to the left side of the annotation
            offsetyt          <int>: Extra margin to add to the top side of the annotation
            offsetxr          <int>: Extra margin to add to the right side of the annotation
            offsetyb          <int>: Extra margin to add to the bottom side of the annotation
            level             <int>: magnification level (dim/2**level)
            plot             <bool>: Whether or not plot the results
            ann_mgr <AnnotationMGR>: AnnotationMGR instance. Default None
            multiple_classes <bool>: Whether or not plot annotations from multiple classes.
                                     Requires ann_mgr != None. Default False

        Returns:
            annotation name <str>, RGBA image_region <np.ndarray>, mask_region<np.ndarray>
        """
        assert isinstance(index, int), type(index)
        assert isinstance(offsetxl, int), type(offsetxl)
        assert isinstance(offsetyt, int), type(offsetyt)
        assert isinstance(offsetxr, int), type(offsetxr)
        assert isinstance(offsetyb, int), type(offsetyb)
        assert isinstance(level, int), type(level)
        assert isinstance(plot, bool), type(plot)
        assert index >= 0, index
        assert offsetxl >= 0, offsetxl
        assert offsetyt >= 0, offsetyt
        assert level >= 0, level

        if ann_mgr:
            assert isinstance(ann_mgr, AnnotationMGR), type(ann_mgr)

        assert isinstance(multiple_classes, bool), type(multiple_classes)

        if multiple_classes and ann_mgr is None:
            raise MultipleClassesButNoAnnotaionMGR

        offsetxr = offsetxl if offsetxr < 0 else offsetxr
        offsetyb = offsetyt if offsetyb < 0 else offsetyb
        name = self.get_aname(index)
        xi, xa, yi, ya, pl_arr = self.get_annotation_bbox(index, return_polygon=True)

        tmpim = self.img.read_region(
            (xi-offsetxl, yi-offsetyt),
            level,
            self.calculate_annotation_region(xi, xa, yi, ya, offsetxl, offsetyt, offsetxr, offsetyb, level)
        )
        tmpim = np.array(tmpim)
        polygon_arrays = [pl_arr]
        polygon_colours = [self.get_mask_color(self.get_cleaned_label(name), ann_mgr, multiple_classes)]
        mask = np.zeros(tmpim.shape[:2][::-1], dtype=np.uint8)

        if ann_mgr:
            if multiple_classes:
                anns = set(ann_mgr.data.keys())
                anns = anns.difference(self.get_cleaned_label(name))

                for label in anns:
                    intersections = ann_mgr.get_intersections(
                        label,
                        BoundingBox(xi-offsetxl, xa+offsetxr, yi-offsetyt, ya+offsetyb)
                    )

                    for _, _, poly_array in intersections:
                        polygon_arrays.append(poly_array)
                        polygon_colours.append(self.get_mask_color(label, ann_mgr, multiple_classes))

            # Getting the main annotations at the end to make sure they are
            # always present (not overwritten by annotations from other classes
            # that could overlap them)
            intersections = ann_mgr.get_intersections(
                self.get_cleaned_label(name),
                BoundingBox(xi-offsetxl, xa+offsetxr, yi-offsetyt, ya+offsetyb)
            )
            polygon_arrays.extend((i[2] for i in intersections))
            polygon_colours.extend([self.get_mask_color(
                self.get_cleaned_label(name), ann_mgr, multiple_classes)]*len(intersections))

        if plot:
            _, (ax1, ax2) = plt.subplots(1, 2)

        assert len(polygon_colours) == len(polygon_arrays), \
            f'{len(polygon_colours)} != {len(polygon_arrays)}'

        for colour, polygon_array in zip(polygon_colours, polygon_arrays):
            polygon_array[:, 0] = polygon_array[:, 0]-xi+offsetxl
            polygon_array[:, 1] = polygon_array[:, 1]-yi+offsetyt
            # scaling/resizing like this only works for single images
            # because the offsetxy and offsetyb cannot be applied to all the images.
            # they could have be designed for only one image
            # So we are using the general way to make it work for single and multiple masks
            # maskx = polygon_array[:, 0].max() + offsetxr
            # masky = polygon_array[:, 1].max() + offsetyb
            # resized_polygon_array = np.empty_like(polygon_array)
            # resized_polygon_array[:, 0] = polygon_array[:, 0] * tmpim.shape[1] / maskx
            # resized_polygon_array[:, 1] = polygon_array[:, 1] * tmpim.shape[0] / masky
            # scaling/resizing like this gives the same results but with some decimals
            resized_polygon_array = polygon_array/2**level
            colored_mask = polygon2mask(tmpim.shape[: 2][:: -1], resized_polygon_array).astype(np.bool)
            mask[colored_mask] = colour

            if plot:
                ax1.plot(resized_polygon_array[:, 0], resized_polygon_array[:, 1], 'y-', linewidth=3)

        mask = mask.T

        if plot:
            ax1.imshow(tmpim)
            # ax1.axis('image')
            # ax1.axis('off')
            ax2.imshow(mask)
            plt.show()

        return name, tmpim, mask

    def calculate_offset(self, di, da, level, target_size):
        """
        Calculates the offset at dimension d to get the target_size at the especified level

        NOTE: di and da could be xi, xa or yi, ya values of the bbox obtained by calling
        xi, xa, yi, ya = self.get_annotation_bbox(index)

        Args:
            di          <int>: initial position at dimension d
            da          <int>: final position at dimension d
            level       <int>: magnification level (dim/2**level)
            target_size <int>: final image length of dimension d at the especified level

        Returns:
            offset <int>
        """
        assert isinstance(di, (int, np.int64))
        assert isinstance(da, (int, np.int64))
        assert isinstance(level, int)
        assert isinstance(target_size, int)
        assert di >= 0
        assert di < da
        assert level >= 0
        assert target_size > 0

        offset = (target_size * (2**level) - (da - di)) / 2

        return ceil(offset)

    def recalculate_offsets_considering_roi_bbox(self, ann_bbox, offsetx, offsety, roi_bbox, inside_roi=True):
        """
        Recalculates the offsets in two possible ways: firstly, making sure the crop lies inside
        the ROI; secondly, making sure to return a crop even if it lies outside outsiede of the
        ROI.

        Args:
            ann_bbox <BoundingBox>: annotation bbox
            offsetx          <int>: offset to be applied to the x axis
            offsety          <int>: offset to be applied to the y axis
            roi_bbox <BoundingBox>: roi bbox
            inside_roi      <bool>: Whether or not have crops only from the inside of the ROI

        Returns:
            offsetx_left, offsetx_right, offsety_top, offsety_bottom
        """
        assert isinstance(inside_roi, bool)

        if inside_roi:
            return self.recalculate_offsets_considering_roi_bbox_1(ann_bbox, offsetx, offsety, roi_bbox)

        return self.recalculate_offsets_considering_roi_bbox_2(ann_bbox, offsetx, offsety, roi_bbox)

    def recalculate_offsets_considering_roi_bbox_1(self, ann_bbox, offsetx, offsety, roi_bbox):
        """
        Makes sure the final ann_bbox size will fit into the ROI bounding box. Thus, the four offsets
        are modified to accomplish that condition.

        Args:
            ann_bbox <BoundingBox>: annotation bbox
            offsetx          <int>: offset to be applied to the x axis
            offsety          <int>: offset to be applied to the y axis
            roi_bbox <BoundingBox>: roi bbox

        Returns:
            offsetx_left, offsetx_right, offsety_top, offsety_bottom
        """
        assert isinstance(ann_bbox, BoundingBox)
        assert isinstance(offsetx, int)
        assert isinstance(offsety, int)
        assert isinstance(roi_bbox, BoundingBox)

        roi_bbox_width, roi_bbox_height = roi_bbox.shape
        ann_bbox_width, ann_bbox_height = ann_bbox.shape

        assert roi_bbox_width >= ann_bbox_width + 2 * offsetx, \
            "roi_bbox_width must be >= ann_bbox_width + 2 * offsetx"
        assert roi_bbox_height >= ann_bbox_height + 2 * offsety, \
            "roi_bbox_height must be >= ann_bbox_height + 2 * offsety"

        new_offsetx_left = new_offsetx_right = offsetx
        new_offsety_top = new_offsety_bottom = offsety

        #######################################################################
        #        making sure the final ann_bbox will be inside the ROI         #
        #######################################################################
        if ann_bbox.xi - offsetx < roi_bbox.xi:
            # case 1: left bbox annotation side out of ROI
            new_offsetx_left = 0
            new_offsetx_right = 2 * offsetx
        elif ann_bbox.xa + offsetx > roi_bbox.xa:
            # case 2: right bbox annotation side out of ROI
            new_offsetx_left = 2 * offsetx
            new_offsetx_right = 0

        if ann_bbox.yi - offsety < roi_bbox.yi:
            # case 3: top bbox annotation side out of ROI
            new_offsety_top = 0
            new_offsety_bottom = 2 * offsety
        elif ann_bbox.ya + offsety > roi_bbox.ya:
            # case 4: bottom bbox annotation side out of ROI
            new_offsety_top = 2 * offsety
            new_offsety_bottom = 0

        return new_offsetx_left, new_offsetx_right, new_offsety_top, new_offsety_bottom

    def recalculate_offsets_considering_roi_bbox_2(self, ann_bbox, offsetx, offsety, roi_bbox):
        """
        Makes sure that annotations touching ROI borders keeps those those positions in the final
        crop. Furthermore, it creates the crops with the fixed width and height even when the final
        crop if bigger than the ROI bounding box.

        NOTE 1: This could lead to annotations going out of the image limits. So far, it seems not to
        happen on the CRLM dataset.

        NOTE 2: After testing this method we saw an increased number of crops. However, due to not
        all the annotations supposed to be touching one or more ROI borders are doing it so,
        some annotations that are supposed to be limited by the ROI borders are being
        shown centred in their crops.

        Args:
            ann_bbox <BoundingBox>: annotation bbox
            offsetx          <int>: offset to be applied to the x axis
            offsety          <int>: offset to be applied to the y axis
            roi_bbox <BoundingBox>: roi bbox

        Returns:
            offsetx_left, offsetx_right, offsety_top, offsety_bottom
        """
        assert isinstance(ann_bbox, BoundingBox)
        assert isinstance(offsetx, int)
        assert isinstance(offsety, int)
        assert isinstance(roi_bbox, BoundingBox)

        # TODO: Verify that the final bbox is still inside the image dimensions
        new_offsetx_left = new_offsetx_right = offsetx
        new_offsety_top = new_offsety_bottom = offsety

        ##########################################################################
        # Making sure taht annotations touching the borders keep those positions #
        ##########################################################################
        if roi_bbox.xi - 50 <= ann_bbox.xi <= roi_bbox.xi + 50:
            # if if ann_bbox.xi == roi_bbox.xi:
            new_offsetx_left = 0
            new_offsetx_right = 2 * offsetx
        elif roi_bbox.xa - 50 <= ann_bbox.xa <= roi_bbox.xa + 50:
            # elif ann_bbox.xa == roi_bbox.xa:
            new_offsetx_left = 2 * offsetx
            new_offsetx_right = 0

        if roi_bbox.yi - 50 <= ann_bbox.yi <= roi_bbox.yi + 50:
            # if ann_bbox.yi == roi_bbox.yi:
            new_offsety_top = 0
            new_offsety_bottom = 2 * offsety
        elif roi_bbox.ya - 50 <= ann_bbox.ya <= roi_bbox.ya + 50:
            # elif ann_bbox.ya == roi_bbox.ya:
            new_offsety_top = 2 * offsety
            new_offsety_bottom = 0

        return new_offsetx_left, new_offsetx_right, new_offsety_top, new_offsety_bottom

    def extract_annotation_using_target_size(
            self, index, min_size, offsetx=128, offsety=128, level=0, roi_bbox=None,
            inside_roi=True, plot=False, ann_mgr=None, multiple_classes=False
    ):
        """
        Processes the annotation with the specified index and returns its name, image region
        and mask region. The returned image and mask will have a width and height >= min_size,
        furthermore, they will always be indside de ROI (if provided)

        Args:
            index             <int>: index of the annotation
            min_size          <int>: minimum size allowed per side (x, y)
            offsetx           <int>: Extra margin to add to the x axis (left and right)
            offsety           <int>: Extra margin to add to the y axis (top and bottom)
            level             <int>: magnification level (dim/2**level)
            roi_bbox  <BoundingBox>: ROI bbox
            inside_roi       <bool>: Whether or not have crops only from the inside of the ROI
            plot             <bool>: Whether or not plot the results
            ann_mgr <AnnotationMGR>: AnnotationMGR instance. Default None
            multiple_classes <bool>: Whether or not plot annotations from multiple classes.
                                     Requires ann_mgr != None. Default False
        Returns:
            annotation name <str>, image_region <np.ndarray>, mask_region<np.ndarray>
        """
        assert isinstance(index, int)
        assert isinstance(min_size, int)
        assert min_size > 0
        assert isinstance(offsetx, int)
        assert isinstance(offsety, int)
        assert isinstance(level, int)
        if roi_bbox is not None:
            assert isinstance(roi_bbox, BoundingBox)
        assert isinstance(inside_roi, bool)
        assert isinstance(plot, bool)

        if ann_mgr:
            assert isinstance(ann_mgr, AnnotationMGR), type(ann_mgr)

        assert isinstance(multiple_classes, bool), type(multiple_classes)

        if multiple_classes and ann_mgr is None:
            raise MultipleClassesButNoAnnotaionMGR

        xi, xa, yi, ya = self.get_annotation_bbox(index)
        width, height = self.calculate_annotation_region(xi, xa, yi, ya, offsetx, offsety, level=level)

        dimx_diff = min_size - width
        dimy_diff = min_size - height

        new_offsetx = self.calculate_offset(xi, xa, level, min_size) if dimx_diff > 0 else offsetx
        new_offsety = self.calculate_offset(yi, ya, level, min_size) if dimy_diff > 0 else offsety

        if roi_bbox is None:
            offsetx_left = offsetx_right = new_offsetx
            offsety_top = offsety_bottom = new_offsety
        else:
            offsetx_left, offsetx_right, offsety_top, offsety_bottom = \
                self.recalculate_offsets_considering_roi_bbox(
                    BoundingBox(xi, xa, yi, ya), new_offsetx, new_offsety, roi_bbox, inside_roi)

        ann_name, ann_img, ann_mask = self.extract_annotation_image(
            index, offsetx_left, offsety_top, offsetx_right, offsety_bottom, level, plot, ann_mgr,
            multiple_classes
        )

        # If any of the offsets was modified, then we make sure that the returning image
        # and mask dimensions are bigger and equal to min_size
        if offsetx_left != offsetx:
            ann_img = ann_img[:min_size, :, :]
            ann_mask = ann_mask[:min_size, :]

        if offsety_top != offsety:
            ann_img = ann_img[:, :min_size, :]
            ann_mask = ann_mask[:, :min_size]

        return ann_name, ann_img, ann_mask

    @staticmethod
    def check_hole_in_tumour(image, central_area_offset=.215, background_threshold=.88):
        """
        Analizes the central area of the image, and returns True if the number of pixels
        with a value above the background_threshold is bigger than half of the number of pixels on
        the central area. Otherwise, returns False

        Args:
            image           <np.ndarray>: image to analize
            central_area_offset  <float>: percentage to ignore before considering the central area
                          e.g.: if dim0=100, dim1=200 and central_area_offset=.2, the central area
                          will be image[20:80, 40:160]
            background_threshold <float>: value between 0 and 1 used to identify background pixels
                                       (the image is turned into gray scale before the comparison)

        Returns:
            bool
        """
        assert isinstance(image, np.ndarray)
        assert isinstance(central_area_offset, float)
        assert 0 < central_area_offset < 1
        assert isinstance(background_threshold, float)
        assert 0 < background_threshold < 1

        bimg = rgb2gray(rgba2rgb(image))
        offset0, offset1 = list(map(lambda x: int(x*central_area_offset), bimg.shape))
        upper0 = bimg.shape[0] - offset0
        upper1 = bimg.shape[1] - offset1
        len0 = upper0 - offset0
        len1 = upper1 - offset1

        if np.sum(bimg[offset0:upper0, offset1:upper1] > background_threshold) > len0*len1/2:
            return True

        return False

    @staticmethod
    def flip_recolor_mix(img, mask=None):
        """
        Performs a random flip and color change in the image and optionaly applies the same flip to
        the mask (if provided).

        Args:
            img  <np.ndarray>: image
            mask <np.ndarray>: image mask

        Returns:
            image <np.ndarray>, mask <np.ndarray or None>
        """
        assert isinstance(img, np.ndarray)

        if mask is not None:
            assert isinstance(mask, np.ndarray)

        direction = np.random.choice([None, 'down', 'up', 'r90', 'r180', 'r270'])

        if np.random.randint(0, 10) < 7:
            rand_color = (
                np.random.random_sample()*0.4+0.7,
                np.random.random_sample()*0.4+0.7,
                np.random.random_sample()*0.4+0.7,
                np.random.random_sample()*0.4-0.2,
            )
        else:
            rand_color = None

        img = np.array(img, dtype=np.float)
        tmask = None

        if direction == 'down':
            timg = np.flipud(img)
            if mask is not None:
                tmask = np.flipud(mask)
        elif direction is None:
            timg = img
            if mask is not None:
                tmask = mask
        elif direction == 'r90':
            timg = np.array(rotate(img, 90))
            if mask is not None:
                tmask = np.array(rotate(mask, 90))
        elif direction == 'r180':
            timg = np.array(rotate(img, 180))
            if mask is not None:
                tmask = np.array(rotate(mask, 180))
        elif direction == 'r270':
            timg = np.array(rotate(img, 270))
            if mask is not None:
                tmask = np.array(rotate(mask, 270))
        else:
            timg = np.fliplr(img)
            if mask is not None:
                tmask = np.fliplr(mask)

        if rand_color is not None:
            timg[:, :, 0] = timg[:, :, 0]*rand_color[0]
            timg[:, :, 2] = timg[:, :, 2]*rand_color[1]
            # timg = timg*rand_color[2]
            timg = timg + rand_color[3]*160
            timg[timg > 255] = 255
            timg[timg < 0] = 0

        return np.array(timg, dtype=np.uint8), tmask

    def check_roi(self, index, ex_roi_list):
        """
        Returns True if the annotation lies inside of any of the ROIs;
        otherwise, returns False

        Args:
            index       <int>: annotation index
            ex_roi_list <list>: list of rois/annotations to review

        Returns:
            bool
        """
        assert isinstance(index, int)
        assert isinstance(ex_roi_list, list)

        if len(ex_roi_list) == 0:
            return False

        ann_bbox = BoundingBox(*self.get_annotation_bbox(index))

        for roi_index in ex_roi_list:
            if ann_bbox.is_inside(BoundingBox(*self.get_annotation_bbox(roi_index)), 0):
                return True

        return False

    def custom_iterator(self, iterable):
        """
        Lazy iterator based on the iterable received

        Args:
            iterable <list>: List of annotation indexes or RoiAnn objects

        Returns:
            roi_index, roi_bbox, ann_index
        """
        for obj in iterable:
            if isinstance(obj, RoiAnn):
                for ann_index in obj.ann_indexes:
                    yield obj.roi_index, obj.roi_bbox, ann_index
            else:
                yield -1, None, obj

    @classmethod
    def get_cleaned_label(cls, ann_name):
        """
        Tries to return  valid annotation label; otherwise, returns an empty string

        Args:
            ann_name <str>: annotation name obtained by self.get_aname method

        Returns:
            label <str>
        """
        assert isinstance(ann_name, str), type(ann_name)

        if ann_name in cls.excluded_ann:
            return ''

        label = Label.label_dict.get(ann_name, '')

        return label

    def get_ann_mgr(self, iterable=None):
        """
        Creates initialized instance of AnnotationMGR

        Args:
            iterable <list>: list created using the objects returned by the
                             self.custom_iterator method
        Returns:
            ann_mgr <AnnotationMGR>
        """
        if iterable is not None:
            assert isinstance(iterable, list), type(iterable)
        else:
            iterable = list(self.custom_iterator([*range(self.root_len)]))

        ann_mgr = AnnotationMGR()
        logger.info("Initializing the annotation manager")

        for _, _, ann_index in tqdm(iterable):
            ann_name = self.get_aname(ann_index)
            label = self.get_cleaned_label(ann_name)

            if not label:
                continue

            xi, xa, yi, ya, pl_arr = self.get_annotation_bbox(ann_index, return_polygon=True)
            ann_mgr.add(ann_index, label, BoundingBox(xi, xa, yi, ya), pl_arr)

        return ann_mgr

    def extract_annotations_and_masks(
            self, level=2, patch_size=640, saving_path='', min_mask_area=.2,
            apply_image_transforms=False, roi_clusters=None, inside_roi=False,
            central_area_offset=.215, background_threshold=.88, img_format='tiff',
            mask_format='png', multiple_ann=False, multiple_classes=False
    ):
        """
        Extracts and saves the image annotations and masks considering the provided arguments.
        If no roi clusters are provided then all the masks will be properly
        centred in the annotation images.
        When roi clusters are provided then some masks could be touching one of the
        annotation image. The aim is only show annotation images that are inside the ROI.

        Args:
            level                   <int>: magnification level (dim/2**level). Default 2
            patch_size              <int>: size of squared patch. Default 640
            saving_path             <str>: folder path to save the extracted annotations,
                                           Default f'{self.fileroot}/../annotations_masks'
            min_mask_area         <float>: min mask area in a annotation patches. Avoids creating
                                           patches with tiny or no mask on them. Default .2
            apply_image_transforms <bool>: whether or not apply image transforms. Default False
            roi_clusters           <list>: List of ROI clusters. Default None.
                                           E.g.: using only the first cluster
                                           CRLM_obj.cluster_annotations_by_roi()[:1]
            inside_roi             <bool>: Whether or not to have crops only from the inside of the ROI.
                                           This means that no crop border will be outside of the ROI.
                                           Default False
            central_area_offset   <float>: percentage to ignore before considering the central area
                            e.g.: if dim0=100, dim1=200 and central_area_offset=.2, the central area
                            will be image[20:80, 40:160]. Default .215
            background_threshold  <float>: value between 0 and 1 used to identify background pixels
                                           (the image is turned into gray scale before the comparison).
                                           Default .88
            img_format              <str>: Saving image extension. Default 'tiff'
            mask_format             <str>: Saving mask extension. Default 'png'
            multiple_ann           <bool>: Whether or not plot multiple annotations with the same label
                                           per crop. Default False
            multiple_classes       <bool>: Whether or not plot annotations from multiple classes per crop.
                                           Requires multiple_ann = True. Default False
        """
        assert isinstance(level, int)
        assert isinstance(patch_size, int)
        assert isinstance(saving_path, str)
        assert isinstance(min_mask_area, float)
        assert level >= 0
        assert patch_size >= 0
        assert min_mask_area > 0
        assert isinstance(apply_image_transforms, bool)
        assert isinstance(central_area_offset, float)
        assert 0 < central_area_offset < 1
        assert isinstance(background_threshold, float)
        assert 0 < background_threshold < 1
        assert isinstance(img_format, str)
        assert img_format != ''
        assert isinstance(mask_format, str)
        assert mask_format != ''
        assert isinstance(multiple_ann, bool), type(multiple_ann)
        assert isinstance(multiple_classes, bool), type(multiple_classes)

        if multiple_classes and not multiple_ann:
            raise MultipleClassesButNoMultipleAnnotationsMGR

        if roi_clusters is not None:
            assert isinstance(roi_clusters, list), type(roi_clusters)

        # NOTE: We're not including the case len(roi_clusters) == 0, because
        # some crlm ndpi files does no have ROIs. so roi_clusters could be
        # just and empty list
        if inside_roi and roi_clusters is None:
            raise NoRoiClusters

        if not saving_path:
            saving_path = os.path.join(self.fileroot, os.path.pardir, 'annotations_masks')

        if not os.path.isdir(saving_path):
            os.makedirs(saving_path)

        count = 0

        if roi_clusters is None or len(roi_clusters) == 0:
            iter_list = [*range(self.root_len)]
        else:
            iter_list = roi_clusters

        iterable = list(self.custom_iterator(iter_list))
        # NOTE: using the annotation manager with all the annotations to avoid missing
        #       extra annotations that partially fall into the crops
        # ann_mgr = self.get_ann_mgr(iterable) if multiple_ann else None
        ann_mgr = self.get_ann_mgr() if multiple_ann else None

        logger.info("Processing annotations")

        for roi_index, roi_bbox, ann_index in tqdm(iterable):
            ann_name = self.get_aname(ann_index)
            label = self.get_cleaned_label(ann_name)

            if not label:
                continue

            # TODO: review and improve how the offset if chosen
            # setting the right offset based on the magnification level
            if level == 2:
                xy_offset = 336 + 48
            elif level == 1:
                xy_offset = 112 + 48
            else:
                xy_offset = 48

            try:
                # TODO: This workaround will fail when the annotation is close to the
                # slide border and no ROI is provided; thus, the offset cannot grow enough.
                # In this border case one the offsets has to be increased more
                # to achieve the path_size on that dimension.
                # NOTE: In the CRLM dataset this border case seems not to be happening
                _, ann_img, ann_mask = self.extract_annotation_using_target_size(
                    ann_index, patch_size, xy_offset, xy_offset, level, roi_bbox, inside_roi,
                    ann_mgr=ann_mgr, multiple_classes=multiple_classes
                )
            except Exception as err:
                logger.warning(
                    f"While processing the annotation {ann_index} from file {self.filename} for the "
                    f"patch size {patch_size}, the following error was raised: {err}"
                )
                continue

            # TODO: should I filter also when the mask ann is too small????
            # Qianni said it's not necessary

            # TODO: review if this is necessary for tumour maybe not for my case
            if label == Label.tumour.file_label and \
               self.check_hole_in_tumour(ann_img, central_area_offset, background_threshold):
                logger.warning(
                    f"Annotation {ann_index} from file {self.filename} for patch size {patch_size} "
                    "looks like background"
                )
                continue

            if ann_img.shape[0] + ann_img.shape[1] <= (patch_size + patch_size) * 2:
                overlapping = ceil(patch_size * .75)
            elif ann_img.shape[0] + ann_img.shape[1] <= (patch_size + patch_size) * 4:
                overlapping = ceil(patch_size * .5)
            else:
                overlapping = ceil(patch_size * .25)

            for ix in get_slices_coords(ann_img.shape[0], patch_size, patch_overlapping=overlapping):
                for iy in get_slices_coords(ann_img.shape[1], patch_size, patch_overlapping=overlapping):
                    tim2_mask = ann_mask[ix:ix+patch_size, iy:iy+patch_size]

                    # making sure the patch contains more than min_mask_area of mask values
                    if np.sum(tim2_mask > 0) > (patch_size**2)*min_mask_area:
                        tim2 = ann_img[ix:ix+patch_size, iy:iy+patch_size]
                        annotation_saving_path = os.path.join(
                            saving_path,
                            '{}_f{:03d}_r{:02.0f}_a{:05d}_c{:05d}'.format(
                                label, self.index, roi_index, ann_index, count)
                        )

                        if apply_image_transforms:
                            tim2_, tim2_mask_ = self.flip_recolor_mix(tim2, tim2_mask)
                            cv2.imwrite(annotation_saving_path + '.ann.{}'.format(img_format),
                                        cv2.cvtColor(tim2_, cv2.COLOR_RGBA2BGRA))
                            cv2.imwrite(annotation_saving_path + '.mask.{}'.format(mask_format),
                                        tim2_mask_)
                        else:
                            cv2.imwrite(annotation_saving_path + '.ann.{}'.format(img_format),
                                        cv2.cvtColor(tim2, cv2.COLOR_RGBA2BGRA))
                            cv2.imwrite(annotation_saving_path + '.mask.{}'.format(mask_format),
                                        tim2_mask)

                        count += 1
