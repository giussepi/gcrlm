# -*- coding: utf-8 -*-
""" constants """

from collections import namedtuple

from matplotlib.colors import ListedColormap

from gcrlm.core.exceptions.crlm import WrongCRLMProcessingType


Detail = namedtuple('Detail', ['colour', 'id', 'name', 'file_label'])


class Label:
    """ Holds tissue colours, ids and label names """
    hepatocyte = Detail('pink', 0, 'Hepatocyte', 'H')
    necrosis = Detail('c', 1, 'Necrosis', 'N')
    fibrosis = Detail('lightgrey', 2, 'Fibrosis', 'F')
    tumour = Detail('saddlebrown', 3, 'Tumour', 'T')
    inflammation = Detail('g', 4, 'Inflammation', 'I')
    mucin = Detail('r', 5, 'Mucin', 'M')
    blood = Detail('purple', 6, 'Blood', 'B')
    foreign_body = Detail('royalblue', 7, 'Foreign body', 'D')   # fb foreign blood reaction
    macrophages = Detail('k', 8, 'Macrophages', 'C')
    background = Detail('white', 9, 'Background', 'G')
    bile_duct = Detail('gold', 10, 'Bile Duct', 'Y')
    # extra label only to identify the non-metastatic cancer
    non_metastatic = Detail('white', 12, 'No metastatic cancer ', '')
    # invasive front labels
    demostic = Detail('green', 13, 'Demostic', '')
    replacement = Detail('blue', 14, 'Replacement', '')
    portal_area = Detail('white', 15, 'Portal Area', '')
    other = Detail('red', 16, 'Other', '')

    ###########################################################################
    #                              IMPORTANT NOTE                             #
    # The labels attribute must follow the order of their ids as defined above#
    # If the order changes, you'll have to retrain your model; and update     #
    # any value that depends on the Label ids. Furthermore, there are several #
    # scripts, that depends on the hardcoded value of file_labels, which have #
    # not been updated to consider main.py as the project root nor have been  #
    # refactored to use file_labels value from this class. To see them just   #
    # look for all the project files containing label_name_list and           #
    # ['H', 'N', 'F', 'T', 'I', 'M', 'B', 'D', 'C', 'G', 'Y'].                #
    #                                                                         #
    # All those files will need to be updated to (if you want them working    #
    # good). Thus, it is better to avoid any modification on the order of the #
    # labels and their ids                                                    #
    ###########################################################################
    labels = (hepatocyte, necrosis, fibrosis, tumour, inflammation, mucin, blood,
              foreign_body, macrophages, background, bile_duct)

    # CMAP = ListedColormap(['pink', 'c', 'lightgrey', 'saddlebrown', 'g', 'r', 'purple',
    #                        'royalblue', 'k', 'white', 'gold'])
    cmap = ListedColormap([label.colour for label in labels])

    # CMAP2 = ListedColormap(['pink', 'c', 'lightgrey', 'saddlebrown', 'g', 'r', 'purple',
    #                         'royalblue', 'k', 'white', 'gold', 'white'])
    cmap2 = ListedColormap(cmap.colors + [background.colour])

    # 'white', 'c', 'lightgrey', 'saddlebrown', 'g'
    short_cmap = ListedColormap([
        background.colour, necrosis.colour, fibrosis.colour, tumour.colour, inflammation.colour])

    ids = tuple(label.id for label in labels)

    # compatible with cmap2 colormap
    metastasis_ids = ids + tuple([non_metastatic.id])

    inv_base = [demostic, replacement, portal_area, other]
    inv_base_label_ids = tuple(elem.id for elem in inv_base)
    inv_base_label_colours = tuple(elem.colour for elem in inv_base)
    # cmap inv front
    inv_cmap = ListedColormap(['indigo', 'gold'] + ['black']*11 + list(inv_base_label_colours))

    inv_front_ids = tuple([demostic.id, replacement.id, portal_area.id, other.id])

    names = tuple(label.name for label in labels)

    # ['H', 'N', 'F', 'T', 'I', 'M', 'B', 'D', 'C', 'G', 'Y']
    file_labels = tuple(label.file_label for label in labels)

    # {'H': 'H', 'N': 'N', 'F': 'F', 'T': 'T', 'I': 'I',
    #  'M': 'M', 'B': 'B', 'FB': 'D', 'MF': 'C', 'G': 'G', 'BD': 'Y'}
    label_dict = dict(zip(['H', 'N', 'F', 'T', 'I', 'M', 'B', 'FB', 'MF', 'G', 'BD'], file_labels))

    label_mask_color_dict = dict(zip(file_labels, range(23, 255, 23)))


class CRLMProcessingTypes:
    """ Holds the types of processing for CRLM dataset """
    ALL_WITHOUT_ROIS = 0
    # Processes all annotations without considering ROIs

    ALL_WITH_ROIS = 1
    # Processes all annotations but considering the ROIs bounding boxes
    # as limits for those annotations lying inside a ROI. In other words,
    # all annotations inside a ROI will have crops only from the inside
    # of their corresponding ROIS. This is good to create masks limited by
    # ROI bouding boxes.

    ONLY_ROIS = 2
    # The processing is similar to the option ALL_WITH_ROIS but this one
    # only processes annotations lying inside a ROI.

    OPTIONS = (ALL_WITHOUT_ROIS, ALL_WITH_ROIS, ONLY_ROIS)

    @classmethod
    def validate_option(cls, option):
        """
        Validate that the provided option is one of the defined ones. It it is not, then
        it raises the WrongCRLMProcessingType exception

        Args:
            option <int>: option to be validated
        """
        assert isinstance(option, int), \
            "The option must be of type int. See Data.dataset_processors.constants.CRLMProcessingTypes"

        if option not in cls.OPTIONS:
            raise WrongCRLMProcessingType()
