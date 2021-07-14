# -*- coding: utf-8 -*-
""" gcrlm/core/exceptions/crlm """


class NoRoiClusters(RuntimeError):
    """
    Exception to be raised by
    Data/Prepare_patches/CRLM.CRLM.extract_annotations_and_masks setting
    inside_roi to True but not providing roi_clusters
    """

    def __init__(self, message=''):
        """ Initializes the instance with a custom message """
        if not message:
            message = 'When inside_roi is set to True, you must provide roi_clusters.'
        super().__init__(message)


class WrongCRLMProcessingType(ValueError):
    """
    Exception to be raised when an undefined option to process the CRLM dataset is
    provided.
    """

    def __init__(self, message=''):
        """ Initializes the instance with a custom message """
        if not message:
            message = ('Invalid option to process the CRLM dataset. See '
                       'Data.dataset_processors.constants.CRLMProcessingTypes.')
        super().__init__(message)


class MultipleClassesButNoAnnotaionMGR(RuntimeError):
    """
    Exception to be raised when multiple_classes is set to True, but no AnnotationMGR
    is provided
    """

    def __init__(self, message=''):
        """ Initializes the instance with a custom message """
        if not message:
            message = ('multiple_classes = True requires an instance of AnnotationMGR in ann_mgr')
        super().__init__(message)


class MultipleClassesButNoMultipleAnnotationsMGR(RuntimeError):
    """
    Exception to be raised when multiple_classes is set to True, but multiple_ann is False
    """

    def __init__(self, message=''):
        """ Initializes the instance with a custom message """
        if not message:
            message = ('multiple_classes = True requires multiple_ann = True')
        super().__init__(message)


class SampleFolderAlreadyExists(RuntimeError):
    """
    Exception to be raised by Data.dataset_processors.crlm.CRLMCropRandSample when the
    sample folder already exists.
    """

    def __init__(self, message=''):
        """ Initializes the instance with a custom message """
        if not message:
            message = (
                'The specified saving_path already exits. If you created that folder by moving '
                'the files, please move its contents back to the original folder, and then '
                'proceed to delete the empty saving_path folder. Else, if you created that folder'
                'by copying the files, then just remove the folder.'
                'Finally, try running this class again.'
            )
        super().__init__(message)
