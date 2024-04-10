
import cv2
import numpy as np
from skimage import transform as trans


REFERENCE_FACIAL_POINTS = [
    [38.2946  , 51.6963  ],
    [73.5318  , 51.5014  ],
    [56.0252  , 71.7366  ],
    [41.5493  , 92.3655  ],
    [70.729904, 92.2041  ]
]


def align_image(
        src_img : np.array, 
        facial_pts : np.array,
        crop_size : tuple = (112, 112)
        ):
    """ Function for aligning input images according to the REFERENCE FACE POINTS.

    Args:
        src_img (np.array): Image which need to be aligned.
        facial_pts (np.array): Detected 5 landmark points.
        crop_size (tuple, optional): The size of the output aligned image. Defaults to (112, 112).

    Returns:
        np.array: Input image aligned according to the provided facial points.
    """

    reference_pts = REFERENCE_FACIAL_POINTS

    ref_pts = np.float32(reference_pts)
    ref_pts_shp = ref_pts.shape

    if ref_pts_shp[0] == 2:
        ref_pts = ref_pts.T

    src_pts = np.float32(facial_pts)
    src_pts_shp = src_pts.shape

    if src_pts_shp[0] == 2:
        src_pts = src_pts.T

    tform = trans.SimilarityTransform()
    tform.estimate(src_pts, ref_pts)
    tfm = tform.params[0:2, :]

    face_img = cv2.warpAffine(src_img, tfm, (crop_size[0], crop_size[1]))

    return face_img