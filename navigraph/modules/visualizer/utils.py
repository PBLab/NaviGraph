import cv2
import numpy as np
from numpy.typing import NDArray
import inspect


def overlay_img(background_img, overlay_img, resize_factor=0.3, opacity=0.5, frame_location='bottom_right', method='on_top') \
        -> NDArray:
    if method == 'side_by_side':
        height_ratio = background_img.shape[0] / overlay_img.shape[0]
        height = int(background_img.shape[0])
        width = int(background_img.shape[1] * height_ratio * 2)
        resized_map = cv2.resize(overlay_img, (width, height), interpolation=cv2.INTER_AREA)

        return np.concatenate((resized_map, background_img), axis=1)

    elif method == 'on_top':
        # convert a single axis then according to it the other to maintain map aspect ratio
        background_img_copy = background_img.copy()
        overlay_img_copy = overlay_img.copy()
        if overlay_img.shape[0] > overlay_img.shape[1]:
            height = int(background_img.shape[0] * resize_factor)
            height_ratio = height / overlay_img_copy.shape[0]
            width = int(overlay_img_copy.shape[1] * height_ratio)
        else:
            width = int(background_img.shape[1] * resize_factor)
            width_ratio = width / overlay_img_copy.shape[1]
            height = int(overlay_img_copy.shape[0] * width_ratio)

        resized_map = cv2.resize(overlay_img_copy, (height, width), interpolation=cv2.INTER_AREA)
        # localize map:
        if frame_location == 'bottom_right':
            background_img_copy[-resized_map.shape[0]:, -resized_map.shape[1]:] = resized_map
        elif frame_location == 'bottom_left':
            background_img_copy[-resized_map.shape[0]:, :resized_map.shape[1]] = resized_map
        else:
            raise ValueError(f'unsupported value for localize parameter - {frame_location}')

        cv2.addWeighted(background_img_copy, opacity, background_img, 1 - opacity, 0, background_img)

        return background_img

    else:
        NotImplementedError('overlay method not supported')


def get_default_args(func):
    signature = inspect.signature(func)
    return {
        k: v.default
        for k, v in signature.parameters.items()
        if v.default is not inspect.Parameter.empty
    }
