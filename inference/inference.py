import logging
import numpy as np
import scipy.signal.windows as w
from .mosaic import from_array
from ..utils.data import normalize_image


def window2d(window_func, window_size, **kwargs):
    window = np.matrix(window_func(M=window_size, sym=False, **kwargs))
    return window.T.dot(window)


def generate_corner_windows(window_func, window_size, **kwargs):
    step = window_size >> 1
    window = window2d(window_func, window_size, **kwargs)
    window_u = np.vstack([np.tile(window[step:step+1, :], (step, 1)), window[step:, :]])
    window_b = np.vstack([window[:step, :], np.tile(window[step:step+1, :], (step, 1))])
    window_l = np.hstack([np.tile(window[:, step:step+1], (1, step)), window[:, step:]])
    window_r = np.hstack([window[:, :step], np.tile(window[:, step:step+1], (1, step))])
    window_ul = np.block([
        [np.ones((step, step)), window_u[:step, step:]],
        [window_l[step:, :step], window_l[step:, step:]]])
    window_ur = np.block([
        [window_u[:step, :step], np.ones((step, step))],
        [window_r[step:, :step], window_r[step:, step:]]])
    window_bl = np.block([
        [window_l[:step, :step], window_l[:step, step:]],
        [np.ones((step, step)), window_b[step:, step:]]])
    window_br = np.block([
        [window_r[:step, :step], window_r[:step, step:]],
        [window_b[step:, :step], np.ones((step, step))]])
    return np.array([
        [ window_ul, window_u, window_ur ],
        [ window_l,  window,   window_r  ],
        [ window_bl, window_b, window_br ],
    ])


def generate_patch_list(image_width, image_height, window_func, window_size, overlapping=False):
    patch_list = []
    if overlapping:
        step = window_size >> 1
        windows = generate_corner_windows(window_func, window_size)
        max_height = int(image_height/step - 1)*step
        max_width = int(image_width/step - 1)*step
        # print("max_height, max_width", max_height, max_width)
    else:
        step = window_size
        windows = np.ones((window_size, window_size))
        max_height = int(image_height/step)*step
        max_width = int(image_width/step)*step
        # print("else max_height, max_width", max_height, max_width)
    
    #for i in range(0, max_height, step):
    #    for j in range(0, max_width, step):
    for i in range(0, image_height-step, step):
        for j in range(0, image_width-step, step):
            if overlapping:
                # Close to border and corner cases
                # Default (1, 1) is regular center window
                border_x, border_y = 1, 1
                if i == 0: border_x = 0
                if j == 0: border_y = 0
                if i == image_height-step: border_x = 2
                if j == image_width-step: border_y = 2
                # Selecting the right window
                current_window = windows[border_x, border_y]
            else:
                current_window = windows

            # The patch is cropped when the patch size is not
            # a multiple of the image size.
            patch_height = window_size
            if i+patch_height > image_height:
                patch_height = image_height - i
            
            patch_width = window_size
            if j+patch_width > image_width:
                patch_width = image_width - j
            
            #print(f'i {i} j {j} patch_height {patch_height} patch_width {patch_width}')

            # Adding the patch
            patch_list.append(
                (j, i, patch_width, patch_height, current_window[:patch_width, :patch_height])
            )
    return patch_list

def sliding_window(
            xraster, model, window_size, tile_size,
            inference_overlap, inference_treshold, batch_size,
            mean, std, n_classes, standardization, normalize,
            use_hanning=True
        ):

    original_shape = xraster[:, :, 0].shape

    xsum = int(((-xraster[:, :, 0].shape[0] % tile_size) + (tile_size * 4)) / 2)
    ysum = int(((-xraster[:, :, 0].shape[1] % tile_size) + (tile_size * 4)) / 2)
    #print("xsum", xsum, "ysum", ysum)

    xraster = np.pad(xraster, ((ysum, ysum), (xsum, xsum), (0, 0)),
       mode='symmetric')#'reflect')

    #print("RASTER SHAPE AFTER PAD", xraster.shape)

    # open rasters and get both data and coordinates
    rast_shape = xraster[:, :, 0].shape  # shape of the wider scene

    # in memory sliding window predictions
    wsy, wsx = window_size, window_size
    # wsy, wsx = rast_shape[0], rast_shape[1]

    # if the window size is bigger than the image, predict full image
    if wsy > rast_shape[0]:
        wsy = rast_shape[0]
    if wsx > rast_shape[1]:
        wsx = rast_shape[1]

    # smooth window
    # this might be problematic since there might be issues on tiles smaller
    # than actual squares
    # spline = spline_window(wsy)

    # print(rast_shape, wsy, wsx)
    prediction = np.zeros(rast_shape + (n_classes,))  # crop out the window
    logging.info(f'wsize: {wsy}x{wsx}. Prediction shape: {prediction.shape}')
    # 2022-03-09 13:47:03; INFO; wsize: 10000x10000. Prediction shape: (38702, 71223, 2)

    window_func = w.hann
    patch_list = generate_patch_list(
        rast_shape[0], rast_shape[1], window_func, wsy, use_hanning)
    pp = len(patch_list)
    #logging.info(f'Patch list done {pp}')

    counter = 0
    for patch in patch_list:
        
        counter += 1

        logging.info(f'{counter} out of {pp}')
        patch_x, patch_y, patch_width, patch_height, window = patch

        #if patch_x + patch_width > rast_shape[1]:
        #    patch_width = 

        #logging.info(f'{patch_x}, {patch_width+patch_x}, {patch_y}, {patch_height+patch_y}')
        #logging.info(f'{psutil.virtual_memory().percent}')
        
        input_path = xraster[
            patch_x:patch_x+patch_width, patch_y:patch_y+patch_height]
        
        #print("firts", input_path.shape)

        if np.all(input_path == input_path[0, 0, 0]):

            input_path_shape = input_path.shape
            prediction[
                patch_x:patch_x+patch_width,
                patch_y:patch_y+patch_height] = np.eye(n_classes)[
                    np.zeros((input_path_shape[0], input_path_shape[1]), dtype=int)]

        else:

            # Normalize values within [0, 1] range
            input_path = normalize_image(input_path, normalize)

            input_path = from_array(
                input_path, (tile_size, tile_size),
                overlap_factor=inference_overlap, fill_mode='reflect')

            input_path = input_path.apply(
                model.predict, progress_bar=False,
                batch_size=batch_size, mean=mean, std=std, standardization=standardization)

            input_path = input_path.get_fusion()

            prediction[
                patch_x:patch_x+patch_width,
                patch_y:patch_y+patch_height] += input_path * np.expand_dims(window, -1)

    if prediction.shape[-1] > 1:
        prediction = np.argmax(prediction, axis=-1)
        #print('Window shape before spline', window.shape, window_spline.shape)
        #window = window * window_spline
    else:
        prediction = np.squeeze(
            np.where(
                prediction > inference_treshold, 1, 0).astype(np.int16)
            )

    print("SHAPR PREDICTION", prediction.shape)

    prediction = prediction[xsum:rast_shape[0] - xsum, ysum:rast_shape[1] - ysum]

    print("SHAPR PREDICTION AFTER CROP", prediction.shape)


    return prediction