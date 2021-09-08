import numpy as np
import cv2
def diezmado(image, D):


    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    image_decimated = image_gray[::D, ::D]

    orientation_mask = np.zeros_like(image_decimated)
    orientation_mask[int(1/D), int(1/D)] = 1
    image_gray_fft = np.fft.fft2(image_decimated)
    image_gray_fft_shift = np.fft.fftshift(image_gray_fft)
    # Aplicacion del filtro a la imagen transformada de fouriere
    mask = orientation_mask  # can also use high or band pass mask
    fft_filtered = image_gray_fft_shift * mask
    image_filtered = np.fft.ifft2(np.fft.fftshift(fft_filtered))
    image_filtered = np.absolute(image_filtered)
    image_filtered /= np.max(image_filtered)

    return image_filtered
def interpolacion(image, i):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image_gray_fft = np.fft.fft2(image_gray)
    image_gray_fft_shift = np.fft.fftshift(image_gray_fft)



    # insert zeros
    rows, cols = image_gray_fft_shift.shape
    num_of_zeros = i-1
    image_zeros = np.zeros((num_of_zeros * rows, num_of_zeros * cols), dtype=image_gray_fft_shift.dtype)
    image_zeros[::num_of_zeros, ::num_of_zeros] = image_gray_fft_shift
    W = 2 * num_of_zeros + 1

    # filtering
    orientation_mask = np.zeros_like(image_gray_fft_shift)
    orientation_mask[int(1 / i), int(1 / i)] = 1
    mask = orientation_mask  # can also use high or band pass mask
    fft_filtered =  image_zeros * mask
    image_filtered = np.fft.ifft2(np.fft.fftshift(fft_filtered))
    image_filtered = np.absolute(image_filtered)
    image_filtered /= np.max(image_filtered)
    return image_filtered
    
