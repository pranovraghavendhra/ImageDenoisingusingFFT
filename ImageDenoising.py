
import cv2 
import numpy as np 
import matplotlib.pyplot as plt 
 
img = cv2.imread("images.jpg", cv2.IMREAD_GRAYSCALE) 
 
f = np.fft.fft2(img) 
fshift = np.fft.fftshift(f) 
 
def gaussian_lowpass(shape, cutoff): 
    rows, cols = shape 
    crow, ccol = rows // 2, cols // 2 
    x = np.linspace(-ccol, ccol, cols) 
    y = np.linspace(-crow, crow, rows) 
    X, Y = np.meshgrid(x, y) 
    d = np.sqrt(X**2 + Y**2) 
    gaussian_filter = np.exp(-(d**2) / (2*(cutoff**2))) 
    return gaussian_filter 
 
gauss_mask = gaussian_lowpass(img.shape, cutoff) 
fshift_filtered = fshift * gauss_mask 
f_ishift = np.fft.ifftshift(fshift_filtered) 
img_back = np.fft.ifft2(f_ishift) 
img_back = np.abs(img_back) 
 
plt.figure(figsize=(10,5)) 
plt.subplot(1,2,1) 
plt.title("Original Noisy Image") 
plt.imshow(img, cmap='gray') 
 
plt.subplot(1,2,2) 
plt.title("Denoised Image (FFT)") 
plt.imshow(img_back, cmap='gray') 
 
plt.tight_layout() 
plt.show()
