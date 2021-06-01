import cv2
import numpy as np

def correctColours(im1, im2, points):
    blurAmount = 0.5 * np.linalg.norm(np.array(points)[38] - np.array(points)[43])
    blurAmount = int(blurAmount)

    if blurAmount % 2 == 0:
        blurAmount += 1
  
    im1Blur = cv2.blur(im1, (blurAmount, blurAmount), 0)
    im2Blur = cv2.blur(im2, (blurAmount, blurAmount), 0)
  
    # Avoid divide-by-zero errors.
    im2Blur += (2 * (im2Blur <= 1)).astype(im2Blur.dtype)
  
    ret = np.uint8((im2.astype(np.float32) * im1Blur.astype(np.float32) / im2Blur.astype(np.float32)).clip(0,255))
    return ret