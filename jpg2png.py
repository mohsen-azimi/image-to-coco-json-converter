import cv2
import glob
import numpy as np

for file in glob.glob('dataset/val_mask/*.jpg'):
    grayImage = cv2.imread(file, cv2.IMREAD_GRAYSCALE) # read in grayscale mode
    thresh = 127
    im_bw = cv2.threshold(grayImage, thresh, 255, cv2.THRESH_BINARY)[1]

    mask = im_bw
    # cv2.imshow("1", grayImage)
    # mask = grayImage<0.8
    (h, w) = mask.shape

    mask_colored = np.zeros([h, w, 3], dtype=np.uint8)
    mask_colored[:, :] = [255, 255, 255]

    mask2 = cv2.bitwise_and(mask_colored, mask_colored, mask=mask)
    # image = originalImage - res
    # image = np.where(image == 0, lab_background, image)


    # cv2.imshow("1",img[:,:,0])
    # cv2.imshow("2",img[:,:,1])
    # cv2.imshow("mask",mask2)
    print(file)
    cv2.imwrite(file[0:-4]+'.png', mask2)

    # cv2.waitKey(0)
# cv2.destroyAllWindows()