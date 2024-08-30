from cv_expt.utils.histogram import hist_eq, hist_gamma
import cv2
import numpy as np

def test_hist_eq():
    img = cv2.imread('./data/test_images/Yaks.jpg')
    assert img is not None
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    hist_eq_img = hist_eq(rgb_img)
    cv2.imwrite('imgs.png', np.hstack([rgb_img, hist_eq_img]))
    
    assert hist_eq_img.shape == rgb_img.shape    
    assert not np.array_equal(hist_eq_img, rgb_img)
    
    # create dummy image of 5 by 5
    test_img = np.array([[0, 1, 2, 3, 4],
                         [5, 6, 7, 8, 9],
                         [10, 11, 12, 13, 14],
                         [15, 16, 17, 18, 19],
                         [20, 21, 22, 23, 24]])
    real_hist_eq = np.array([[ 10,  20,  30,  40,  51],
                            [ 61,  71,  81,  91, 102],
                            [112, 122, 132, 142, 152],
                            [163, 173, 183, 193, 204],
                            [214, 224, 234, 244, 255]])   
    hist_eq_img = hist_eq(test_img)
    # print(hist_eq_img)
    # check if equalized value is correct
    assert np.array_equal(hist_eq_img, real_hist_eq)
    

def test_hist_gamma():
    img = cv2.imread('./data/test_images/Yaks.jpg')
    assert img is not None
    