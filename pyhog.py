
import numpy as np
import features_pedro_py

try:
    from scipy.misc import imrotate
    imrotate_available = True
except ImportError:
    imrotate_available = False

def features_pedro(img, sbin):
    imgf = img.copy('F')
    hogf = features_pedro_py.process(imgf, sbin)
    return hogf

def hog_picture(w, bs=20):
    """ Visualize positive HOG weights.
    ported to numpy from https://github.com/CSAILVision/ihog/blob/master/showHOG.m
    """
    if not imrotate_available:
        raise RuntimeError('This function requires scipy')
    bim1 = np.zeros((bs, bs))
    bim1[:,round(bs/2)-1:round(bs/2)] = 1
    bim = np.zeros((9,)+bim1.shape)
    for i in range(9):
      bim[i] = imrotate(bim1, -i*20)/255.0
    s = w.shape
    w = w.copy()
    w[w < 0] = 0
    im = np.zeros((bs*s[0], bs*s[1]))
    for i in range(s[0]):
      iis = slice( i*bs, (i+1)*bs )
      for j in range(s[1]):
        jjs = slice( j*bs, (j+1)*bs )
        for k in range(9):
          im[iis,jjs] += bim[k] * w[i,j,k+18]
    return im/np.max(w)
