import pdb
import features_pedro_py

def features_pedro(img, sbin):
    imgf = img.copy('F')
    hogf = features_pedro_py.process(imgf, sbin)
    return hogf
