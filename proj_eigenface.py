import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import re
import numpy
import os
from sklearn import svm

def read_pgm(path, filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(path + filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    return numpy.frombuffer(buffer,
                            dtype='u1' if int(maxval) < 256 else byteorder+'u2',
                            count=int(width)*int(height),
                            offset=len(header)
                            ).reshape((int(height), int(width)))


def read_image_folder(folderpath='.'):
    subname = 'sunglasses'
    label_sunglasses = []

    images = []
    h = 0;
    w = 0;
    num_images = 0
    listing = os.listdir(folderpath)
    for file in listing:
        name, extension = os.path.splitext(file)
        if extension == '.pgm':
            im = read_pgm(folderpath, file)
            im_vector = np.reshape(im, (1, im.shape[0] * im.shape[1]))
            images.append(im_vector)
            if subname in name:
                label_sunglasses.append(1)
            else:
                label_sunglasses.append(0)
            if num_images == 0:
                h = im.shape[0]
                w = im.shape[1]
            num_images += 1
    images = np.asarray(images).reshape((num_images, h * w))
    label_sunglasses = np.asarray(label_sunglasses)
    #label_sunglasses = label_sunglasses.T
    #print (label_sunglasses)
    return images, num_images, h, w, label_sunglasses

## input: original data matrix [n by m] (n is number of images; m is the size of each image vector)
##        n_PCA_components (number of dimensions you want in the end; default 20)
## output: mean_face (a mean face vector of size m)
##         features (a new feature matrix [n by n_PCA_components])
def PCA_down_dim(images, n_PCA_components=20):
    # n_PCA_components = min(n, height * width)
    print("Features compressed to %d eigenfaces" % n_PCA_components)
    mean_face = np.mean(images, axis=0)
    images_centered = images - mean_face
    pca = PCA(n_PCA_components)
    pca.fit(images_centered)

    #i = 2
    eigenvectors = []
    for eigenvector in pca.components_:
        eigenvectors.append(eigenvector)
        #plt.figure(i)
        #plt.imshow((eigenvector + mean_face).reshape((height, width)), plt.cm.gray)
        #i += 1
    eigenvectors = np.asarray(eigenvectors)
    #plt.show()

    features = images.dot(eigenvectors.T)

    return mean_face, features

if __name__ == "__main__":

    #images, n, height, width, labels = read_image_folder('/Users/ANNIE/DOWNLOADS/faces_4/an2i/')
    images, n, height, width, labels = read_image_folder('./FACES/')
    print("Dataset consists of %d faces" % n)

    mean_face, features = PCA_down_dim(images, 10)
    #plt. figure(1)
    #plt.imshow(mean_face.reshape((height, width)), plt.cm.gray)

    clf = svm.SVC(kernel='linear', C=1.0)


    print('learning EFS features...')
    clf.fit(features, labels)
    print('predicting...')
    pred_by_efs = clf.predict(features)
    print(pred_by_efs)
    loss = np.sum( np.abs( np.subtract(pred_by_efs, labels) ) ) / float(n)
    print('0/1 loss of EFS feature predicting is: %f' % loss)

    print('learning PXL features...')
    clf.fit(images, labels)
    print('predicting...')
    pred_by_pxl = clf.predict(images)
    print(pred_by_efs)
    loss = np.sum(np.subtract(pred_by_pxl, labels)) / float(n)
    print('0/1 loss of PXL feature predicting is: %f' % loss)



