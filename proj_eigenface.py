import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

import os
from sklearn import svm
import time


def read_pgm(path, filename, byteorder='>'):
    """Return image data from a raw PGM file as numpy array.

    Format specification: http://netpbm.sourceforge.net/doc/pgm.html

    """
    with open(path + filename, 'rb') as pgmf:
        pgm_type = pgmf.readline()
        (width, height) = [int(i) for i in pgmf.readline().split()]
        maxval = int(pgmf.readline())
        assert maxval <= 255
        raster = []

        if pgm_type == 'P2\n':   # plain pgm
            line = pgmf.readline()
            while line:
                line = line.split()
                for x in line:
                    raster.append(int(x))
                line = pgmf.readline()

        else:
            if pgm_type == 'P5\n':
                for y in range(height):
                    row = []
                    for x in range(width):
                        row.append(ord(pgmf.read(1)))
                    raster.append(row)

        raster = np.asarray(raster).reshape((height, width))
        #plt.imshow(raster, plt.cm.gray)
        return raster


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

    i = 2
    for eigenvector in pca.components_:
        #plt.figure(i)
        #plt.imshow(eigenvector.reshape((height, width)), plt.cm.gray)
        i += 1
    #plt.show()

    features = images_centered.dot(pca.components_.T)

    return mean_face, features, pca.components_


if __name__ == "__main__":
    start = time.time() * 1000
    #images, n, height, width, labels = read_image_folder('/Users/ANNIE/DOWNLOADS/faces_4/an2i/')
    images, n, height, width, labels = read_image_folder('./FACES/')
    #images, n, height, width, labels = read_image_folder('./FACES_M/')
    print("Dataset consists of %d faces" % n)
    print('Loading images time: %f' % (time.time() * 1000 - start))

    efs_perfm = []
    axis = []
    for n_efs in range(4, 52):
        mean_face, features, eigenvectors = PCA_down_dim(images, n_efs)
        # plt. figure(1)
        # plt.imshow(mean_face.reshape((height, width)), plt.cm.gray)

        clf = svm.SVC(kernel='linear', C=1.0)

        #print('learning EFS features...')
        clf.fit(features, labels)
        #print('predicting...')
        pred_by_efs = clf.predict(features)
        # print(pred_by_efs)
        loss = np.sum(np.abs(np.subtract(pred_by_efs, labels))) / float(n)
        #print('0/1 loss of EFS feature predicting is: %f' % loss)

        images_test, n_test, height_test, width_test, labels_test = read_image_folder('./VALIDATION/')
        #print("Test Dataset consists of %d faces" % n_test)
        #print('On validation set...')
        mean_face_test = np.mean(images_test, axis=0)
        images_centered_test = images_test - mean_face_test
        features_test = images_centered_test.dot(eigenvectors.T)
        #print('predicting...')
        pred_by_efs_test = clf.predict(features_test)
        # print(pred_by_efs)
        loss = np.sum(np.abs(np.subtract(pred_by_efs_test, labels_test))) / float(n_test)
        #print('0/1 loss of EFS feature predicting is: %f' % loss)
        efs_perfm.append(loss)
        axis.append(n_efs)

    print('learning PXL features...')
    clf.fit(images, labels)
    print('predicting...')
    pred_by_pxl = clf.predict(images)
    # print(pred_by_efs)
    loss = np.sum(np.abs(np.subtract(pred_by_pxl, labels))) / float(n)
    print('0/1 loss of PXL feature predicting is: %f' % loss)

    print('predicting on validation set...')
    pred_by_pxl_test = clf.predict(images_test)
    # print(pred_by_efs)
    loss = np.sum(np.abs(np.subtract(pred_by_pxl_test, labels_test))) / float(n_test)
    print('0/1 loss of PXL feature predicting is: %f' % loss)

    plt.figure(2)
    plt.plot(axis, efs_perfm)
    plt.xlabel('number of eigenfaces')
    plt.ylabel('prediction 0/1 loss on validation set')
    plt.show()