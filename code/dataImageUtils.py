# methods related to dealing with image data, from reading in images to predicting on them with models

import numpy as np
import plotEvalUtils as peu
from pyrsgis import raster
from pyrsgis.convert import changeDimension
from sklearn.utils import resample
import math

# global variables

input_bands = [i+1 for i in range(0,7)]
nBands = len(input_bands)
ndvi_band = 9
labels_band = 8

def setGlobalVariables(inputBands, n_Bands):
    
    global input_bands
    global nBands
    
    input_bands = inputBands
    nBands = n_Bands
    

def removeOuterEdges(x):
    '''Something is off with the top row in the satellite data, and sometimes the other edges, remove them.'''
    if x.ndim == 2: 
        x = np.delete(x, [0, x.shape[0]-1], axis=0) # remove top and bottom edges
        x = np.delete(x, [0, x.shape[1]-1], axis=1) # remove left and right edges
    else: 
        x = np.delete(x, [0, x.shape[1]-1], axis=1) # remove top and bottom edges
        x = np.delete(x, [0, x.shape[2]-1], axis=2) # remove left and right edges

    return x

def normalizeUInt16Band(band):
    '''Bands 1-7 are uint16, ranging from 0-65535, normalize them by dividing by the max.'''
    return band/65535.0


def processImage(image):
    '''Process an image and it's data for the fitWithBasicNN.ipynb notebook.'''
    
    # read in band data
    ds_features, features = raster.read(image, bands=input_bands) # if just inputting one band, do NOT put the single number in a list to pass to "bands", it causes some issue under the hood
    ds_labels, labels = raster.read(image, bands=labels_band)

    # remove outer edges of data (which sometimes have issues)
    features = removeOuterEdges(features)
    labels = removeOuterEdges(labels)

    # fill NaNs with 0s
    features = np.nan_to_num(features)
    labels = np.nan_to_num(labels)

    # normalize bands - doing this here caused issues with the model fitting for some reason
    # features = normalizeUInt16Band(features)

    # convert labels to int for classification
    labels = (labels == 1).astype(int)


    return features, labels, ds_labels

# method for loading multiple images as training data (with some portion set aside for testing data with same images)
def loadTrainingImages(images_list, downsampleMajority):
    '''Load images from list as training data, separately process each image and then concatenate the numpy arrays of training data.'''
    
    # initialize empty arrays which will concatenate data from all training images
    training_image_data = np.empty((0,nBands))
    training_image_labels = np.empty((0,))

    for i, image in enumerate(images_list):

        features, labels, _ = processImage(image)

        # make some plots just for the first training image
#         if i == 0:
#             ds_ndvi, features_ndvi = raster.read(image, bands=ndvi_band)
#             features_ndvi = removeOuterEdges(features_ndvi)
#             features_ndvi = np.nan_to_num(features_ndvi)
#             print('\nFirst training image NDVI band:')
#             peu.plotNVDIBand(features_ndvi) # plot NDVI band

#             print('\nFirst training image mangroves from labels: ')
#             peu.plotMangroveBand(labels) # plot label (mangrove) band

        # change dimensions for input into neural net
        features_input = changeDimension(features)
        labels_input = changeDimension(labels)

        # append image inputs together
        training_image_data = np.append(training_image_data, features_input, axis=0)
        training_image_labels = np.append(training_image_labels, labels_input, axis=0)

        # end for loop

        
    if downsampleMajority:

        # separate classes for downsampling

        mangrove_features = training_image_data[training_image_labels==1]
        mangrove_labels = training_image_labels[training_image_labels==1]

        non_mangrove_features = training_image_data[training_image_labels==0]
        non_mangrove_labels = training_image_labels[training_image_labels==0]

        # down-sample the non-mangrove class (which should be the majority)
        # sample without replacement, match minority n, make sure random seed is the same for features and labels
        non_mangrove_features = resample(non_mangrove_features, replace = False, n_samples = mangrove_features.shape[0], random_state = 6792)
        non_mangrove_labels = resample(non_mangrove_labels, replace = False, n_samples = mangrove_features.shape[0], random_state = 6792)

        # recombine features and labels

        features = np.concatenate((mangrove_features, non_mangrove_features), axis=0)
        labels = np.concatenate((mangrove_labels, non_mangrove_labels), axis=0)

    else:

        features = training_image_data
        labels = training_image_labels

    # check balance of classes
    training_data_length = len(features)
    print('Using training data of length: ', training_data_length)
    print(f"Class 0: {np.count_nonzero(labels==0)} Class 1: {np.count_nonzero(labels==1)}")
    print(f"Class 0: {100 * np.count_nonzero(labels==0)/training_data_length : .1f}% Class 1: {100 * np.count_nonzero(labels==1)/training_data_length : .1f}%")

    return features, labels




# method for predicting on an image with the trained model
def predictOnImage(model, image):
    '''Take trained model and apply it to a new image.'''
    
    print('\n\nPredicting for image:', image)
    name = image.split("/")[-1].split(".")[0][:-5]
    year = int(image.split("/")[-1].split("_")[2].split(".")[0])

    features_new, labels_new, ds_labels_new = processImage(image)
    
    # plot NDVI band (if using it)
    ds_ndvi, features_ndvi = raster.read(image, bands=ndvi_band)
    features_ndvi = removeOuterEdges(features_ndvi)
    features_ndvi = np.nan_to_num(features_ndvi)
    print(f'\nImage {year} NDVI band:')
    peu.plotNVDIBand(features_ndvi, name, year, "BasicNN") # plot NDVI band

    # plot labeled Mangrove band
    print('\nLabel mangroves from 2000 data:')
    peu.plotMangroveBand(labels_new, name, 2000, False, "BasicNN")

    # change dimensions of input
    features_new_1D = changeDimension(features_new)
    labels_new_1D = changeDimension(labels_new)

    # reshape it as an additional step for input into the NN
    features_new_1D = features_new_1D.reshape((features_new_1D.shape[0], 1, nBands))

    # normalize bands for new image
    features_new_1D = normalizeUInt16Band(features_new_1D)

    # predict on new image
    predicted_new_image_prob = model.predict(features_new_1D)
    predicted_new_image_prob = predicted_new_image_prob[:,1]

    # set probability threshold 
    probThresh = 0.5
    
    # reshape prediction into 2D for plotting
    predicted_new_image_aboveThresh = (predicted_new_image_prob > probThresh).astype(int)
    prediction_new_image_2d = np.reshape(predicted_new_image_aboveThresh, (ds_labels_new.RasterYSize-2, ds_labels_new.RasterXSize-2)) # need the -2s since I removed the outer edges

    # plot predicted mangroves
    print('\nPredicted mangroves:')
    peu.plotMangroveBand(prediction_new_image_2d, name, year, True, "BasicNN")

    # plot difference in predicted and labeled, or future vs past labeled
    if year > 2000: print(f'\nDifference between {year} predicted and 2000 labeled mangroves:')
    else: print('\nDifference between predicted and labeled mangroves from the year 2000:')
    peu.plotDifference(labels_new, prediction_new_image_2d, name, year, "BasicNN")

    # print classification metrics
    if year == 2000:
        peu.printClassificationMetrics(labels_new_1D, predicted_new_image_prob, probThresh)
        peu.makeROCPlot(labels_new_1D, predicted_new_image_prob, name, year, "BasicNN")
    

def processImageCNN(image, kSize, stride):
    '''Process an image and it's data for the fitWithCNN.ipynb notebook.'''

    margin = math.floor(kSize/2)

    # read in band data
    ds_features, features = raster.read(image, bands=input_bands)
    ds_labels, labels = raster.read(image, bands=labels_band)
    
    # remove outer edges of data (which sometimes have issues)
    features = removeOuterEdges(features)
    labels = removeOuterEdges(labels)

    # fill NaNs with 0s
    features = np.nan_to_num(features)
    labels = np.nan_to_num(labels)

    # normalize bands
    features = normalizeUInt16Band(features)
    
    # turn labels to ints
    labels = (labels == 1).astype(int)
          
    # get dimensions for creating 7x7 feature arrays
    _, rows, cols = features.shape
    features = np.pad(features, margin, mode='constant')[margin:-margin, :, :]
    
    
    # perhaps can somehow use these shapes to get number of points for final features vector (but still doesn't work for stride=1 or stride=2)
#     fill_rows = int((features.shape[1]-margin)/stride)
#     fill_cols = int((features.shape[2]-margin)/stride)
    
    
    # init empty array for filling with the proper shape for input into the CNN
        # something not quite right with the current implementation, hence the if statements and exception raising
        # works for odd and even strides in some instances but not all
    if stride == 1: features_shaped = np.empty((rows*cols, kSize, kSize, nBands))
#     elif stride % 2 == 1: features_shaped = np.empty((int((rows+margin)/stride)*int((cols+margin)/stride), kSize, kSize, nBands))
#     else: raise Exception("Stride cannot be even with the current implementation.")
    else: raise Exception("Stride cannot be a number different than 1 with the current implementation.")
    
    n = 0
    for row in range(margin, rows+margin, stride):    
        for col in range(margin, cols+margin, stride):            
            feat = features[:, row-margin:row+margin+1, col-margin:col+margin+1]

            b1, b2, b3, b4, b5, b6, b7 = feat # this is hardcoded at the moment which isn't great
            feat = np.dstack((b1, b2, b3, b4, b5, b6, b7))

            features_shaped[n, :, :, :] = feat
            n += 1
        
        
    return features_shaped, labels, ds_labels
    


# # method for loading multiple images as training data (with some portion set aside for testing data with same images)
def loadTrainingImagesCNN(images_list, downsampleMajority, kSize, stride):
    '''Load images from list as training data, separately process each image and produce image chips.'''
    
    # initialize empty arrays which will concatenate data from all training images
    training_image_data = np.empty((0, kSize, kSize, nBands))
    training_image_labels = np.empty((0,))

    for i, image in enumerate(images_list):
        
        print('Processing image: ', i)

        features, labels, _ = processImageCNN(image, kSize, stride)

        # make some plots just for the first training image
#         if i == 0:
#             ds_ndvi, features_ndvi = raster.read(image, bands=ndvi_band)
#             features_ndvi = removeOuterEdges(features_ndvi)
#             features_ndvi = np.nan_to_num(features_ndvi)
#             print('\nFirst training image NDVI band:')
#             peu.plotNVDIBand(features_ndvi) # plot NDVI band

#             print('\nFirst training image mangroves from labels: ')
#             peu.plotMangroveBand(labels) # plot label (mangrove) band

            
        # apply stride to labels
        labels = labels[::stride,::stride]

        # change dimension of labels array
        labels = changeDimension(labels)  
            
        # append image inputs together
        training_image_data = np.append(training_image_data, features, axis=0)
        training_image_labels = np.append(training_image_labels, labels, axis=0)

        # end for loop

        
    if downsampleMajority:

        # separate classes for downsampling

        mangrove_features = training_image_data[training_image_labels==1]
        mangrove_labels = training_image_labels[training_image_labels==1]

        non_mangrove_features = training_image_data[training_image_labels==0]
        non_mangrove_labels = training_image_labels[training_image_labels==0]

        # down-sample the non-mangrove class (which should be the majority) - sample without replacement, match minority n, make sure random seed is the same for features and labels
        non_mangrove_features = resample(non_mangrove_features, replace = False, n_samples = mangrove_features.shape[0], random_state = 6792)
        non_mangrove_labels = resample(non_mangrove_labels, replace = False, n_samples = mangrove_features.shape[0], random_state = 6792)

        # recombine features and labels

        final_features = np.concatenate((mangrove_features, non_mangrove_features), axis=0)
        final_labels = np.concatenate((mangrove_labels, non_mangrove_labels), axis=0)

    else:

        final_features = training_image_data
        final_labels = training_image_labels

    # check balance of classes
    training_data_length = len(final_features)
    print('Using training data of length: ', training_data_length)
    print(f"Class 0: {np.count_nonzero(final_labels==0)} Class 1: {np.count_nonzero(final_labels==1)}")
    print(f"Class 0: {100 * np.count_nonzero(final_labels==0)/training_data_length : .1f}% Class 1: {100 * np.count_nonzero(final_labels==1)/training_data_length : .1f}%")

    return final_features, final_labels



def predictOnImageCNN(model, image, kSize):
    '''Take trained CNN model and apply it to a new image.'''
    
    print('\n\nPredicting for image:', image)
    name = image.split("/")[-1].split(".")[0][:-5]
    year = int(image.split("/")[-1].split("_")[2].split(".")[0])
    
    features_new, labels_new, ds_labels_new = processImageCNN(image, kSize, 1)

    # plot NDVI band
    ds_ndvi, features_ndvi = raster.read(image, bands=ndvi_band)
    features_ndvi = removeOuterEdges(features_ndvi)
    features_ndvi = np.nan_to_num(features_ndvi)
    print(f'\nImage {year} NDVI band:')
    peu.plotNVDIBand(features_ndvi, name, year, "CNN") # plot NDVI band

    # plot labeled Mangrove band
    print('\nLabel mangroves from 2000 data:')
    peu.plotMangroveBand(labels_new, name, 2000, False, "CNN")
        
    # change dimension of labels array
    labels_new_1D = changeDimension(labels_new)
    
    # predict on new image
    predicted_new_image_prob = model.predict(features_new)
    predicted_new_image_prob = predicted_new_image_prob[:,1]

    # set probability threshold
    probThresh = 0.5

    # reshape prediction into 2D for plotting
    predicted_new_image_aboveThresh = (predicted_new_image_prob > probThresh).astype(int)
    prediction_new_image_2d = np.reshape(predicted_new_image_aboveThresh, (ds_labels_new.RasterYSize-2, ds_labels_new.RasterXSize-2)) # need the -2s since I removed the outer edges
    
    # plot predicted mangroves
    print('\nPredicted mangroves:')
    peu.plotMangroveBand(prediction_new_image_2d, name, year, True, "CNN")

    # plot difference in predicted and labeled, or future vs past labeled
    if year > 2000: print(f'\nDifference between {year} predicted and 2000 labeled mangroves:')
    else: print('\nDifference between predicted and labeled mangroves from the year 2000:')
    peu.plotDifference(labels_new, prediction_new_image_2d, name, year, "CNN")

    # print classification metrics
    if year == 2000:
        peu.printClassificationMetrics(labels_new_1D, predicted_new_image_prob, probThresh)
        peu.makeROCPlot(labels_new_1D, predicted_new_image_prob, name, year, "CNN")
    