import numpy as np
import plotEvalUtils as peu
from pyrsgis import raster
from pyrsgis.convert import changeDimension
from sklearn.utils import resample


# global variables

main_bands = [i+1 for i in range(0,7)]
ndvi_band = 9
labels_band = 8
input_bands = main_bands
nBands = len(input_bands)
downsampleMajority = False

def setGlobalVariables(inputBands, n_Bands, downsample_Majority):
    
    global input_bands
    global nBands
    global downsampleMajority
    
    input_bands = inputBands
    nBands = n_Bands
    downsampleMajority = downsample_Majority
    
# define functions for use in the notebook - these should be moved to a separate file but making that work with colab is turning out to be a real headache

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
    return band/65535.



# method for loading multiple images as training data (with some portion set aside for testing data with same images)
def loadTrainingImages(images_list):
    '''Load images from list as training data, separately process each image and then concatenate the numpy arrays of training data.'''
    
    # initialize empty arrays which will concatenate data from all training images
    training_image_data = np.empty((0,nBands))
    training_image_labels = np.empty((0,))

    for i, image in enumerate(images_list):
        # read in band data
        ds_features, features = raster.read(image, bands=input_bands) # if just inputting one band, do NOT put the single number in a list to pass to "bands", it causes some issue under the hood
        ds_labels, labels = raster.read(image, bands=labels_band)

        # remove outer edges of data (which sometimes have issues)
        features = removeOuterEdges(features)
        labels = removeOuterEdges(labels)

        # fill NaNs with 0s
        features = np.nan_to_num(features)
        labels = np.nan_to_num(labels)

        # print('Feature shape: ', features.shape)

        # make some plots just for the first training image
        if i == 0:
            if input_bands == ndvi_band: 
                print('\nFirst training image NDVI band:')
                peu.plotNVDIBand(features) # plot NDVI band
            print('\nFirst training image mangroves from labels: ')
            peu.plotMangroveBand(labels) # plot label (mangrove) band

        # change dimensions for input into neural net
        features_input = changeDimension(features)
        labels_input = changeDimension(labels)
        # print(features_input.shape)
        # print(labels_input.shape)

        # convert labels to int for classification
        labels_input = (labels_input == 1).astype(int)

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

        # down-sample the non-mangrove class (which should be the majority) - sample without replacement, match minority n, make sure random seed is the same for features and labels
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
    # read in band data
    ds_features_new, features_new = raster.read(image, bands=input_bands)
    ds_labels_new, labels_new = raster.read(image, bands=labels_band)

    # remove outer edges of data (which sometimes have issues)
    features_new = removeOuterEdges(features_new)
    labels_new = removeOuterEdges(labels_new)

    # fill NaNs with 0s
    features_new = np.nan_to_num(features_new)
    labels_new = np.nan_to_num(labels_new)

    # change label from float to int
    labels_new = (labels_new == 1).astype(int)

    # print('Check shapes:', features_new.shape, labels_new.shape)

    # plot NDVI band (if using it)
    if input_bands == ndvi_band: 
        print('\nNDVI band:')
        peu.plotNVDIBand(features_new)

    # plot Mangrove band
    print('\nLabel mangroves from 2000 data:')
    peu.plotMangroveBand(labels_new)

    # change dimensions of input
    features_new_input = changeDimension(features_new)
    labels_new_input = changeDimension(labels_new)

    # reshape it as an additional step for input into the NN
    features_new_input = features_new_input.reshape((features_new_input.shape[0], 1, nBands))
    # print('Check transformed shapes:', features_new_input.shape, labels_new_input.shape)

    # normalize bands for new image if using the main bands
    if input_bands == main_bands:
        features_new_input = normalizeUInt16Band(features_new_input)

    # predict on new image
    predicted_new_image_prob = model.predict(features_new_input)
    predicted_new_image_prob = predicted_new_image_prob[:,1]

    # print classification metrics
    probThresh = 0.5
    peu.printClassificationMetrics(labels_new_input, predicted_new_image_prob, probThresh)
    peu.makeROCPlot(labels_new_input, predicted_new_image_prob)

    # reshape prediction into 2D for plotting
    predicted_new_image_aboveThresh = (predicted_new_image_prob > probThresh).astype(int)
    prediction_new_image_2d = np.reshape(predicted_new_image_aboveThresh, (ds_labels_new.RasterYSize-2, ds_labels_new.RasterXSize-2)) # need the -2s since I removed the outer edges

    # plot predicted mangroves
    print('\nPredicted mangroves:')
    peu.plotMangroveBand(prediction_new_image_2d)

    # plot difference in predicted and labeled, or future vs past labeled
    print('\nDifference between predicted and labeled mangroves:')
    peu.plotDifference(labels_new, prediction_new_image_2d)
