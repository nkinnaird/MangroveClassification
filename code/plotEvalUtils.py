# methods related to plotting


import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import matplotlib.pylab as pylab
params = {'figure.figsize': (20, 5),
          'legend.fontsize': 'x-large',
          'axes.labelsize': 'x-large',
          'axes.titlesize': 'x-large',
          'xtick.labelsize': 'x-large',
          'ytick.labelsize': 'x-large',
          'legend.framealpha': 1}
pylab.rcParams.update(params)

from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score, classification_report, roc_curve, auc



# define colormap for plots
discrete_cmap = mpl.colors.ListedColormap(["red", "cornflowerblue", "gold", "olivedrab"], name="discrete_cmap")
vmin=-1
vmax=2

# define patches for custom legends
mangrove_patch = mpatches.Patch(color='gold', label='Mangrove')
non_mangrove_patch = mpatches.Patch(color='cornflowerblue', label='Non-Mangrove')

mangrove_patch_pred = mpatches.Patch(color='gold', label='Mangrove (No Change)')
non_mangrove_patch_pred = mpatches.Patch(color='cornflowerblue', label='Non-Mangrove (No Change)')
loss_patch_pred = mpatches.Patch(color='red', label='Loss')
growth_patch_pred = mpatches.Patch(color='olivedrab', label='Growth')

fn_patch_pred = mpatches.Patch(color='red', label='False Negative')
fp_patch_pred = mpatches.Patch(color='olivedrab', label='False Positive')


def plotNVDIBand(input_data, name, year, modelFolder):
    '''Make a plot of the NDVI information for a satellite image.'''
    
    plt.figure()
    plt.imshow(input_data, cmap='RdYlGn', vmin=-1, vmax=1)
    plt.title("NDVI for " + name + " in " + str(year))
    
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)

    image_path = f"SavedPlots/{modelFolder}/{name}/NDVI_{name}_{year}.png"
    print("Saving image: ", image_path)
    plt.savefig(image_path, bbox_inches='tight')
    
    plt.show()

def plotMangroveBand(input_data, name, year, predicted, modelFolder):
    '''Make a plot of the mapped mangrove areas for a satellite image.'''
    
    plt.figure()
    plt.imshow(input_data, cmap=discrete_cmap, vmin=vmin, vmax=vmax)
    if not predicted: plt.title("Labeled Mangroves for " + name + " in " + str(year))
    else: plt.title("Predicted Mangroves for " + name + " in " + str(year))
    # plt.colorbar()
    plt.legend(handles=[mangrove_patch, non_mangrove_patch])
    
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    
    if not predicted: image_path = f"SavedPlots/{modelFolder}/{name}/LabeledMangroves_{name}_{year}.png"
    else: image_path = f"SavedPlots/{modelFolder}/{name}/PredictedMangroves_{name}_{year}.png"
    
    print("Saving image: ", image_path)
    plt.savefig(image_path, bbox_inches='tight')   
    
    plt.show()

def plotDifference(labels_data, predicted_data, name, year, modelFolder):
    '''
    Plot difference in predicted (or future predicted) mangroves and labeled (past) mangroves.
    # multiply first array by 2 in order to get 4 values for difference plot:
    # pred - label -> output
    # 0 - 0 -> 0, predicted and label/past are not mangroves
    # 1 - 1 -> 1, predicted and label/past are mangroves
    # 1 - 0 -> 2, predicted was mangrove, label/past was not -> growth/false positive
    # 0 - 1 -> 0, predicted was not mangrove, label/past was -> loss/false negative
    '''
    
    image_difference = 2 * predicted_data - labels_data
    plt.figure()
    plt.imshow(image_difference, cmap=discrete_cmap, vmin=vmin, vmax=vmax)
#     plt.colorbar()
    if year == 2000: 
        plt.title("Predicted vs Actual Mangroves for " + name + " in " + str(year))
        plt.legend(handles=[mangrove_patch_pred, non_mangrove_patch_pred, fp_patch_pred, fn_patch_pred])
    else: 
        plt.title("Change in Mangroves for " + name + " in " + str(year) + " vs 2000")
        plt.legend(handles=[mangrove_patch_pred, non_mangrove_patch_pred, growth_patch_pred, loss_patch_pred])
        
    ax = plt.gca()
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
        
    if year == 2000: image_path = f"SavedPlots/{modelFolder}/{name}/PvA_{name}_{year}.png"
    else: image_path = f"SavedPlots/{modelFolder}/{name}/GaL_{name}_{year}.png"
    
    print("Saving image: ", image_path)
    plt.savefig(image_path, bbox_inches='tight')   
        
    plt.show()
    
    
# methods related to classification evaluation
    
def printClassificationMetrics(y_actual, y_predicted_prob, input_prob=0.5):
    '''Print various classification metrics.'''

    y_predicted = (y_predicted_prob > input_prob).astype(int) # convert prediction probabilities to 0 or 1 values depending on threshold
    cMatrix = confusion_matrix(y_actual, y_predicted)
    pScore = precision_score(y_actual, y_predicted)
    rScore = recall_score(y_actual, y_predicted)
    aScore = accuracy_score(y_actual, y_predicted)
    f1Score = f1_score(y_actual, y_predicted)

    print("Confusion matrix:\n", cMatrix)
    print("\nP-Score: %.3f, R-Score: %.3f" % (pScore, rScore))
    print('Accuracy: %.3f' % (aScore))
    print('f1: %.3f' % (f1Score))

    print(classification_report(y_actual, y_predicted))
    
    return f1Score

def makeROCPlot(y_actual, y_predicted_prob, name, year, modelFolder, saveImage=True):
    '''Make an ROC plot from classification results.'''
    
    fpr, tpr, thresholds = roc_curve(y_actual, y_predicted_prob)
    auc_score = auc(fpr, tpr)

    plt.figure(1, figsize=(6,5))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='AUC = {:.3f}'.format(auc_score))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    
    if saveImage:
        image_path = f"SavedPlots/{modelFolder}/{name}/ROC_{name}_{year}.png"
        print("Saving image: ", image_path)
        plt.savefig(image_path, bbox_inches='tight')    
    
    plt.show()
    
    return auc_score
