import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, accuracy_score, f1_score, classification_report, roc_curve, auc


# methods related to plotting

# define colormap for plots
discrete_cmap = mpl.colors.ListedColormap(["red", "cornflowerblue", "gold", "olivedrab"], name="discrete_cmap")
vmin=-1
vmax=2

def plotNVDIBand(input_data):
    plt.figure(figsize=(24,6))
    plt.imshow(input_data, cmap='RdYlGn', vmin=-1, vmax=1)
    plt.colorbar()
    plt.show()

def plotMangroveBand(input_data):
    plt.figure(figsize=(16,6))
    plt.imshow(input_data, cmap=discrete_cmap, vmin=vmin, vmax=vmax)
    # plt.colorbar()
    plt.show()

def plotDifference(labels_data, predicted_data):
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
    plt.figure(figsize=(24,6))
    plt.imshow(image_difference, cmap=discrete_cmap, vmin=vmin, vmax=vmax)
    plt.colorbar()
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

def makeROCPlot(y_actual, y_predicted_prob):
    fpr, tpr, thresholds = roc_curve(y_actual, y_predicted_prob)
    auc_score = auc(fpr, tpr)

    plt.figure(1)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.plot(fpr, tpr, label='AUC = {:.3f}'.format(auc_score))
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='best')
    plt.show()
    
    return auc_score
