import numpy as np



def calculate_variance(prediction_probs):
    mean_for_every_classifier = []
    variance_for_every_classifier=[]
    for collumn in range(len(prediction_probs[1])):#collumns
        mean_for_this_classifier=0.0
        for row in range(len(prediction_probs)):#rows
            mean_for_this_classifier=mean_for_this_classifier+prediction_probs[row][collumn]
        mean_for_this_classifier=mean_for_this_classifier/len(prediction_probs)
        mean_for_every_classifier.append(mean_for_this_classifier)
    #print(mean_for_every_classifier)
    preds_sub_mean=np.subtract(prediction_probs, mean_for_every_classifier)
    #print(preds_sub_mean)
    preds_sub_mean_squared=np.square(preds_sub_mean)
    #print(preds_sub_mean_squared)
    for  collumn in range(len(prediction_probs[1])):
        variance = 0
        for row in range(len(prediction_probs)):
            variance=variance+preds_sub_mean_squared[row][collumn]
        variance=variance/len(prediction_probs)
        variance_for_every_classifier.append(variance)
    #print(variance_for_every_classifier)
    return variance_for_every_classifier

def calculate_deviation(variance_for_every_classifier):
    deviation_for_every_classifier=np.sqrt(variance_for_every_classifier)
    return deviation_for_every_classifier