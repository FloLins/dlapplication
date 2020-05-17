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


def compute_shannon_entropie(summed_probs):
    #print(summed_probs)
    information=-np.log2(summed_probs)
    mul = np.multiply(summed_probs,information)
    #print(mul)
    sum = np.sum(mul)
    #print(sum)
    return sum


def compute_min_entropy(summed_probs):
    p1 = np.amax(summed_probs)
    information=-np.log2(p1)
    #print(information)
    return information


def compute_guessing_entropy(summed_probs):
    summed_probs=-np.sort(-summed_probs)
    g_entropy=0
    for i in range(len(summed_probs)):
        g_entropy=i*summed_probs[i]
    return g_entropy