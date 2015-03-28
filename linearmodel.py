import utils
import math
import numpy as np

# Converts the hypothesis to predicted labels [0,1]
def get_predictions(test_data, weights):
    predictions = []
    for td in test_data:
        d = np.insert(td, 0, 1)
        h = np.dot(d, weights)
        if h < 0.5:
            predictions.append(0)
        else:
            predictions.append(1)
    return predictions

# Computes the accuracy
def compute_accuracy(predictions, actual_values):
    n = len(predictions)
    correct = 0.0
    for i in range(n):
        if predictions[i] == actual_values[i]:
            correct += 1.0
    return (correct/n)*100.0

# Computes the MSE cost
def compute_cost(data, labels, weights):
    m = len(labels)
    sum_errors = 0.0
    for i in range(m):
        d = np.insert(data[i], 0, 1)
        l = labels[i]
        hypothesis = np.dot(d, weights)
        error = hypothesis - l
        sq_error = (error * error)/2
        sum_errors += sq_error
    return sum_errors/m

# Stochastic gradient descent
def sgd(data, labels, alpha):
    m = len(labels)
    errors = []
    weights = np.zeros(len(data[0])+1) # Initialise weights to 0
    mse = 1.0
    prev_mse = 100.0
    while mse/prev_mse < 0.999999:
        prev_mse = mse

        for i in range(m):
            d = np.insert(data[i], 0, 1)
            l = labels[i]
            h = np.dot(d, weights)
            error = h - l
            weights = weights - (alpha * error * d)

        mse = compute_cost(data, labels, weights)
        errors.append(mse)
        print "MSE %0.8f after iteration %d" % (mse, len(errors))

        if mse > prev_mse:
            break
    return weights, errors

# Batch gradient descent
def bgd(data, labels, alpha):
    weights = np.zeros(np.shape(data[0]+1)) # Initialise weights to 0
    m = len(labels)
    mse = 1.0
    prev_mse = 100.0
    errors = []
    while mse/prev_mse < 0.9999:
        prev_mse = mse

        hypothesis = np.dot(data, weights)
        loss = hypothesis - labels
        gradient = np.dot(loss, data)/m
        weights = weights - (alpha*gradient)

        # Calculate cost using current weights
        mse = compute_cost(data, labels, weights)
        errors.append(mse)
        print "MSE %0.8f after iteration %d" % (mse, len(errors))
        #if mse > prev_mse:
            #return None, None
    return weights, errors

def linear_regression(folds, alpha):
    accuracy = []
    fold_errors = []
    for i, _ in enumerate(folds):
        print "linearmodel.py: Fold %d" %(i)
        train_data, train_labels, test_data, test_labels = utils.split_fold(folds, i)
        #weights, errors = bgd(train_data, train_labels, alpha)
        weights, errors = sgd(train_data, train_labels, alpha)

        if np.array_equal(weights,None):
            print "No convergence with learning rate %f" % (alpha)
            break
        else:
            # Evaluate function using weights produced on test set
            #mse = compute_cost(test_data, test_labels, weights)
            predictions = get_predictions(test_data, weights)
            accuracy.append(compute_accuracy(predictions, test_labels))
            fold_errors.append(errors)
    return accuracy, fold_errors

def main():
    folds = utils.get_data()
    sgd_alphas = [0.01, 0.001, 0.0001] #0.001, 0.0001]
    bgd_alphas = []
    rates = {}
    for alpha in sgd_alphas:
        accuracy, errors = linear_regression(folds, alpha)
        rates[alpha] = errors
        print np.mean(accuracy)
    utils.plot_errors(rates)


if __name__ == '__main__':
    main()