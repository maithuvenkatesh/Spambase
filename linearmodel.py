import utils
import numpy as np

def get_predictions(test_data, weights):
    predictions = np.dot(test_data, weights)
    for p in range(len(predictions)):
        if predictions[p] < 0.5:
            predictions[p] = 0
        else:
            predictions[p] = 1
    return predictions

def compute_accuracy(predictions, actual_values):
    n = len(predictions)
    correct = 0.0
    for i in range(n):
        if predictions[i] == actual_values[i]:
            correct += 1.0
    return (correct/n)*100.0

def compute_cost(data, labels, weights):
    m = len(labels)
    hypothesis = np.dot(data, weights)
    sq_errors = (hypothesis - labels) ** 2
    mse = (np.sum(sq_errors))/(2.0*m)
    return mse

def cost(hypothesis, labels, weights):
    m = len(labels)
    #hypothesis = np.dot(data, weights)
    sq_errors = (hypothesis - labels) ** 2
    mse = (np.sum(sq_errors))/(2.0*m)
    return mse

# Stochastic gradient descent
def sgd(data, labels, alpha):
    m = len(labels)
    errors = []
    weights = np.zeros(np.shape(data[0])) # Initialise weights to 0
    mse = 1.0
    prev_mse = 100.0
    grad = []
    while mse/prev_mse < 0.999999:
        prev_mse = mse

        for i in range(m):
            d = data[i]
            hypothesis = np.dot(d, weights)
            gradient = np.dot((hypothesis - labels[i]), d)
            weights = weights - (alpha * gradient)

        #predictions = get_predictions(data, weights)
        #mse = cost(predictions, labels, weights)
        mse = compute_cost(data, labels, weights)
        errors.append(mse)
        print "MSE %0.8f after iteration %d" % (mse, len(errors))
    return weights, errors

# Batch gradient descent
def bgd(data, labels, alpha):
    weights = np.zeros(np.shape(data[0])) # Initialise weights to 0
    m = len(labels)
    mse = 1.0
    prev_mse = 100.0
    errors = []
    while mse/prev_mse < 0.999999:
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
    scores = []
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
            mse = compute_cost(test_data, test_labels, weights)
            predictions = get_predictions(test_data, weights)
            accuracy.append(compute_accuracy(predictions, test_labels))
            fold_errors.append(errors)
    return accuracy, fold_errors

def main():
    folds = utils.get_data()
    alphas = [0.001]
    #alphas = [0.1, 0.01, 0.001]
    rates = {}
    for alpha in alphas:
        accuracy, errors = linear_regression(folds, alpha)
        rates[alpha] = errors
    print np.mean(accuracy)
    utils.plot_errors(rates)


if __name__ == '__main__':
    main()