import random
import numpy as np
import matplotlib.pyplot as plt

# Extracts data from file
def get_raw_data():
    print "utils.py: Getting data"
    data = []
    filepath = "spambase.data"
    with open(filepath, "r") as f:
        for l in f:
            tmp = []
            l = l.strip()
            for i in l.split(","):
                tmp.append(float(i))
            data.append(tmp)
    return data

# Normalises data by computing z-scores
def normalise(data):
    print "utils.py: Normalising data"
    means = np.mean([x[:-1] for x in data], axis=0) # Mean of features
    stds = np.std([x[:-1] for x in data], axis=0) # Standard deviation of features
    norms = []
    for d in data:
        tmp = [(d[i] - means[i])/stds[i] for i in range(len(d)-1)]
        tmp.append(d[-1])
        norms.append(tmp)
    return norms

# Helper function for splitting data into groups
def get_group(data, k, i):
    for x in range(i, len(data), k):
        yield data[x]

# Splits dataset into k folds
def split_data(data, k):
    print "utils.py: Splitting data into %d folds" %(k)
    groups = []
    for i in range(k):
        g = np.random.permutation((list((get_group(data, k, i)))))
        groups.append(g)
    return groups

# Splits fold into training and testing sets
def split_fold(folds, i):
    print "utils.py: Splitting fold"
    train = list(folds[:i]) + list(folds[i+1:])
    train = [x for t in train for x in t] # Converts into a 2d array

    train_data = [x[:-1] for x in train]
    train_labels = [x[-1] for x in train]

    test = folds[i]
    test_data = [x[:-1] for x in test]
    test_labels = [x[-1] for x in test]

    return train_data, train_labels, test_data, test_labels

def get_data():
    raw_data = get_raw_data()
    data_norm = normalise(raw_data)
    groups = split_data(data_norm, 10)
    return groups

def get_avg_mse(errors):
    max_iterations = np.max([len(x) for x in errors])
    iterations = [x for x in range(0,max_iterations)]
    avg_errors = []
    for i in iterations:
        tmp = 0.0
        for j in errors:
            if i < len(j):
                tmp += j[i]
            else:
                tmp += j[-1]
        avg_errors.append(tmp/len(errors))
    return iterations, avg_errors

def plot_errors(rates):
    print "utilities.py: Plotting errors"
    colours = ["-b", "-r", "-g"]
    for idx, alpha in enumerate(rates):
        iterations, avg_errors = get_avg_mse(rates[alpha])
        plt.plot(iterations, avg_errors, colours[idx], label=str(alpha))
        #plt.plot(range(len(rates[alpha])), rates[alpha], colours[idx], label=str(alpha))
    plt.xlabel("No. of Iterations")
    plt.ylabel("Mean Squared Error (MSE)")
    plt.title("Stochastic Gradient Descent")
    plt.legend(loc="upper right")
    plt.show()

def roc_curve(rates):
    return ''



def main():
    folds = get_data()
    split_fold(folds, 2)

if __name__ == '__main__':
    main()