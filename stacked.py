# Test stacking on the sonar dataset
from random import seed
from random import randrange
from csv import reader
from math import sqrt
from math import pi
from math import exp


# Load a CSV file
def load_csv(filename):
        print('hello')
        dataset = list()
        with open(filename, 'r') as file1:
            csv_reader = reader(file1)
            for row in csv_reader:
                if not row:
                    continue
                dataset.append(row)
        return dataset


# Convert string column to float
def str_column_to_float(dataset, column):
        for row in dataset:
            row[column] = float(row[column].strip())


# Convert string column to integer
def str_column_to_int(dataset, column):
        class_values = [row[column] for row in dataset]
        unique = set(class_values)
        lookup = dict()
        for i, value in enumerate(unique):
            lookup[value] = i
        for row in dataset:
            row[column] = lookup[row[column]]
        return lookup


# Split a dataset into k folds
def cross_validation_split(dataset, n_folds):
        dataset_split = list()
        dataset_copy = list(dataset)
        fold_size = int(len(dataset) / n_folds)
        for i in range(n_folds):
            fold = list()
            while len(fold) < fold_size:
                index = randrange(len(dataset_copy))
                fold.append(dataset_copy.pop(index))
                dataset_split.append(fold)
        return dataset_split


# Calculate accuracy percentage
def accuracy_metric(actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0


# Evaluate an algorithm using a cross validation split
def evaluate_algorithm(dataset, algorithm, n_folds, *args):
    folds = cross_validation_split(dataset, n_folds)
    scores = list()
    for fold in folds:
        train_set = list(folds)
        #print(train_set)
        train_set.remove(fold)
        train_set = sum(train_set, [])
        test_set = list()
        for row in fold:
            row_copy = list(row)
            test_set.append(row_copy)
            row_copy[-1] = None
        predicted = algorithm(train_set, test_set, *args)
        print predicted
        actual = [row[-1] for row in fold]
        accuracy = accuracy_metric(actual, predicted)
        scores.append(accuracy)
    return scores


def predict(summaries, inputVector):
        probabilities = calculateClassProbabilities(summaries, inputVector)
        bestLabel, bestProb = None, -1
        for classValue, probability in list(probabilities.items()):
            if bestLabel is None or probability > bestProb:
                bestProb = probability
                bestLabel = classValue
        return bestLabel


def calculateClassProbabilities(summaries, inputVector):
        probabilities = {}
        for classValue, classSummaries in list(summaries.items()):
            probabilities[classValue] = 1
            for i in range(len(classSummaries)):
                mean, stdev = classSummaries[i]
                x = inputVector[i]
                #p[]= calculateProbability(x, mean, stdev)
                probabilities[classValue] *= calculateProbability(x, mean, stdev)
        return probabilities


def calculateProbability(x, mean, stdev):
        exponent = exp(- (pow(x - mean, 2) / (2 * pow(stdev, 2))))
        return (1 / (sqrt(2 * pi) * stdev)) * exponent


def mean(numbers):
        return sum(numbers) / float(len(numbers))


def stdev(numbers):

        avg = mean(numbers)
        #print((numbers))
        variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
        return sqrt(variance)


def summarize(dataset):
        summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
        del summaries[-1]
        return summaries


def summarizeByClass(dataset):
        separated = separateByClass(dataset)
        summaries = {}
        for classValue, instances in list(separated.items()):
            summaries[classValue] = summarize(instances)
        return summaries


def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated


def naiveBayesian_model(train):
        summaries = summarizeByClass(train)
        return summaries


def naiveBayesian_predict(NaiveBayesianModel, row):
        return predict(NaiveBayesianModel, row)


# Calculate the Euclidean distance between two vectors
def euclidean_distance(row1, row2):
    distance = 0.0
    for i in range(len(row1) - 1):
        distance += (row1[i] - row2[i]) ** 2
    return sqrt(distance)


# Locate neighbors for a new row
def get_neighbors(train, test_row, num_neighbors):
    distances = list()
    for train_row in train:
        dist = euclidean_distance(test_row, train_row)
        distances.append((train_row, dist))
    distances.sort(key=lambda tup: tup[1])
    neighbors = list()
    for i in range(num_neighbors):
        neighbors.append(distances[i][0])
    return neighbors


# Make a prediction with kNN
def knn_predict(model, test_row, num_neighbors=2):
        neighbors = get_neighbors(model, test_row, num_neighbors)
        output_values = [row[-1] for row in neighbors]
        prediction = max(set(output_values), key=output_values.count)
        return prediction


# Prepare the kNN model
def knn_model(train):
    #print(train)
    return train


# Make a prediction with coefficients
def logistic_regression_predict(model, row):
		print row
		yhat = model[0]
		for i in range(len(row) - 1):
			yhat += model[i + 1] * row[i]
		return 1.0 / (1.0 + exp(-yhat))


# Estimate logistic regression coefficients using stochastic gradient descent
def logistic_regression_model(train, l_rate=0.01, n_epoch=5000):
        coef = [0.0 for i in range(len(train[0]))]
        for epoch in range(n_epoch):
            for row in train:
                yhat = logistic_regression_predict(coef, row)
                error = row[-1] - yhat
                coef[0] = coef[0] + l_rate * error * yhat * (1.0 - yhat)
                for i in range(len(row) - 1):
                    abcd = l_rate * error * yhat * (1.0 - yhat) * row[i]
                    coef[i + 1] = coef[i + 1] + abcd
        return coef


# Make predictions with sub-models and construct a new stacked row
def to_stacked_row(models, predict_list, row):
	stacked_row = list()
	for i in range(len(models)):
		prediction = predict_list[i](models[i], row)
		stacked_row.append(prediction)
	#stacked_row.append(row[-1])
	return stacked_row

# Stacked Generalization Algorithm
def stacking(train, test):
        model_list = [knn_model, naiveBayesian_model]
        predict_list = [knn_predict, naiveBayesian_predict]
        models = list()
        for i in range(len(model_list)):
            print(model_list[i])
            #print train
            model = model_list[i](train)
            models.append(model)
        stacked_dataset = list()
        for row in train:
            stacked_row = to_stacked_row(models, predict_list, row)
            stacked_dataset.append(stacked_row)
        stacked_model = logistic_regression_model(stacked_dataset)
        print stacked_model
        predictions = list()
        weight=0
        for row in test:
            stacked_row = to_stacked_row(models, predict_list, row)
            stacked_dataset.append(stacked_row)
            prediction = logistic_regression_predict(stacked_model, stacked_row)
            print "prediction"
            weight
            prediction = round(prediction)
            print prediction
            predictions.append(prediction)
        return predictions


seed(1)
# load and prepare data
filename = 'sample_data.csv'
dataset = load_csv(filename)
# convert string attributes to integers
for i in range(len(dataset[0]) - 1):
    str_column_to_float(dataset, i)
# convert class column to integers
str_column_to_int(dataset, len(dataset[0]) - 1)
n_folds = 3
scores = evaluate_algorithm(dataset, stacking, n_folds)
print(('Scores: %s' % scores))
print(('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores)))))