from collections import Counter
from linear_algebra import distance
from statistics import mean
import math, random
import matplotlib.pyplot as plt
from data import cities
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

def majority_vote(labels):
    """assumes that labels are ordered from nearest to farthest"""
    vote_counts = Counter(labels)
    winner, winner_count = vote_counts.most_common(1)[0]
    num_winners = len([count
                       for count in vote_counts.values()
                       if count == winner_count])

    if num_winners == 1:
        return winner                     # unique winner, so return it
    else:
        return majority_vote(labels[:-1]) # try again without the farthest


def knn_classify(k, labeled_points, new_point):
    """each labeled point should be a pair (point, label)"""

    # order the labeled points from nearest to farthest
    by_distance = sorted(labeled_points,
                         key=lambda point_label: distance(point_label[0], new_point))

    # find the labels for the k closest
    k_nearest_labels = [label for _, label in by_distance[:k]]

    # and let them vote
    return majority_vote(k_nearest_labels)


def predict_preferred_language_by_city(k_values, cities):
    """
    TODO
    predicts a preferred programming language for each city using above knn_classify() and 
    counts if predicted language matches the actual language.
    Finally, print number of correct for each k value using this:
    print(k, "neighbor[s]:", num_correct, "correct out of", len(cities))
    """
    
    for i in k_values:
        num_correct=0
        for values in cities:
            new_cities = cities[:]
            new_cities.remove(values)
            predict = knn_classify(i,new_cities,values[0]) 
            if (predict == values[1]):
                num_correct+=1
        print (i,"neighbor[s]:", num_correct, "correct out of", len(cities))  

def get_val(cities):
    labels = []
    language = []
    for city in cities:
        labels.append([city[0][0],city[0][1]])
        language.append(city[1])

    return np.array(labels),np.array(language)

def predict_preferred_language_by_city_scikit(k_values, cities):
    for k in k_values:
        num_correct = 0
        knn = KNeighborsClassifier(n_neighbors=k) 
        for values in cities:
            new_cities = cities[:]
            new_cities.remove(values)
            X_train,Y_train = get_val(new_cities)
            knn.fit(X_train,Y_train)
            predict = knn.predict(np.array([[values[0][0],values[0][1]]]))

            if predict == values[1]:
                num_correct += 1

        print(k,"neighbor[s]:", num_correct, "correct out of", len(cities))          

if __name__ == "__main__":
    k_values = [1, 3, 5, 7]
    # TODO
    # Import cities from data.py and pass it into predict_preferred_language_by_city(x, y).
    print("K values with original code are:")
    predict_preferred_language_by_city(k_values, cities)
    print("K values with scikit code are:")
    predict_preferred_language_by_city_scikit(k_values, cities)