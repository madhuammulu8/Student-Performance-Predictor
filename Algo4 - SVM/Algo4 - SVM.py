import os
import numpy as np
import configparser
import pandas as pd
import random as rnd
from pathlib import Path
from regex import S
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
          
class Support_vector_machines:
    
    def __init__(self, config):
        self.config = config
        self.max_iter = int(self.config.get('svm_initialize','max_iter'))
        self.C = float(self.config.get('svm_initialize','C'))
        self.epsilon = float(self.config.get('svm_initialize','epsilon'))

    def congruous(self, x_coordinate, y_coordinate):
        structure = x_coordinate.shape[0]
        a = np.zeros((structure))
        loop = 0
        while (1):           
            alpha_prev = np.copy(a)
            for step in range(0, structure): 
                index = self.random_number_generator(0, structure-1, step)
                x_coordinator_1, x_coordinator_2, y_coordinator_1, y_coordinator_2 = x_coordinate[index,:], x_coordinate[step,:], y_coordinate[index], y_coordinate[step]
                k_ij = self.kernel(x_coordinator_1, x_coordinator_1) + self.kernel(x_coordinator_2, x_coordinator_2) - 2 * self.kernel(x_coordinator_1, x_coordinator_2)
                if k_ij == 0:
                    continue
                alpha_prime_j, alpha_prime_i = a[step], a[index]
                if y_coordinator_1 == y_coordinator_2:
                    (L, support_vectors) = max(0, alpha_prime_i + alpha_prime_j - self.C), min(self.C, alpha_prime_i + alpha_prime_j)
                else:
                    (L, support_vectors) = max(0, alpha_prime_j - alpha_prime_i), min(self.C, self.C - alpha_prime_i + alpha_prime_j)
                self.normal_direction_of_plane = np.dot(a * y_coordinate, x_coordinate)
                self.form_of_threshold = np.mean(y_coordinate - np.dot(self.normal_direction_of_plane.T, x_coordinate.T))
                s = np.sign(np.dot(self.normal_direction_of_plane.T, x_coordinator_1.T) + self.form_of_threshold).astype(int) - y_coordinator_1
                t = np.sign(np.dot(self.normal_direction_of_plane.T, x_coordinator_2.T) + self.form_of_threshold).astype(int) - y_coordinator_2
                a[step] = alpha_prime_j + float(y_coordinator_2 * (s - t))/k_ij
                a[step] = max(a[step], L)
                a[step] = min(a[step], support_vectors)
                a[index] = alpha_prime_i + y_coordinator_1*y_coordinator_2 * (alpha_prime_j - a[step])
            difference = np.linalg.norm(a - alpha_prev)
            if difference < self.epsilon:
                break  
        loop += 1
 
    def support_vectors_calculation(self, value):
        return np.sign(np.dot(self.normal_direction_of_plane.T, value.T) + self.form_of_threshold).astype(int)
        
    def random_number_generator(self, value1,value2,compare):
        result = compare
        count=0
        while result == compare and count<1000:
            result = rnd.randint(value1,value2)
            count=count+1
        return result
    
    def kernel(self, value1, value2):
        return np.dot(value1, value2.T)

    def accuracy(self, data, y_hat):
        correct_counter = 0
        increment = 0
        length =len(data)
        while(increment!=length):
            if(data[increment] == -1 and y_hat[increment] == -1):
                correct_counter = correct_counter + 1
            if(data[increment] == 1 and y_hat[increment] == 1):
                correct_counter = correct_counter + 1
            increment+=1      
        return (correct_counter/length)

    def result(self):
        return self.C, self.max_iter, self.epsilon
        
def main():
    print("\nStudent Performance Predictor")
    print("\nAlgorithm: SVM")
    path = Path(__file__)
    ROOT_DIR = path.parent.absolute()
    config_path = os.path.join(ROOT_DIR, "configfile.properties")
    config = configparser.ConfigParser()
    config.read(config_path)
    data_file = os.path.join(ROOT_DIR, 'student_data.csv')
    pandas_dataframe = pd.read_csv(data_file)
    pandas_dataframe["passed"] = pandas_dataframe["passed"].astype('category')
    pandas_dataframe["passed"] = pandas_dataframe["passed"].cat.codes
    prediction_feature = pandas_dataframe['passed']
    hand_picked_feature = pandas_dataframe.iloc[:,[int(config.get('svm_initialize','efficient_feature'))]]
    hand_picked_feature = hand_picked_feature.dropna()
    hand_picked_feature = hand_picked_feature.apply(lambda col: pd.factorize(col, sort=True)[0])
    multiplicative = preprocessing.MinMaxScaler()
    multiplicative.fit(hand_picked_feature)
    scaled = multiplicative.transform(hand_picked_feature)
    trained_data = pd.DataFrame(scaled, columns=hand_picked_feature.columns)
    duplicate_trained_data = trained_data
    duplicate_prediction_feature = prediction_feature
    performance_prediction = Support_vector_machines(config)
    x_train, x_test, y_train, y_test = train_test_split(duplicate_trained_data.values, duplicate_prediction_feature.values, test_size = 0.2, random_state = 0)
    performance_prediction.congruous(x_train,y_train)
    y_hat = performance_prediction.support_vectors_calculation(x_test)
    predicted_accuracy = performance_prediction.accuracy(y_test, y_hat)
    c, maximum_iterations, epsilon = performance_prediction.result()
    predicted_accuracy = predicted_accuracy*100
    print("\nAccuracy: {:.2f} %".format(predicted_accuracy))
    print("\nDegree of Correct Classification: {0}".format(c))
    print("\nMaximum number of Iterations: {0}".format(maximum_iterations))
    print("\nEpsilon: {0}".format(epsilon))
    print("\nKernel: Linear\n")


if __name__ == "__main__":
    main()