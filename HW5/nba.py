import pandas as pd
import numpy as np


def main():
    symptoms = pd.read_csv("symptom.csv", sep=";")
    diseases = pd.read_csv("disease.csv", sep=";")
    init_symptoms = np.random.randint(0, 2, symptoms.shape[0])  # 23
    print(init_symptoms)
    probability = np.ones(diseases.shape[0] - 1)  # 9
    print(len(probability))
    print(symptoms.iloc[0][1])
    print("Симптомы:")
    for j in range(len(init_symptoms)):
        if init_symptoms[j] == 1:
            print(symptoms.iloc[j][0])
    for i in range(len(probability)):
        for j in range(len(init_symptoms)):
            if init_symptoms[j] == 1:
                probability[i] *= symptoms.iloc[j][i+1]
        probability[i] *= diseases.iloc[i][1] / diseases.iloc[diseases.shape[0] - 1][1]
    most_prob = -1
    index = -1
    for i in range(len(probability)):
        if probability[i] > most_prob:
            most_prob = probability[i]
            index = i
    print(probability)
    print("Болеет ", diseases.iloc[index+1][0], ", с вероятностью ", most_prob)


if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
