import os
import csv
import random

root_dir = "./train"

columns = ["id", "label"]

labels = []

labels_dict = ["apple", "avocado", "banana", "cherry", "kiwi",
          "mango", "orange", "pineapple", "strawberries", "watermelon"]

'''
0: apple
1: avocado
2: banana
3: cherry
4: kiwi
5: mango
6: orange
7: pineapple
8: strawberries
9: watermelon
'''

for child_dir in os.listdir(root_dir):
    current_path = os.path.join(root_dir, child_dir)
    for file_name in os.listdir(current_path):
        labels.append([file_name, labels_dict.index(child_dir)])

random.shuffle(labels)

csv_file_name = "train.csv"

with open(csv_file_name, "w", newline='') as file:
    writer = csv.writer(file)

    writer.writerow(columns)

    writer.writerows(labels)
