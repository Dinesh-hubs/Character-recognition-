import numpy as np
import csv
import matplotlib.pyplot as plt
from docx import Document
from collections import Counter
import time

def label_conversion(input):
    label_to_index = {label: idx for idx, label,in enumerate(input)}
    return label_to_index

def index_convsion(input):
    labels = index_to_label(input)
    index_to_label = {idx: label for label,idx in labels.items()}   

def mapping(input):
    unique_labels = sorted(set(input))
    return unique_labels

def read_csv(filename):
    labels = []
    bitarrays = []
    with open(filename,'r') as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            if len(row)<2:
                continue
            labels.append(row[0])
            bitarrays.append([int(bit.strip()) for bit in row[1].split(',')])
    return labels,bitarrays

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def fnn_forward(input,weights1,weights2):
    h1_in = np.dot(input,weights1)
    h1_out = sigmoid(h1_in)

    h2_in = np.dot(h1_out,weights2)
    h2_out = sigmoid(h2_in)

    return h2_out,h1_out

def fnn_backprop(data,labels,weights1,weights2,lr):
    h2_out,h1_out = fnn_forward(data,weights1,weights2)

    output_loss = h2_out - labels
    hidden_loss = np.multiply(np.dot(output_loss,weights2.T), h1_out * (1 - h1_out))

    weights1_adj = np.outer(data,hidden_loss)
    weights2_adj = np.outer(h1_out,output_loss)

    weights1 -= lr * weights1_adj
    weights2 -= lr * weights2_adj

    return weights1,weights2

def generate_wt(x,y):
    return np.random.randn(x,y)

def mean_square(out,Y):
    return np.sum(np.square(out - Y)) / len(Y)

def train_fnn(input,labels,weights1,weights2,lr,epochs):
    accuracies = []
    losses = []
    for epoch in range(epochs):
        l = []
        correct = 0
        for i in range(len(input)):
            out,_ = fnn_forward(input[i],weights1,weights2)
            loss = mean_square(out,labels[i])
            l.append(loss)
            if np.argmax(out) == np.argmax(labels[i]):
                correct += 1
            weights1,weights2 = fnn_backprop(input[i],labels[i],weights1,weights2,lr)

        avg_loss = sum(l)/len(input)
        accuracy = (correct/len(input)) * 100
        print(f"Epoch {epoch + 1}: Accuracy = {accuracy:.2f}% , Avg Loss = {avg_loss:.4f}")
        
        accuracies.append(accuracy)
        losses.append(avg_loss)
    return accuracies,losses,weights1,weights2

def fnn_predict(input,weights1,weights2,label,axis = 1):
    if axis == 1:
        input = input[0]
    
    original_label = label
    out,_ = fnn_forward(input,weights1,weights2)
    out_percentage = out * 100
    max_value = np.max(out_percentage)
    label_idx = np.argmax(max_value)
    print(label_idx)
    return label_idx,max_value,original_label


    