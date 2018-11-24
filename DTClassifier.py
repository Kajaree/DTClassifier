#!/usr/bin/env python2

import re
import csv
import os
import pandas as pd
from math import log
from graphviz import Source
from pprint import pprint

#path to file
path_to_file = os.getcwd()

class DTClassifier:
    def __init__(self, maxDepth = 3):
        self.nodes = [DTNode()]
        self.maxDepth = maxDepth
        
    #classifying the training data
    def classifyTrainingData(self, dataset, currentNode, depth): 
        #initialise the current node
        node = self.nodes[currentNode]
        #initialise other values
        rows, columns = dataset.axes
        ppos, pneg = classifyfeature(dataset) #no of positive and negative samples
        feature = 0         #index of the feature
        splitAtNode = 0
        maxInfoGain = 0
        maxValue = 0

        for column in range(len(columns)-1):
            sortedData = dataset.sort_values(by=[columns[column]])
            fpos = 0
            fneg = 0
            maxGain = 0
            for row in range(len(rows)):
                if sortedData.iat[row,-1] == 1.0:
                    fpos += 1
                    infoGain = information_gain(fpos, fneg, ppos-fpos, pneg-fneg)
                    if infoGain > maxGain:
                        maxGain = infoGain
                        currentSplit = row
                        if row > 0:
                            splitAtValue = (sortedData.iat[row, column] + sortedData.iat[row-1, column])/2
                        else:
                            splitAtValue =  sortedData.iat[row, column]
                else:
                    fneg += 1
            if maxGain > maxInfoGain:
                maxInfoGain = maxGain
                feature = column
                splitAtNode = currentSplit
                maxValue = splitAtValue
        node.accepted = ppos
        node.denied = pneg
        node.origin = maxValue
        node.feature = columns[feature]
        # The array is sorted and split
        sortedData = dataset.sort_values(by=[node.feature])
        set1 = sortedData.iloc[:splitAtNode, :]
        set2 = sortedData.iloc[splitAtNode:, :]
        if ppos == 0 or pneg == 0 or splitAtNode == 0 or  depth >= self.maxDepth:
            node.isLeaf = True
            node.decision = (ppos > pneg)
            return
        else:
            node.isLeaf = False
        leftChildNode = DTNode(currentNode)
        rightChildNode = DTNode(currentNode)
        node.leftChild = len(self.nodes)
        node.rightChild = len(self.nodes) + 1
        leftChildNode.identifier = node.leftChild
        rightChildNode.identifier = node.rightChild
        leftChildNode.parentFeature = "yes"
        rightChildNode.parentFeature = "no"

        self.nodes.append(leftChildNode)
        self.nodes.append(rightChildNode)
        self.nodes[node.leftChild].identifier = node.leftChild
        self.nodes[node.rightChild].identifier = node.rightChild
        self.classifyTrainingData(set1,node.leftChild, depth+1)
        self.classifyTrainingData(set2,node.rightChild, depth+1)
        return sortedData

                
    def classifyTestData(self, test_data, index):
        currentNode = self.nodes[0]
        while not currentNode.isLeaf:
            if (test_data.loc[index,str(currentNode.feature)] < currentNode.origin):
                currentNode = self.nodes[currentNode.leftChild]
            else:
                currentNode = self.nodes[currentNode.rightChild]
        return currentNode.decision

    def predictOutcome(self, dataset):
        rows = dataset.axes[0]
        predictions = {}
        for row in range(len(rows)):
            prediction = self.classifyTestData(dataset, row)
            p = {}
            p["predicted"] = prediction
            p["actual"] = dataset.iat[row, -1]
            predictions[row] = p
        return predictions

    def show_tree(self, file):
        filename = os.path.join(path_to_file, file)
        fo = open(filename,"w")
        fo.write("digraph Tree {\nnode [shape=box, style=\"filled\", color=\"black\"] ;\n")
        self.writeToDotFile(0, fo)
        fo.write("}")
        fo.close()
        graph = Source.from_file(filename)
        graph.view()

    def writeToDotFile(self, index, filename):
        if index == None:
            return
        self.updateToDotFile(self.nodes[index], filename)
        self.writeToDotFile(self.nodes[index].leftChild, filename)
        self.writeToDotFile(self.nodes[index].rightChild, filename)

    def updateToDotFile(self, node, filename):
        if node.isLeaf:
		    filename.write(str(node.identifier)+" [ label=\""+"Samples: "+ str(node.accepted + node.denied) +"\\nAccpeted: "+str(node.accepted)+" rejected: "+str(node.denied)+"\" , fillcolor=bisque2] ;\n")
            #filename.write(node_description)
        else:
            node_description=str(node.identifier)+" [ label=\""+node.feature+" <= "+str(node.origin)+"\\nSamples: "+ str(node.accepted + node.denied) +"\\nAccpeted: "+str(node.accepted)+" rejected: "+str(node.denied)+"\" , fillcolor=white] ;\n"
            filename.write(node_description)

	    

        if(node.parent!=-1):
        #create edge
            condition = node.identifier % 2
            if(condition):
                edge = str(node.parent)+"->"+str(node.identifier) + " [labeldistance=2.5, labelangle=45, headlabel=Yes] ;\n"
            else:
                edge = str(node.parent)+"->"+str(node.identifier) + " [labeldistance=2.5, labelangle=-45, headlabel=No] ;\n"
            filename.write(edge)
	return

    def accuracy(self, predictions):
        posMatches = 0
        for index in range(len(predictions)):
            if predictions[index]["actual"] == predictions[index]["predicted"]:
                posMatches += 1
        return float(posMatches)/len(predictions)

class DTNode:
    def __init__(self, parent=-1, leftChild = None, rightChild = None):
        self.isLeaf = False
        self.parent = parent
        self.isLeft = False
        self.leftChild = leftChild
        self.rightChild = rightChild
        self.classfeature = None
        self.accepted = None
        self.denied = None
        self.feature = None
        self.origin = 0
        self.decision = None #outcome
        self.parentFeature = None 
        self.identifier = 0
    def __repr__(self):
        return "{} {} {} ".format(self.feature, self.origin, self.decision)
    
def calculate_entropy(p, n):
    log2=lambda x:log(x)/log(2)  
    if n == 0:
        return p*log2(1.0/p)
    elif p == 0:
        return n*log2(1.0/n)
    return p*log(1.0/p) + n*log(1.0/n)

def information_gain(lPos, lNeg, rPos, rNeg):
    #lPos denotes no of accepted samples by left child
    #lNeg denotes no of denied samples by left child
    #rPos denotes no of accepted samples by right child
    #rNeg denotes no of denied samples by right child

    total = float(lPos + lNeg + rPos + rNeg) #total no of samples
    p_total = float(lPos + lNeg)    
    n_total = float(rPos + rNeg)    

    #entropy of the entire set of samples
    information_gain = calculate_entropy((lPos+rPos)/total,(lNeg + rNeg)/total) 

    #conditional entropy by the set split by some attribute
    if p_total > 0:
        information_gain -= p_total/total * calculate_entropy(lPos/p_total,lNeg/p_total) 
    if n_total > 0:
        information_gain -= n_total/total * calculate_entropy(rPos/n_total,rNeg/n_total)
    return information_gain


def classifyfeature(dataset):
    rows = dataset.axes[0]
    posSample = 0
    negSample = 0
    for i in range(len(rows)):
        if(dataset.iat[i,-1] == 1.0):
            posSample += 1
        else :
            negSample += 1
    return posSample, negSample 

#Read csv file into a dataframe using pandas
def read_data(filename):
    return pd.read_csv(filename)
