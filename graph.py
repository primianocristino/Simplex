import numpy as np


def minCostFlowConstraintConversion(edgesData):
    nodes = []
    numEdges = 0
    edgesDict = {}
    c = []
    for data in edgesData:
        if data[0] != None and data[1] != None:
            numEdges += 1

    i = 0
    inverseEdgesDict = {}
    for data in edgesData:

        if data[0] != None and data[1] != None:
            inverseEdgesDict[i] = (data[0], data[1])
            nodes.append(data[0])
            nodes.append(data[1])
            edgesDict[(data[0], data[1])] = i
            c.append(data[2]["weight"])
            i += 1

    nodes = np.unique(nodes).flatten()
    A = []
    b = [0]*(nodes.size)

    typeConstraint = []
    for (i, node) in list(zip(range(len(nodes)), nodes)):

        rowConstraint = list(np.zeros((1, numEdges), dtype=object).flatten())

        for data in edgesData:
            if node == data[0]:
                # arco uscente
                if data[1] is None:
                    b[i] = -1*data[2]["demand"]
                else:

                    rowConstraint[edgesDict[(
                        node, data[1])]] = +1

            elif node == data[1]:
                # arco entrante
                if data[0] is None:
                    b[i] = -1*data[2]["demand"]
                else:
                    rowConstraint[edgesDict[(
                        data[0], node)]] = -1
        A.append(rowConstraint)
        typeConstraint.append("=")

    # if realProblem == "max flow":
    for data in edgesData:
        if "capacity" in data[2].keys() and data[2]["capacity"] != np.inf:
            rowConstraint = list(
                np.zeros((1, numEdges), dtype=object).flatten())

            rowConstraint[edgesDict[(data[0], data[1])]] = 1

            A.append(rowConstraint)
            b.append(data[2]["capacity"])
            typeConstraint.append("<=")

    A = np.array(A, dtype=object)
    b = np.array(b, dtype=object)
    c = np.array(c, dtype=object)

    return A, b, c, typeConstraint, inverseEdgesDict


def maxFlowConstraintConversion(edgesData):
    # if problem == "max flow":
    typeConstraint = []
    c = np.zeros((1, len(edgesData)), dtype=object).flatten()
    b = np.zeros((1, len(edgesData)), dtype=object).flatten()
    numArcs = len(edgesData)
    nodes = []

    i = 0
    edgesDict = {}
    inverseEdgesDict = {}
    nodeDict = {}
    for edge in edgesData:
        nodes.append(edge[0])
        nodes.append(edge[1])
        if edge[0] not in nodeDict.keys():
            nodeDict[edge[0]] = [[], []]

        if edge[1] not in nodeDict.keys():
            nodeDict[edge[1]] = [[], []]

        nodeDict[edge[0]][1].append((edge[0], edge[1]))  # archi uscenti
        nodeDict[edge[1]][0].append((edge[0], edge[1]))

        edgesDict[(edge[0], edge[1])] = i
        inverseEdgesDict[i] = (edge[0], edge[1])
        c[i] = edge[2]["weight"]
        b[i] = edge[2]["capacity"]
        typeConstraint.append("<=")
        i += 1

    numNodes = np.unique(nodes).size
    A = np.identity(numArcs, dtype=object)

    for node in nodeDict.keys():
        if node != 0 and node != numNodes-1:
            entryArcs = nodeDict[node][0]
            exitArcs = nodeDict[node][1]

            rowEquality = np.zeros((1, numArcs), dtype=object)
            for edge in entryArcs:
                rowEquality[0, edgesDict[edge]] = -1

            for edge in exitArcs:
                rowEquality[0, edgesDict[edge]] = 1

            A = np.vstack((A, rowEquality))
            b = np.hstack((b, np.array([0], dtype=object)))
            typeConstraint.append("=")

    return A, b, c, typeConstraint, inverseEdgesDict
