import GaussSimplex as gs
import numpy as np
import sys


# # # STARTING EXAMPLES


def Ex1():
    print("---------------------------------------------")
    # maniezzo example
    A = np.array([[5, 15], [4, 4], [35, 20]], dtype=int)
    b = np.array([480, 160, 1190], dtype=int)
    c = np.array([13, 23], dtype=int)
    typeConstraint = ["<=", "<=", "<="]
    typeP = "max"
    showNum = False
    axis_range = (4, 40)
    return A, b, c, axis_range, showNum, typeP, typeConstraint


def Ex2():
    print("---------------------------------------------")
    # ugo problem in domain max

    A = np.array([[0, -1], [-1, 0], [1, 0], [-1, 1], [1, -1],
                  [0, 1], [0, -1]], dtype=int)
    b = np.array([3, 1, 3, 1, 1, 4, 1], dtype=int)
    c = np.array([1, 1], dtype=int)
    typeConstraint = ["<="]*7
    typeP = "max"
    showNum = True
    axis_range = (4, 10)
    return A, b, c, axis_range, showNum, typeP, typeConstraint


def Ex3():
    print("---------------------------------------------")
    # ugo problem not in domain min destra

    A = np.array([[0, -1], [-1, 0], [1, 0], [-1, 1], [1, -1],
                  [0, 1], [0, -1], [-1, -1]], dtype=int)
    b = np.array([3, 1, 3, 1, 1, 4, 1, -3], dtype=int)
    c = np.array([1, 2], dtype=int)
    typeConstraint = ["<="]*8
    typeP = "min"
    showNum = True
    axis_range = (4, 10)
    return A, b, c, axis_range, showNum, typeP, typeConstraint


def Ex4():
    print("---------------------------------------------")
    # ugo problem not in domain min sinistra

    A = np.array([[0, -1], [-1, 0], [1, 0], [-1, 1], [1, -1],
                  [0, 1], [0, -1], [-1, -1]], dtype=int)
    b = np.array([3, 1, 3, 1, 1, 4, 1, -3], dtype=int)
    c = np.array([3, 1], dtype=int)
    typeConstraint = ["<="]*8
    typeP = "min"
    showNum = True
    axis_range = (4, 10)
    return A, b, c, axis_range, showNum, typeP, typeConstraint


def Ex5():
    print("---------------------------------------------")
    # ugo problem not in domain max

    A = np.array([[0, -1], [-1, 0], [1, 0], [-1, 1], [1, -1],
                  [0, 1], [0, -1], [-1, -1]], dtype=int)
    b = np.array([3, 1, 3, 1, 1, 4, 1, -3], dtype=int)
    c = np.array([1, 2], dtype=int)
    typeConstraint = ["<="]*8
    typeP = "max"
    showNum = True
    axis_range = (4, 10)
    return A, b, c, axis_range, showNum, typeP, typeConstraint


def run(A, b, c, axis_range, showNum, typeP, typeConstraint):
    solutionSteps = gs.simplex(A, b, c, typeConstraint, typeP=typeP)
    if len(c) == 2:
        gs.showSimplexPolyedre(A, b, c, typeConstraint, typeP, solutionSteps, time=0.5, axis_range=axis_range,
                               fontsize=11, showNum=showNum)


if __name__ == "__main__":

    exercises = {1: Ex1, 2: Ex2, 3: Ex3, 4: Ex4, 5: Ex5}
    try:
        ex = int(sys.argv[1])
        result = exercises[ex]()
    except:
        result = exercises[1]()

    A, b, c, axis_range, showNum, typeP, typeConstraint = result
    run(A, b, c, axis_range, showNum, typeP, typeConstraint)
