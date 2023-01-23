import numpy as np
import GaussSimplex as gs
import graph
import sys


def minCostEx():
    print("----------------------------------------------------------------")
    # min cost example (with infinity flow capacity)
    edgesData = [
        (None, 0, {'demand': -10}),
        (None, 1, {'demand': -4}),
        (0, 1, {'capacity': np.inf, 'weight': 10}),
        (0, 2, {'capacity': np.inf, 'weight': 8}),
        (0, 3, {'capacity': np.inf, 'weight': 1}),
        (1, 2, {'capacity': np.inf, 'weight': 2}),
        (2, 3, {'capacity': np.inf, 'weight': 1}),
        (2, 4, {'capacity': np.inf, 'weight': 4}),
        (3, 4, {'capacity': np.inf, 'weight': 12}),
        (3, None, {'demand': 6}),
        (4, None, {'demand': 8})]

    # un problema di minimo con solo uguaglianze
    A, b, c, t, varAlias = graph.minCostFlowConstraintConversion(edgesData)

    A2, b2, c2, t2 = gs.fromMinToPrimal(A, b, c, t)
    A3, b3, c3, t3 = gs.fromMaxToDual(A2, b2, c2, t2)

    A4, b4, c4, t4 = gs.convert2DualConstraint(A, b, c, t)

    solution1 = gs.calculate(A, b, c, t, "min", varAlias=varAlias)
    solution2 = gs.calculate(A2, b2, c2, t2, "max")
    solution3 = gs.calculate(A3, b3, c3, t3, "min")
    solution4 = gs.calculate(A4, b4, c4, t4, "min")

    print("\nMIN COST FLOW SIMPLEX SOLUTION VARIANT")
    print("\tNormal solution")
    print("\t\t", solution1)
    print("\tPrimal solution")
    print("\t\t", solution2)
    print("\tFrom primal to dual solution")
    print("\t\t", solution3)
    print("\tFrom normal to dual solution")
    print("\t\t", solution4)


def maxFlowEx():
    print("----------------------------------------------------------------")
    # max flow example

    edgesData = [
        (0, 1, {'capacity': 10, 'weight': 1}),
        (0, 2, {'capacity': 12, 'weight': 1}),
        (0, 3, {'capacity': 7, 'weight': 1}),
        (1, 5, {'capacity': 7, 'weight': 0}),
        (1, 4, {'capacity': 6, 'weight': 0}),
        (2, 1, {'capacity': 5, 'weight': 0}),
        (2, 3, {'capacity': 5, 'weight': 0}),
        (3, 5, {'capacity': 10, 'weight': 0}),
        (4, 5, {'capacity': 4, 'weight': 0})
    ]

    A, b, c, t, varAlias = graph.maxFlowConstraintConversion(edgesData)
    A2, b2, c2, t2 = gs.fromMaxToDual(A, b, c, t)
    A3, b3, c3, t3 = gs.fromMinToPrimal(A2, b2, c2, t2)

    A4, b4, c4, t4 = gs.convert2PrimalConstraint(A, b, c, t)

    solution1 = gs.calculate(A, b, c, t, "max", varAlias=varAlias)
    solution2 = gs.calculate(A2, b2, c2, t2, "min")
    solution3 = gs.calculate(A3, b3, c3, t3, "max")
    solution4 = gs.calculate(A4, b4, c4, t4, "max")

    print("\nMAX FLOW SIMPLEX SOLUTION:")
    print("\tNormal solution")
    print("\t\t", solution1)
    print("\tDual solution")
    print("\t\t", solution2)
    print("\tFrom dual to primal solution")
    print("\t\t", solution3)
    print("\tFrom normal to primal solution")
    print("\t\t", solution4)


def maxFlowAsMinCost():
    print("------------------------------------------------------")
    # max flow to min cost

    edgesData = [
        (0, 1, {'capacity': 10, 'weight': 0}),
        (0, 2, {'capacity': 12, 'weight': 0}),
        (0, 3, {'capacity': 7, 'weight': 0}),
        (1, 5, {'capacity': 7, 'weight': 0}),
        (1, 4, {'capacity': 6, 'weight': 0}),
        (2, 1, {'capacity': 5, 'weight': 0}),
        (2, 3, {'capacity': 5, 'weight': 0}),
        (3, 5, {'capacity': 10, 'weight': 0}),
        (4, 5, {'capacity': 4, 'weight': 0}),
        (5, 0, {'capacity': 9999, 'weight': -1})
    ]

    A, b, c, t, varAlias = graph.minCostFlowConstraintConversion(
        edgesData)
    simplexSolution = gs.calculate(A, b, c, t, "min", varAlias=varAlias)

    print("\nSIMPLEX MAX FLOW AS MIN COST:")

    # questo perché è di base il flussso negativo dell'arco negativo dal pozzo alla sorgente.
    # Quindi in realta è un flusso positivo dalla sorgente al pozzo.
    simplexSolution["flow"] *= -1
    print(simplexSolution)


if __name__ == "__main__":
    exercice = {1: minCostEx, 2: maxFlowEx, 3: maxFlowAsMinCost}
    try:
        ex = int(sys.argv[1])
        exercice[ex]()
    except:
        exercice[1]()
