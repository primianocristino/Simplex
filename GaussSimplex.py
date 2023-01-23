import numpy as np
import copy as cpy
import math
from matplotlib import pyplot as plt


class Fract:
    def __init__(self, num=1, denum=1, dtype=int):

        if type(num) is Fract:
            self.__num = num.getNum()
            self.__denum = num.getDenum()
        else:
            self.__num = num
            self.__denum = denum

        if dtype == float:
            self.simplifyFloat()
        else:
            self.simplifyInt()

    def getNum(self):
        return self.__num

    def getDenum(self):
        return self.__denum

    def __str__(self):
        if self.__num == 0:
            if self.__denum == 0:
                return "inf"
            return "0"
        elif self.__denum == 1:
            return str(self.__num)
        else:
            return str(self.__num)+"/"+str(self.__denum)

    def __float__(self):
        return float(self.__num/self.__denum)

    def __repr__(self):
        return self.__str__()

    def gcd(self, m, n):
        while m % n != 0:
            old_m = m
            old_n = n
            m = old_n
            n = old_m % old_n
        return n

    def simplifyInt(self):
        gcd = self.gcd(self.__num, self.__denum)
        self.__num = int(self.__num // gcd)
        self.__denum = int(self.__denum // gcd)

    def simplifyFloat(self):
        gcd = self.gcd(self.__num, self.__denum)
        self.__num = self.__num / gcd
        self.__denum = self.__denum / gcd

    def __mul__(self, ob1):
        if type(ob1) is not Fract:
            return Fract(self.__num*ob1, self.__denum)
        return Fract(self.__num*ob1.getNum(), self.__denum*ob1.getDenum())

    def __rmul__(self, ob1):
        return self.__mul__(ob1)

    def __truediv__(self, ob1):
        if type(ob1) is not Fract:
            return Fract(self.__num, self.__denum*ob1)
        return Fract(self.__num * ob1.getDenum(), self.__denum*ob1.getNum())

    def __mont__(self, obj):
        pass

    def __add__(self, obj):

        newnum = 0
        newdenom = 0
        if type(obj) is not Fract:
            newnum = self.__num*obj.getDenum() + self.__denum*obj
            newdenom = self.__denum
        else:
            newnum = self.__num*obj.getDenum() + self.__denum*obj.getNum()
            newdenom = self.__denum*obj.getDenum()

        return Fract(newnum, newdenom)

    def __radd__(self, ob1):
        return self.__add__(ob1)

    def __sub__(self, ob1):
        return self.__add__(-1*ob1)

    def __eq__(self, obj):
        if type(obj) is not Fract:
            return self.__num == obj and self.__denum == 1
        return self.__num == obj.getNum() and self.__denum == obj.getDenum()

    def __ne__(self, obj):
        return not self.__eq__(obj)

    def __lt__(self, obj):
        if type(obj) is not Fract:
            obj = Fract(obj, 1)

        return not (self > obj or self == obj)

    def __gt__(self, obj):
        if type(obj) is not Fract:
            obj = Fract(obj, 1)
        return self.__num*obj.getDenum() - self.__denum*obj.getNum() > 0

    def __le__(self, obj):
        if type(obj) is not Fract:
            obj = Fract(obj, 1)

        return self < obj or self == obj

    def __ge__(self, obj):
        if type(obj) is not Fract:
            obj = Fract(obj, 1)

        return self > obj or self == obj


def buildConstraint(Ai, Bi, typeC):
    if typeC == "=" or typeC == "==":
        return ((Ai == Bi))
    elif typeC == "<=" or typeC == "=<":
        return ((Ai <= Bi))
    elif typeC == ">=" or typeC == "=>":
        return ((Ai >= Bi))

    return ((Ai <= Bi))


def showSimplexPolyedre(A, b, c, typeConstraint, typeP, solutionSteps, time=0.8, axis_range=(7, 7), fontsize=11, showNum=True, addAxisCostraints=True):

    if addAxisCostraints:
        typeConstraint = ([">="]*2)+typeConstraint
        axis_constraint = np.array([[1, 0], [0, 1]], dtype=int)
        b_constraint = np.array([0, 0], dtype=int)
        A = np.vstack((axis_constraint, A))
        b = np.hstack((b_constraint, b))
    Afloat = numpyFract(A, convert=float)
    d = np.linspace(-int(axis_range[0]), int(axis_range[1]), 300)

    x, y = np.meshgrid(d, d)

    fig, ax = plt.subplots()

    ax.set_aspect('equal', adjustable='box')
    ax.grid(True)
    if showNum:
        ax.set_xticks(
            np.array(list(range(-int(axis_range[0]), int(axis_range[1]), 1))))
        ax.set_yticks(
            np.array(list(range(-int(axis_range[0]), int(axis_range[1]), 1))))
    plt.pause(time)
    axisX = plt.contour(x, y, y, [0], colors="red", linestyles="dashed")
    axisY = plt.contour(x, y, x, [0], colors="red", linestyles="dashed")

    # disequation = (A[0, 0]*x + A[0, 1]*y <= b[0])
    disequation = None

    for i in range(A.shape[0]):
        plt.pause(time)

        coef0 = A[i, 0]
        coef1 = A[i, 1]

        msg = "Drawing: "

        if coef0 != 0:
            if coef0 == -1:
                msg += "-x"
            elif coef0 != 1:
                msg += str(coef0)+"x"
            else:
                msg += "x"

            if coef1 == -1:
                msg += " - y"
            elif coef1 == 1:
                msg += " + "+"y"
            elif coef1 != 0 and coef1 != 1:
                msg += " + "+str(coef1)+"y"

        else:
            if coef1 == -1:
                msg += "-y"
            elif coef1 == 1:
                msg += "y"
            elif coef1 != 0:
                msg += str(coef1)+"y"

        if typeConstraint[i] == "<=" or typeConstraint[i] == "=<":
            msg += " <= " + str(b[i])
        elif typeConstraint[i] == ">=" or typeConstraint[i] == "=>":
            msg += " >= " + str(b[i])
        elif typeConstraint[i] == "=" or typeConstraint[i] == "==":
            msg += " = " + str(b[i])

        ax.set_title(msg, loc="left", fontsize=fontsize)

        ax.set_title("Costraints plot", loc="right", fontsize=fontsize)
        Ai = Afloat[i, 0]*x + Afloat[i, 1]*y
        Bi = b[i]

        if i <= 0:
            disequation = buildConstraint(
                Ai, Bi, typeConstraint[i])  # ((Ai <= Bi))  # primal
        else:
            disequation &= buildConstraint(Ai, Bi, typeConstraint[i])

        ax.imshow(disequation,
                  extent=(x.min(), x.max(), y.min(), y.max()), origin="lower", cmap="Greens", alpha=0.5)

        Costraint = ax.contour(
            x, y, Ai, [Bi], colors="blue", linestyles="solid")
        plt.pause(time)
    ax.imshow(disequation,
              extent=(x.min(), x.max(), y.min(), y.max()), origin="lower", cmap="Greens", alpha=1)
    ax.set_title("Drawing: finished", loc="left", fontsize=fontsize)

    axisX = plt.contour(x, y, y, [0], colors="red", linestyles="solid")
    axisY = plt.contour(x, y, x, [0], colors="red", linestyles="solid")
    plt.pause(time)

    ##plot scatter##

    ax.set_title(typeP+" " +
                 str(c[0])+"x1"+"%+d" % ((c[1]))+"x2", loc="right", fontsize=fontsize)

    prev_annotate = None

    i = 0
    while True:
        if i >= len(solutionSteps):
            break

        cp = (solutionSteps[i]["x1"], solutionSteps[i]["x2"])
        cp = (fractTofloat(cp[0]), fractTofloat(cp[1]))

        if i > 0:
            if prev_annotate is not None:
                prev_annotate.remove()
                plt.pause(time)

            cur_ann = ax.annotate(text='', xy=oldp, xytext=cp, arrowprops=dict(
                arrowstyle='<-', zorder=4, linewidth=3, animated=True))
            prev_annotate = cur_ann
            plt.pause(time)
            ax.scatter(oldp[0], oldp[1], marker='o',
                       color="yellow", alpha=1, zorder=3)
            plt.pause(time)

        ax.set_title(
            "Drawing: (x1= "+str(solutionSteps[i]["x1"])+", x2= "+str(solutionSteps[i]["x2"])+")", loc="left", fontsize=fontsize)
        ax.scatter(cp[0], cp[1], marker='o', color="green", alpha=1, zorder=3)
        plt.pause(time)
        oldp = cp

        i = i+1

    if prev_annotate is not None:
        prev_annotate.remove()
    plt.pause(0.1)
    plt.show()

    return None


def Gauss(A, pivot=[0, 1]):
    # print("pivot--->: ", pivot)
    denom = A[pivot[0], pivot[1]]
    pivotRow = cpy.deepcopy(A[pivot[0]])
    for i in range(A.shape[0]):
        if i != pivot[0]:

            num = A[i, pivot[1]]
            k = num/denom
            for j in range(A.shape[1]):
                A[i, j] = A[i, j]-(k*pivotRow[j])
        else:

            k = cpy.deepcopy(A[pivot[0], pivot[1]])
            num = A[i, pivot[1]]
            if k != 0:
                for i in range(A.shape[1]):
                    A[pivot[0], i] /= k

    return A


def convert2DualConstraint(A, b, c, typeConstraint):

    A = cpy.deepcopy(A)
    b = cpy.deepcopy(b)
    c = cpy.deepcopy(c)
    typeConstraint = cpy.deepcopy(typeConstraint)
    rowAtoAdd = []
    rowBtoAdd = []

    for i in range(len(typeConstraint)):
        typeC = typeConstraint[i]
        if typeC == "==" or typeC == "=":
            typeConstraint[i] = ">="
            sameArow = cpy.deepcopy(A[i, :])
            sameBrow = cpy.deepcopy(b[i])
            sameArow *= -1
            sameBrow *= -1
            rowAtoAdd.append(sameArow)
            rowBtoAdd.append(sameBrow)

    for el in range(len(rowAtoAdd)):
        A = np.vstack((A, rowAtoAdd[el]))
        b = np.hstack((b, rowBtoAdd[el]))
        typeConstraint.append(">=")

    for i in range(len(typeConstraint)):
        typeC = typeConstraint[i]
        if typeC == ">=" or typeC == "=>":
            pass
        if typeC == "<=" or typeC == "<=":
            A[i, :] = A[i, :]*-1
            b[i] = b[i]*-1
            typeConstraint[i] = ">="

    return A, b, c, typeConstraint


def fromMinToPrimal(A, b, c, typeConstraint):

    A1, b1, c1, t1 = convert2DualConstraint(A, b, c, typeConstraint)

    c2 = b1[:].flatten()
    b2 = c1[:].reshape((c.size, 1)).flatten()
    A2 = A1.T
    t2 = ["<="]*A2.shape[0]

    return A2, b2, c2, t2


def convert2PrimalConstraint(A, b, c, typeConstraint):
    A = cpy.deepcopy(A)
    b = cpy.deepcopy(b)
    c = cpy.deepcopy(c)
    typeConstraint = cpy.deepcopy(typeConstraint)
    rowAtoAdd = []
    rowBtoAdd = []
    for i in range(len(typeConstraint)):
        typeC = typeConstraint[i]
        if typeC == "==" or typeC == "=":
            typeConstraint[i] = ">="
            sameArow = cpy.deepcopy(A[i, :])
            sameBrow = cpy.deepcopy(b[i])
            sameArow *= -1
            sameBrow *= -1
            rowAtoAdd.append(sameArow)
            rowBtoAdd.append(sameBrow)

    for el in range(len(rowAtoAdd)):
        A = np.vstack((A, rowAtoAdd[el]))
        b = np.hstack((b, rowBtoAdd[el]))
        typeConstraint.append(">=")

    for i in range(len(typeConstraint)):
        typeC = typeConstraint[i]
        if typeC == ">=" or typeC == "=>":
            A[i, :] = A[i, :]*-1
            b[i] = b[i]*-1
            typeConstraint[i] = "<="
        if typeC == "<=" or typeC == "<=":
            pass

    return A, b, c, typeConstraint


def fromMaxToDual(A, b, c, typeConstraint):

    A1, b1, c1, t1 = convert2PrimalConstraint(A, b, c, typeConstraint)
    c2 = b1[:].flatten()
    b2 = c1[:].reshape((c.size, 1)).flatten()
    A2 = A1.T
    t2 = [">="]*A2.shape[0]

    return A2, b2, c2, t2


def build_dictPivots(E1, slackCoordList, artCoordList,
                     Psol, PsolSlack, PsolSlackSurplus, PsolSlackSurplusArt, typeP):
    dictPivots = {}
    strColDict = {}
    solutionDict = {}
    #lenColTableu = E1.shape[1]
    lenRowTableu = E1.shape[0]
    # per tutte le colonne di tableu tranne quella di b

    dictPivots[(lenRowTableu-1, 0)] = (lenRowTableu-1, -1)  # P

    totalPivots = slackCoordList+artCoordList
    for i in range(len(totalPivots)):
        totalPivots[i] = (totalPivots[i][0], totalPivots[i]
                          [1]+1)  # aggiungo Psize

    for coord in totalPivots:
        dictPivots[coord] = (coord[0], -1)

    for i in range(E1.shape[1]-1):
        if i == 0:
            strColDict["P"] = i
        if i >= 1 and i < Psol:
            strColDict["x"+str(i-1+1)] = i

        if i >= Psol and i < PsolSlack:
            strColDict["s"+str(i-Psol+1)] = i

        if i >= PsolSlack and i < PsolSlackSurplus:
            strColDict["s"+str(i-PsolSlack+1)] = i

        if i >= PsolSlackSurplus and i < PsolSlackSurplusArt:
            strColDict["a"+str(i-PsolSlackSurplus+1)] = i

        dictPivotKeys = dictPivots.keys()

        for k2 in strColDict.keys():
            solutionDict[k2] = 0
            for k in (dictPivotKeys):
                if k[1] == strColDict[k2]:
                    solutionDict[k2] = E1[(k[0], -1)]

    return dictPivots, strColDict, solutionDict, Psol, PsolSlack, PsolSlackSurplus, PsolSlackSurplusArt


def numpyFract(A, convert=Fract):
    B = np.zeros(A.shape, dtype=object).flatten()

    shape = A.shape
    A = A.flatten()

    for i in range(A.size):
        if convert is Fract:
            B[i] = Fract(A[i])
        elif A[i] is not int:  # A[i] è float
            # se il den ha .0 diventa int, altrimenti lo lascio cosi com'è
            if math.modf(A[i])[0] == float(0):
                B[i] = int(A[i])
            else:
                B[i] = float(A[i])
        else:
            B[i] = int(A[i])

    A = A.reshape(shape)
    return B.reshape(shape)


def fractTofloat(obj):

    if type(obj) == Fract:
        return float(obj.getNum()/obj.getDenum())
    else:
        return obj


def controlConstraint(typeC, i, A, b, forcedArt):

    if typeC == ">=" or typeC == "=>":
        A[i] *= -1

        if b[i] >= 0:
            forcedArt.append(i)

        b[i] *= -1

    elif typeC == "<=" or typeC == "=<":
        if b[i] < 0:
            forcedArt.append(i)

    return A, b, forcedArt


def convert2Tableu(A, b, c, typeP, typeConstraint):

    M = 1000

    c = c.astype('object')
    A = A.astype('object')
    b = b.astype('object')

    countSol = c.size
    A1 = cpy.deepcopy(A)

    # STEP 1
    for i in range(b.size):
        if b[i] < 0:
            A1[i] *= -1
            b[i] *= -1

            if typeConstraint[i] == "<=" or typeConstraint[i] == "=<":

                typeConstraint[i] = ">="
            elif typeConstraint[i] == ">=" or typeConstraint[i] == "=>":
                typeConstraint[i] = "<="

    # STEP 2

    slackCoordinates = []
    countSlack = 0
    for i in range(len(typeConstraint)):
        if typeConstraint[i] == "<=" or typeConstraint[i] == "=<":
            column = np.zeros((b.size, 1), dtype=object)
            column[i] = 1
            A1 = np.hstack((A1, column))
            slackCoordinates.append((i, c.size+countSlack))
            countSlack += 1

    A1.astype("object")
    # STEP 3

    countSlackSuper = countSlack
    countArtinSuper = 0
    artificialCoord = []

    for i in range(len(typeConstraint)):
        if typeConstraint[i] == ">=" or typeConstraint[i] == "=>":
            column = np.zeros((b.size, 1), dtype=object)
            column[i] = -1
            A1 = np.hstack((A1, column))  # aggiungo la surplus
            countSlackSuper += 1

            artificialCoord.append((i, 0))
            countArtinSuper += 1

    A1.astype("object")

    # STEP 4

    countSlackSuperArt = countSlackSuper + countArtinSuper
    for i in range(len(typeConstraint)):

        if typeConstraint[i] == "=" or typeConstraint[i] == "==":
            artificialCoord.append((i, 0))
            countSlackSuperArt += 1

    # STEP 4.5 [ADD ARTIFICIAL VARIABLES]

    restC = np.zeros((1, countSlackSuperArt), dtype=object).flatten()
    c1 = np.hstack((c, restC))
    c1.astype("object")

    for i in range(len(artificialCoord)):
        column = np.zeros((b.size, 1), dtype=object)
        column[artificialCoord[i][0], 0] = 1
        A1 = np.hstack((A1, column))

        if typeP == "max":
            c1[countSol + countSlackSuper+i] = -M
        else:
            c1[countSol + countSlackSuper+i] = M

        artificialCoord[i] = (artificialCoord[i][0],
                              countSol + countSlackSuper+i)

    A1.astype("object")

    c1 *= -1

    D = np.vstack((A1, c1))
    D = D.astype('object')

    pcolumn = np.zeros((b.size+1, 1), dtype=object)
    pcolumn[-1, 0] = 1

    E = np.hstack((pcolumn, D))
    E = E.astype('object')
    E.astype("object")

    b1 = np.hstack((b, np.array([0], dtype=object)))
    b1 = b1.reshape((b1.size, 1))

    F = np.hstack((E, b1))
    F = F.astype('object')

    if typeP != "max":
        for coo in artificialCoord:
            F[-1, :] = (M)*F[coo[0], :] + F[-1, :]
    else:

        for coo in artificialCoord:
            F[-1, :] = (-M)*F[coo[0], :] + F[-1, :]

    tableu = F

    dictPivots, strColDict, solutionDict, Psol, PsolSlack, PsolSlackSurplus, PsolSlackSurplusArt = build_dictPivots(
        tableu, slackCoordinates, artificialCoord,
        countSol+1, countSol+1+countSlack, countSol+1+countSlackSuper, countSol+1+countSlackSuperArt, typeP)

    return numpyFract(tableu), dictPivots, strColDict, solutionDict, Psol


def update_SolutionDict(tableu, dictPivots, strColDict, solutionDict, pivot, typeP):

    dict_pop = None
    pivot_i, pivot_j = pivot

    for k in dictPivots.keys():  # scambio nuovo pivot<->vecchio pivot
        if pivot_i == k[0]:
            dictPivots[(pivot_i, pivot_j)] = dictPivots[k]
            dict_pop = k
            del dictPivots[k]
            break
            print()

    for k in solutionDict:
        solutionDict[k] = 0
    print("popped pivot to swap with", pivot, "<-->", dict_pop, "\n")

    dictPivotKeys = dictPivots.keys()
    for k2 in strColDict.keys():
        solutionDict[k2] = 0
        for k in (dictPivotKeys):
            if k[1] == strColDict[k2]:
                solutionDict[k2] = tableu[(k[0], -1)]

    return dictPivots, solutionDict


def buildGraphicStep(solutionDict, PsolSize):
    i = 0

    nonBasicDict = {}
    for k in solutionDict:
        if i < PsolSize:  # and i > 0:
            nonBasicDict[k] = solutionDict[k]
        elif i >= PsolSize:
            break

        i = i+1

    return [nonBasicDict]


def simplex(A, b, c, typeConstraint, typeP="max"):

    A1 = cpy.deepcopy(A)
    b1 = cpy.deepcopy(b)
    c1 = cpy.deepcopy(c)
    typeConstraint1 = cpy.deepcopy(typeConstraint)
    result = convert2Tableu(A1, b1, c1, typeP, typeConstraint1)

    tableu, dictPivots, strColDict, solutionDict, Psol = result
    print()
    print("==================START=====================")
    print()
    print("Initial matrix")
    print(tableu)

    print()
    print("{(Pivots_coord):(b_coord)}: ", dictPivots)
    print()
    print("Initial solutionDict", solutionDict)
    print()

    stepSolution = buildGraphicStep(solutionDict, Psol)

    debug = True
    while True:
        if typeP != "max":
            pivot_j = np.argmax(tableu[-1:, 1:-1])
            pivot_j += 1
        else:
            pivot_j = np.argmin(tableu[-1:, 1:-1])
            pivot_j += 1
        #####################se la funzione obiettivo(pivot_j) ha tutti i valori non negativi, termina##################
        if (tableu[-1, pivot_j] >= 0 and typeP == "max") or (tableu[-1, pivot_j] <= 0 and typeP != "max"):
            break  # sempre ci deve essere =
        else:
            ################scelta del pivot con cui applicare Gauss#####################
            column = cpy.deepcopy(tableu[:, pivot_j])
            column = column.flatten()

            b1 = cpy.deepcopy(tableu[:, -1]).flatten()
            b1 = b1[:-1]
            column = column[:-1]
            ratio = np.array([-1]*b1.size, dtype=object)
            pivot_i = None

            ratio = np.array([-1]*column.size, dtype=object)
            for i in range(column.size):
                if ((column[i] is not None) and column[i] > 0):  # se il denominatore è 0, rip
                    ratio[i] = b1[i]/column[i]

                    if ratio[i] >= 0:  # se b è 0, il minratio avrà 0, quindi è un caso da escludere il ==0
                        if pivot_i is None:
                            pivot_i = i
                        elif pivot_i is not None and ratio[i] < ratio[pivot_i]:
                            pivot_i = i

            pivot = (pivot_i, pivot_j)
            print("-----------------------------\n")
            tableu = Gauss(tableu, pivot)
            print("Gauss on pivot: ", pivot)
            print("Current Matrix")
            print(tableu)
            print()
            #########################creation of intermadiate solution##############################

            dictPivots, solutionDict = update_SolutionDict(
                tableu, dictPivots, strColDict, solutionDict, pivot, typeP)

            print("{(Pivots_coord):(b_coord)}: ", dictPivots)
            print()
            print("Intermediate solutionDict", solutionDict)
            print()

            stepSolution += buildGraphicStep(solutionDict, Psol)

    print("\n============================\n\nSolution Step")
    print(stepSolution)
    print("\n============================\n\nFinal Solution")
    print(stepSolution[-1])
    print()
    print("THE END")
    return stepSolution


def calculate(A, b, c, typeConstraint, typeP, varAlias=None, showNum=True, axis_range=(7, 7)):
    solutionSteps = simplex(A, b, c, typeConstraint, typeP=typeP)

    finalSolution = solutionSteps[-1]

    outputSolution = {}

    outputSolution["flow"] = finalSolution["P"]
    outputSolution["edges"] = {}

    if len(c) == 2:
        showSimplexPolyedre(A, b, c, typeConstraint, typeP, solutionSteps, time=0.5, axis_range=axis_range,
                            fontsize=11, showNum=showNum)

    if varAlias is not None:
        for variable, value in finalSolution.items():
            if variable != "P":
                index = int(variable.split("x")[1])
                outputSolution["edges"][varAlias[index-1]] = value

        return outputSolution

    return finalSolution
