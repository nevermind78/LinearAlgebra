#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 16:42:29 2019

@author: jecker
"""

from __future__ import division
import numpy as np
import sympy as sp
from sympy.solvers.solveset import linsolve
from IPython.display import display, Latex
import plotly
import plotly.graph_objs as go

plotly.offline.init_notebook_mode(connected=True)
from IPython.core.magic import register_cell_magic
from IPython.display import HTML
import ipywidgets as widgets
import random
from ipywidgets import interact_manual
import matplotlib.pyplot as plt
from scipy.integrate import quad, nquad
from operator import mul
from itertools import cycle, chain
from collections.abc import Iterable


@register_cell_magic
def bgc(color):
    script = (
        "var cell = this.closest('.jp-CodeCell');"
        "var editor = cell.querySelector('.jp-Editor');"
        "editor.style.background='{}';"
        "this.parentNode.removeChild(this)"
    ).format(color)

    display(HTML('<img src onerror="{}">'.format(script)))


###############################################################################

def printMonomial(coeff, index=None, include_zeros=False):
    """Prints the monomial coeff*x_{index} in optimal way

    :param coeff: value of the coefficient
    :type coeff: float
    :param index: index of the monomial. If None, only the numerical value of the coefficient is displayed
    :type index: int or NoneType
    :param include_zeros: if True, monomials of type 0x_n are printed. Defaults to False
    :type include_zeros: bool
    :return: string representative of the monomial
    :rtype: str
    """

    if index is not None:
        coeff = abs(coeff)

    if coeff % 1:
        return str(round(coeff, 3)) + ('x_' + str(index) if index is not None else "")
    elif not coeff:
        if index is None:
            return str(0)
        else:
            return str(0) + 'x_' + str(index) if include_zeros else ""
    elif coeff == 1:
        return 'x_' + str(index) if index is not None else str(int(coeff))
    elif coeff == -1:
        return 'x_' + str(index) if index is not None else str(int(coeff))
    else:
        return str(int(coeff)) + ('x_' + str(index) if index is not None else "")


def printPlusMinus(coeff, include_zeros=False):
    """Prints a plus or minus sign, depending on the sign of the coefficient

    :param coeff: value of the coefficient
    :type coeff: float
    :param include_zeros: if True, 0-coefficients are assigned a "+" sign
    :type include_zeros: bool
    :return: "+" if the coefficient is positive, "-" if it is negative, "" if it is 0
    :rtype: str
    """
    if coeff > 0:
        return "+"
    elif coeff < 0:
        return "-"
    else:
        return "+" if include_zeros else ""


def strEq(n, coeff):
    """Method that provides the Latex string of a linear equation, given the number of unknowns and the values
    of the coefficients. If no coefficient value is provided, then a symbolic equation with `n` unknowns is plotted.
    In particular:

        * **SYMBOLIC EQUATION**: if the number of unknowns is either 1 or 2, then all the equation is
          displayed while, if the number of unknowns is higher than 2, only the first and last term of the equation
          are displayed
        * **NUMERICAL EQUATION**: whichever the number of unknowns, the whole equation is plotted. Numerical values
          of the coefficients are rounded to the third digit

    :param n: number of unknowns of the equation
    :type n: int
    :param coeff: coefficients of the linear equation. It must be [] if a symbolic equation is desired
    :type: list[float]
    :return: Latex string representing the equation
    :rtype: str
    """

    Eq = ''
    if not len(coeff):
        if n is 1:
            Eq = Eq + 'a_1x_1 = b'
        elif n is 2:
            Eq = Eq + 'a_1x_1 + a_2x_2 = b'
        else:
            Eq = Eq + 'a_1x_1 + \ldots + ' + 'a_' + str(n) + 'x_' + str(n) + '= b'
    else:
        all_zeros = len(set(coeff[:-1])) == 1 and not coeff[0]  # check if all lhs coefficients are 0
        start_put_sign = all_zeros
        if n is 1:
            Eq += "-" if coeff[0] < 0 else ""
            Eq += printMonomial(coeff[0], index=1, include_zeros=all_zeros) + "=" + printMonomial(coeff[-1])
        else:
            Eq += "-" if coeff[0] < 0 else ""
            Eq += printMonomial(coeff[0], index=1, include_zeros=all_zeros)
            start_put_sign = start_put_sign or coeff[0] is not 0
            for i in range(1, n):
                Eq += printPlusMinus(coeff[i], include_zeros=all_zeros) if start_put_sign \
                      else "-" if coeff[i] < 0 else ""
                Eq += printMonomial(coeff[i], index=i+1, include_zeros=all_zeros)
                start_put_sign = start_put_sign or coeff[i] is not 0
            Eq += "=" + printMonomial(coeff[-1])
    return Eq


def printEq(coeff, b, *args):
    """Method that prints the Latex string of a linear equation, given the values of the coefficients. If no coefficient
     value is provided, then a symbolic equation with `n` unknowns is plotted. In particular:

        * **SYMBOLIC EQUATION**: if the number of unknowns is either 1 or 2, then all the equation is
          displayed while, if the number of unknowns is higher than 2, only the first and last term of the equation
          are displayed
        * **NUMERICAL EQUATION**: whichever the number of unknowns, the whole equation is plotted. Numerical values
          of the coefficients are rounded to the third digit

    :param coeff: coefficients of the left-hand side of the linear equation
    :type: list[float]
    :param b: right-hand side coefficient of the linear equation
    :type b: float
    :param *args: optional; if passed, it contains the number of unknowns to be considered. If not passed, all the
        unknowns are considered, i.e. n equals the length of the coefficients list
    :type: *args: list
    """

    if len(args) == 1:
        n = args[0]
    else:
        n = len(coeff)
    coeff = coeff + b
    texEq = '$'
    texEq = texEq + strEq(n, coeff)
    texEq = texEq + '$'
    display(Latex(texEq))
    return


def printSyst(A, b, *args):
    """Method that prints a linear system of `n` unknowns and `m` equations. If `A` and `b` are empty, then a symbolic
    system is printed; otherwise a system containing the values of the coefficients stored in `A` and `b`, approximated
    up to their third digit is printed.

    :param A: left-hand side matrix. It must be [] if a symbolic system is desired
    :type: list[list[float]]
    :param b: right-hand side vector. It must be [] if a symbolic system is desired
    :type b: list[float]
    :param args: optional; if not empty, it is a list of two integers representing the number of equations of the
        linear system (i.e. `m`) and the number of unknowns of the system (i.e. `n`)
    :type: list
    """

    if (len(args) == 2) or (len(A) == len(b)):  # ensures that MatCoeff has proper dimensions
        if len(args) == 2:
            m = args[0]
            n = args[1]
        else:
            m = len(A)
            n = len(A[0])

        texSyst = '$\\begin{cases}'
        Eq_list = []
        if len(A) and len(b):
            if type(b[0]) is list:
                b = np.array(b).astype(float)
                A = np.concatenate((A, b), axis=1)
            else:
                A = [A[i] + [b[i]] for i in range(0, m)]  # becomes augmented matrix
            A = np.array(A)  # just in case it's not

        for i in range(m):
            if not len(A) or not len(b):
                Eq_i = ''
                if n is 1:
                    Eq_i = Eq_i + 'a_{' + str(i + 1) + '1}' + 'x_1 = b_' + str(i + 1)
                elif n is 2:
                    Eq_i = Eq_i + 'a_{' + str(i + 1) + '1}' + 'x_1 + ' + 'a_{' + str(i + 1) + '2}' + 'x_2 = b_' + str(
                        i + 1)
                else:
                    Eq_i = Eq_i + 'a_{' + str(i + 1) + '1}' + 'x_1 + \ldots +' + 'a_{' + str(i + 1) + str(
                        n) + '}' + 'x_' + str(n) + '= b_' + str(i + 1)
            else:
                Eq_i = strEq(n, A[i, :])  # attention A is (A|b)
            Eq_list.append(Eq_i)
            texSyst = texSyst + Eq_list[i] + '\\\\'
        texSyst = texSyst + '\\end{cases}$'
        display(Latex(texSyst))
    else:
        print("La matrice des coefficients n'a pas les bonnes dimensions")

    return


def texMatrix(*args):
    """Method which produces the Latex string corresponding to the input matrix.

    .. note:: if two inputs are passed, they represent A and b respectively; as a result the augmented matrix A|B is
      plotted. Otherwise, if the input is unique, just the matrix A is plotted

    :param args: input arguments; they could be either a matrix and a vector or a single matrix
    :type args: list[list] or list[numpy.ndarray]
    :return: Latex string representing the input matrix or the input matrix augmented by the input vector
    :rtype: str
    """

    if len(args) == 2:  # matrice augmentée
        if not type(args[0]) is np.ndarray:
            A = np.array(args[0]).astype(float)
        else:
            A = args[0].astype(float)
        if len(A.shape) <= 1:
            raise ValueError("If two input arguments are passed, the first one must be either a matrix or a column "
                             "vector! Row vectors or empty vectors are not accepted.")
        m = A.shape[1]
        if not type(args[1]) is np.array:
            b = np.array(args[1]).astype(float)
        else:
            b = args[1].astype(float)
        if len(b.shape) <= 1:
            raise ValueError("If two input arguments are passed, the second one must be either a matrix or a column "
                             "vector! Row vectors or empty vectors are not accepted.")
        try:
            assert A.shape[0] == b.shape[0]
        except AssertionError:
            raise ValueError(f"If two input arguments are passed, they must both be either matrices or column vectors, "
                             f"with the same number of rows. In this case, instead, the first input argument has "
                             f"{A.shape[0]} rows, while the second one has {b.shape[0]}")

        A = np.concatenate((A, b), axis=1)
        texApre = '\\left(\\begin{array}{'
        texA = ''
        for i in np.asarray(A):
            texALigne = ''
            texALigne = texALigne + str(round(i[0], 4) if i[0] % 1 else int(i[0]))
            if texA == '':
                texApre = texApre + 'c'
            for j in i[1:m]:
                if texA == '':
                    texApre = texApre + 'c'
                texALigne = texALigne + ' & ' + str(round(j, 4) if j % 1 else int(j))
            if texA == '':
                texApre = texApre + '| c'
            for j in i[m:]:
                if texA == '':
                    texApre = texApre + 'c'
                texALigne = texALigne + ' & ' + str(round(j, 4) if j % 1 else int(j))
            texALigne = texALigne + ' \\\\'
            texA = texA + texALigne
        texA = texApre + '}  ' + texA[:-2] + ' \\end{array}\\right)'
    elif len(args) == 1:  # matrice des coefficients
        if not type(args[0]) is np.ndarray:
            A = np.array(args[0]).astype(float)
        else:
            A = args[0].astype(float)

        texApre = '\\left(\\begin{array}{'
        texApost = ' \\end{array}\\right)'
        texA = ''
        if len(A.shape) == 0 or A.shape[0] == 0:
            return texApre + '}' + texA + texApost
        elif len(A.shape) == 1:
            A = np.expand_dims(A, 0)

        for i in np.asarray(A):
            texALigne = ''
            texALigne = texALigne + str(round(i[0], 4) if i[0] % 1 else int(i[0]))
            if texA == '':
                texApre = texApre + 'c'
            for j in i[1:]:
                if texA == '':
                    texApre = texApre + 'c'
                texALigne = texALigne + ' & ' + str(round(j, 4) if j % 1 else int(j))
            texALigne = texALigne + ' \\\\'
            texA = texA + texALigne
        texA = texApre + '}  ' + texA[:-2] + texApost
    else:
        print("Ce n'est pas une matrice des coefficients ni une matrice augmentée")
        texA = ''

    return texA


def printA(*args, name=None):
    """Method which prints the input matrix.

    .. note:: if two inputs are passed, they represent A and b respectively; as a result the augmented matrix A|B is
      plotted. Otherwise, if the input is unique, just the matrix A is plotted

    :param args: input arguments; they could be either a matrix and a vector or a single matrix
    :type args: list[numpy.ndarray] or list[list]
    :param name: if not None, it is the name of the matrix; what is printed is then {name} = {value}. If None, only the
      matrix value is displayed. Defaults to None
    :type name: str or NoneType
    """

    if name is not None and type(name) is str:
        texA = '$' + name + ' = ' + texMatrix(*args) + '$'
    else:
        texA = '$' + texMatrix(*args) + '$'
    display(Latex(texA))
    return


def printEquMatrices(*args):
    """Method which prints the list of input matrices.

    .. note:: if two inputs are passed, they represent the list of coefficient matrices A and the list of rhs b
      respectively; as a result the augmented matrices A|B are plotted. Otherwise, if the input is unique, just the
      matrices A are plotted

    :param args: input arguments; they could be either a list of matrices and a list of vectors or
        a single list of matrices
    :type args: list
    """

    # list of matrices is M=[M1, M2, ..., Mn] where Mi=(Mi|b)
    if len(args) == 2:
        listOfMatrices = args[0]
        listOfRhS = args[1]
        texEqu = '$' + texMatrix(listOfMatrices[0], listOfRhS[0])
        for i in range(1, len(listOfMatrices)):
            texEqu = texEqu + '\\quad \\sim \\quad' + texMatrix(listOfMatrices[i], listOfRhS[i])
        texEqu = texEqu + '$'
        display(Latex(texEqu))
    else:
        listOfMatrices = args[0]
        texEqu = '$' + texMatrix(listOfMatrices[0])
        for i in range(1, len(listOfMatrices)):
            texEqu = texEqu + '\\quad \\sim \\quad' + texMatrix(listOfMatrices[i])
        texEqu = texEqu + '$'
        display(Latex(texEqu))
    return


def printEquMatricesAug(listOfMatrices, listOfRhS):  # list of matrices is M=[M1, M2, ..., Mn] where Mi=(Mi|b)
    texEqu = '$' + texMatrix(listOfMatrices[0], listOfRhS[0])
    for i in range(1, len(listOfMatrices)):
        texEqu = texEqu + '\\quad \\sim \\quad' + texMatrix(listOfMatrices[i], listOfRhS[i])
    texEqu = texEqu + '$'
    display(Latex(texEqu))


def printLUMatrices(LList, UList):
    """Method which prints the list of L and U matrices, constructed during an interactive LU decomposition routine

   :param LList: list of lower triangular matrices
   :type LList: list[numpy.ndarray]
   :param UList: list of upper triangular matrices
   :type UList: list[numpy.ndarray]
    """

    try:
        assert len(LList) == len(UList)
    except AssertionError:
        print("The lists of lower and upper traingular matrices must have the same length!")
        raise ValueError

    texEqu = '\\begin{align*}'
    for i in range(len(LList)):
        texEqu += 'L &= ' + texMatrix(LList[i]) + '\\qquad & U &= ' + texMatrix(UList[i])
        if i < len(LList) - 1:
            texEqu += ' \\\\ '
    texEqu += '\\end{align*}'
    display(Latex(texEqu))

    return


# %% Functions to enter something

def EnterInt(n=None):
    """Function to allow the user to enter a non-negative integer

    :param n: first integer, passed to the function. If null or negative or None, an integer is requested to the user.
       Defaults to None
    :type n: int or NoneType
    :return: positive integer
    :rtype: int
    """

    while type(n) is not int or (type(n) is int and n <= 0):
        try:
            n = int(n)
            if n <= 0:
                print("Le nombre ne peut pas être négatif o zero!")
                print("Entrez à nouveau: ")
                n = input()
        except:
            if n is not None:
                print("Ce n'est pas un entier!")
                print("Entrez à nouveau:")
                n = input()
            else:
                print("Entrez un entier positif")
                n = input()
    return n


def EnterListReal(n):
    """Function which allows the user to enter a list of `n` real numbers

    :param n: number of real numbers in the desired list
    :type n: int
    :return: list of `n` real numbers
    :rtype: list[float]
    """

    if n < 0:
        print(f"Impossible de générer une liste de {n} nombres réels")
    elif n == 0:
        return []
    else:
        print(f"Entrez une liste de {n} nombres réels")
        coeff = None
        while type(coeff) is not list:
            try:
                coeff = input()
                coeff = [float(eval(x)) for x in coeff.split(',')]
                if len(coeff) != n:
                    print("Vous n'avez pas entré le bon nombre de réels!")
                    print("Entrez à nouveau : ")
                    coeff = input()
            except:
                print("Ce n'est pas le bon format!")
                print("Entrez à nouveau")
                coeff = input()
        return coeff


def SolOfEq(sol, coeff, i):
    """Method that verifies if `sol` is a solution to the linear equation `i`with coefficients `coeff`

    :param sol: candidate solution vector
    :type sol: list
    :param coeff: coefficients of the linear equation
    :type coeff: list
    :param i: index of the equation
    :type i: int
    :return: True if `sol` is a solution, False otherwise
    :rtype: bool
    """

    try:
        assert len(sol) == len(coeff)-1
    except AssertionError:
        print(f"La suite entrée n'est pas une solution de l'équation {i}; Les dimensions ne correspondent pas")
        return False

    A = np.array(coeff[:-1])
    isSol = abs(np.dot(A, sol) - coeff[-1]) < 1e-8
    if isSol:
        print(f"La suite entrée est une solution de l'équation {i}")
    else:
        print(f"La suite entrée n'est pas une solution de l'équation {i}")
    return isSol


def SolOfSyst(solution, A, b):
    """Method that verifies if `solution` is a solution to the linear system with left-hand side matrix `A` and
    right-hand side vector `b`

    :param solution: candidate solution vector
    :type solution: list
    :param A: left-hand side matrix of the linear system
    :type A: list[list[float]] or numpy.ndarray
    :param b: right-hand side vector of the linear system
    :type b: list[float] or numpy.ndarray
    :return: True if `sol` is a solution, False otherwise
    :rtype: bool
    """

    try:
        assert len(solution) == (len(A[0]) if type(A) is list else A.shape[1])
    except AssertionError:
        print(f"La suite entrée n'est pas une solution du système; Les dimensions ne correspondent pas")
        return False

    A = [A[i] + [b[i]] for i in range(0, len(A))]
    A = np.array(A)
    isSol = [SolOfEq(solution, A[i, :], i+1) for i in range(len(A))]
    if all(isSol):
        print("C'est une solution du système")
        return True
    else:
        print("Ce n'est pas une solution du système")
        return False


# PLOTS WITH PLOTLY #

def scatterPlot(x1=None, x2=None, y=None, color=None):
    """Method that realized an interactive scatter plot in 2D or 3D

    :param x1: x data. If None, no plot is realized
    :type x1: list or np.ndarray or NoneType
    :param x2: y data. If None, no pot is realized
    :type x2: list or np.ndarray or NoneType
    :param y: z data. If None, a 2D scatter plot is realized. Defaults to None
    :type y: list or np.ndarray or NoneType
    :param color: color of the points in the scatter plot. If None, defaults to cyan.
    :type color: str or NoneType
    :return: generated plot
    :rtype: plotly.Figure
    """

    if x1 is None or y is None:
        raise ValueError("x1 and y must be given as input to display the data in a scatter plot!")

    col = 'rgb(51, 214, 255)' if color is None else color

    data = []
    if x2 is None:
        points = go.Scatter(x=x1, y=y, marker=dict(symbol='circle', size=8, color=col),
                            mode='markers', name='Data')
    else:
        points = go.Scatter3d(x=x1, y=x2, z=y, marker=dict(symbol='circle', size=5, color=col),
                              mode='markers', name='Data')
    data.append(points)

    layout = go.Layout(
        showlegend=True,
        legend=dict(orientation="h"),
        autosize=True,
        width=600,
        height=600,
        scene=go.layout.Scene(
            xaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230, 230)'
            ),
            yaxis=dict(
                gridcolor='rgb(255, 255, 255)',
                zerolinecolor='rgb(255, 255, 255)',
                showbackground=True,
                backgroundcolor='rgb(230, 230, 230)'
            ),
            zaxis=None if x2 is None else dict(
                                    gridcolor='rgb(255, 255, 255)',
                                    zerolinecolor='rgb(255, 255, 255)',
                                    showbackground=True,
                                    backgroundcolor='rgb(230, 230, 230)'
                                ),
        )
    )

    fig = go.FigureWidget(data=data, layout=layout)
    plotly.offline.iplot(fig)

    return fig


def drawLine(p, d):
    """Method which allows to plot lines, points and arrows in the 2D-place or 3D-space, using plotly library

    :param p: point
    :type p: list[list[float]]
    :param d: direction vector. If made of all zeros, just the reference point is plotted; if different from 0 a line
      passing through `p` and with direction `d` is plotted
    :type d: list[list[float]]
    :return: generated plot
    :rtype: plotly.Figure
    """

    blue = 'rgb(51, 214, 255)'
    colors = [blue]
    colorscale = [[0.0, colors[0]],
                  [0.1, colors[0]],
                  [0.2, colors[0]],
                  [0.3, colors[0]],
                  [0.4, colors[0]],
                  [0.5, colors[0]],
                  [0.6, colors[0]],
                  [0.7, colors[0]],
                  [0.8, colors[0]],
                  [0.9, colors[0]],
                  [1.0, colors[0]]]
    vec = 0.9 * np.array(d)
    if len(p) == 2:
        data = []
        t = np.linspace(-5, 5, 51)
        s = np.linspace(0, 1, 10)
        if all(dd == [0] for dd in d):
            vector = go.Scatter(x=p[0] + s*0, y=p[1] + s*0, marker=dict(symbol=6, size=12, color=colors[0]),
                                name ='Point')
        else:
            trace = go.Scatter(x=p[0] + t * d[0], y=p[1] + t * d[1], name='Droite')
            peak = go.Scatter(x=d[0], y=d[1], marker=dict(symbol=6, size=12, color=colors[0]), showlegend=False)
            vector = go.Scatter(x=p[0] + s * d[0], y=p[1] + s * d[1], mode='lines',
                                line=dict(width=5, color=colors[0]), name='Vecteur directeur')
        zero = go.Scatter(x=t*0, y=t*0, name='Origine', marker=dict(symbol=6, size=12, color=colors[0]),
                          showlegend=False)

        data.append(vector)
        data.append(zero)
        if not all(dd == [0] for dd in d):
            data.append(trace)
            data.append(peak)

        fig = go.FigureWidget(data=data)
        plotly.offline.iplot(fig)
    elif len(p) == 3:
        data = [
            {
                'type': 'cone',
                'x': [1], 'y': vec[1], 'z': vec[2],
                'u': d[0], 'v': d[1], 'w': d[2],
                "sizemode": "absolute",
                'colorscale': colorscale,
                'sizeref': 1,
                "showscale": False,
                'hoverinfo': 'none'
            }
        ]
        t = np.linspace(-5, 5, 51)
        s = np.linspace(0, 1, 10)
        zero = go.Scatter3d(x=t*0, y=t*0, z=t*0, name='Origine', marker=dict(size=3), showlegend=False)
        if all(dd == [0] for dd in d):
            vector = go.Scatter3d(x=p[0] + s*0, y=p[1] + s*0, z=p[2] + s*0, marker=dict(size=5),
                                  name='Point')
        else:
            trace = go.Scatter3d(x=p[0] + t * d[0], y=p[1] + t * d[1], z=p[2] + t * d[2], mode='lines', name='Droite')
            vector = go.Scatter3d(x=p[0] + s * d[0], y=p[1] + s * d[1], z=p[2] + s * d[2], mode='lines',
                                  line=dict(width=5,color=colors[0], dash='solid'), name='Vecteur directeur',
                                  hoverinfo='none')
        data.append(zero)
        data.append(vector)
        if not all(dd == [0] for dd in d):
            data.append(trace)
        layout = {
            'scene': {
                'camera': {
                    'eye': {'x': -0.76, 'y': 1.8, 'z': 0.92}
                }
            }
        }
        fig = go.FigureWidget(data=data, layout=layout)
        plotly.offline.iplot(fig)
    return fig


def Plot2DSys(xL, xR, p, A, b, with_sol=True, with_sol_lstsq=False):
    """Function for the graphical visualization of a 2D system of equations, plotting the straight lines characterizing
    the different equations appearing in the system

    :param xL: left limit of the plot in both coordinates
    :type xL: int or float
    :param xR: right limit of the plot in both coordinates
    :type xR: int or float
    :param p: number of points used to draw the straight lines
    :type p: int
    :param A: matrix of the linear system
    :type A: list[list[float]] or numpy.ndarray
    :param b: right-hand side vector of the linear system
    :type b: list[float] or numpy.ndarray
    :param with_sol: if True, also the solution is displayed. Defaults to True
    :type with_sol: bool
    :param with_sol_lstsq: if True, the Least-Squares solution is displayed, in case the system does not admit any
       solution. Defaults to False.
    :type with_sol_lstsq: bool
    """

    M = [A[i] + [b[i]] for i in range(len(A))]
    A = np.array(A)
    b = np.array(b)
    M = np.array(M)

    t = np.linspace(xL, xR, p)
    data = []
    for i in range(1, len(M)+1):
        if abs(M[i-1, 1]) > abs(M[i-1, 0]):
            trace = go.Scatter(x=t, y=(M[i-1, 2] - M[i-1, 0] * t) / M[i-1, 1], mode='lines', name='Droite %d' % i)
        else:
            trace = go.Scatter(x=(M[i-1, 2] - M[i-1, 1] * t) / M[i-1, 0], y=t, mode='lines', name='Droite %d' % i)
        data.append(trace)

    has_solution = False
    if with_sol:
        A_sp = sp.Matrix(A)
        b_sp = sp.Matrix(b)
        x,y = sp.symbols('x, y')
        sys = A_sp, b_sp

        x_sp = linsolve(sys, x, y)

        if isinstance(x_sp, sp.sets.EmptySet):
            display(Latex(r"Le systéme n'admet pas des solutions!"))
        elif isinstance(x_sp, sp.sets.FiniteSet):
            has_solution = True
            x_sp = x_sp.args[0]
            used_symbols = x_sp.atoms(sp.Symbol)

            if len(used_symbols) == 0:
                trace = go.Scatter(x=[float(x_sp[0])], y=[float(x_sp[1])], mode='markers', name='Solution')
            elif len(used_symbols) == 1:
                if x in used_symbols:
                    yy = sp.lambdify(x, x_sp[1])(t)
                    trace = go.Scatter(x=t, y=yy if isinstance(yy, Iterable) else yy*np.ones_like(t), name='Solution')
                elif y in used_symbols:
                    xx = sp.lambdify(y, x_sp[0])(t)
                    trace = go.Scatter(x=xx if isinstance(xx, Iterable) else xx*np.ones_like(t), y=t, name='Solution')
            else:
                display(Latex(r"Chaque vecteur de $\mathbb{R}^2$ est une solution du systéme!"))

            data.append(trace)

    if with_sol_lstsq and not has_solution:
        A_sp = sp.Matrix(A.T@A)
        b_sp = sp.Matrix(A.T@b)
        x, y = sp.symbols('x, y')
        sys = A_sp, b_sp

        x_sp = linsolve(sys, x, y)

        if isinstance(x_sp, sp.sets.EmptySet):
            display(Latex("Le systeme n'admet pas des solutions au sens du moidres carrées!"))
        elif isinstance(x_sp, sp.sets.FiniteSet):
            display(Latex("La solution du système peut être exprimée comme une solution au sens du moindres carrés."))
            x_sp = x_sp.args[0]
            used_symbols = x_sp.atoms(sp.Symbol)

            if len(used_symbols) == 0:
                trace = go.Scatter(x=[float(x_sp[0])], y=[float(x_sp[1])], mode='markers', name='Solution (MC)')
            elif len(used_symbols) == 1:
                if x in used_symbols:
                    yy = sp.lambdify(x, x_sp[1])(t)
                    trace = go.Scatter(x=t, y=yy if isinstance(yy, Iterable) else yy * np.ones_like(t),
                                       name='Solution (MC)')
                elif y in used_symbols:
                    xx = sp.lambdify(y, x_sp[0])(t)
                    trace = go.Scatter(x=xx if isinstance(xx, Iterable) else xx * np.ones_like(t), y=t,
                                       name='Solution (MC)')
            else:
                display(Latex(r"Chaque vecteur de $\mathbb{R}^2$ est une solution du systéme!"))

            data.append(trace)

    layout = go.Layout(yaxis=dict(scaleanchor="x", scaleratio=1))

    fig = go.Figure(data=data, layout=layout)
    plotly.offline.iplot(fig)

    return


def Plot3DSys(xL, xR, p, A, b, with_sol=True, with_sol_lstsq=False):
    """Function for the graphical visualization of a 3D system of equations, plotting the straight lines characterizing
    the different equations appearing in the system

    :param xL: left limit of the plot in all coordinates
    :type xL: int or float
    :param xR: right limit of the plot in all coordinates
    :type xR: int or float
    :param p: number of points used to draw the straight lines
    :type p: int
    :param A: matrix of the linear system
    :type A: list[list[float]] or numpy.ndarray
    :param b: right-hand side vector of the linear system
    :type b: list[float] or numpy.ndarray
    :param with_sol: if True, also the solution is displayed. Defaults to True
    :type with_sol: bool
    :param with_sol_lstsq: if True, the Least-Squares solution is displayed, in case the system does not admit any
       solution. Defaults to False.
    :type with_sol_lstsq: bool
    """

    A = np.array(A)
    b = np.array(b)
    
    gr = 'rgb(102,255,102)'
    org = 'rgb(255,117,26)'
    cyan = 'rgb(51, 214, 255)'
    yellow = 'rgb(255, 255, 0)'
    purple = 'rgb(255, 0, 255)'
    blue = 'rgb(0, 0, 255)'
    red = 'rgb(255, 0, 0)'
    colors = cycle([cyan, gr, org, yellow, purple, blue])

    s = np.linspace(xL, xR, p)
    t = np.linspace(xL, xR, p)
    tGrid, sGrid = np.meshgrid(s, t)
    data = []
    for i in range(len(A)):
        color = next(colors)
        colorscale = [[0.0, color],
                      [0.1, color],
                      [0.2, color],
                      [0.3, color],
                      [0.4, color],
                      [0.5, color],
                      [0.6, color],
                      [0.7, color],
                      [0.8, color],
                      [0.9, color],
                      [1.0, color]]
        j = i + 1
        
        arg = np.argmax(np.abs(A[i,:]))
        
        # All the coefficient of equation are 0: b==0 -> every combinations are solution, b!=0 -> No solution
        if A[i, arg] == 0:
            if b[i] == 0:
                print("No constraints on equation", j, "of the system.")
            else:
                print("No solution for equation", j, "of the system.")
        # At least a coefficient is different from 0, plot the one with largest coeff in magnitude
        else:
            if arg == 2:  # z en fonction de x,y
                x = sGrid
                y = tGrid
                surface = go.Surface(x=x, y=y, z=(b[i] - A[i, 0] * x - A[i, 1] * y) / A[i, 2],
                                     showscale=False, showlegend=True, colorscale=colorscale, opacity=1,
                                     name='Plan %d' % j)

            elif arg == 1:  # y en fonction de x,z
                x = sGrid
                z = tGrid
                surface = go.Surface(x=x, y=(b[i]-A[i, 0]*x - A[i, 2]*z)/A[i, 1], z=z,
                                     showscale=False, showlegend=True, colorscale=colorscale, opacity=1,
                                     name='Plan %d' % j)

            elif arg == 0:  # x en fonction de y,z
                y = sGrid
                z = tGrid
                surface = go.Surface(x=(b[i] - A[i, 1] * y - A[i,2] * z)/A[i, 0], y=y, z=z,
                                     showscale=False, showlegend=True, colorscale=colorscale, opacity=1,
                                     name='Plan %d' % j)

            data.append(surface)

            layout = go.Layout(
                showlegend=True,  # not there WHY???? --> LEGEND NOT YET IMPLEMENTED FOR SURFACE OBJECTS!!
                legend=dict(orientation="h"),
                autosize=True,
                width=800,
                height=800,
                scene=go.layout.Scene(
                    xaxis=dict(
                        gridcolor='rgb(255, 255, 255)',
                        zerolinecolor='rgb(255, 255, 255)',
                        showbackground=True,
                        backgroundcolor='rgb(230, 230,230)'
                    ),
                    yaxis=dict(
                        gridcolor='rgb(255, 255, 255)',
                        zerolinecolor='rgb(255, 255, 255)',
                        showbackground=True,
                        backgroundcolor='rgb(230, 230,230)'
                    ),
                    zaxis=dict(
                        gridcolor='rgb(255, 255, 255)',
                        zerolinecolor='rgb(255, 255, 255)',
                        showbackground=True,
                        backgroundcolor='rgb(230, 230,230)'
                    )
                ),
                yaxis=dict(scaleanchor="x", scaleratio=1),
            )

    colorscale = [[0.0, red],
                  [0.1, red],
                  [0.2, red],
                  [0.3, red],
                  [0.4, red],
                  [0.5, red],
                  [0.6, red],
                  [0.7, red],
                  [0.8, red],
                  [0.9, red],
                  [1.0, red]]

    has_solution = False
    if with_sol:
        A_sp = sp.Matrix(A)
        b_sp = sp.Matrix(b)
        x, y, z = sp.symbols('x, y, z')
        sys = A_sp, b_sp

        x_sp = linsolve(sys, x, y, z)

        if isinstance(x_sp, sp.sets.EmptySet):
            display(Latex(r"Le systéme n'admet pas des solutions!"))
        elif isinstance(x_sp, sp.sets.FiniteSet):
            has_solution = True
            x_sp = x_sp.args[0]
            used_symbols = x_sp.atoms(sp.Symbol)

            if len(used_symbols) == 0:
                trace = go.Scatter3d(x=[float(x_sp[0])], y=[float(x_sp[1])], z=[float(x_sp[2])],
                                     mode='markers', marker=dict(color=red), name='Solution')
            elif len(used_symbols) == 1:
                if x in used_symbols:
                    yy = sp.lambdify([x], x_sp[1])(s)
                    zz = sp.lambdify([x], x_sp[2])(s)
                    trace = go.Scatter3d(x=s,
                                         y=yy if isinstance(yy, Iterable) else yy * np.ones_like(s),
                                         z=zz if isinstance(zz, Iterable) else zz * np.ones_like(s),
                                         name='Solution',
                                         marker=dict(color=red),
                                         line=go.scatter3d.Line(color=red))
                elif y in used_symbols:
                    xx = sp.lambdify([y], x_sp[0])(s)
                    zz = sp.lambdify([y], x_sp[2])(s)
                    trace = go.Scatter3d(x=xx if isinstance(xx, Iterable) else xx * np.ones_like(s),
                                         y=s,
                                         z=zz if isinstance(zz, Iterable) else zz * np.ones_like(s),
                                         name='Solution',
                                         marker=dict(color=red),
                                         line=go.scatter3d.Line(color=red))
                elif z in used_symbols:
                    xx = sp.lambdify([z], x_sp[0])(s)
                    yy = sp.lambdify([z], x_sp[1])(s)
                    trace = go.Scatter3d(x=xx if isinstance(xx, Iterable) else xx * np.ones_like(s),
                                         y=yy if isinstance(yy, Iterable) else yy * np.ones_like(s),
                                         z=s,
                                         name='Solution',
                                         marker=dict(color=red),
                                         line=go.scatter3d.Line(color=red))
            elif len(used_symbols) == 2:
                if x in used_symbols and y in used_symbols:
                    zz = sp.lambdify([x, y], x_sp[2])(sGrid, tGrid)
                    trace = go.Surface(x=sGrid,
                                       y=tGrid,
                                       z=zz if isinstance(zz, Iterable) else zz * np.ones_like(tGrid),
                                       showscale=False, showlegend=True, colorscale=colorscale, opacity=1,
                                       name='Solution')
                elif x in used_symbols and z in used_symbols:
                    yy = sp.lambdify([x, z], x_sp[1])(sGrid, tGrid)
                    trace = go.Surface(x=sGrid,
                                       y=yy if isinstance(yy, Iterable) else yy * np.ones_like(tGrid),
                                       z=tGrid,
                                       showscale=False, showlegend=True, colorscale=colorscale, opacity=1,
                                       name='Solution')
                elif y in used_symbols and z in used_symbols:
                    xx = sp.lambdify([y, z], x_sp[0])(sGrid, tGrid)
                    trace = go.Surface(x=xx if isinstance(xx, Iterable) else xx * np.ones_like(tGrid),
                                       y=sGrid,
                                       z=tGrid,
                                       showscale=False, showlegend=True, colorscale=colorscale, opacity=1,
                                       name='Solution')
            else:
                display(Latex(r"Chaque vecteur de $\mathbb{R}^3$ est une solution du systéme!"))

            data.append(trace)

    if with_sol_lstsq and not has_solution:
        A_sp = sp.Matrix(A.T @ A)
        b_sp = sp.Matrix(A.T @ b)
        x, y, z = sp.symbols('x, y, z')
        sys = A_sp, b_sp

        x_sp = linsolve(sys, x, y, z)

        if isinstance(x_sp, sp.sets.EmptySet):
            display(Latex("Le systeme n'admet pas des solutions au sens du moidres carrées!"))
        elif isinstance(x_sp, sp.sets.FiniteSet):
            display(Latex("La solution du système peut être exprimée comme une solution au sens du moindres carrés."))
            x_sp = x_sp.args[0]

            used_symbols = x_sp.atoms(sp.Symbol)

            if len(used_symbols) == 0:
                trace = go.Scatter3d(x=[float(x_sp[0])], y=[float(x_sp[1])], z=[float(x_sp[2])],
                                     mode='markers', marker=dict(color=red), name='Solution (MC)')
            elif len(used_symbols) == 1:
                if x in used_symbols:
                    yy = sp.lambdify([x], x_sp[1])(s)
                    zz = sp.lambdify([x], x_sp[2])(s)
                    trace = go.Scatter3d(x=s,
                                         y=yy if isinstance(yy, Iterable) else yy * np.ones_like(s),
                                         z=zz if isinstance(zz, Iterable) else zz * np.ones_like(s),
                                         name='Solution (MC)',
                                         marker=dict(color=red),
                                         line=go.scatter3d.Line(color=red))
                elif y in used_symbols:
                    xx = sp.lambdify([y], x_sp[0])(s)
                    zz = sp.lambdify([y], x_sp[2])(s)
                    trace = go.Scatter3d(x=xx if isinstance(xx, Iterable) else xx * np.ones_like(s),
                                         y=s,
                                         z=zz if isinstance(zz, Iterable) else zz * np.ones_like(s),
                                         name='Solution (MC)',
                                         marker=dict(color=red),
                                         line=go.scatter3d.Line(color=red))
                elif z in used_symbols:
                    xx = sp.lambdify([z], x_sp[0])(s)
                    yy = sp.lambdify([z], x_sp[1])(s)
                    trace = go.Scatter3d(x=xx if isinstance(xx, Iterable) else xx * np.ones_like(s),
                                         y=yy if isinstance(yy, Iterable) else yy * np.ones_like(s),
                                         z=s,
                                         name='Solution (MC)',
                                         marker=dict(color=red),
                                         line=go.scatter3d.Line(color=red))
            elif len(used_symbols) == 2:
                if x in used_symbols and y in used_symbols:
                    zz = sp.lambdify([x, y], x_sp[2])(sGrid, tGrid)
                    trace = go.Surface(x=sGrid,
                                       y=tGrid,
                                       z=zz if isinstance(zz, Iterable) else zz * np.ones_like(tGrid),
                                       showscale=False, showlegend=True, colorscale=colorscale, opacity=1,
                                       name='Solution (MC)')
                elif x in used_symbols and z in used_symbols:
                    yy = sp.lambdify([x, z], x_sp[1])(sGrid, tGrid)
                    trace = go.Surface(x=sGrid,
                                       y=yy if isinstance(yy, Iterable) else yy * np.ones_like(tGrid),
                                       z=tGrid,
                                       showscale=False, showlegend=True, colorscale=colorscale, opacity=1,
                                       name='Solution (MC)')
                elif y in used_symbols and z in used_symbols:
                    xx = sp.lambdify([y, z], x_sp[0])(sGrid, tGrid)
                    trace = go.Surface(x=xx if isinstance(xx, Iterable) else xx * np.ones_like(tGrid),
                                       y=sGrid,
                                       z=tGrid,
                                       showscale=False, showlegend=True, colorscale=colorscale, opacity=1,
                                       name='Solution (MC)')
            else:
                display(Latex(r"Chaque vecteur de $\mathbb{R}^3$ est une solution du systéme "
                              r"au sens du moidres carrées!"))

            data.append(trace)
    
    fig = go.Figure(data=data, layout=layout)
    plotly.offline.iplot(fig)

    return


def vector_plot_2D(vects, is_vect=True, orig=None, labels=None, show=True):
    """Method to plot 2D vectors in plotly

    :param vects: vectors to plot
    :type vects: list[list] or list[np.ndarray]
    :param is_vect:
    :type is_vect: bool
    :param orig: location of the origin
    :type orig: list
    :param labels: labels of the vectors to plot. If None, the vectors are named as vector1, vector2 etc.
    :type labels: list[str] or NoneType
    :param show: if True, the plot is displayed. Defaults to True
    :type show: bool
    """

    if orig is None:
        orig = [0, 0]

    if is_vect:
        if not hasattr(orig[0], "__iter__"):
            coords = [[orig, np.sum([orig, v], axis=0)] for v in vects]
        else:
            coords = [[o, np.sum([o,v], axis=0)] for o,v in zip(orig, vects)]
    else:
        coords = vects

    data = []
    for i,c in enumerate(coords):
        X1, Y1 = zip(c[0])
        X2, Y2 = zip(c[1])
        vector = go.Scatter(x=[X1[0], X2[0]],
                            y=[Y1[0], Y2[0]],
                            marker=dict(size=[0, 2],
                                        color=['blue'],
                                        line=dict(width=2,
                                                  color='DarkSlateGrey')),
                            name='Vector'+str(i+1) if labels is None else labels[i])
        data.append(vector)

    layout = go.Layout(
             margin=dict(l=4, r=4, b=4, t=4),
             width=600,
             height=400,
             yaxis=dict(scaleanchor="x", scaleratio=1),
             scene=go.layout.Scene(
                xaxis=dict(
                    gridcolor='rgb(255, 255, 255)',
                    zerolinecolor='rgb(255, 255, 255)',
                    showbackground=True,
                    backgroundcolor='rgb(230, 230,230)'
                ),
                yaxis=dict(
                    gridcolor='rgb(255, 255, 255)',
                    zerolinecolor='rgb(255, 255, 255)',
                    showbackground=True,
                    backgroundcolor='rgb(230, 230,230)'
                ),
                zaxis=dict(
                    gridcolor='rgb(255, 255, 255)',
                    zerolinecolor='rgb(255, 255, 255)',
                    showbackground=True,
                    backgroundcolor='rgb(230, 230,230)'
                )
             )
                  )
    fig = go.Figure(data=data, layout=layout)

    if show:
        fig.show()

    return fig


def vector_plot_3D(vects, is_vect=True, orig=None, labels=None, show=True):
    """Method to plot 3D vectors in plotly

    :param vects: vectors to plot
    :type vects: list[list] or list[np.ndarray]
    :param is_vect:
    :type is_vect: bool
    :param orig: location of the origin(s)
    :type orig: list or list[list]
    :param labels: labels of the vectors to plot. If None, the vectors are named as vector1, vector2 etc.
    :type labels: list[str] or NoneType
    :param show: if True, the plot is displayed. Defaults to True
    :type show: bool
    """

    if orig is None:
        orig = [0, 0, 0]

    if is_vect:
        if not hasattr(orig[0], "__iter__"):
            coords = [[orig, np.sum([orig, v], axis=0)] for v in vects]
        else:
            coords = [[o, np.sum([o,v], axis=0)] for o,v in zip(orig, vects)]
    else:
        coords = vects

    data = []
    for i,c in enumerate(coords):
        X1, Y1, Z1 = zip(c[0])
        X2, Y2, Z2 = zip(c[1])
        vector = go.Scatter3d(x=[X1[0], X2[0]],
                              y=[Y1[0], Y2[0]],
                              z=[Z1[0], Z2[0]],
                              marker=dict(size=[0, 2],
                                          color=['blue'],
                                          line=dict(width=2,
                                                    color='DarkSlateGrey')),
                              name='Vector'+str(i+1) if labels is None else labels[i])
        data.append(vector)

    layout = go.Layout(
             margin=dict(l=4, r=4, b=4, t=4),
             yaxis=dict(scaleanchor="x", scaleratio=1),
             scene=go.layout.Scene(
                xaxis=dict(
                    gridcolor='rgb(255, 255, 255)',
                    zerolinecolor='rgb(255, 255, 255)',
                    showbackground=True,
                    backgroundcolor='rgb(230, 230,230)'
                ),
                yaxis=dict(
                    gridcolor='rgb(255, 255, 255)',
                    zerolinecolor='rgb(255, 255, 255)',
                    showbackground=True,
                    backgroundcolor='rgb(230, 230,230)'
                ),
                zaxis=dict(
                    gridcolor='rgb(255, 255, 255)',
                    zerolinecolor='rgb(255, 255, 255)',
                    showbackground=True,
                    backgroundcolor='rgb(230, 230,230)'
                )
             )
                  )
    fig = go.Figure(data=data, layout=layout)

    if show:
        fig.show()

    return fig


def plot_line(B, data=None, layout=None, label=""):
    """Function to plot a plane in the 2D-3D space

    :param B: list of 2D-3D vectors generating the line
    :type B: list[list]
    :param data: data of the figure. defaults to None
    :type data: list or NoneType
    :param layout: layout of the figure. Defaults to None
    :type layout: plotly.layout or NoneType
    :param label: name of the line in the legend. Defaults to "".
    :type label: str
    :return: figure
    :rtype: plotly.figure
    """

    if data is None:
        data = []

    if layout is None:
        layout = go.Layout(
            margin=dict(l=4, r=4, b=4, t=4),
            yaxis=dict(scaleanchor="x", scaleratio=1),
            scene=go.layout.Scene(
                xaxis=dict(
                    gridcolor='rgb(255, 255, 255)',
                    zerolinecolor='rgb(255, 255, 255)',
                    showbackground=True,
                    backgroundcolor='rgb(230, 230,230)'
                ),
                yaxis=dict(
                    gridcolor='rgb(255, 255, 255)',
                    zerolinecolor='rgb(255, 255, 255)',
                    showbackground=True,
                    backgroundcolor='rgb(230, 230,230)'
                ),
                zaxis=dict(
                    gridcolor='rgb(255, 255, 255)',
                    zerolinecolor='rgb(255, 255, 255)',
                    showbackground=True,
                    backgroundcolor='rgb(230, 230,230)'
                )
            )
        )

    try:
        assert len(B) == 1 and \
               (all([len(B[i]) == 2 for i in range(len(B))]) or all([len(B[i]) == 3 for i in range(len(B))]))
    except AssertionError:
        raise TypeError("L'entrée doit être une liste d'un élément, "
                        "dont chacun est une liste de deux ou trois nombres.")

    x = np.linspace(-1.5 * abs(B[0][0]),
                    1.5 * abs(B[0][0]), 2)
    y = np.linspace(-1.5 * abs(B[0][1]),
                    1.5 * abs(B[0][1]), 2)
    if len(B[0]) == 3:
        z = np.linspace(-1.5 * abs(B[0][2]),
                        1.5 * abs(B[0][2]), 2)

    coeffs = [B[0][0], B[0][1]]
    if len(B[0]) == 3:
        coeffs.append(B[0][2])

    if coeffs[0] != 0:
        new_coeffs = [coeffs[1] / coeffs[0]] if len(coeffs) == 2 \
            else [coeffs[1] / coeffs[0], coeffs[2] / coeffs[0]]
        ref_index = 0
    elif coeffs[1] != 0:
        new_coeffs = [coeffs[0] / coeffs[1]] if len(coeffs) == 2 \
            else [coeffs[0] / coeffs[1], coeffs[2] / coeffs[1]]
        ref_index = 1
    elif len(coeffs) == 3 and coeffs[2] != 0:
        new_coeffs = [coeffs[0] / coeffs[2]] if len(coeffs) == 2 \
            else [coeffs[0] / coeffs[2], coeffs[1] / coeffs[2]]
        ref_index = 2
    else:
        return go.Figure(data=data, layout=layout)

    if len(B[0]) == 2:
        line = go.Scatter(x=x if ref_index == 0 else np.zeros_like(x),
                          y=new_coeffs[0] * x if ref_index == 0 else y,
                          marker=dict(size=[0, 0],
                                      color=['rgb(255,165,0)'],
                                      line=dict(width=1,
                                                color='rgb(255,165,0)')),
                          name=label)
    elif len(B[0]) == 3:
        line = go.Scatter3d(x=x if ref_index == 0 else np.zeros_like(x),
                            y=new_coeffs[0] * x if ref_index == 0
                            else y if ref_index == 1 else np.zeros_like(x),
                            z=new_coeffs[1] * x if ref_index == 0
                            else new_coeffs[1] * y if ref_index == 1
                            else z,
                            marker=dict(size=[0, 0],
                                        color=['rgb(255,165,0)'],
                                        line=dict(width=1,
                                                  color='rgb(255,165,0)')),
                            name=label)

    data.append(line)

    return go.Figure(data=data, layout=layout)


def plot_plane(B, color='rgb(0,255,255)', data=None, layout=None, label=""):
    """Function to plot a plane in the 3D space

    :param B: list of 3D vectors generating the plane
    :type B: list[list]
    :param data: data of the figure. defaults to None
    :type data: list or NoneType
    :param layout: layout of the figure. Defaults to None
    :type layout: plotly.layout or NoneType
    :param color: color of the plane to be drawn. Defaults to 'rgb(0,255,255)' (orange)
    :type color: str
    :param label: name of the plane in the legend
    :type label: str
    :return: figure
    :rtype: plotly.figure
    """

    if data is None:
        data = []

    if layout is None:
        layout = go.Layout(
            margin=dict(l=4, r=4, b=4, t=4),
            yaxis=dict(scaleanchor="x", scaleratio=1),
            scene=go.layout.Scene(
                xaxis=dict(
                    gridcolor='rgb(255, 255, 255)',
                    zerolinecolor='rgb(255, 255, 255)',
                    showbackground=True,
                    backgroundcolor='rgb(230, 230,230)'
                ),
                yaxis=dict(
                    gridcolor='rgb(255, 255, 255)',
                    zerolinecolor='rgb(255, 255, 255)',
                    showbackground=True,
                    backgroundcolor='rgb(230, 230,230)'
                ),
                zaxis=dict(
                    gridcolor='rgb(255, 255, 255)',
                    zerolinecolor='rgb(255, 255, 255)',
                    showbackground=True,
                    backgroundcolor='rgb(230, 230,230)'
                )
            )
        )

    colorscale = [[0.0, color],
                  [0.1, color],
                  [0.2, color],
                  [0.3, color],
                  [0.4, color],
                  [0.5, color],
                  [0.6, color],
                  [0.7, color],
                  [0.8, color],
                  [0.9, color],
                  [1.0, color]]

    try:
        assert len(B) == 2 and all([len(B[i]) == 3 for i in range(len(B))])
    except AssertionError:
        raise TypeError("L'entrée doit être une liste d'un élément, dont chacun est une liste de trois nombres.")

    coeffs = [(B[0][1] * B[1][2] - B[1][1] * B[0][2]),
              (B[1][0] * B[0][2] - B[0][0] * B[1][2]),
              (B[0][0] * B[1][1] - B[1][0] * B[0][1])]
    length = np.linspace(-1.5 * max(abs(B[0][0]), abs(B[1][0])),
                         1.5 * max(abs(B[0][0]), abs(B[1][0])), 100)
    width = np.linspace(-1.5 * max(abs(B[0][1]), abs(B[1][1])),
                        1.5 * max(abs(B[0][1]), abs(B[1][1])), 100)
    height = np.linspace(-1.5 * max(abs(B[0][2]), abs(B[1][2])),
                         1.5 * max(abs(B[0][2]), abs(B[1][2])), 100)

    if coeffs[2] != 0:
        Grid1, Grid2 = np.meshgrid(length, width)
        x = Grid1
        y = Grid2
        new_coeffs = [-coeffs[0] / coeffs[2], -coeffs[1] / coeffs[2]]
        ref_index = 2
    elif coeffs[1] != 0:
        Grid1, Grid2 = np.meshgrid(length, height)
        x = Grid1
        z = Grid2
        new_coeffs = [-coeffs[0] / coeffs[1], -coeffs[2] / coeffs[1]]
        ref_index = 1
    elif coeffs[0] != 0:
        Grid1, Grid2 = np.meshgrid(width, height)
        y = Grid1
        z = Grid2
        new_coeffs = [-coeffs[1] / coeffs[0], -coeffs[2] / coeffs[0]]
        ref_index = 0
    else:
        return go.Figure(data=data, layout=layout)

    surface = go.Surface(x=x if ref_index != 0 else new_coeffs[0] * y + new_coeffs[1] * z,
                         y=y if ref_index != 1 else new_coeffs[0] * x + new_coeffs[1] * z,
                         z=z if ref_index != 2 else new_coeffs[0] * x + new_coeffs[1] * y,
                         showscale=False, showlegend=True, opacity=0.25, colorscale=colorscale, name=label)

    data.append(surface)

    return go.Figure(data=data, layout=layout)


def isDiag(M):
    """Method which checks if a matrix is diagonal

    :param M: input matrix
    :type M: list[list[float]] or numpy.ndarray
    :return: True if M is diagonal else False
    :rtype: bool
    """

    if not type(M) is np.ndarray:
        M = np.array(M)

    i, j = M.shape
    try:
        assert i == j
    except AssertionError:
        print("A non-squared matrix cannot be diagonal!")
        return False

    test = M.reshape(-1)[:-1].reshape(i - 1, j + 1)
    return ~np.any(test[:, 1:] >= 1e-10)


def isSym(M):
    """Method which checks if a matrix is symmetric

    :param M: input matrix
    :type M: list[list[float]] or numpy.ndarray
    :return: True if M is symmetric else False
    :rtype: bool
    """

    if not type(M) is np.ndarray:
        M = np.array(M)

    i, j = M.shape
    try:
        assert i == j
    except AssertionError:
        print("A non-squared matrix cannot be symmetric!")
        return False

    return ~np.any(M - np.transpose(M) >= 1e-10)


# ECHELONNAGE #

def echZero(indice, M):
    """Method which sets to zero the entries of matrix M that correspond to a True value in the boolean vector indice

    :param indice: vector of booleans; if an element is True, it means that the element with the corresponding index in
       matrix M must be set to 0
    :type indice: list[bool]
    :param M: matrix to be processed
    :type: numpy.ndarray
    :return: processed matrix M, where the given entries have been properly set to 0
    :rtype: numpy.ndarray
    """

    Mat = M[np.logical_not(indice), :].ravel()
    Mat = np.concatenate([Mat, M[indice, :].ravel()])
    Mat = Mat.reshape(len(M), len(M[0, :]))
    return Mat


def Eij(M, i, j, get_E_inv=False):
    """Method to swap line `i` and line `j` of matrix `M`

    :param M: matrix to be processed
    :type M: numpy.ndarray
    :param i: first line index
    :type i: int
    :param j: second line index
    :type j: int
    :param get_E_inv: if True, the inverse matrix of the applied elementary operation is returned. Defaults to False
    :type get_E_inv: bool
    :return: processed matrix, with line `i` and `j` having been swapped
    :rtype: numpy.ndarray
    """

    M = np.array(M)
    M[[i, j], :] = M[[j, i], :]

    if get_E_inv:
        L = np.eye(M.shape[0], M.shape[0]).astype(float)
        L[[i, j]] = L[[j, i]]
        
    if get_E_inv:
        return M, L
    else:
        return M


def Ealpha(M, i, alpha, get_E_inv=False):
    """Method to multiply line `i` of matrix `M` by the scalar coefficient `alpha`

    :param M: matrix to be processed
    :type M: numpy.ndarray
    :param i: reference line index
    :type i: int
    :param alpha: scalar coefficient
    :type alpha: float
    :param get_E_inv: if True, the inverse matrix of the applied elementary operation is returned. Defaults to False
    :type get_E_inv: bool
    :return: processed matrix, with line `i` multiplied by the scalar `alpha`
    :rtype: numpy.ndarray
    """

    M = np.array(M)
    M[i, :] = alpha * M[i, :]

    if get_E_inv:
        L = np.eye(M.shape[0], M.shape[0]).astype(float)
        L[i ,i] = 1 / alpha

    if get_E_inv:
        return M, L
    else:
        return M


def Eijalpha(M, i, j, alpha, get_E_inv=False):
    """Method to add to line `i` of matrix `M` line `j` of the same matrix, multiplied by the scalar coefficient `alpha`

    :param M: matrix to be processed
    :type M: numpy.ndarray
    :param i: line to be modified
    :type i: int
    :param j: line whose multiple has tobe added to line `i`
    :type j: int
    :param alpha: scalar coefficient
    :type alpha: float
    :param get_E_inv: if True, the inverse matrix of the applied elementary operation is returned. Defaults to False
    :type get_E_inv: bool
    :return: processed matrix, with line `i` being summed up with line `j` multiplied by `alpha`
    :rtype: numpy.ndarray
    """

    M = np.array(M)
    M[i, :] = M[i, :] + alpha * M[j, :]

    if get_E_inv:
        L = np.eye(M.shape[0], M.shape[0]).astype(float)
        L[i, j] = -alpha

    if get_E_inv:
        return M, L
    else:
        return M


def echelonMat(ech, *args):
    """Method to perform Gauss elimination on either the matrix of the coefficients (if `len(args)==1`) or on the
    augmented matrix (if `len(args)==2`); the elimination can be either in standard form (if `ech=='E` or in reduced
    form (if `ech=='ER'`).

    :param ech:
    :type ech:
    :param args:
    :type args:
    :return:
    :rtype:
    """

    if len(args) == 2:  # matrice augmentée
        A = np.array(args[0]).astype(float)
        m = A.shape[0]
        n = A.shape[1]
        b = np.array(args[1])

        b = np.reshape(b, (m, 1))
        A = np.concatenate((A, b), axis=1)

    else:  # matrice coeff
        A = np.array(args[0]).astype(float)
        m = A.shape[0]
        n = A.shape[1]

    if ech in {'E', 'ER'}:  # Echelonnée

        Mat = np.array(A)
        Mat = Mat.astype(float)  # in case the array in int instead of float.
        numPivot = 0
        for i in range(len(Mat)):
            j = i
            goOn = True
            if all(abs(Mat[j:, i]) < 1e-15) and j != len(Mat[0, :]) - 1:  # if column (or rest of) is 0, take next
                goOn = False
            if j == len(Mat[0, :]) - 1:
                Mat[i+1:len(Mat), :] = 0
                break
            if goOn:
                if abs(Mat[i, j]) < 1e-15:
                    Mat[i, j] = 0
                    zero = abs(Mat[i:, j]) < 1e-15
                    M = echZero(zero, Mat[i:, :])
                    Mat[i:, :] = M
                Mat = Ealpha(Mat, i, 1 / Mat[i, j])  # usually Mat[i,j]!=0
                for k in range(i + 1, len(A)):
                    Mat = Eijalpha(Mat, k, i, -Mat[k, j])
                    # Mat[k,:]=[0 if abs(Mat[k,l])<1e-15 else Mat[k,l] for l in range(len(MatCoeff[0,:]))]
                numPivot += 1
                Mat[abs(Mat) < 1e-15] = 0

        print("La forme échelonnée de la matrice est")
        if len(args) == 2:
            printEquMatrices([A[:, :n], Mat[:, :n]], [A[:, n:], Mat[:, n:]])
        else:
            printEquMatrices([A, Mat])

    if ech == 'ER':  # Echelonnée réduite

        Mat = np.array(Mat)
        i = len(Mat) - 1
        while i >= 1:
            j = i  # we can start at pos ij at least the pivot is there
            goOn = True
            if all(abs(Mat[i, :len(Mat[0]) - 1]) < 1e-15) and i != 0:  # if row (or rest of) is zero, take next
                goOn = False
            if goOn:
                # we have a row with one non-nul element
                if abs(Mat[i, j]) < 1e-15:  # if element Aij=0 take next one --> find pivot
                    j += 1
                # Aij!=0 and Aij==1 if echelonMat worked
                for k in range(i):  # put zeros above pivot (which is 1 now)
                    Mat = Eijalpha(Mat, k, i, -Mat[k, j])
            i -= 1

        print("La forme échelonnée réduite de la matrice est")
        if len(args) == 2:
            printEquMatrices([A[:, :n], Mat[:, :n]], [A[:, n:], Mat[:, n:]])
        else:
            printEquMatrices([A, Mat])

        if (Mat[:min(m,n), :min(m,n)] == np.eye(min(m,n))).all():
            print("La matrice peut être réduite à la matrice d'identité")
        else:
            print("La matrice ne peut pas être réduite à la matrice d'identité")

    if ech != 'ER' and ech != 'E':
        print(f"Méthode d'échelonnage non reconnue {ech}. Méthodes disponibles: 'E' (pour la forme échelonnée standard)"
              f", 'ER' (pour la forme échelonnée réduite))")

    return np.asmatrix(Mat)


def randomA():
    """Method which generates a random matrix with rows and columns within 1 and 10 and integer entries between -100
    and 100

    :return: generated random matrix
    :rtype: numpy.ndarray
    """
    n = random.randint(1, 10)
    m = random.randint(1, 10)
    A = [[random.randint(-100, 100) for i in range(n)] for j in range(m)]
    printA(A)
    return np.array(A)


def dimensionA(A):
    """Method which allows the user to enter the matrix dimensions and verifies whether they are correct or not

    :param A: reference matrix
    :type A: numpy.ndarray
    """
    m = widgets.IntText(
        value=1,
        step=1,
        description='m:',
        disabled=False
    )
    n = widgets.IntText(
        value=1,
        step=1,
        description='n:',
        disabled=False
    )

    display(m)
    display(n)

    def f():
        if m.value == A.shape[0] and n.value == A.shape[1]:
            print('Correct!')
        else:
            print('Incorrect, entrez de nouvelles valeurs')

    interact_manual(f)
    return


def manualEch(*args):
    """Method which allows the user to perform the Gauss elimination method on the given input matrix, eventually
    extended by the right-hand side vector.

    :param args:
    :type args:
    :return:
    :rtype:
    """

    if len(args) == 2:  # matrice augmentée
        A = np.array(args[0]).astype(float)
        m = A.shape[0]
        b = args[1]
        if type(b[0]) is list:
            b = np.array(b).astype(float)
            A = np.concatenate((A, b), axis=1)
        else:
            b = [b[i] for i in range(m)]
            A = [A[i] + [b[i]] for i in range(m)]
    else:
        A = np.array(args[0]).astype(float)
        m = A.shape[0]

    j = widgets.BoundedIntText(
        value=1,
        min=1,
        max=m,
        step=1,
        description='Ligne j:',
        disabled=False
    )
    i = widgets.BoundedIntText(
        value=1,
        min=1,
        max=m,
        step=1,
        description='Ligne i:',
        disabled=False
    )

    r = widgets.RadioButtons(
        options=['Tij', 'Di(alpha)', 'Lij(alpha)', 'Revert'],
        description='Opération:',
        disabled=False
    )

    alpha = widgets.Text(
        value='1',
        description='Coeff. alpha:',
        disabled=False
    )
    print("Régler les paramètres et évaluer la cellule suivante")
    print("Répéter cela jusqu'à obtenir une forme échelonnée réduite")
    display(r)
    display(i)
    display(j)
    display(alpha)
    return i, j, r, alpha


def echelonnage(i, j, r, alpha, A, m=None, *args):
    """Method which performs the Gauss elimination method step described by `r.value` with parameters `ì`, `j` and
    `alpha` on matrix `A`

    :param i: first reference line
    :type i: ipywidgets.Text
    :param j: second reference line
    :type j: ipywidgets.Text
    :param r: RadioButton describing the elementary matricial operation to be performed
    :type r: ipywidgets.radioButton
    :param alpha: scalar coefficient
    :type alpha: ipywidgets.Text
    :param A: starting matrix
    :type A: numpy.ndarray
    :param m: starting augmented matrix. If None, it equals A. Defaults to None.
    :type m: numpy.ndarray or NoneType
    :param args: either the list of matrices or both the list of matrices and rhs having bee built during the
       application of the methos
    :type args: list[numpy.ndarray] or tuple(list[numpy.ndarray], list[numpy.ndarray])
    :return: processed matrix
    :rtype: numpy.ndarray
    """

    if m is None:
        m = A.copy()
    m = np.array(m).astype(float)
    if alpha.value == 0 and r.value in {'Di(alpha)', 'Lij(alpha)'}:
        print('Le coefficient alpha doit être non-nul!')

    if r.value == 'Tij':
        m = Eij(m, i.value-1, j.value-1)
    if r.value == 'Di(alpha)':
        m = Ealpha(m, i.value-1, eval(alpha.value))
    if r.value == 'Lij(alpha)':
        m = Eijalpha(m, i.value-1, j.value-1, eval(alpha.value))

    if len(args) == 2:
        A = np.asmatrix(A)
        MatriceList = args[0]
        RhSList = args[1]
        if r.value != 'Revert':
            MatriceList.append(m[:, :A.shape[1]])
            RhSList.append(m[:, A.shape[1]:])
        else:
            if len(MatriceList) > 1 and len(RhSList) > 1:
                MatriceList.pop()
                RhSList.pop()
                mat = MatriceList[-1]
                rhs = RhSList[-1]
                m = np.concatenate((mat,rhs), axis=1)
            else:
                print("Impossible de revenir sur l'opération!")
        printEquMatrices(MatriceList, RhSList)
    elif len(args) == 1:
        MatriceList = args[0]
        if r.value != 'Revert':
            MatriceList.append(m)
        else:
            if len(MatriceList) > 1:
                MatriceList.pop()
                m = MatriceList[-1]
            else:
                print("Impossible de revenir sur l'opération!")
        printEquMatrices(MatriceList)
    else:
        print("La liste des matrices ou des matrices et des vecteurs connus doit être donnée en entrée de la fonction!")
        raise ValueError
    return m


def LU_interactive(i, j, r, alpha, *args):
    """Method which performs the Gauss elimination method step described by `r.value` with parameters `ì`, `j` and
    `alpha` on matrix `A`

    :param i: first reference line
    :type i: ipywidgets.Text
    :param j: second reference line
    :type j: ipywidgets.Text
    :param r: RadioButton describing the elementary matricial operation to be performed
    :type r: ipywidgets.radioButton
    :param alpha: scalar coefficient
    :type alpha: ipywidgets.Text
    :param A: starting matrix
    :type A: numpy.ndarray
    :param m: starting augmented matrix. If None, it equals A. Defaults to None.
    :type m: numpy.ndarray or NoneType
    :param args: either the list of matrices or both the list of matrices and rhs having bee built during the
       application of the method
    :type args: list[numpy.ndarray] or tuple(list[numpy.ndarray], list[numpy.ndarray])
    :return: processed matrix
    :rtype: numpy.ndarray
    """

    if len(args) == 2:
        U = np.array(args[1][-1]).astype(float)
    else:
        print("La liste des matrices diagonales inférieures et supérieures déjà calculées doit être donnée en entrée de"
              " la fonction")
        raise ValueError

    if alpha.value == 0 and r.value in {'Di(alpha)', 'Lij(alpha)'}:
        print('Le coefficient alpha doit être non-nul!')

    is_valid_operation = True
    if r.value == 'Tij':
        print("Échanger deux lignes n'est pas une option si on veut une décomposition LU et non pas PLU (ou P est une matrice de permutation)")
        is_valid_operation = False
    if r.value == 'Di(alpha)':
        U, L = Ealpha(U, i.value-1, eval(alpha.value), get_E_inv=True)
    if r.value == 'Lij(alpha)':
        U, L = Eijalpha(U, i.value-1, j.value-1, eval(alpha.value), get_E_inv=True)

    if is_valid_operation:
        LList = args[0]
        UList = args[1]
        if r.value != 'Revert':
            UList.append(U)
            LList.append(np.dot(LList[-1], L))
        else:
            if len(UList) > 1 and len(LList) > 1:
                UList.pop()
                U = UList[-1]
                LList.pop()
                L = LList[-1]
            else:
                print("Impossible de revenir sur l'opération!")
        printLUMatrices(LList, UList)
    else:
        L = args[0][-1]
        U = args[1][-1]
    return L, U


def LU_no_pivoting(A, ptol=1e-5):
    """Method that computes the LU decomposition of a matrix, without using pivoting. If the matrix cannot be
    decomposed, the method raises a ValueError.

    :param A: matrix to be decomposed
    :type A: list[list] or numpy.ndarray
    :param ptol: tolerance on the pivot values; if a pivot with value smaller (in absolute value) than ptol is found,
      a ValueError is raised. Defaults to 1e-5
    :type ptol: float
    :return: lower triangular matrix L and upper triangular matrix U such that A = LU, if they exist
    :rtype: tuple(numpy.ndarray, numpy.ndarray) or NoneType
    """

    A = np.array(A).astype(float)
    m, n = A.shape

    n_ops = 0
    n_steps = 0

    try:
        assert m <= n
    except AssertionError:
        raise ValueError("La décomposition LU n'est pas implémentée pour les matrices rectangulaires "
                         "ayant plus de lignes que de colonnes")
    for i in range(m):
        if (A[i+1:, :] == 0).all():
            break
        pivot = A[i, i]
        if abs(pivot) <= ptol:
            print("Pivot avec la valeur 0 rencontré. Cette matrice n'admet pas de décomposition LU (sans permutation)")
            return None, None
        for k in range(i+1, m):
            lam = A[k, i] / pivot
            n_ops += 1
            if lam:
                A[k, i+1:n] = A[k, i+1:n] - lam * A[i, i+1:n]
                n_ops += 2*(n-i-1)
                A[k, i] = lam
                n_steps += 1

    L = np.eye(m) + np.tril(A, -1)[:m, :m]
    U = np.triu(A)

    n_ops += m**2

    print(f"Nombre d'opérations élémentaires (I, II, III): {n_steps}")
    print(f"Coût de la décomposition LU (nombre total d'additions, soustractions, "
          f"multiplications et divisions): {n_ops}")

    return L, U


def manualOp(*args):
    """Method which allows the user to perform elementary operations on the given input matrix, eventually extended by
    the right-hand side vector.

    :param args:
    :type args:
    :return:
    :rtype:
    """

    if len(args) == 2:  # matrice augmentée
        A = np.array(args[0]).astype(float)
        M = A.shape[0]
        b = args[1]
        if type(b[0]) is list:
            b = np.array(b).astype(float)
            A = np.concatenate((A, b), axis=1)
        else:
            b = [b[i] for i in range(M)]
            A = [A[i] + [b[i]] for i in range(M)]
    else:
        A = np.array(args[0]).astype(float)
        M = A.shape[0]
    A = np.array(A)  # just in case it's not

    i = widgets.BoundedIntText(
        value=1,
        min=1,
        max=M,
        step=1,
        description='Ligne i:',
        disabled=False
    )

    j = widgets.BoundedIntText(
        value=1,
        min=1,
        max=M,
        step=1,
        description='Ligne j:',
        disabled=False
    )

    r = widgets.RadioButtons(
        options=['Tij', 'Di(alpha)', 'Lij(alpha)'],
        description='Opération:',
        disabled=False
    )

    alpha = widgets.Text(
        value='1',
        description='Coeff. alpha:',
        disabled=False
    )

    print("Régler les paramètres et cliquer sur RUN INTERACT pour effectuer votre opération")

    def f(r, i, j, alpha):
        m = A
        MatriceList = [A[:, :len(A[0])-1]]
        RhSList = [A[:, len(A[0])-1:]]
        if alpha == 0 and r != 'Tij':
            print('Le coefficient alpha doit être non-nul!')
        if r == 'Tij':
            m = Eij(m, i-1, j-1)
        if r == 'Di(alpha)':
            m = Ealpha(m, i-1, eval(alpha))
        if r == 'Lij(alpha)':
            m = Eijalpha(m, i-1, j-1, eval(alpha))
        MatriceList.append(m[:, :len(A[0])-1])
        RhSList.append(m[:, len(A[0])-1:])
        printEquMatricesAug(MatriceList, RhSList)
        return

    interact_manual(f, r=r, i=i, j=j, alpha=alpha)
    return


def __check_vector_2D(vec):
    """Method that, given a vector, checks if it is a valid 2D vector and returns it as a numpy column array

    :param vec: vector to be analyzed
    :type vec: numpy.ndarray or list or lits[list]
    :return: input vector, as a column numpy array, provided that it is a valid 2D array. Else it raises error
    :rtype: numpy.ndarray or NoneType
    """

    if type(vec) is not np.ndarray:
        vec = np.array(vec)

    try:
        assert (len(vec.shape) == 1 and vec.shape[0] == 2) or \
               (len(vec.shape) == 2 and vec.shape[0] + vec.shape[1] == 3)
    except AssertionError:
        print("Erreur: le vecteur entré doit être un vecteur 2D, ligne ou colonne")
    else:
        if len(vec.shape) == 2:
            vec = np.squeeze(vec)

    return vec


def __check_vector_3D(vec):
    """Method that, given a vector, checks if it is a valid 3D vector and returns it as a numpy column array

    :param vec: vector to be analyzed
    :type vec: numpy.ndarray or list or lits[list]
    :return: input vector, as a column numpy array, provided that it is a valid 3D array. Else it raises error
    :rtype: numpy.ndarray or NoneType
    """

    if type(vec) is not np.ndarray:
        vec = np.array(vec)

    try:
        assert (len(vec.shape) == 1 and vec.shape[0] == 3) or \
               (len(vec.shape) == 2 and (vec.shape[0] == 1 or vec.shape[1] == 1) and
                                        (vec.shape[0] == 3 or vec.shape[1] == 3))
    except AssertionError:
        print("Erreur: le vecteur entré doit être un vecteur 3D, ligne ou colonne")
    else:
        if len(vec.shape) == 2:
            vec = np.squeeze(vec)

    return vec


def get_vector_info_2D(vec, show=False, return_results=False):
    """Method that, given a 2D vector, computes and displays its norm and its angle with respect to the x-axis.
    Basically it gives the representation of the vector in polar coordinates. If 'show' is passed as True, then it also
    displays the vector in the plan

    :param vec: vector to be analyzed
    :type vec: numpy.ndarray or list or list[list]
    :param show: if True, a plot of the vector as an arrow in the 2D plane is performed. Defaults to False
    :type show: bool
    :param return_results: if True, the results are also returned and not only displayed. Defaults to False
    :type return_results: bool
    :return: norm and angle of the vector, if return_results is True, else None.
    :rtype: tuple(float, float) or NoneType
    """

    vec = __check_vector_2D(vec)

    norm = np.linalg.norm(vec)
    if norm > 0:
        angle = (np.arccos(vec[0] / norm) / np.pi * 180.0) * (np.sign(vec[1]) if vec[1] != 0 else 1)
    else:
        if show:
            print("Le vecteur donné est nul: impossible de définir son angle")
        angle = np.nan

    if show:
        display(Latex("$||v|| = % 10.4f; \\qquad \\theta = % 10.4f °$" % (norm, angle)))

    if show:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.gca()
        origin = [0], [0]
        limit = 1.25 * np.max(np.abs(vec))
        theta = np.linspace(0, angle / 180.0 * np.pi, 100)
        r = 0.15 * np.max(np.abs(vec))
        x1 = r * np.cos(theta)
        x2 = r * np.sin(theta)

        ax.quiver(*origin, vec[0], vec[1], color='r', angles='xy', scale_units='xy', scale=1)
        ax.plot(*origin, marker='o', color='b')
        ax.plot(x1, x2, '-', color='k', linewidth=1)
        ax.text(x1[50] * 1.5, x2[50] * 1.5, r'$\Theta$')
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.grid()

        # Move left y-axis and bottom x-axis to centre, passing through (0,0)
        ax.spines['left'].set_position('center')
        ax.spines['bottom'].set_position('center')

        # Eliminate upper and right axes
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')

        # Show ticks in the left and lower axes only
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        plt.show()

    if return_results:
        return norm, angle
    else:
        return


def get_couple_vectors_info_2D(vec1, vec2, show=False, return_results=False):
    """Method that, given a couple of 2D vectors, computes and displays their norms, their inner product and the angle
    between them. If 'show' is passed as True, then it also displays the vectors in the plan.

    :param vec1: vector 1 to be analyzed
    :type vec1: numpy.ndarray or list or list[list]
    :param vec2: vector 1 to be analyzed
    :type vec2: numpy.ndarray or list or list[list]
    :param show: if True, a plot of the vectors as a arrows in the 2D plane is performed. Defaults to False
    :type show: bool
    :param return_results: if True, the results are also returned and not only displayed. Defaults to False
    :type return_results: bool
    :return: norms of the two vectors, inner product and angle, if return_results is True else None
    :rtype: tuple(float,float,float,float) or NoneType
    """

    vec1 = __check_vector_2D(vec1)
    vec2 = __check_vector_2D(vec2)

    norm1, angle1 = get_vector_info_2D(vec1, show=False, return_results=True)
    if show:
        display(Latex("$||v_1|| = % 10.4f; \\qquad \\theta_1 = % 10.4f °$" % (norm1, angle1)))

    norm2, angle2 = get_vector_info_2D(vec2, show=False, return_results=True)
    if show:
        display(Latex("$||v_2|| = % 10.4f; \\qquad \\theta_2 = % 10.4f °$" % (norm2, angle2)))

    if norm1 > 0 and norm2 > 0:
        inner_product = np.dot(vec1, vec2)
        diff_angle = np.arccos(inner_product / (norm1 * norm2)) * 180.0 / np.pi
    else:
        inner_product = 0
        diff_angle = np.nan
    if show:
        display(Latex("$v_1 \\cdot v_2 = % 10.4f; \\qquad \\Delta\\theta = % 10.4f °$" %(inner_product, diff_angle)))

    if show:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.gca()
        origin = [0], [0]

        limit = 1.25 * np.max(np.hstack((np.abs(vec1), np.abs(vec2))))
        if not np.isnan(diff_angle):
            if np.abs(angle2 - angle1) <= np.pi:
                theta = np.linspace(angle1 / 180.0 * np.pi, angle2 / 180.0 * np.pi, 100)
            else:
                if angle1 < 0:
                    theta = np.linspace((angle1 + 360.0) / 180.0 * np.pi, angle2 / 180.0 * np.pi, 100)
                elif angle2 < 0:
                    theta = np.linspace(angle1 / 180.0 * np.pi, (angle2 + 360.0) / 180.0 * np.pi, 100)
                else:
                    theta = np.linspace(angle1 / 180.0 * np.pi, angle2 / 180.0 * np.pi, 100)
            r = 0.15 * np.max(np.hstack((np.abs(vec1), np.abs(vec2))))
            x1 = r * np.cos(theta)
            x2 = r * np.sin(theta)

        ax.plot(*origin, marker='o', color='b')
        ax.quiver(*origin, vec1[0], vec1[1], color='r', angles='xy', scale_units='xy', scale=1, label="Vector 1")
        ax.quiver(*origin, vec2[0], vec2[1], color='g', angles='xy', scale_units='xy', scale=1, label="Vector 2")

        if not np.isnan(diff_angle):
            ax.plot(x1, x2, '-', color='k', linewidth=1)
            ax.text(x1[50] * 1.75, x2[50] * 1.75, r'$\Delta\theta$')

        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.legend(loc="best")
        ax.grid()

        # Move left y-axis and bottom x-axis to centre, passing through (0,0)
        ax.spines['left'].set_position('center')
        ax.spines['bottom'].set_position('center')

        # Eliminate upper and right axes
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')

        # Show ticks in the left and lower axes only
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')

        plt.show()

    if return_results:
        return norm1, norm2, inner_product, diff_angle
    else:
        return


def get_vector_info_3D(vec, show=False):
    """Method that, given a 3D vector, computes, displays its norm and its angle with respect to the x-axis.
    Basically it gives the representation of the vector in polar coordinates. If 'show' is passed as True, then it also
    displays the vector in the plan

    :param vec: vector to be analyzed
    :type vec: numpy.ndarray or list or list[list]
    :param show: if True, a plot of the vector as an arrow in the 2D lan is performed. Defaults to False
    :type show: bool
    """

    vec = __check_vector_3D(vec)

    norm = np.linalg.norm(vec)
    if norm > 0:
        polar_angle = np.arccos(vec[2] / norm) / np.pi * 180.0
        azimuthal_angle = np.arctan2(vec[1], vec[0]) /np.pi * 180.0
    else:
        print("Le vecteur donné est nul: impossible de définir son angle")
        polar_angle = np.nan
        azimuthal_angle = np.nan

    display(Latex("$||v|| = % 10.4f; \\qquad \\theta = % 10.4f °$ (angle polaire); "
                  "$\\qquad \\phi = % 10.4f °$ (angle azimutal)" %
                  (norm, polar_angle, azimuthal_angle)))

    if show:
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure()
        ax = fig.gca(projection='3d')

        origin = [0], [0], [0]
        limit = 1.25 * np.max(np.abs(vec))

        ax.quiver(*origin, vec[0], vec[1], vec[2], color='r', normalize=False, linewidth=2)
        ax.plot(*origin, marker='o', color='b')
        ax.set_xlim(-limit, limit)
        ax.set_ylim(-limit, limit)
        ax.set_zlim(-limit, limit)
        ax.grid()

        ax.quiver(0, 0, 0,limit, 0, 0,
                  color='k', linewidth=1)
        ax.quiver(0, 0, 0, 0,limit, 0,
                  color='k', linewidth=1)
        ax.quiver(0, 0, 0, 0, 0,limit,
                  color='k', linewidth=1)
        ax.quiver(0, 0, 0, -limit, 0, 0,
                  arrow_length_ratio=0, color='k', linewidth=1)
        ax.quiver(0, 0, 0, 0, -limit, 0,
                  arrow_length_ratio=0, color='k', linewidth=1)
        ax.quiver(0, 0, 0, 0, 0, -limit,
                  arrow_length_ratio=0, color='k', linewidth=1)

        plt.show()

    return


def get_couple_matrices_info(A, B, show=False):
    """Function that, given two matrices, returns their norms, the norm of their sum, the inner product and the angle
    between those, all with respect to the trace inner product. Angle is expressed in radians.

    :param A: first matrix
    :type A: list[list[int]] or numpy.ndarray
    :param B: second matrix
    :type B: list[list[int]] or numpy.ndarray
    :param show: if True, results are displayed. Defaults to False
    :type show: bool
    :return: dictionary containing the norms of the matrices, the norm of their sum, the inner product and the angle
    :rtype: dict{str:float}
    """

    if type(A) is not np.ndarray:
        A = np.array(A)

    if type(B) is not np.ndarray:
        B = np.array(B)

    result = dict()
    result['norm_1'] = np.sqrt(np.trace(np.dot(A.T, A)))
    result['norm_2'] = np.sqrt(np.trace(np.dot(B.T, B)))
    result['norm_sum'] = np.sqrt(np.trace(np.dot((A+B).T, A+B)))
    result['inner_product'] = np.trace(np.dot(A.T, B))
    result['angle'] = np.arccos(result['inner_product'] / (result['norm_1'] * result['norm_2'])) \
                      if result['norm_1'] > 0 and result['norm_2'] > 0 else np.nan

    if show:
        display(Latex(r"$||A||$ = % 10.4f; $||B||$ = % 10.4f"
                      %(result['norm_1'], result['norm_2'])))
        display(Latex(r"$||A+B||$ = % 10.4f; $\langle A,B \rangle$ = % 10.4f"
                      %(result['norm_sum'], result['inner_product'])))
        display(Latex(r"$\Delta\theta$ = % 10.4f °; $||A+B||^2$ = % 10.4f; $||A||^2 + ||B||^2$ = % 10.4f"
                      % (result['angle'] * 180.0 / np.pi, result['norm_sum']**2, result['norm_1']**2+result['norm_2']**2)))

    return result


def get_couple_functions_info(f, g, int_limits, weight_function=None, show=False):
    """Function that, given two continuous functions, returns their norms, the norm of their sum, the inner product and
    the angle between those, all with respect to the weighted L2 inner product. Angle is expressed in radians.

    :param f: first function
    :type f: function
    :param g: second function
    :type g: function
    :param int_limits: integration limits
    :type int_limits: list[float, float]
    :param weight_function: weighing function of the inner product. If None, it defaults to 1.0
    :type weight_function: function or NoneType
    :param show: if True, results are displayed. Defaults to False
    :type show: bool
    :return: dictionary containing the norms of the functions, the norm of their sum, the inner product and the angle
    :rtype: dict{str:float}
    """

    if weight_function is None:
        weight_function = lambda x: 1.0 + 0*x

    try:
        x = np.linspace(int_limits[0], int_limits[1], 1000)
        w = weight_function(x)
        assert (w >= 0).all() and (w > 0).any()
    except AssertionError:
        raise ValueError("The weight function must be non-negative and non-null within the integration limits!")

    result = dict()
    result['norm_1'] = np.sqrt(quad(lambda x: weight_function(x)*f(x)**2, int_limits[0], int_limits[1])[0])
    result['norm_2'] = np.sqrt(quad(lambda x: weight_function(x)*g(x)**2, int_limits[0], int_limits[1])[0])
    result['norm_sum'] = np.sqrt(quad(lambda x: weight_function(x)*(f(x)+g(x))**2, int_limits[0], int_limits[1])[0])
    result['inner_product'] = quad(lambda x: weight_function(x)*f(x)*g(x), int_limits[0], int_limits[1])[0]
    result['angle'] = np.arccos(result['inner_product'] / (result['norm_1'] * result['norm_2'])) \
        if result['norm_1'] > 0 and result['norm_2'] > 0 else np.nan

    if show:
        display(Latex(r"$||f||$ = % 10.4f; $||g||$ = % 10.4f"
                      % (result['norm_1'], result['norm_2'])))
        display(Latex(r"$||f+g||$ = % 10.4f; $\langle f,g \rangle$ = % 10.4f"
                      % (result['norm_sum'], result['inner_product'])))
        display(Latex(r"$\Delta\theta$ = % 10.4f °; $||f+g||^2$ = % 10.4f; $||f||^2 + ||g||^2$ = % 10.4f"
                      % (result['angle'] * 180.0 / np.pi, result['norm_sum'] ** 2,
                         result['norm_1'] ** 2 + result['norm_2'] ** 2)))

    return result


def couple_functions_plotter(f, g, limits, pi_formatting=False):
    """Function that, given two functions, generates a subplot with a plot of the functions and a plot of their
    pointwise product

    :param f: first function
    :type f: function
    :param g: second function
    :type g: function
    :param limits: plotting limits
    :type limits: list[float, float]
    :param pi_formatting: if True, the x-axis is formatted with ticks relative to pi. Defaults to False
    :type pi_formatting: bool
    """

    x = np.linspace(limits[0], limits[1], 1000)

    fg = lambda x: f(x) * g(x)

    fig, axs = plt.subplots(1, 2, figsize=(16, 8))

    axs[0].plot(x, f(x), label='f', color='b')
    axs[0].plot(x, g(x), label='g', color='r')
    min_val = np.min(np.hstack([f(x), g(x)]))
    max_val = np.max(np.hstack([f(x), g(x)]))
    axs[0].set_ylim([min_val - 0.1*np.abs(min_val), max_val + 0.1*np.abs(max_val)])
    axs[0].set_xlim([limits[0], limits[1]])
    x_axis_pos = 0.0 if min_val >= 0 else 1.0 if max_val <= 0 else np.abs(min_val) / (max_val - min_val)
    y_axis_pos = 0.0 if limits[0] >= 0 else 1.0 if limits[1] <= 0 else np.abs(limits[0]) / (limits[1] - limits[0])
    axs[0].grid(linestyle='--', linewidth=0.5)
    axs[0].set_title('Fonctions f et g')
    axs[0].legend(loc='best', fontsize=10)
    axs[0].spines['left'].set_position(('axes', y_axis_pos))
    axs[0].spines['bottom'].set_position(('axes', x_axis_pos))
    axs[0].spines['right'].set_color('none')
    axs[0].spines['top'].set_color('none')
    axs[0].xaxis.set_ticks_position('bottom')
    axs[0].yaxis.set_ticks_position('left')

    if pi_formatting:
        axs[0].xaxis.set_major_locator(plt.MultipleLocator(np.pi/2))
        axs[0].xaxis.set_minor_locator(plt.MultipleLocator(np.pi/12))
        axs[0].xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
        axs[0].xaxis.set_tick_params(labelsize=13)

    axs[1].plot(x, fg(x), color='g')
    axs[1].grid(linestyle='--', linewidth=0.5)
    axs[1].set_title('Produit entre f et g')
    min_val = np.min(fg(x))
    max_val = np.max(fg(x))
    axs[1].set_ylim([min_val - 0.1 * np.abs(min_val), max_val + 0.1 * np.abs(max_val)])
    axs[1].set_xlim([limits[0], limits[1]])
    x_axis_pos = 0.0 if min_val >= 0 else 1.0 if max_val <= 0 else np.abs(min_val) / (max_val - min_val)
    y_axis_pos = 0.0 if limits[0] >= 0 else 1.0 if limits[1] <= 0 else np.abs(limits[0]) / (limits[1] - limits[0])
    axs[1].spines['left'].set_position(('axes', y_axis_pos))
    axs[1].spines['bottom'].set_position(('axes', x_axis_pos))
    axs[1].spines['right'].set_color('none')
    axs[1].spines['top'].set_color('none')
    axs[1].xaxis.set_ticks_position('bottom')
    axs[1].yaxis.set_ticks_position('left')

    if pi_formatting:
        axs[1].xaxis.set_major_locator(plt.MultipleLocator(np.pi/2))
        axs[1].xaxis.set_minor_locator(plt.MultipleLocator(np.pi/12))
        axs[1].xaxis.set_major_formatter(plt.FuncFormatter(multiple_formatter()))
        axs[1].xaxis.set_tick_params(labelsize=13)

    fig.suptitle(f"Tracés des deux fonctions et de leur produit")

    return


def multiple_formatter(denominator=2, number=np.pi, latex='\pi'):
    def gcd(a, b):
        while b:
            a, b = b, a%b
        return a

    def _multiple_formatter(x, pos):
        den = denominator
        num = np.int(np.rint(den*x/number))
        com = gcd(num,den)
        (num,den) = (int(num/com),int(den/com))
        if den == 1:
            if num == 0:
                return r'$0$'
            if num == 1:
                return r'$%s$'%latex
            elif num == -1:
                return r'$-%s$'%latex
            else:
                return r'$%s%s$'%(num,latex)
        else:
            if num == 1:
                return r'$\frac{%s}{%s}$'%(latex,den)
            elif num == -1:
                return r'$-\frac{%s}{%s}$'%(latex,den)
            else:
                return r'$\frac{%s%s}{%s}$'%(num,latex,den)

    return _multiple_formatter


def visualize_Cauchy_Schwarz(u, v, limits=None):
    """Method that allows to plot the parabola explaining the Cauchy-Schwarz inequality for the two 'vectors' u and v.
    u and v can be n-dimensional vectors (in which case the standard inner product is adopted), MxN matrices (in which
    case the trace-induced inner product is used) or continuous functions (in which case the integral-induced inner
    product is used). Also, the LHS and the RHS terms appearing in the Cauchy-Schwarz inequality are returned

    :param u: first 'vector'
    :type u: list[float] or list[list[float]] or numpy.ndarray or function
    :param v: second 'vector'
    :type v: list[float] or list[list[float]] or numpy.ndarray or function
    :param limits: if u and v are functions, those are the integration limits; otherwise they are useless and
      they are defaulted to None. They must be expressed as a either a list of 2 floats (for 1D integration) or as
      a list of n lists of 2 floats (for n-dimensional integration)
    :type limits: list[list[float, float]] or list[float, float] NoneType
    :return: LHS and RHS terms of the Cauchy-Schwarz inequality, in a dictionary
    :rtype: dict(str:float, str:float)
    """

    try:
        assert type(u) is type(v)
    except AssertionError:
        raise ValueError(f"The two input vectors must be of the same type, while here the first input is of type "
                         f"{type(u)}, while he second one is of type {type(v)}")

    if not callable(u):
        u = np.squeeze(np.array(u))
        v = np.squeeze(np.array(v))
    else:
        try:
            assert limits is not None and type(limits) is list and len(limits)
        except AssertionError:
            raise ValueError("If the given inputs are functions, then the extrema of integration must be passed as a "
                             "list made of lists of two floats via the keyword argument 'limits'. If the list"
                             "has n sublists, then the functions must have n arguments, otherwise an error is thrown."
                             "Also, 'limits' can be a list of 2 floats in 1D integration has to be performed")

    if type(u) is np.ndarray:
        try:
            assert u.shape == v.shape
        except AssertionError:
            raise ValueError(f"The two input elements must be of the same shape, while here the first input has"
                             f"shape {u.shape}, while the second one has shape {v.shape}")

    if callable(u):
        if type(limits[0]) is not list:
            limits = [limits]

        a = nquad(lambda *x: v(*x)**2, limits)[0]
        b = 2 * nquad(lambda *x: u(*x)*v(*x), limits)[0]
        c = nquad(lambda *x: u(*x)**2, limits)[0]

    elif len(u.shape) > 1:
        a = np.trace(np.dot(v.T, v))
        b = 2 * np.trace(np.dot(u.T, v))
        c = np.trace(np.dot(u.T, u))

    else:
        a = np.linalg.norm(v)**2
        b = 2 * np.inner(u,v)
        c = np.linalg.norm(u)**2

    x_v = -b / (2*a) if a > 0 else 0.0
    t = np.linspace(x_v-5, x_v+5, 100)
    plot_limits = [x_v-5, x_v+5]

    parab = lambda t: a*t**2 + b*t + c

    fig, axs = plt.subplots(1,1)
    axs.plot(t, parab(t), color='r')
    axs.grid(linestyle='--', linewidth=0.5)
    axs.set_title('Parabole de Cauchy-Schwarz')
    min_val = np.min(np.hstack([0, parab(t)]))
    max_val = np.max(parab(t))
    axs.set_ylim([min_val - 0.1 * np.abs(min_val), max_val + 0.1 * np.abs(max_val)])
    axs.set_xlim([plot_limits[0], plot_limits[1]])
    x_axis_pos = 0.0 if min_val >= 0 else 1.0 if max_val <= 0 else np.abs(min_val) / (max_val - min_val)
    y_axis_pos = 0.0 if plot_limits[0] >= 0 else 1.0 if plot_limits[1] <= 0 else \
                 np.abs(plot_limits[0]) / (plot_limits[1] - plot_limits[0])
    axs.spines['left'].set_position(('axes', y_axis_pos))
    axs.spines['bottom'].set_position(('axes', x_axis_pos))
    axs.spines['right'].set_color('none')
    axs.spines['top'].set_color('none')
    axs.xaxis.set_ticks_position('bottom')
    axs.yaxis.set_ticks_position('left')

    out = dict()
    out['LHS'] = np.abs(b / 2)
    out['RHS'] = np.sqrt(a) * np.sqrt(c)

    display(Latex(r"Cauchy-Schwarz Inequality: $|\langle u,v \rangle| = % 10.4f \leq ||u|| \ ||v|| = % 10.4f$"
                  % (out['LHS'], out['RHS'])))

    return out


def compute_expansion_coefficients(B, v, W=None, int_limits=None, weight_function=None):
    """Method that, given a basis and an element, computes the expansion coefficients of the element with respect to
    the basis

    :param B: basis of the vector space. Here B is assumed to be a basis!!
    :type B: list[function] or list[list] or list[tuple] or list[numpy.ndarray]
    :param v: element
    :type v: function or list or tuple or numpy.ndarray
    :param W: weighting matrix. If None, it defaults to the identity
    :type W: numpy.ndarray or NoneType
    :param int_limits: integration limits. If None, they are not used
    :type int_limits: list[float, float] or NoneType
    :param weight_function: weighting function. If None, it defaults to 1.0
    :type weight_function: function or NoneType
    :return: coordinates of the element with respect to the basis
    :rtype: list
    """

    try:
        assert type(B) is list and len(B)
    except AssertionError:
        raise TypeError(f"The basis must be passed as a non-empty list of elements, while its type is {type(B)} "
                        f"in this case.")

    try:
        assert type(B[0]) == type(v)
    except AssertionError:
        raise TypeError(f"The target vector and the basis elements must be of the same type, while they are of types"
                        f"{type(B[0])} and {type(v)} respectively in this case.")

    if callable(v):

        try:
            assert int_limits is not None
        except AssertionError:
            raise ValueError("Impossible to compute the inner product between functions if no integration limits "
                             "are defined")

        if weight_function is None:
            weight_function = lambda x: 1.0 + 0*x

        try:
            x = np.linspace(int_limits[0], int_limits[1], 1000)
            w = weight_function(x)
            assert (w >= 0).all() and (w > 0).any()
        except AssertionError:
            raise ValueError("The weight function must be non-negative and non-null within the integration limits!")

        A = np.zeros((len(B), len(B)))
        b = np.zeros(len(B))
        for i in range(len(B)):
            b[i] = quad(lambda x: weight_function(x)*B[i](x)*v(x), int_limits[0], int_limits[1])[0]
            for j in range(i+1):
                A[i,j] = quad(lambda x: weight_function(x)*B[i](x)*B[j](x), int_limits[0], int_limits[1])[0]

    elif type(v) is list or type(v) is tuple or type(v) is np.ndarray:

        if W is None:
            W = np.eye(len(v))
        else:
            if type(W) is not np.ndarray:
                W = np.array(W)
            try:
                assert W.shape[0] == W.shape[1] == len(v) == len(B[0])
            except AssertionError:
                raise ValueError("The weighting matrix must be a square matrix with the same dimensionality of the "
                                 "basis elements and the target element")

            try:
                assert isSym(W)
            except AssertionError:
                raise ValueError("The weighting matrix must be symmetric to define an inner product")

            lam, _ = np.linalg.eig(W)
            try:
                assert (np.real(lam) > 0).all()
            except AssertionError:
                raise ValueError("The weighting matrix must be positive definite to define an inner product!")

        A = np.zeros((len(B), len(B)))
        b = np.zeros(len(B))
        for i in range(len(B)):
            b[i] = np.inner(v, np.dot(W, B[i]))
            for j in range(i + 1):
                A[i, j] = np.inner(B[i], np.dot(W, B[j]))

    else:
        raise TypeError(f"The type of the target element is {type(v)} and it is not supported")

    A = A + A.T - np.diag(np.diag(A))
    v_B = np.linalg.solve(A, b)

    return v_B


def project(u, v, W=None, int_limits=None, weight_function=None):
    """Method that computes the projection of 'u' over 'v', considering both the cases in which u and v are vectors or
    functions. In the former case, it used the inner product defined by 'W' (the standard one if W is None); in the
    latter case it uses the integral inner product, within the limits specified by 'int_limits' and considering
    'weight_function' as weight function

    :param u: element to project
    :type u: list or numpy.ndarray or sympy.function
    :param v: element on which projecting
    :type v: list or numpy.ndarray or sympy.function
    :param W: weighting matrix
    :type W: list or numpy.ndarray or Nonetype
    :param int_limits: integration limits
    :type int_limits: list or NoneType
    :param weight_function: weighting function
    :type weight_function: sympy.function or NoneType
    :return: projection of u onto v
    :rtype: numpy.ndarray or sympy.function
    """

    try:
        assert type(u) == type(v) or (isinstance(u, sp.Basic) and isinstance(v, sp.Basic))
    except AssertionError:
        raise TypeError(f"The vector to project and the one on which projecting must be of the same type, while here "
                        f"u is of type {type(u)} and v is of type {type(v)}")

    if isinstance(u, sp.Basic):

        x = sp.Symbol('x')

        try:
            assert int_limits is not None
        except AssertionError:
            raise ValueError("Impossible to compute the inner product between functions if no integration limits "
                             "are defined")

        if weight_function is None:
            weight_function = 1.0 + 0*x

        try:
            xx = np.linspace(int_limits[0], int_limits[1], 1000)
            weight_function_lam = np.vectorize(sp.lambdify(x, weight_function, "numpy"))
            w = weight_function_lam(xx)
            assert (w >= 0).all() and (w > 0).any()

        except AssertionError:
            raise ValueError("The weight function must be non-negative and non-null within the integration limits!")

        coeff = sp.integrate(weight_function * u * v, (x, int_limits[0], int_limits[1])) / \
                sp.integrate(weight_function * v * v, (x, int_limits[0], int_limits[1]))

        proj_u_v = coeff * v

    elif type(u) is list or type(u) is np.ndarray:

        if W is None:
            W = np.eye(len(v))
        else:
            if type(W) is not np.ndarray:
                W = np.array(W)
            try:
                assert W.shape[0] == W.shape[1] == len(v) == len(u)
            except AssertionError:
                raise ValueError("The weighting matrix must be a square matrix with the same dimensionality of the "
                                 "basis elements and the target element")

            try:
                assert isSym(W)
            except AssertionError:
                raise ValueError("The weighting matrix must be symmetric to define an inner product")

            lam, _ = np.linalg.eig(W)
            try:
                assert (np.real(lam) > 0).all()
            except AssertionError:
                raise ValueError("The weighting matrix must be positive definite to define an inner product!")

        coeff = np.inner(u, np.dot(W, v)) / np.inner(v, np.dot(W, v))

        proj_u_v = coeff * np.array(v)

    else:
        raise TypeError(f"The type of the target element is {type(u)} and it is not supported")

    return proj_u_v


def gram_schmidt(vects, W=None, return_norms=False, show=False):
    """Method that, given a list of vectors, applies the Gram-Schmidt process to it. If the list
    of elements is a basis for some subspace, then this algorithm computes an orthonormal basis.

    :param vects: initial list of elements
    :type vects: list[list] or list[numpy.ndarray]
    :param W: weighting matrix
    :type W: list or numpy.ndarray or NoneType
    :param return_norms: if True, the norms of the generated orthogonal vectors are returned. Defaults to False
    :type return_norms: bool
    :param show: if True, the final vectors are plotted, in case thei are 2D or 3D. It defaults to False
    :type show: bool
    :return: list of elements after the application of the Gram-Schmidt algorithm
    :rtype: list[numpy.ndarray]
    """

    if W is None:
        W = np.eye(len(vects[0]))

    norms = [np.sqrt(np.inner(np.dot(W, vects[0]), vects[0]))]
    result = [np.array(vects[0], dtype='float32') / norms[0]]

    for index in range(1, len(vects)):
        tmp = np.array(vects[index], dtype='float32')
        for item in range(index):
            tmp -= project(tmp, result[item], W=W)

        norms.append(np.sqrt(np.inner(np.dot(W, tmp), tmp)))
        if norms[-1] > 1e-4:
            result.append(tmp / norms[-1])
        else:
            result.append(tmp)

    if show:
        if len(result[0]) == 2:
            vector_plot_2D(result,
                           labels=[f'vector {i}' for i in range(len(result))])
        elif len(result[0] == 3):
            vector_plot_3D(result,
                           labels=[f'old vector {i}' for i in range(len(result))])
        else:
            display(Latex(r"La représentation graphique n'est pas possible dans $\mathbb{R}^n$, avec $n \geq 4$!"))

    if return_norms:
        return result, norms
    else:
        return result


def gram_schmidt_func(vects, int_limits=None, weight_function=None, return_norms=False, show=False):
    """Method that, given a list of uni-variate functions, applies the Gram-Schmidt process to it. If the list
    of elements is a basis for some subspace, then this algorithm computes an orthonormal basis.

    :param vects: initial list of elements
    :type vects: list[sympy.function]
    :param int_limits: integration limits
    :type int_limits: list[float float] or NoneType
    :param weight_function: weighting function. If None, it defaults to 1
    :type weight_function: function or NoneType
    :param return_norms: if True, the norms of the generated orthogonal vectors are returned. Defaults to False
    :type return_norms: bool
    :param show: if True, the computed functions are plotted. It defaults to False
    :type show: bool.
    :return: list of elements after the application of the Gram-Schmidt algorithm
    :rtype: list[numpy.ndarray]
    """

    x = sp.Symbol('x')

    if int_limits is None:
        int_limits = [-1, 1]

    if weight_function is None:
        weight_function = 1 + 0*x

    norms = [sp.sqrt(sp.integrate(weight_function * vects[0] * vects[0], (x, int_limits[0], int_limits[1])))]
    result = [vects[0] / norms[0]]

    for index in range(1, len(vects)):
        for item in range(index):
            tmp = vects[index]
            vects[index] -= project(tmp, result[item],
                                    weight_function=weight_function,
                                    int_limits=int_limits)

        curr_norm = sp.sqrt(sp.integrate(weight_function * vects[index] * vects[index],
                                         (x, int_limits[0], int_limits[1])))
        norms.append(curr_norm)

        if norms[-1] > 1e-4:
            result.append(vects[index] / curr_norm)
        else:
            result.append(vects[index])

    if show:
        p = sp.plotting.plot(show=False, xlim=(int_limits[0], int_limits[1]), ylim=(-2, 2),
                             title="Fonctions", legend=True)
        colors = cycle(["b", "g", "r", "c", "m", "y", "k"])
        for item in range(len(result)):
            p.append(sp.plotting.plot(result[item], show=False, xlim=(int_limits[0], int_limits[1]), ylim=(-2, 2),
                                      title="Fonctions", legend=True, line_color=next(colors))[0])
        p.show()

    if return_norms:
        return result, norms
    else:
        return result


def project_on_subspace(u, B, W=None, int_limits=None, weight_function=None, plot=True, show=True):
    """Method that allows to compute the projection of 'u' over a subspace, whose basis is 'B'. It works both on
    vectors and functions. In the former case, it used the inner product defined by 'W' (the standard one if W is None);
    in the latter case it uses the integral inner product, within the limits specified by 'int_limits' and considering
    'weight_function' as weight function

    :param u: element to project
    :type u: list or numpy.ndarray or sympy.function
    :param B: basis of the subspace
    :type B: list[list] or list[numpy.ndarray] or list[sympy.function]
    :param W: weighting matrix
    :type W: list or numpy.ndarray or Nonetype
    :param int_limits: integration limits
    :type int_limits: list[float] or NoneType
    :param weight_function: weighting function
    :type weight_function: sympy.function or NoneType
    :param plot: if True and if possible, the results are displayed in a plot. Defaults to True.
    :type plot: bool
    :param show: if True, the results are displayed. Defaults to True
    :type show: bool
    :return: projection of u onto the subspace having B as basis
    :rtype: numpy.ndarray or sympy.function
    """

    try:
        assert all([type(u) == type(B[i]) for i in range(len(B))]) or \
               all([isinstance(u, sp.Basic) and isinstance(B[i], sp.Basic) for i in range(len(B))])
    except AssertionError:
        raise TypeError(f"Le vecteur à projeter et ceux générant le sous-espace vectoriel sur lequel la projection "
                        f"doit être calculée doivent être du même type!")

    if type(u) in {list, np.ndarray}:
        B_norm = gram_schmidt(B, W=W, return_norms=False, show=False)
        u = np.array(u, dtype='float64')
    elif isinstance(u, sp.Basic):
        if int_limits is None:
            int_limits = [-1, 1]
        x = sp.Symbol('x')
        B_norm = gram_schmidt_func(B, int_limits=int_limits, weight_function=weight_function, return_norms=False, show=False)
    else:
        raise TypeError

    res = np.zeros_like(u, dtype='float64') if type(u) in {list, np.ndarray} \
          else 0 + 0*x if isinstance(u, sp.Basic) \
          else None

    for v in B_norm:
        res += project(u, v, W=W, int_limits=int_limits, weight_function=weight_function)

    if show:
        display(Latex(f'La projection de $u$ sur $W$ est: '
                      f'{np.round(res, 4) if type(u) is np.ndarray else res}'))
        display(Latex(f'La projection de $u$ sur $W^\perp$ est: '
                      f'{np.round(u-res, 4) if type(u) is np.ndarray else u-res}'))

    if plot:
        if type(res) is np.ndarray:
            if len(res) >= 4 or len(res) <= 1:
                display(Latex(r"La représentation graphique n'est pas possible dans $\mathbb{R}^n$, avec $n \geq 4$!"))
                return np.round(res, 4) if type(u) is np.ndarray else res

            elif len(res) == 2:
                fig = vector_plot_2D(list(chain(*[[u], B, [res], [u-res]])),
                                     orig=list(chain(*[[[0,0]], [[0,0] for _ in range(len(B))], [[0,0]], [res]])),
                                     labels=list(chain(*[['u'], [f'$B_{i}$' for i in range(len(B))],
                                                         [r'$proj_uW$'], [r'$proj_uW^\perp$']])),
                                     show=False)

            elif len(res) == 3:
                fig = vector_plot_3D(list(chain(*[[u], B, [res], [u-res]])),
                                     orig=list(chain(*[[[0,0,0]], [[0,0,0] for _ in range(len(B))], [[0,0,0]], [res]])),
                                     labels=list(chain(*[['u'], [f'$B_{i}$' for i in range(len(B))],
                                                        [r'$proj_uW$'], [r'$proj_uW^\perp$']])),
                                     show=False)

            if len(B) == 2:
                colorscale = [[0.0, 'rgb(0,255,255)'],
                              [0.1, 'rgb(0,255,255)'],
                              [0.2, 'rgb(0,255,255)'],
                              [0.3, 'rgb(0,255,255)'],
                              [0.4, 'rgb(0,255,255)'],
                              [0.5, 'rgb(0,255,255)'],
                              [0.6, 'rgb(0,255,255)'],
                              [0.7, 'rgb(0,255,255)'],
                              [0.8, 'rgb(0,255,255)'],
                              [0.9, 'rgb(0,255,255)'],
                              [1.0, 'rgb(0,255,255)']]
                coeffs = [(B[0][1] * B[1][2] - B[1][1] * B[0][2]),
                          (B[1][0] * B[0][2] - B[0][0] * B[1][2]),
                          (B[0][0] * B[1][1] - B[1][0] * B[0][1])]
                length = np.linspace(-1.5*max(abs(B[0][0]), abs(B[1][0]), abs(res[0])),
                                     1.5*max(abs(B[0][0]), abs(B[1][0]), abs(res[0])), 100)
                width = np.linspace(-1.5*max(abs(B[0][1]), abs(B[1][1]), abs(u[1])),
                                    1.5*max(abs(B[0][1]), abs(B[1][1]), abs(res[1])), 100)
                height = np.linspace(-1.5 * max(abs(B[0][2]), abs(B[1][2]), abs(u[2])),
                                     1.5 * max(abs(B[0][2]), abs(B[1][2]), abs(res[2])), 100)

                if coeffs[2] != 0:
                    Grid1, Grid2 = np.meshgrid(length, width)
                    x = Grid1
                    y = Grid2
                    new_coeffs = [-coeffs[0]/coeffs[2], -coeffs[1]/coeffs[2]]
                    ref_index = 2
                elif coeffs[1] != 0:
                    Grid1, Grid2 = np.meshgrid(length, height)
                    x = Grid1
                    z = Grid2
                    new_coeffs = [-coeffs[0] / coeffs[1], -coeffs[2] / coeffs[1]]
                    ref_index = 1
                elif coeffs[0] != 0:
                    Grid1, Grid2 = np.meshgrid(width, height)
                    y = Grid1
                    z = Grid2
                    new_coeffs = [-coeffs[1] / coeffs[0], -coeffs[2] / coeffs[0]]
                    ref_index = 0
                else:
                    plotly.offline.iplot(fig)
                    return np.round(res, 4) if type(u) is np.ndarray else res

                surface = go.Surface(x=x if ref_index != 0 else new_coeffs[0]*y + new_coeffs[1]*z,
                                     y=y if ref_index != 1 else new_coeffs[0]*x + new_coeffs[1]*z,
                                     z=z if ref_index != 2 else new_coeffs[0]*x + new_coeffs[1]*y,
                                     showscale=False, showlegend=True, opacity=0.25, colorscale=colorscale, name='W')
                data = list(fig.data)
                layout = fig.layout
                data.append(surface)
                fig = go.Figure(data=data, layout=layout)
                plotly.offline.iplot(fig)

            elif len(B) == 1:
                x = np.linspace(-1.5*max(abs(B[0][0]), abs(res[0])),
                                1.5*max(abs(B[0][0]), abs(res[0])), 2)
                y = np.linspace(-1.5 * max(abs(B[0][1]), abs(res[1])),
                                1.5 * max(abs(B[0][1]), abs(res[1])), 2)
                if len(B[0]) == 3:
                    z = np.linspace(-1.5 * max(abs(B[0][2]), abs(res[2])),
                                    1.5 * max(abs(B[0][2]), abs(res[2])), 2)

                coeffs = [B[0][0], B[0][1]]
                if len(B[0]) == 3:
                    coeffs.append(B[0][2])

                if coeffs[0] != 0:
                    new_coeffs = [coeffs[1] / coeffs[0]] if len(coeffs) == 2 \
                                 else [coeffs[1] / coeffs[0], coeffs[2]/coeffs[0]]
                    ref_index = 0
                elif coeffs[1] != 0:
                    new_coeffs = [coeffs[0] / coeffs[1]] if len(coeffs) == 2 \
                        else [coeffs[0] / coeffs[1], coeffs[2] / coeffs[1]]
                    ref_index = 1
                elif len(coeffs) == 3 and coeffs[2] != 0:
                    new_coeffs = [coeffs[0] / coeffs[2]] if len(coeffs) == 2 \
                                 else [coeffs[0] / coeffs[2], coeffs[1] / coeffs[2]]
                    ref_index = 2
                else:
                    plotly.offline.iplot(fig)
                    return np.round(res, 4) if type(u) is np.ndarray else res

                if len(B[0]) == 2:
                    line = go.Scatter(x=x if ref_index == 0 else np.zeros_like(x),
                                      y=new_coeffs[0]*x if ref_index == 0 else y,
                                      marker=dict(size=[0, 0],
                                                  color=['rgb(255,165,0)'],
                                                  line=dict(width=1,
                                                            color='rgb(255,165,0)')),
                                      name='W')
                elif len(B[0]) == 3:
                    line = go.Scatter3d(x=x if ref_index == 0 else np.zeros_like(x),
                                        y=new_coeffs[0]*x if ref_index == 0
                                          else y if ref_index == 1 else np.zeros_like(x),
                                        z=new_coeffs[1]*x if ref_index == 0
                                          else new_coeffs[1]*y if ref_index == 1
                                          else z,
                                        marker=dict(size=[0, 0],
                                                    color=['rgb(255,165,0)'],
                                                    line=dict(width=1,
                                                                color='rgb(0255,165,0)')),
                                        name='W')

                data = list(fig.data)
                layout = fig.layout
                data.append(line)
                fig = go.Figure(data=data, layout=layout)
                plotly.offline.iplot(fig)

        if isinstance(res, sp.Basic):
            p = sp.plotting.plot(show=False, xlim=(int_limits[0], int_limits[1]), ylim=(-2, 2),
                                 title="Fonctions", legend=True)
            colors = cycle(chain(*[["g" for _ in range(len(B))], ["b"], ["c"], ["r"]]))
            plt_vects = list(chain(*[B, [u], [res], [u-res]]))
            labels = list(chain(*[[f'$B_{i}$' for i in range(len(B))], ['u'], [r'$proj_uW$'], [r'$proj_uW^\perp']]))
            for cnt, item in enumerate(plt_vects):
                p.append(
                    sp.plotting.plot(item, show=False, xlim=(int_limits[0], int_limits[1]), ylim=(-2, 2),
                                     title="Fonctions", label=labels[cnt], legend=True, line_color=next(colors))[0])
            p.show()

    return np.round(res, 4) if type(u) is np.ndarray else res


def manual_GS(dim):
    """Interactive code, that allows to perform the interactive GS process on vectors

    :type dim: dimensionality of the vectors' set
    :type dim: int
    :return:
    :rtype:
    """

    assert dim > 0 and type(dim) is int

    style = {'description_width': 'initial'}

    step_number = widgets.BoundedIntText(
                                value=1,
                                step=1,
                                min=1,
                                max=dim,
                                description='Numéro du élément courant',
                                disabled=False,
                                style=style
    )

    norm_coeff = widgets.BoundedFloatText(
                                    value=1.0,
                                    step=0.1,
                                    min=0.0,
                                    max=1e10,
                                    description='Coefficient de Normalisation',
                                    disabled=False,
                                    style=style
    )

    proj_coeffs = [None] * (dim-1)
    for item in range(dim-1):
        proj_coeffs[item] = widgets.FloatText(
                                        value=0.0,
                                        step=0.1,
                                        description=f'Coefficient de Projection {item+1}',
                                        disabled=False,
                                        style=style
        )

    operation = widgets.RadioButtons(
                    options=['Projection', 'Normalization', 'Revert'],
                    description='Opération:',
                    disabled=False,
                    style=style
    )

    display(Latex("Régler les paramètres et évaluer la cellule suivante"))
    display(step_number)
    display(operation)
    display(norm_coeff)
    for item in range(dim-1):
        display(proj_coeffs[item])

    return norm_coeff, proj_coeffs, operation, step_number


def interactive_gram_schmidt(norm_coeff, proj_coeffs, operation, step_number, old_vects, VectorsList, W=None):
    """Method that, given a list of vectors, allows to perform the Gram-Schmidt process interactively.

    :param norm_coeff:
    :type norm_coeff:
    :param proj_coeffs:
    :type proj_coeffs:
    :param operation:
    :type operation:
    :param step_number:
    :type step_number:
    :param VectorsList:
    :type VectorsList:
    :param old_vects:
    :type old_vects:
    :param W:
    :type W:
    :return:
    :rtype:
    """

    if type(VectorsList[0][0]) is not np.ndarray or type(VectorsList[0][0][0]) is not float:
        for elem in range(len(VectorsList[0])):
            VectorsList[0][elem] = np.array(VectorsList[0][elem], dtype='float32')

    if type(old_vects[0]) is not np.ndarray or type(old_vects[0][0]) is not float:
        for elem in range(len(old_vects)):
            old_vects[elem] = np.array(old_vects[elem], dtype='float32')

    step = step_number.value
    new_vects = old_vects.copy()

    if step <= 1 and operation.value == 'Projection':
        display(Latex("Opération invalide. Le premier vecteur ne peut pas être projeté!"))
        return new_vects

    display(Latex(f"Construction du vecteur numéro {step}"))

    if operation.value == 'Revert':
        if len(VectorsList) > 1:
            VectorsList.pop()
            new_vects = VectorsList[-1].copy()
        else:
            display(Latex("Impossible de revenir sur l'opération!"))

    elif operation.value == 'Normalization':
        _, true_val = gram_schmidt(new_vects[:step], W=W, return_norms=True)
        if np.abs(true_val[-1] - norm_coeff.value) <= 1e-4 and norm_coeff.value > 0.0:
            display(Latex("La valuer entrée est correct!"))
            new_vects[step-1] /= norm_coeff.value
            VectorsList.append(new_vects)
        else:
            display(Latex("La valeur entrée n'est pas correct! "
                          "Réessayez et n'oubliez pas d'insérer les résultats avec 4 chiffres après la virgule."
                          "Vous pouvez annuler votre opération en sélectionnant le bouton 'Revert'"))

    elif operation.value == 'Projection':
        true_res, true_norms = gram_schmidt(new_vects[:step], W=W, return_norms=True)
        true_res_2 = list(map(mul, true_res, true_norms))
        for item in range(step-1):
            new_vects[step-1] -= proj_coeffs[item].value * new_vects[item]
        if all([(np.linalg.norm(new_vects[item] - true_res_2[item]) <= 1e-4) for item in range(step)]):
            display(Latex("Le valuers entrées sont correctes!"))
            VectorsList.append(new_vects)
        else:
            display(Latex("Le valuer entrées ne sont pas correctes! "
                          "Réessayez et n'oubliez pas d'insérer les résultats avec 4 chiffres après la virgule."
                          "Vous pouvez annuler votre opération en sélectionnant le bouton 'Revert'"))

    else:
        raise ValueError(f"Opération {operation.value} non reconnue!")

    new_vects_mat = np.array(new_vects)
    orthonormality_mat = np.array([[np.dot(new_vects_mat[i], new_vects_mat[j])
                                    for i in range(len(new_vects_mat))]
                                    for j in range(len(new_vects_mat))])

    for dim in range(len(new_vects), 0, -1):
        orthonormality_check = np.linalg.norm(orthonormality_mat[:dim, :dim] - np.eye(dim)) <= 1e-4 and \
                               np.linalg.norm(orthonormality_mat[dim:, dim:]) <= 1e-4

        if orthonormality_check:
            display(Latex("OK! L'algorithme de Gram-Schmidt est terminé!"))
            break

    display(Latex(
        f"Vecteurs courantes: {[np.around(VectorsList[-1][item].tolist(), 4) for item in range(len(VectorsList[-1]))]}"))

    if len(new_vects[0]) == 2:
        if len(VectorsList) >= 2:
            vector_plot_2D(VectorsList[-2] + new_vects,
                           labels=[f'old vector {i}' for i in range(len(old_vects))] +
                                  [f'new vector {i}' for i in range(len(new_vects))])
        else:
            vector_plot_2D(new_vects,
                           labels=[f'old vector {i}' for i in range(len(new_vects))])
    elif len(new_vects[0]) == 3:
        if len(VectorsList) >= 2:
            vector_plot_3D(VectorsList[-2] + new_vects,
                           labels=[f'old vector {i}' for i in range(len(old_vects))] +
                                  [f'new vector {i}' for i in range(len(new_vects))])
        else:
            vector_plot_3D(new_vects,
                           labels=[f'old vector {i}' for i in range(len(new_vects))])
    else:
        display(Latex(r"La représentation graphique n'est pas possible dans $\mathbb{R}^n$, avec $n \geq 4$!"))

    return new_vects


def interactive_gram_schmidt_func(norm_coeff, proj_coeffs, operation, step_number, old_vects, VectorsList,
                                  int_limits=None, weight_function=None):
    """Method that, given a list of vectors, allows to perform the Gram-Schmidt process interactively.

    :param norm_coeff:
    :type norm_coeff:
    :param proj_coeffs:
    :type proj_coeffs:
    :param operation:
    :type operation:
    :param step_number:
    :type step_number:
    :param VectorsList:
    :type VectorsList:
    :param old_vects:
    :type old_vects:
    :param int_limits: integration limits
    :type int_limits: list[float,float] or NoneType
    :param weight_function: weighting function
    :type weight_function: sympy.function
    :return:
    :rtype:
    """

    x = sp.Symbol('x')

    if int_limits is None:
        int_limits = [-1, 1]

    if weight_function is None:
        weight_function = 1 + 0*x

    xx = np.linspace(int_limits[0], int_limits[1], 500)

    step = step_number.value
    new_vects = old_vects.copy()

    if step <= 1 and operation.value == 'Projection':
        display(Latex("Opération invalide. Le premier vecteur ne peut pas être projeté!"))
        return new_vects

    display(Latex(f"Construction de la fonction numéro {step}"))

    if operation.value == 'Revert':
        if len(VectorsList) > 1:
            VectorsList.pop()
            new_vects = VectorsList[-1].copy()
        else:
            display(Latex("Impossible de revenir sur l'opération!"))

    elif operation.value == 'Normalization':
        _, true_val = gram_schmidt_func(new_vects[:step], int_limits=int_limits, weight_function=weight_function,
                                        return_norms=True)
        if np.abs(true_val[-1] - norm_coeff.value) <= 1e-4 and norm_coeff.value > 0.0:
            display(Latex("La valuer entrée est correct!"))
            new_vects[step-1] = new_vects[step-1] / norm_coeff.value
            VectorsList.append(new_vects)
        else:
            display(Latex("La valeur entrée n'est pas correct! "
                          "Réessayez et n'oubliez pas d'insérer les résultats avec 4 chiffres après la virgule."))

    elif operation.value == 'Projection':
        true_res, true_norms = gram_schmidt_func(new_vects[:step], int_limits=int_limits,
                                                 weight_function=weight_function,
                                                 return_norms=True)
        true_res_2 = [true_res[i] * true_norms[i] for i in range(len(true_norms))]
        for item in range(step-1):
            new_vects[step-1] = new_vects[step-1] - proj_coeffs[item].value * new_vects[item]

        if all([(np.linalg.norm(np.vectorize(sp.lambdify(x, new_vects[item] - true_res_2[item], "numpy"))(xx)) <= 1e-4)
                for item in range(step)]):
            display(Latex("Le valuers entrées sont correctes!"))
            VectorsList.append(new_vects)
        else:
            display(Latex("Le valuer entrées ne sont pas correctes! "
                          "Réessayez et n'oubliez pas d'insérer les résultats avec 4 chiffres après la virgule."
                          "Vous pouvez annuler votre opération en sélectionnant le bouton 'Revert'"))

    else:
        raise ValueError(f"Opération {operation.value} non reconnue!")

    display(Latex(f"Fonctions courantes: {new_vects}"))

    p = sp.plotting.plot(show=False, xlim=(int_limits[0], int_limits[1]), ylim=(-2, 2),
                         title="Fonctions")
    colors = cycle(["b", "g", "r", "c", "m", "y", "k"])
    for item in range(len(new_vects)):
        p.append(sp.plotting.plot(new_vects[item], show=False, xlim=(int_limits[0], int_limits[1]), ylim=(-2, 2),
                                  title="Fonctions", line_color=next(colors))[0])
    p.show()

    orthonormality_mat = np.array([[float(sp.integrate(weight_function * new_vects[i] * new_vects[j],
                                                       (x, int_limits[0], int_limits[1])))
                                    for i in range(len(new_vects))]
                                   for j in range(len(new_vects))])

    for dim in range(len(new_vects), 1, -1):
        orthonormality_check = np.linalg.norm(orthonormality_mat[:dim, :dim] - np.eye(dim)) <= 1e-4 and \
                               np.linalg.norm(orthonormality_mat[dim:, dim:]) <= 1e-4

        if orthonormality_check:
            display(Latex("OK! L'algorithme de Gram-Schmidt est terminé!"))
            break

    return new_vects


def integrate_sp_function(func, x, int_limits, show=True):
    """Easy-to-use interface to the sympy integrate command, used to integrate uni-variate functions

    :param func: function to integrate
    :type func: sympy.function
    :param x: integration variable
    :type x: sympy.Symbol
    :param int_limits: integration limits
    :type int_limits: list[float,float]
    :param show: if True, the result of the integration is displayed. Defaults to True.
    :type show: bool
    :return: result of the integration
    :rtype: float
    """

    I = sp.integrate(func, (x, int_limits[0], int_limits[1]))

    if show:
        display(Latex(f"Résultat de l'intégration: {I}"))

    return I


def extract_vectors_from_sympy_FiniteSet(S):
    """Method to extract characteristic vectors from a sympy FiniteSet.
    NOTICE: here the FiniteSet is assumed to have a unique element, as the solutions to linear systems!

    :param S: sympy FiniteSet to be evaluated
    :type S: sp.sets.FiniteSet
    :return: characteristic vectors of the set, in the form (characteristic solution, basis vectors)
    :rtype: list[np.ndarray]
    """

    try:
        assert isinstance(S, sp.sets.FiniteSet) or isinstance(S, sp.sets.EmptySet)
    except AssertionError:
        raise ValueError(f"L'ensemble doit être de type sympy.sets.FiniteSet ou sympy.sets.EmptySet, "
                         f"alors qu'ici il est de type {type(S)}")

    if len(S) > 0 and isinstance(S.args[0], Iterable):
        S = S.args[0]

    if isinstance(S, sp.sets.EmptySet) or len(S) == 0:
        return []

    used_symbols = list(S.atoms(sp.Symbol))
    num_symbols = len(used_symbols)

    sol = []
    S = list(S)

    base_sol = np.zeros(len(S))
    for elem in range(len(S)):
        if isinstance(S[elem], sp.Number):
            base_sol[elem] = S[elem]
            S[elem] -= base_sol[elem]
        else:
            values = (0,) * num_symbols
            base_sol[elem] = sp.lambdify(list(used_symbols), S[elem])(*values)
            S[elem] -= base_sol[elem]
    sol.append(base_sol)

    for count, symbol in enumerate(used_symbols):
        expansion_sol = np.zeros(len(S))
        for elem in range(len(S)):
            if isinstance(S[elem], sp.Number):
                expansion_sol[elem] = 0
            else:
                values = [0,] * num_symbols
                values[count] = 1
                values = tuple(values)
                expansion_sol[elem] = sp.lambdify(list(used_symbols), S[elem])(*values)
        sol.append(expansion_sol)

    return sol


def compare_sympy_FiniteSets(set1, set2):
    """Method that compares two sympy FiniteSet (or, more trivially, EmptySet) and determines whether they are
    equivalent or not

    :param set1: first set
    :type set1: sp.sets.FiniteSet or sp.sets.EmptySet
    :param set2: second set
    :type set2: sp.sets.FiniteSet or sp.sets.EmptySet
    :return: True if the two sets are equivalent, False otherwise
    :rtype: bool
    """

    vectors1 = extract_vectors_from_sympy_FiniteSet(set1)
    vectors2 = extract_vectors_from_sympy_FiniteSet(set2)

    if len(vectors1) != len(vectors2) :
        is_equal = False
    elif all([np.linalg.norm(vectors1[i] - vectors2[i]) <= 1e-5 for i in range(len(vectors1))]):
        is_equal = True
    elif len(vectors1) > 1:
        is_equal = True

        test_vectors = np.array([vectors1[i] - vectors1[0] + vectors2[0]
                                 for i in range(1, len(vectors1))]).T
        basis_vectors = np.array(vectors2[1:]).T

        rank1 = np.linalg.matrix_rank(basis_vectors)
        rank2 = np.linalg.matrix_rank(np.hstack((basis_vectors, test_vectors)))
        if rank1 != rank2:
            is_equal = False

        if is_equal:
            test_vectors = np.array([vectors2[i] - vectors2[0] + vectors1[0]
                                     for i in range(1, len(vectors2))]).T
            basis_vectors = np.array(vectors1[1:]).T

            rank1 = np.linalg.matrix_rank(basis_vectors)
            rank2 = np.linalg.matrix_rank(np.hstack((basis_vectors, test_vectors)))
            if rank1 != rank2:
                is_equal = False
    else:
        is_equal = False

    return is_equal


def unique_qr(A, only_valid=True):
    """Method to compute a unique QR factorization of a matrix A (if possible, else it raises a LinAlg error),
    imposing the diagonal entries of R to be strictly positive.

    :param A: matrix to factorise
    :type A: list[list[float]] or np.ndarray
    :param only_valid: if True, the method raises a ValueError if A does not respect the criteria for the existance
       of a QR decomposition. Defaults to True
    :type only_valid: bool
    :return: Q and R, resulting from QR factorization of A, with positivity constraint on the diagonal entries of R.
       If A has more columns than rows or if its columns are linearly dependent, a ValueError is raised if 'only_valid'
       is set to True
    :rtype: tuple(np.ndarray, np.ndarray) ore NoneType
    """

    if only_valid:
        if A.shape[1] > A.shape[0]:
            raise ValueError("Cette méthode calcule uniquement la factorisation QR pour les matrices de taille MxN "
                             "avec M>=N, car cela garantit l'existence d'une factorisation QR.")

        if np.linalg.matrix_rank(A) < A.shape[1]:
            raise ValueError("Cette méthode ne calcule la factorisation QR que pour les matrices ayant "
                             "colonnes linéairement indépendantes, car cela garantit l'existence de"
                             "une factorisation QR.")

    Q, R = np.linalg.qr(A, mode='reduced')
    signs = 2 * (np.diag(R) >= 0) - 1
    Q = Q * signs[np.newaxis, :]
    R = R * signs[:, np.newaxis]

    R[np.abs(R) < 1e-10] = 0

    return Q, R


def check_QR_step(Q, R):
    """Method to check the step number of the interactive QR algorithm, based on the expressions of Q and R

    :type Q: temporary orthogonal matrix
    :type Q: list[list] or np.ndarray
    :param R: temporary upper triangular matrix
    :type R: list[list] or np.ndarray
    :return: step number
    :rtype: int
    """

    orthonormality_mat = np.array([[np.dot(Q[:, i], Q[:, j])
                                    for i in range(Q.shape[1])]
                                   for j in range(Q.shape[1])])

    step_check = False
    step_number = 0
    for dim in range(Q.shape[1], 0, -1):
        orthonormality_check = np.linalg.norm(orthonormality_mat[:dim, :dim] - np.eye(dim)) <= 1e-4 and \
                               np.linalg.norm(orthonormality_mat[dim:, dim:]) <= 1e-4

        if orthonormality_check:
            step_number = Q.shape[1] + 1
            break

        if not orthonormality_check:
            step_check = np.linalg.norm(orthonormality_mat[:dim, :dim] - np.eye(dim)) <= 1e-4

            step_check_2 = False
            if step_check:
                for dim2 in range(1, Q.shape[1] - dim):
                    step_check_2 = np.linalg.norm(orthonormality_mat[dim:dim+dim2, dim:dim+dim2]) <= 1e-4

                    if step_check_2:
                        step_number = dim + dim2 + 1
                        break

                if not step_check_2:
                    step_number = dim + 1
                    break

    if not step_check:
        step_number = 1

    R_check = True
    if step_number > 1:
       R_check = np.allclose(R[:step_number-1, :step_number-1], np.triu(R[:step_number-1, :step_number-1])) and \
                 np.all([R[i, i] >= 0 for i in range(step_number-1)])

    if not R_check:
        raise ValueError(f"Sur la base de la valeur de la matrice Q, l'algorithme doit être à l'étape {step_number+1}, "
                         f"mais la matrice R correspondante n'est pas triangulaire supérieure et avec des valeurs "
                         f"non-négatives le long de la diagonale principale!")

    return step_number


def manual_QR(Q, R, A):
    """Interactive code, that allows to perform the interactive QR factorization of a matrix

    :type Q: temporary orthogonal matrix
    :type Q: list[list] or np.ndarray
    :param R: temporary upper triangular matrix
    :type R: list[list] or np.ndarray
    :param A: original matrix to be factorized
    :type A: list[list] or np.ndarray
    :return:
    :rtype:
    """

    style = {'description_width': 'initial'}

    display(Latex(f"MATRICES COURANTES"))
    printA(A, name="A")
    printA(Q, name="Q")
    printA(R, name="R")

    step_number = check_QR_step(Q, R)

    if step_number == Q.shape[1] + 1:
        display(Latex("OK! La factorisation QR est terminé!"))
        return

    display(Latex(f"En regardant les matrices Q et R, vous êtes à l'étape {step_number} de l'algorithme de "
                  f"factorisation QR!"))
    if step_number == 1:
        display(Latex(f"Il suffit de normaliser le premier vecteur colonne de la matrice $A$!"))
    else:
        display(Latex(f"Considérez la colonne nombre {step_number} de la matrice $A$!"))
        display(Latex(r"Calculez son produit scalaire avec les vecteurs orthonormaux déjà dérivés "
                      r"$\{\langle c_i, w_j \rangle\}_{j=1}^{i-1}$ et insérez les valeurs dans les espaces données."))
        display(Latex(r"Enfin, dérivez sa projection sur l'espace généré par les vecteurs déjà considérés "
                      r"(via l'algorithme de Gram-Schmidt: "
                      r"$\tilde{w}_i = c_i - \sum\limits_{j=1}^{i-1} \langle c_i, w_j \rangle w_j $"
                      r"calculez sa norme $||\tilde{w}_i||$ et insérez-la dans le dernier espace!"))

    proj_coeffs = [None] * (step_number-1)
    for item in range(step_number-1):
        proj_coeffs[item] = widgets.FloatText(
                                        value=0.0,
                                        step=0.1,
                                        description=f'Coefficient de Projection {item+1}',
                                        disabled=False,
                                        style=style
                                        )

    norm_coeff = widgets.BoundedFloatText(
                                        value=1.0,
                                        step=0.1,
                                        min=0.0,
                                        max=1e10,
                                        description='Coefficient de Normalisation',
                                        disabled=False,
                                        style=style
                                        )


    display(Latex("Régler les paramètres et évaluer la cellule suivante"))
    for item in range(step_number-1):
        display(proj_coeffs[item])
    display(norm_coeff)

    return norm_coeff, proj_coeffs


def interactive_QR(norm_coeff, proj_coeffs, QList, RList, A):
    """Method that allows to perform the QR factorization algorithm interactively

    :param norm_coeff:
    :type norm_coeff:
    :param proj_coeffs:
    :type proj_coeffs:
    :param QList:
    :type QList:
    :param RList:
    :type RList:
    :param A:
    :type A:
    :return:
    :rtype:
    """

    try:
        assert len(QList) > 0 and len(RList) > 0
    except AssertionError:
        raise ValueError("Les listes de matrices Q et R ne doivent pas être vides!")

    step_number = len(proj_coeffs) + 1

    Q_new = QList[-1].copy()
    Q_new[:, step_number-1] = A[:, step_number-1]
    for index in range(step_number-1):
        Q_new[:, step_number-1] -= proj_coeffs[index].value * Q_new[:, index]
    Q_new[:, step_number-1] /= norm_coeff.value

    R_new = RList[-1].copy()
    R_new[:step_number-1, step_number-1] = np.array([proj_coeffs[index].value for index in range(step_number-1)])
    R_new[step_number:, step_number-1] = 0
    R_new[step_number-1, step_number-1] = norm_coeff.value

    Q_true, R_true = unique_qr(A, only_valid=False)

    Q_check = np.linalg.norm(Q_new[:, :step_number] - Q_true[:, :step_number]) <= 1e-4 * np.max(A)
    R_check = np.linalg.norm(R_new[:, :step_number] - R_true[:, :step_number]) <= 1e-4 * np.max(A)

    # printA(Q_new, name="Q_new")
    # printA(Q_true, name="Q_true")
    # printA(R_new, name="R_new")
    # printA(R_true, name="R_true")

    if Q_check and R_check:
        display(Latex("C'est correct!"))
        QList.append(Q_new)
        RList.append(R_new)
        display(Latex(f"MATRICES COURANTES"))
        printA(A, name="A")
        printA(Q_new, name="Q")
        printA(R_new, name="R")
        if step_number == A.shape[1]:
            display(Latex("OK! La factorisation QR est terminé!"))
    else:
        display(Latex("C'est faux!"))

    return

