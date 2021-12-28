import sys
sys.path.insert(0, './../')
import numpy as np
import plotly
plotly.offline.init_notebook_mode(connected=True)
import ipywidgets as widgets
from IPython.display import display, Latex
from ipywidgets import interact_manual, Layout
from Librairie.AL_Fct import printA, texMatrix, isDiag, isSym


def Ex2Chapitre2_1():
    """Provides the correction of exercise 2 of notebook 2_1
    """

    a = widgets.Checkbox(
        value=False,
        description=r'Il existe \(\lambda\in \mathbb{R}\) tel que \((A-\lambda B)^T\) soit échelonnée réduite',
        disabled=False,
        layout=Layout(width='80%', height='30px')
    )
    b = widgets.Checkbox(
        value=False,
        description=r'Il existe \(\lambda\in \mathbb{R}\) tel que \((A-\lambda B)^T\) soit échelonnée (mais pas réduite)',
        disabled=False,
        layout=Layout(width='80%', height='30px')

    )
    c = widgets.Checkbox(
        value=False,
        description=r"Il n'existe pas de \(\lambda\in \mathbb{R}\) tel que \((A-\lambda B)^T\) soit échelonnée",
        disabled=False,
        layout=Layout(width='80%', height='30px')
    )

    def correction(a, b, c):
        if a and not c and not b:
            display(Latex("C'est correct! Pour $\lambda=-3$ La matrice échelonnée réduite est:"))
            A = [[1, 0, -2, 3], [0, 1, -1, 7]]
            printA(A)
        else:
            display(Latex("C'est faux."))

    interact_manual(correction, a=a, b=b, c=c)

    return


def Ex3Chapitre2_1():
    """Provides the correction of exercise 3 of notebook 2_1
    """

    a = widgets.Checkbox(
        value=False,
        description=r'\(C_{32}\) vaut -14',
        disabled=False,
        layout=Layout(width='80%', height='30px')
    )
    b = widgets.Checkbox(
        value=False,
        description=r'\(C_{32}\) vaut 14',
        disabled=False,
        layout=Layout(width='80%', height='30px')

    )
    c = widgets.Checkbox(
        value=False,
        description=r'\(C_{32}\) vaut -10',
        disabled=False,
        layout=Layout(width='80%', height='30px')
    )
    d = widgets.Checkbox(
        value=False,
        description=r"\(C_{32}\) n'existe pas",
        disabled=False,
        layout=Layout(width='80%', height='30px')
    )

    def correction(a, b, c, d):
        if c and not a and not b and not d:
            display(Latex("C'est correct! La matrice C vaut:"))
            C = [[-6, 64], [-32, -22], [28, -10], [-2, 6]]
            printA(C)
        else:
            display(Latex("C'est faux."))

    interact_manual(correction, a=a, b=b, c=c, d=d)

    return


def Ex1Chapitre2_2():
    """Provides the correction of exercise 1 of notebook 2_2
    """

    a = widgets.Checkbox(
        value=False,
        description=r'Le produit $AB$ vaut: <br>'
                    r'\begin{equation*} \qquad AB = \begin{pmatrix}-1 & 4\\-3& -3\\2 & 0\end{pmatrix}\end{equation*}',
        disabled=False,
        layout=Layout(width='100%', height='110px')
    )
    b = widgets.Checkbox(
        value=False,
        description=r'Le produit $AB$ vaut: <br>'
                    r'\begin{equation*} \qquad AB =\begin{pmatrix}-1 & 8\\-3& 3\\-2 & 4\end{pmatrix}\end{equation*}',
        disabled=False,
        layout=Layout(width='100%', height='110px')

    )
    c = widgets.Checkbox(
        value=False,
        description=r'Le produit $AB$ vaut: <br>'
                    r'\begin{equation*} \qquad AB =\begin{pmatrix}5 & -4\\1 & 0\end{pmatrix}\end{equation*}',
        disabled=False,
        layout=Layout(width='100%', height='90px')
    )
    d = widgets.Checkbox(
        value=False,
        description=r"Le produit $AB$ n'est pas défini",
        disabled=False,
        layout=Layout(width='100%', height='60px')
    )

    def correction(a, b, c, d):
        if b and not a and not c and not d:
            display(Latex("C'est correct!"))
        else:
            display(Latex("C'est faux."))

    interact_manual(correction, a=a, b=b, c=c, d=d)

    return


def Ex2Chapitre2_2():
    """Provides the correction of exercise 2 of notebook 2_2
    """

    a = widgets.Checkbox(
        value=False,
        description=r'Le produit $AB$ appartient à $\mathcal{M}_{3 \times 3}(\mathbb{R})$',
        disabled=False,
        layout=Layout(width='80%', height='50px')
    )
    b = widgets.Checkbox(
        value=False,
        description=r'Le produit $AB$ appartient à $\mathcal{M}_{3 \times 2}(\mathbb{R})$',
        disabled=False,
        layout=Layout(width='80%', height='50px')

    )
    c = widgets.Checkbox(
        value=False,
        description=r'Le produit $AB$ appartient à $\mathcal{M}_{2 \times 1}(\mathbb{R})$',
        disabled=False,
        layout=Layout(width='80%', height='50px')
    )
    d = widgets.Checkbox(
        value=False,
        description=r"$AB$ n'est pas définie",
        disabled=False,
        layout=Layout(width='80%', height='50px')
    )

    def correction(a, b, c, d):
        if c and not a and not b and not d:
            A = [[14], [6]]
            texA = '$' + texMatrix(A) + '$'
            display(Latex(r"C'est correct! Le produit $ AB$ vaut: $AB$ = " + texA))
        else:
            display(Latex("C'est faux."))

    interact_manual(correction, a=a, b=b, c=c, d=d)

    return


def Ex3Chapitre2_2():
    """Provides the correction of exercise 3 of notebook 2_2
    """
    display(Latex("Insérez les valeurs de a et b"))
    a = widgets.FloatText(
        value=0.0,
        step=0.1,
        description='a:',
        disabled=False
    )
    b = widgets.FloatText(
        value=0.0,
        step=0.1,
        description='b:',
        disabled=False
    )

    display(a)
    display(b)

    def f():
        A = np.array([[-1, 2], [5, -2]])
        B = np.array([[-1, 1], [a.value, b.value]])

        AB = np.dot(A,B)
        texAB = '$' + texMatrix(AB) + '$'
        BA = np.dot(B,A)
        texBA = '$' + texMatrix(BA) + '$'

        if a.value == 5/2 and b.value == -3/2:
            display(Latex(r"Correcte! Le produits $AB$ et $BA$ valent chacun: " + texAB))
        else:
            display(Latex(r"Incorrecte! Le produit $AB$ vaut " + texAB + r"et par contre le produit "
                          r"$BA$ vaut " + texBA + r". Entrez de nouvelles valeurs!"))

    interact_manual(f)

    return


def Ex1Chapitre2_3(A, B, C):
    """Provides the correction to exercise 1 of notebook 2_3

    :param A: original matrix
    :type A: list[list] or numpy.ndarray
    :param B: matrix such that A+B should be diagonal
    :type B: list[list] or numpy.ndarray
    :param C: matrix such that A+C should be symmetric and not diagonal
    :type C: list[list] or numpy.ndarray
    :return:
    :rtype:
    """

    if not type(A) is np.ndarray:
        A = np.array(A)
    if not type(B) is np.ndarray:
        B = np.array(B)
    if not type(C) is np.ndarray:
        C = np.array(C)

    ans1 = isDiag(A+B)
    ans2 = isSym(A+C) and not isDiag(A+C)

    if ans1 and ans2:
        display(Latex('Correcte!'))
    else:
        display(Latex('Incorrecte! Entrez des nouvelles valeurs pur le matrices B et C!\n'))

    if ans1:
        display(Latex("A+B est bien diagonale!"))
    else:
        display(Latex("A+B est n'est pas diagonale!"))
    texAB = '$' + texMatrix(A+B) + '$'
    display(Latex(r"A+B=" + texAB))

    if ans2:
        display(Latex("A+C est bien symétrique et non diagonale!"))
    elif isSym(A + C) and isDiag(A + C):
        display(Latex("A+C est bien symétrique mais elle est aussi diagonale!"))
    else:
        display(Latex("A + C n'est pas symétrique"))
    texAC = '$' + texMatrix(A + C) + '$'
    display(Latex(r"A+C=" + texAC))

    return


def Ex2Chapitre2_3():
    """Provides the correction to exercise 2 of notebook 2_3
    """

    a = widgets.Checkbox(
        value=False,
        description=r'$(A^{-1})^T$ et $(A^T)^{-1}$ sont triangulaires supérieures mais différentes',
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )
    b = widgets.Checkbox(
        value=False,
        description=r'$(A^{-1})^T$ et $(A^T)^{-1}$ sont triangulaires inférieures mais différentes',
        disabled=False,
        layout=Layout(width='80%', height='40px')

    )
    c = widgets.Checkbox(
        value=False,
        description=r'$(A^{-1})^T$ et $(A^T)^{-1}$ sont triangulaires inférieures et identiques',
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )
    d = widgets.Checkbox(
        value=False,
        description=r'$(A^{-1})^T$ et $(A^T)^{-1}$ sont triangulaires supérieures et identiques',
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )

    def correction(a, b, c, d):
        if d and not a and not c and not b:
            A = np.array(([-1, 0, 0], [3, 1/2, 0], [1, 2, 1]))
            res = np.transpose(np.linalg.inv(A))
            texAres = '$' + texMatrix(res) + '$'
            display(Latex("C'est correct! $(A^T)^{-1}$ est donnée par: $\quad$" + texAres))
        else:
            display(Latex("C'est faux."))

    interact_manual(correction, a=a, b=b, c=c, d=d)

    return

def Ex1Chapitre2_4():
    """Provides the correction to exercise 2 of notebook 2_4
    """

    a = widgets.Checkbox(
        value=False,
        description=r'Le système admet une solution unique et elle est:'
                    r'$$\qquad \qquad x = \begin{pmatrix} 1&4/3&4/3\end{pmatrix}^T$$',
        disabled=False,
        layout=Layout(width='80%', height='70px')
    )
    b = widgets.Checkbox(
        value=False,
        description=r"Le système n'admet aucune solution",
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )
    c = widgets.Checkbox(
        value=False,
        description=r'Le système admet une solution unique et elle est:'
                    r'$$\qquad \qquad x = \begin{pmatrix} 1&4/3&8/3\end{pmatrix}^T$$',
        disabled=False,
        layout=Layout(width='80%', height='70px')

    )
    d = widgets.Checkbox(
        value=False,
        description=r'Le système admet plusieurs solutions',
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )
    e = widgets.Checkbox(
        value=False,
        description=r'$A$ est inversible et son inverse est: <br>'
                    r'$$\qquad \qquad A^{-1} = \begin{pmatrix} 1/2&0&1/2\\1/2&-1/3&5/3\\1/2&-2/3&5/6\end{pmatrix}$$',
        disabled=False,
        layout=Layout(width='80%', height='100px')
    )
    f = widgets.Checkbox(
        value=False,
        description=r"$A$ n'est pas inversible",
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )
    g = widgets.Checkbox(
        value=False,
        description=r'$A$ est inversible et son inverse est:'
                    r'$$\qquad \qquad A^{-1} = \begin{pmatrix} 1/2&0&1/2\\1/2&-1/3&5/3\\1/2&-2/3&-1/2\end{pmatrix}$$',
        disabled=False,
        layout=Layout(width='80%', height='100x')
    )

    def correction(a, b, c, d, e, f, g):
        if c and e and not a and not b and not d and not f and not g:
            display(Latex("C'est correct!"))
        else:
            display(Latex("C'est faux."))

    interact_manual(correction, a=a, b=b, c=c, d=d, e=e, f=f, g=g)

    return


def Ex2Chapitre2_4():
    """Provides the correction to exercise 3 of notebook 2_4
    """

    a = widgets.Checkbox(
        value=False,
        description=r"Le système n'admet une solution unique que si $\alpha < 2$",
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )
    b = widgets.Checkbox(
        value=False,
        description=r"Le système n'admet une solution unique que si $\alpha \geq 2$",
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )
    c = widgets.Checkbox(
        value=False,
        description=r'Le système admet une solution unique $\forall \alpha \in \mathbb{R}$',
        disabled=False,
        layout=Layout(width='80%', height='40px')

    )
    d = widgets.Checkbox(
        value=False,
        description=r"Le système n'admet aucune solution si $\alpha < 2$, alors qu'il admet une solution unique si "
                    r"$\alpha \geq 2$",
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )
    e = widgets.Checkbox(
        value=False,
        description=r"Le système admet plusieurs solutions si $\alpha \neq 2$, alors qu'il admet une solution unique si"
                    r" $\alpha = 2$",
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )
    f = widgets.Checkbox(
        value=False,
        description=r"Le système n'admet jamais une solution unique, quelle que soit  $\alpha \in \mathbb{R}$",
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )

    def correction(a, b, c, d, e, f):
        if f and not a and not b and not c and not d and not e:
            display(Latex("C'est correct!"))
        else:
            display(Latex("C'est faux."))

    interact_manual(correction, a=a, b=b, c=c, d=d, e=e, f=f)

    return


def Ex1aChapitre2_5():
    """Provides the correction to exercise 1a of notebook 2_5
    """

    a = widgets.Checkbox(
        value=False,
        description=r'\(E_1E_2\) multiplie la ligne 4 par -6 et échange les lignes 2 et 3',
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )
    b = widgets.Checkbox(
        value=False,
        description=r'\(E_1E_2\) ajoute 6 fois la ligne 4 à la ligne 2 et échange les lignes 1 et 3',
        disabled=False,
        layout=Layout(width='80%', height='40px')

    )
    c = widgets.Checkbox(
        value=False,
        description=r'\(E_1E_2\) échange les lignes 1 et 3 et ajoute -6 fois la ligne 4 à la ligne 2',
        disabled=False,
       layout=Layout(width='80%', height='40px')
    )
    d = widgets.Checkbox(
        value=False,
        description=r"\(E_1E_2\) ajoute -6 fois la ligne 4 à la ligne 2 et échange les lignes 1 et 2",
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )

    def correction(a, b, c, d):
        if c and not(a) and not(d) and not(b):
            display(Latex("C'est correct! Par exemple, si on applique le produit à la matrice ci-dessous"))
            A=[[1,-1,0,0], [0,0,0,1], [1,2,1,2], [1,0,0,1]]
            B=[[1,0,0,0], [0,1,0,-6], [0,0,1,0], [0,0,0,1]]
            C=[[0,0,1,0], [0,1,0,0], [1,0,0,0], [0,0,0,1]]
            BCA = np.linalg.multi_dot([B,C,A])
            texA = '$' + texMatrix(A) + '$'
            texBCA = '$' + texMatrix(BCA) + '$'
            display(Latex('$\qquad A = $' + texA))
            display(Latex("on obtient"))
            display((Latex('$\qquad \hat{A} = $' + texBCA)))
        else:
            display(Latex("C'est faux."))


    interact_manual(correction,a=a,b=b,c=c,d=d)

    return


def Ex1bChapitre2_5(inv):
    """Provides the correction to exercise 1b of notebook 2_5

    :param inv: inverse of the matrix to be calculated
    :type inv: list[list]
    """

    if inv == [[0, 0, 1, 0], [0, 1, 0, 6], [1, 0, 0, 0], [0, 0, 0, 1]]:
        display(Latex("C'est correct!"))
    else:
         display(Latex("C'est faux."))

    return


def Ex2aChapitre2_5(A, B, T, D, L):
    """Provides the correction to exercise 2a of notebook 2_5

    :param A: starting matrix
    :type A: list[list]
    :param B: target matrix
    :type B: list[list]
    :param T: permutation (type I) matrix
    :type T: list[list]
    :param D: scalar multiplication (type II) matrix
    :type D: list[list]
    :param L: linear combination (type III) matrix
    :type L: list[list]
    """

    if ~(B - np.linalg.multi_dot([L, D, T, A])).any():
        display(Latex("C'est correct!"))
    else:
        display(Latex("C'est faux."))
        str = 'Il faut entrer la/les matrice(s) {'
        if (np.asmatrix(T) - np.asmatrix([[0, 0, 1], [0, 1, 0], [1, 0, 0]])).any():
            str = str + ' T, '
        if (np.asmatrix(D) - np.asmatrix([[1, 0, 0], [0, 1, 0], [0, 0, 5]])).any():
            str = str + ' D, '
        if (np.asmatrix(L) - np.asmatrix([[1, 0, 0], [-4, 1, 0], [0, 0, 1]])).any():
            str = str + ' L, '
        str = str + '}. Le produit des matrices entrées vaut:'
        display(Latex(str))
        tmp = np.linalg.multi_dot([L, D, T, A])
        texM = '$' + texMatrix(tmp) + '$'
        display(Latex('$\qquad \hat{B} = $' + texM))

    return


def Ex2bChapitre2_5(inv):
    """Provides the correction to exercise 2b of notebook 2_5

    :param inv: inverse of the matrix to be calculated
    :type inv: list[list]
    """

    if inv == [[0, 0, 1/5], [4, 1, 0], [1, 0, 0]]:
        display(Latex("C'est correct!"))
    else:
        display(Latex("C'est faux."))


    return


def Ex1Chapitre2_6_7():
    """Provides the correction to exercise 1 of notebook 2_6-7
    """

    a = widgets.Checkbox(
        value=False,
        description=r'$A^{-1}$ existe et le système admet plusieurs solutions, quelle que soit la valeur de $b$',
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )
    b = widgets.Checkbox(
        value=False,
        description=r'$A^{-1}$ existe et le système admet une solution unique ou plusieurs solutions en fonction '
                    r'de la valeur de $b$',
        disabled=False,
        layout=Layout(width='80%', height='40px')

    )
    c = widgets.Checkbox(
        value=False,
        description=r'$A$ est inversible et le système admet une solution unique, quelle que soit la valeur de $b$',
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )
    d = widgets.Checkbox(
        value=False,
        description=r'$A^{-1}$ existe et le système admet au moins une solution, quelle que soit la valeur de $b$',
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )
    e = widgets.Checkbox(
        value=False,
        description=r"$A$ n'est pas inversible et le système admet une unique solution ou plusieurs solutions, "
                    r"selon la valeur de $b$ ",
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )
    f = widgets.Checkbox(
        value=False,
        description=r"$A$ n'est pas inversible et le système admet une solution unique, "
                    r"quelle que soit la valeur de $b$",
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )
    g = widgets.Checkbox(
        value=False,
        description=r"Le système admet une solution unique et $A$ n'est pas inversible",
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )
    h = widgets.Checkbox(
        value=False,
        description=r'Le système admet une solution unique et $A$ est inversible',
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )

    def correction(a, b, c, d, e, f, g, h):
        if c and d and h and not a and not b and not e and not f and not g:
            display(Latex("C'est correct!"))
        else:
            display(Latex("C'est faux."))

    interact_manual(correction, a=a, b=b, c=c, d=d, e=e, f=f, g=g, h=h)

    return


def Ex2Chapitre2_6_7():
    """Provides the correction to exercise 2 of notebook 2_6-7
    """

    a = widgets.Checkbox(
        value=False,
        description=r'$A_1$ est inversible',
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )
    b = widgets.Checkbox(
        value=False,
        description=r'$A_2$ est inversible',
        disabled=False,
        layout=Layout(width='80%', height='40px')

    )
    c = widgets.Checkbox(
        value=False,
        description=r'$A_3$ est inversible',
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )

    def correction(a, b, c):
        if a and not b and not c:
            A1 = np.array([[2, 0, 1], [0, 6, 4], [2, 2, 1]])
            A1_inv = np.linalg.inv(A1)
            texA1inv = '$' + texMatrix(A1_inv) + '$'
            display(Latex("C'est correct! $A_1$ est la seule matrice inversible et son inverse est: $\quad A_1^{-1} = $"
                          + texA1inv))
        else:
            display(Latex("C'est faux."))

    interact_manual(correction, a=a, b=b, c=c)

    return


def Ex3Chapitre2_6_7():
    """Provides the correction to exercise 3 of notebook 2_6-7
    """

    style = {'description_width': 'initial'}
    a = widgets.Checkbox(
        value=False,
        description=r"Si $\alpha = 4$ et $\beta = 2$, alors $A$ n'est pas inversible et le système linéaire "
                    r"admet une infinité de solutions",
        disabled=False,
        style=style,
        layout=Layout(width='80%', height='40px')
    )
    b = widgets.Checkbox(
        value=False,
        description=r"Si $\alpha=8$ et $\beta=-1$, alors $A$ n'est pas inversible et le système linéaire n'admet pas de"
                    r" solutions",
        disabled=False,
        style=style,
        layout=Layout(width='80%', height='40px')

    )
    c = widgets.Checkbox(
        value=False,
        description=r"Si $\alpha=-2$ et $\beta=-4$, alors $A$ n'est pas inversible et le système linéaire admet "
                    r"une infinité de solutions",
        disabled=False,
        style=style,
        layout=Layout(width='80%', height='40px')
    )
    d = widgets.Checkbox(
        value=False,
        description=r"Si $\alpha=8$ et $\beta=1$, alors $A$ n'est pas inversible et le système linéaire admet "
                    r"une infinité de solutions",
        disabled=False,
        style=style,
        layout=Layout(width='80%', height='40px')
    )
    e = widgets.Checkbox(
        value=False,
        description=r"Si $\alpha=-4$ et $\beta=-2$, alors $A$ n'est pas inversible et le système linéaire n'admet pas"
                    r"de solutions",
        disabled=False,
        style=style,
        layout=Layout(width='80%', height='40px')
    )
    f = widgets.Checkbox(
        value=False,
        description=r'Si $\alpha=4$ et $\beta=1$, alors $A$ est inversible et le système linéaire admet une infinité '
                    r'de solutions',
        disabled=False,
        style=style,
        layout=Layout(width='80%', height='40px')
    )
    g = widgets.Checkbox(
        value=False,
        description=r'Si $\alpha=4$ et $\beta=1$, alors $A$ est inversible et le système linéaire admet une solution'
                    r' unique',
        disabled=False,
        style=style,
        layout=Layout(width='80%', height='40px')
    )
    h = widgets.Checkbox(
        value=False,
        description=r"Pour infinite de valeurs de $\alpha$ et $\beta$ $A$ n'est pas inversible, mais seulement pour l'un"
                    r" d'eux le système admet une infinité de solutions",
        disabled=False,
        style=style,
        layout=Layout(width='100%', height='40px')
    )

    def correction(a, b, c, d, e, f, g, h):
        if c and e and h and not a and not b and not d and not f and not g:
            display(Latex(r"C'est correct! En effet $A$ n'est pas inversible si $\alpha = \dfrac{8}{\beta}$ (résultat "
                          r"obtenu en divisant par élément les lignes de A les unes par les autres et en imposant que "
                          r"les résultats des divisions soient les mêmes). De plus, si $\alpha = -2$ et $\beta = -4$, "
                          r"alors le système admet une infinité de solutions, puisque les rapports obtenus par la "
                          r"division est $-\dfrac{1}{2}$)."))
        else:
            display(Latex("C'est faux."))

    interact_manual(correction, a=a, b=b, c=c, d=d, e=e, f=f, g=g, h=h)

    return


def Ex4Chapitre2_6_7():
    """Provides the correction of exercise 4 of notebook 2_6-7
    """

    display(Latex("Insérez les valeurs de a et b"))
    a = widgets.FloatText(
        value=0.0,
        step=0.1,
        description='a:',
        disabled=False
    )
    b = widgets.FloatText(
        value=0.0,
        step=0.1,
        description='b:',
        disabled=False
    )

    display(a)
    display(b)

    def f():
        A = np.array([[0.5, a.value, 1], [0, 2, -1], [-2, 1, b.value]])
        B = np.array([[-6, -2, -2], [4, 2, 1], [8, 3, 2]])

        AB = np.dot(A, B)
        texAB = '$' + texMatrix(AB) + '$'
        BA = np.dot(B, A)
        texBA = '$' + texMatrix(BA) + '$'

        if a.value == -1 and b.value == -2:
            display(Latex(r"Correcte! Le produits $AB$ et $BA$ valent chacun: $I$ = " + texAB))
        else:
            display(Latex(r"Incorrecte! Le produit $AB$ vaut  " + texAB + r" et le produit "
                          r"$BA$ vaut  " + texBA + r",  donc $A$ ne peut pas être l'inverse de $B$. "
                          r"Entrez de nouvelles valeurs!"))

    interact_manual(f)

    return


def Ex1Chapitre2_8_9(E1, E2, E3, E4):
    """Provides the correction of exercise 2 of notebook 2_8_9

    :param E1:
    :type E1:
    :param E2:
    :type E2:
    :param E3:
    :type E3:
    :param E4:
    :type E4:
    :return:
    :rtype:
    """

    # MATRIX A1
    E_pre_1 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    E_post_1 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [-1, 0, 0, 1]]

    # MATRIX A2
    E_pre_2 = [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    E_post_2 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1/2, 0], [0, 0, 0, 1]]

    # MATRIX A3
    E_pre_3 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    E_post_3 = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, -1], [0, 0, 0, 1]]

    # MATRIX A4
    E_pre_4 = [[1/2, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    E_post_4 = [[1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 0, 1]]

    E_bool = np.zeros(4).astype(bool)
    E_bool[0] = E1[0] == E_pre_1 and E1[1] == E_post_1
    E_bool[1] = E2[0] == E_pre_2 and E2[1] == E_post_2
    E_bool[2] = E3[0] == E_pre_3 and E3[1] == E_post_3
    E_bool[3] = E4[0] == E_pre_4 and E4[1] == E_post_4

    correct = set(np.where(E_bool)[0]+1)
    wrong = set(np.arange(1,5)) - correct

    if wrong:
        if correct:
            display(Latex(f"Corrects: {correct}"))
        else:
            display((Latex("Corrects: {}")))
        display(Latex(f"Manqué: {wrong}"))
    else:
        display(Latex("C'est correcte."))

    return

def Ex3Chapitre2_8_9():
    """Provides the correction of exercise 3 of notebook 2_8_9
    """
    
    
    a_1 = widgets.Checkbox(
        value=False,
        description=r'La matrice $A_1$ admet la décomposition LU',
        disabled=False,
        layout=Layout(width='80%', height='30px')
    )
    
    a_2 = widgets.Checkbox(
        value=False,
        description=r'La matrice $A_2$ admet la décomposition LU',
        disabled=False,
        layout=Layout(width='80%', height='30px')
    )
    
    a_3 = widgets.Checkbox(
        value=False,
        description=r'La matrice $A_3$ admet la décomposition LU',
        disabled=False,
        layout=Layout(width='80%', height='30px')
    )
    
    def correction(a_1, a_2, a_3):
        if not a_1 and a_2 and not a_3:
            display(Latex("C'est correct! Plus précisément, la matrice $A_1$ n'admet pas décomposition LU car elle n'est pas inversible, la matrice $A_2$ admet décomposition LU et la matrice $A_3$ n'admet pas décomposition LU car elle ne peut pas être réduite sans échanger deux lignes pendant la méthode d'élimination de Gauss"))         
        else:
            display(Latex("C'est faux."))

    interact_manual(correction, a_1=a_1, a_2=a_2, a_3=a_3)

    return


def Ex1Chapitre2_10():
    """Provides the correction to exercise 1 of notebook 2_10
    """

    a = widgets.Checkbox(
        value=False,
        description=r"Le système linéaire n'admet aucune solution",
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )
    b = widgets.Checkbox(
        value=False,
        description=r"Le système linéaire admet une solution unique",
        disabled=False,
        layout=Layout(width='80%', height='40px')

    )
    c = widgets.Checkbox(
        value=False,
        description=r"Le système linéaire admet deux solutions distinctes",
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )
    d = widgets.Checkbox(
        value=False,
        description=r"Le système linéaire admet une infinité de solutions",
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )
    e = widgets.Checkbox(
        value=False,
        description=r"La décomposition LU de $A$ ne peut pas être calculée",
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )
    f = widgets.Checkbox(
        value=False,
        description=r"La dernière colonne de $U$ est entièrement composée de zéros",
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )
    g = widgets.Checkbox(
        value=False,
        description=r'La dernière ligne de $U$ est entièrement composée de zéros',
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )
    h = widgets.Checkbox(
        value=False,
        description=r'La première entrée de la première ligne de $L$ est égale à 1',
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )

    def correction(a, b, c, d, e, f, g, h):
        if not a and not b and not c and d and not e and not f and g and not h:
            display(Latex("C'est correct. En effet, $A$ n'est clairement pas inversible, car la dernière ligne est "
                          "égale à la seconde moins la première, et il en va de même pour le vecteur de droite $b$. "
                          "Par conséquent, la dernière ligne de $U$ est entièrement composée de zéros (réponse 7) et la "
                          "dernière entrée du vecteur de droite $b$, après l'application de la méthode d'élimination de "
                          "Gauss, est également égale à 0. Ainsi, la dernière équation du système linéaire résultant a "
                          "tous les coefficients égaux à 0, ce qui donne lieu à une infinité de solutions "
                          "(réponse 4)."))
        else:
            display(Latex("C'est faux."))

    interact_manual(correction, a=a, b=b, c=c, d=d, e=e, f=f, g=g, h=h)

    return


def Ex2Chapitre2_10(L, U, b, x, y):
    """Provides the correction to exercise 2 of notebook 2_10

    :param L: lower triangular matrix from LU decomposition
    :type L: list[list]
    :param U: upper triangular matrix from LU decomposition
    :type U: list[list]
    :param b: right-hand side vector
    :type b: list[list]
    :param x: system solution
    :type x: list[list]
    :param y: temporary variable
    :type y: list[list]
    """

    if type(L) is list:
        L = np.array(L)

    if type(U) is list:
        U = np.array(U)

    if type(x) is list:
        x = np.array(x)

    if type(y) is list:
        y = np.array(y)

    y_true = np.linalg.solve(L, b)
    x_true = np.linalg.solve(U, y)

    res_x = np.linalg.norm(x - x_true) / np.linalg.norm(x_true) <= 1e-4
    res_y = np.linalg.norm(y - y_true) / np.linalg.norm(y_true) <= 1e-4

    if res_x and res_y:
        display(Latex("C'est correct"))
    else:
        display(Latex("C'est faux"))

    return


def Ex3Chapitre2_10():
    """Provides the correction to exercise 3 of notebook 2_10

    :return:
    :rtype:
    """

    a = widgets.Checkbox(
        value=False,
        description=r"If L is such that all its diagonal elements equal 1, then the temporary variable y is a vector "
                    r"of ones as well",
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )

    b = widgets.Checkbox(
        value=False,
        description=r"Matrix L is diagonal",
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )

    c = widgets.Checkbox(
        value=False,
        description=r"Matrix U is diagonal",
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )

    d = widgets.Checkbox(
        value=False,
        description=r"The second entry of the solution is always 2.5 times the fourth one",
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )

    e = widgets.Checkbox(
        value=False,
        description=r"The second entry of the solution is always 5 times the fourth one",
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )

    f = widgets.Checkbox(
        value=False,
        description=r"The sum of all the entries always equals 2.5",
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )

    g = widgets.Checkbox(
        value=False,
        description=r"The sum of all the entries but the second one always equals 0",
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )

    h = widgets.Checkbox(
        value=False,
        description=r"The vector $\hat{x} = (1, 0 -1, 0)$ is one of the solutions to the linear system",
        disabled=False,
        layout=Layout(width='80%', height='40px')
    )

    def correction(a, b, c, d, e, f, g, h):
        if a and not b and not c and d and not e and not f and g and h:
            display(Latex("C'est correct. Indeed the set of all possible solutions can be written as "
                          "$x = [1-4a, 2.5a, 3a-1, a]$. From this, it is clear that the second entry is 2.5 times the "
                          "fourth one (answer 4), that the sum of all the entries but the second one equals $0$ "
                          "(answer 7) and that $x^* = [1,0,-1,0]$ is a solution, in case $a$ is set to 0 (answer 8). "
                          "Also, if L is computed so that it has ones on its diagonal, it is immediate to deduce that "
                          "the temporary vector $y$ that solves the system $Ly=b$ is actually all made of ones."))
        else:
            display(Latex("C'est faux."))

    interact_manual(correction, a=a, b=b, c=c, d=d, e=e, f=f, g=g, h=h)

    return






