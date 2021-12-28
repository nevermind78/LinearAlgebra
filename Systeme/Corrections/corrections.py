import sys
sys.path.insert(0, './../')

import plotly
import numpy as np
plotly.offline.init_notebook_mode(connected=True)
import ipywidgets as widgets
import plotly.graph_objs as go

from ipywidgets import interact_manual, Layout
from Librairie.AL_Fct import drawLine
from IPython.display import display, Latex, display_latex

from ipywidgets import interactive, HBox, VBox, widgets, interact, FloatSlider
###############   CHAPITRE 1_3_4    ###############


def Ex3Chapitre1_3_4():
    """Provides the correction to exercise 3 of notebook 1_3-4
    """

    print("Cliquer sur CTRL (ou CMD) pour sélectionner plusieurs réponses")

    style = {'description_width': 'initial'}
    res = widgets.SelectMultiple(
        options=['a)', 'b)', 'c)'],
        description='Systèmes avec le même ensemble de solutions:',
        style=style,
        layout=Layout(width='35%', height='170px'),
        disabled=False,
    )

    def correction(res):
        if 'a)' in res and 'c)' in res :
            print("C'est correct!")
            print('Pour le système a), on peut par exemple faire\n')
            sola= '$\\left(\\begin{array}{cc|c} 1 & 1 & 3\\\\ -1& 4 & -1 \\end{array}\\right) \\stackrel{E_{12}}{\sim}\\left(\\begin{array}{cc|c} -1& 4 & -1\\\\ 1 & 1 & 3 \\end{array}\\right)\\stackrel{E_{1}(-2)}{\sim}\\left(\\begin{array}{cc|c} 2& -8 & 2\\\\ 1 & 1 & 3 \\end{array}\\right)$'
            display(Latex(sola))

            print("Pour le système b), les systèmes ne sont pas équivalents. Comme solution on peut exprimer x1 en fonction de x2 et on obtient deux droites (parallèles) de pente 1 mais de hauteurs -2 et 2.$")
            print('Pour le système c), on peut par exemple faire\n')
            sola= '$\\left(\\begin{array}{ccc|c} \dfrac{1}{4} & -2 & 1& 5\\\\ 0& 1 & -1 & 0\\\\ 1 & 2 & -1 & 0 \\end{array}\\right) \\stackrel{E_{1}(4)}{\sim}\\left(\\begin{array}{ccc|c} 1 & -8 & 4& 20\\\\ 0& 1 & -1 & 0\\\\ 1 & 2 & -1 & 0\\end{array}\\right)\\stackrel{E_{31}(-1)}{\sim}\\left(\\begin{array}{ccc|c} 1& -8 & 4&20\\\\ 0 & 1 & -1&0\\\\ 0&10 &-5 & -20\\end{array}\\right)\\stackrel{E_{3}\\big({\\tiny\dfrac{1}{5}}\\big)}{\sim}\\left(\\begin{array}{ccc|c}1& -8 & 4&20\\\\ 0 & 1 & -1&0\\\\ 0&2&-1 & -4\\end{array}\\right)$'
            display(Latex(sola))
            
        else:
            print("C'est faux. Veuillez rentrer d'autres valeurs")

    interact_manual(correction, res=res)

    return



def Ex4Chapitre1_3_4():
    """Provides the correction to exercise 4 of notebook 1_3-4
    """

    print("Cliquer sur CTRL (ou CMD) pour sélectionner plusieurs réponses")

    style = {'description_width': 'initial'}
    res = widgets.SelectMultiple(
        options=['a)', 'b)', 'c)', 'd)', 'e)', 'f)', 'g)', 'h)'],
        description='Systèmes avec le même ensemble de solutions:',
        style=style,
        layout=Layout(width='15%', height='170px'),
        disabled=False,
    )

    def correction(res):
        if 'a)' in res and 'c)' in res and 'd)' in res and 'h)' in res:
            print("C'est correct!")
        else:
            print("C'est faux. Veuillez rentrer d'autres valeurs")

    interact_manual(correction, res=res)

    return



###############   CHAPITRE 1_5_6    ###############

def Ex1Chapitre1_5_6(data):
    """Provides the correction to exercise 1 of notebook 1_5-6
    e=matrice qui sont échelonnée, er=échelonnée réduite et r=rien
    """

    e=data[0].value
    er=data[1].value
    r=data[2].value
    r=list(r.split(','))
    r=[elem.strip() for elem in r if elem.strip()]
    er=list(er.split(','))
    er=[elem.strip() for elem in er if elem.strip()]
    e=list(e.split(','))
    e=[elem.strip() for elem in e if elem.strip()]

    corr_e=['C','D','E','G','H','I','J']
    corr_er=['D','H','I','J']
    corr_r=['A','B','F']

    if set(corr_r) == set(r) and set(corr_er) == set(er) and set(corr_e) == set(e):
        print('Correct')
    else:
        if not set(corr_r) == set(r):
            print("Les matrices n'étant ni échelonnées, ni échelonnées-réduites sont fausses. ")
        if not set(corr_e) == set(e):
            print("Les matrices étant échelonnées sont fausses. ")
        if not set(corr_er) == set(er):
            print("Les matrices étant échelonnées-réduite sont fausses. ")
    return


###############   CHAPITRE 1_7   ###############


def Ex2Chapitre1_7():
    """Provides the correction to exercise 2 of notebook 1_7
    """

    print("Cliquer sur CTRL pour sélectionner plusieurs réponses")

    style = {'description_width': 'initial'}
    inc = widgets.SelectMultiple(
        options=['a)', 'b)', 'c)', 'd)'],
        description='Incompatibles:',
        style=style,
        layout=Layout(width='15%', height='90px'),
        disabled=False,
    )
    comp = widgets.SelectMultiple(
        options=['a)', 'b)', 'c)', 'd)'],
        description='Compatibles:',
        layout=Layout(width='15%', height='90px'),
        disabled=False
    )

    def correction(inc, c):
        if 'a)' in c and 'c)' in c and 'd)' in c and 'b)' in inc:
            print("C'est correct!")
            print("En particulier, les systèmes a) et d) admettent une infinité de solutions, tandis que le système c) "
                  "admet une solution unique.")
        else:
            print("C'est faux. Veuillez rentrer d'autres valeurs")

    interact_manual(correction, inc=inc, c=comp)

    return




def Ex3Chapitre1_7():
    """Provides the correction of exercise 3 of notebook 1_7
    """

    systa = widgets.Select(
        options=['Point', 'Droite', 'Plan', 'Incompatible'],
        description='Système a):',
        disabled=False,
    )
    systb = widgets.Select(
        options=['Point', 'Droite', 'Plan', 'Incompatible'],
        description='Système b):',
        disabled=False
    )
    systc = widgets.Select(
        options=['Point', 'Droite', 'Plan', 'Espace', 'Incompatible'],
        description='Système c):',
        disabled=False
    )
    systd = widgets.Select(
        options=['Point', 'Droite', 'Plan', 'Espace', 'Incompatible'],
        description='Système d):',
        disabled=False
    )
    choice = widgets.Dropdown(
        options=['a)', 'b)', 'c)', 'd)'],
        value='a)',
        description='Système:',
        disabled=False,
    )

    def plot(c):
        if c == 'a)':
            drawLine([[0], [0]], [[4], [1]])
        if c == 'b)':
            print("Le système est incompatible, donc il n'y a pas de solutions")
        if c == 'c)':
            drawLine([[-17], [5], [-10]], [[0], [0], [0]])
        if c == 'd)':
            drawLine([[1], [0], [0]], [[0], [-1], [1]])

    def correction(a, b, c, d):
        if 'Droite' in a and 'Incompatible' in b and 'Point' in c and 'Droite' in d:
            print("C'est correct!")
            print("Sélectionnez le système souhaité et appuyez sur 'Run Interact'"
                  " pour visualiser son ensemble de solution(s), le cas échéant")
            interact_manual(plot, c=choice)
        else:
            print("C'est faux. Veuillez rentrer d'autres valeurs")

    interact_manual(correction, a=systa, b=systb, c=systc, d=systd)

    return



def Ex3Chapitre1_2(cas):
    if cas=='a':
        A=[[1, 5], [-2, -10]] # we initialize the problem. The values of h and k must be fixed to one specific value here
        b=[0,18]

        data=[]
        x=np.linspace(-5,5,11)
        m=len(A)
        MatCoeff = [A[i]+[b[i]]for i in range(m)] #becomes augmented matrix
        MatCoeff=np.array(MatCoeff)

        for i in range(len(MatCoeff)):
            trace=go.Scatter(x=x,  y=(MatCoeff[i,2]-MatCoeff[i,0]*x)/MatCoeff[i,1], name='a) Droite %d'%(i+1))
            data.append(trace)

        f=go.FigureWidget(data=data,
            layout=go.Layout(xaxis=dict(
                range=[-5, 5]
            ),
            yaxis=dict(
                range=[-10, 10]
            ) )

        )

        def update_y(l):
            MatCoeff = [[1, 5, l]]
            MatCoeff=np.array(MatCoeff)
            f.data[0].y=(MatCoeff[0,2]-MatCoeff[0,0]*x)/MatCoeff[0,1]

        freq_slider = interactive(update_y, l=(-15, 15, 1/2)) 
        vb = VBox((f, freq_slider))
        vb.layout.align_items = 'center'
        return vb
    elif cas=='b':
        A=[[1, 1], [-2,-1]] # we initialize the problem. The value of h is fixed
        b=[3, -1]

        m=len(A)
        MatCoeff = [A[i]+[b[i]]for i in range(m)] #becomes augmented matrix
        MatCoeff=np.array(MatCoeff)
        data=[]
        x=np.linspace(-5,5,11)
        MatCoeff=np.array(MatCoeff)
        for i in range(len(MatCoeff)):
            trace=go.Scatter(x=x,  y= (MatCoeff[i,2]-MatCoeff[i,0]*x)/MatCoeff[i,1], name='b) Droite %d'%(i+1))
            data.append(trace)

        f=go.FigureWidget(data=data,
            layout=go.Layout(xaxis=dict(
                range=[-5, 5]
            ),
            yaxis=dict(
                range=[-10, 10]
            ) )

        )

        def mat(k):
            l=k
            MatCoeff=[[l,1,3]]
            return MatCoeff
        def update_y(l):
            MatCoeff= mat(l)
            MatCoeff=np.array(MatCoeff)
            f.data[0].y=(MatCoeff[0,2]-MatCoeff[0,0]*x)/MatCoeff[0,1]

        freq_slider = interactive(update_y, l=(-10, 10, 1/2)) 

        vb = VBox((f, freq_slider))
        vb.layout.align_items = 'center'
        return vb
    elif cas=='c':
        A=[[3, 1], [1, 3]] # we initialize the problem. The values of h and k are fixed
        b=[1, 1]

        m=len(A)
        MatCoeff = [A[i]+[b[i]]for i in range(m)] #becomes augmented matrix
        MatCoeff=np.array(MatCoeff)
        data=[]
        x=np.linspace(-25,25,101)
        MatCoeff=np.array(MatCoeff)
        for i in range(len(MatCoeff)):
            trace=go.Scatter(x=x,  y= (MatCoeff[i,2]-MatCoeff[i,0]*x)/MatCoeff[i,1], name='c) Droite %d'%(i+1))
            data.append(trace)

        f=go.FigureWidget(data=data,
            layout=go.Layout(xaxis=dict(
                range=[-25, 25]
            ),
            yaxis=dict(
                range=[-50, 50]
            ) )

        )

        def mat(l):
            MatCoeff= [[l, 3, 1],[3, 1, l] ]
            return MatCoeff

        def update_y(l):
            MatCoeff= mat(l)
            MatCoeff=np.array(MatCoeff)
            f.data[0].y=(MatCoeff[0,2]-MatCoeff[0,0]*x)/MatCoeff[0,1]
            f.data[1].y=(MatCoeff[1,2]-MatCoeff[1,0]*x)/MatCoeff[1,1]

        freq_slider = interactive(update_y, l=(-20, 20, 1)) 

        vb = VBox((f, freq_slider))
        vb.layout.align_items = 'center'
        return vb        
    elif cas=='d':
        A=[[1, 1], [3, -5]] # we initialize the problem. The values of h and k are fixed
        b=[1, 2]

        m=len(A)
        MatCoeff = [A[i]+[b[i]]for i in range(m)] #becomes augmented matrix
        MatCoeff=np.array(MatCoeff)
        data=[]
        x=np.linspace(-15,15,101)
        MatCoeff=np.array(MatCoeff)
        for i in range(len(MatCoeff)):
            trace=go.Scatter(x=x,  y= (MatCoeff[i,2]-MatCoeff[i,0]*x)/MatCoeff[i,1], name='d) Droite %d'%(i+1))
            data.append(trace)

        f=go.FigureWidget(data=data,
            layout=go.Layout(xaxis=dict(
                range=[-15, 15]
            ),
            yaxis=dict(
                range=[-10, 10]
            ) )

        )

        def update_y(l, k):
            MatCoeff= [[k, 1, l],[3, -5, 2] ]
            MatCoeff=np.array(MatCoeff)
            f.data[0].y=(MatCoeff[0,2]-MatCoeff[0,0]*x)/MatCoeff[0,1]


            f.data[1].y=(MatCoeff[1,2]-MatCoeff[1,0]*x)/MatCoeff[1,1]

        freq_slider = interactive(update_y, l=(-10, 10, 1/10),k=(-10, 10, 1/10))

        vb = VBox((f, freq_slider))
        vb.layout.align_items = 'center'
        return vb 
    else:
        print('Veuillez entrer a,b,c ou d')
        
    return 

def corr3Chapitre1_2(cas): 
    if cas=='a':
        a = widgets.Checkbox(
        value=False,
        description=r'Le système possède toujours au moins une solution',
        disabled=False,
        layout=Layout(width='80%', height='30px')
         )
        b = widgets.Checkbox(
        value=False,
        description=r'Le système admet une infinité de solution pour une unique valeur de \(l\)',
        disabled=False,
        layout=Layout(width='80%', height='30px')
         )
        c = widgets.Checkbox(
        value=False,
        description=r'Quand le système admet une solution, alors il en admet toujours une infinité',
        disabled=False,
        layout=Layout(width='80%', height='30px')
         )
        d = widgets.Checkbox(
        value=False,
        description=r'Le système admet une unique solution pour toutes les valeurs de \(l\)',
        disabled=False,
        layout=Layout(width='80%', height='30px')
         )
        e = widgets.Checkbox(
        value=False,
        description=r'Le système ne possède jamais de solutions',
        disabled=False,
        layout=Layout(width='80%', height='30px')
         )
        
        def correction(a, b, c,d,e):
            if b and c and not a and not e and not d:
                print("C'est correct!")
                display(Latex('Le système admet une infinité de solution pour $l=...$'))
                m = widgets.IntText(
                        value=1,
                        step=1,
                        description='l=',
                        disabled=False
                    )
    
                display(m)

                def f():
                    if m.value == -9:
                        print("C'est correct! La hauteur des deux doites doit être la même! Ici la hauteur est -9/5 et la pente -1/5" )
                        print("Si l est différent de -9, alors le système n'admet pas de solutions car les deux droites sont parrallèles" )
                    else:
                        print('Incorrect, entrez une nouvelle valeur')
                interact_manual(f)
            else:
                display(Latex("C'est faux."))

        interact_manual(correction, a=a, b=b, c=c, d=d, e=e)
        
    elif cas=='b':
        a = widgets.Checkbox(
        value=False,
        description=r'Le système possède toujours au moins une solution',
        disabled=False,
        layout=Layout(width='80%', height='30px')
         )
        b = widgets.Checkbox(
        value=False,
        description=r"Le système n'admet pas de solutions pour une unique valeur de \(l\)",
        disabled=False,
        layout=Layout(width='80%', height='30px')
         )
        c = widgets.Checkbox(
        value=False,
        description=r"Le système n'admet pas de solutions pour une infinité de valeurs de \(l\)",
        disabled=False,
        layout=Layout(width='80%', height='30px')
         )
        d = widgets.Checkbox(
        value=False,
        description=r'Le système admet une unique solution pour une unique valeur de \(l\)',
        disabled=False,
        layout=Layout(width='80%', height='30px')
         )
        e = widgets.Checkbox(
        value=False,
        description=r'Le système admet une unique solution pour une infinité de valeurs de \(l\)',
        disabled=False,
        layout=Layout(width='80%', height='30px')
         )
        
        def correction(a, b, c,d,e):
            if b and e and not a and not c and not d:
                print("C'est correct!")
                display(Latex("Le système n'admet pas de solution pour $l=...$"))
                m = widgets.IntText(
                    value=1,
                    step=1,
                    description='l=',
                    disabled=False
                    )
    
                display(m)
                def f():
                        if m.value == 2:
                            print("C'est correct!")
                            print("Pour que les deux droites soient parallèles, il faut que leur pente soit la même! Ici la pente est de -2 et les hauteurs valent 3 (équation 1) et 1 (équation 2)" )
                        else:
                            print('Incorrect, entrez une nouvelle valeur')
                interact_manual(f) 
            else:
                display(Latex("C'est faux."))  
        interact_manual(correction, a=a, b=b, c=c, d=d, e=e)
    elif cas=='c':
        a = widgets.Checkbox(
        value=False,
        description=r'Le système ne possède jamais une infinité de solutions',
        disabled=False,
        layout=Layout(width='80%', height='30px')
         )
        b = widgets.Checkbox(
        value=False,
        description=r"Le système n'admet pas de solutions pour une unique valeur de \(l\)",
        disabled=False,
        layout=Layout(width='80%', height='30px')
         )
        c = widgets.Checkbox(
        value=False,
        description=r"Le système n'admet pas de solutions pour une infinité de valeurs de \(l\)",
        disabled=False,
        layout=Layout(width='80%', height='30px')
         )
        d = widgets.Checkbox(
        value=False,
        description=r'Le système admet une unique solution pour deux valeurs de \(l\)',
        disabled=False,
        layout=Layout(width='80%', height='30px')
         )
        e = widgets.Checkbox(
        value=False,
        description=r'Le système admet une unique solution pour une infinité de valeurs de \(l\)',
        disabled=False,
        layout=Layout(width='80%', height='30px')
         )
        
        def correction(a, b, c,d,e):
            if a and b and d and e and not c :
                print("C'est correct!")
                display(Latex("Le système n'admet pas de solution pour $l=...$"))
                m = widgets.IntText(
                        value=1,
                        step=1,
                        description='l=',
                        disabled=False
                    )
                display(m)
                def f():
                        if m.value == 9:
                            print("C'est correct!")
                            print("Pour que les deux droites soient parallèles, il faut que leur pente soit la même! Ici la pente est de 1/3 et les hauteurs valent 1/9 (équation 1) et 3 (équation 2)" )
                        else:
                            print('Incorrect, entrez une nouvelle valeur')
                interact_manual(f) 
            else:
                display(Latex("C'est faux."))  
        interact_manual(correction, a=a, b=b, c=c, d=d, e=e)
    elif cas=='d':
        a = widgets.Checkbox(
        value=False,
        description=r'Le système admet une infinité de solutions pour un unique couple \((l,k)\)',
        disabled=False,
        layout=Layout(width='80%', height='30px')
         )
        b = widgets.Checkbox(
        value=False,
        description=r"Le système admet toujours au moins une solution",
        disabled=False,
        layout=Layout(width='80%', height='30px')
         )
        c = widgets.Checkbox(
        value=False,
        description=r"Pour \(k\) fixé, on peut toujours trouver une valeur de \(l\) pour que le système admette une unique solution",
        disabled=False,
        layout=Layout(width='80%', height='30px')
         )
        d = widgets.Checkbox(
        value=False,
        description=r'Si \(k=l\), alors le système admet toujours une solution',
        disabled=False,
        layout=Layout(width='80%', height='30px')
         )
        e = widgets.Checkbox(
        value=False,
        description=r'Si \(l\le k\), le système admet toujours une solution',
        disabled=False,
        layout=Layout(width='80%', height='30px')
         )
        
        def correction(a, b, c,d,e):
            if a and not b and not d and not e and not c :
                print("C'est correct!")
                display(Latex("Si $k=-\dfrac{3}{5}$ et $l=-\dfrac{2}{5}$, le système admet une infinité de solutions."))
                display(Latex("Si $k=-\dfrac{3}{5}$ et $l\\neq -\dfrac{2}{5}$ le système n'admet pas de solution."))
                display(Latex("Si $k\\neq-\dfrac{3}{5}$ le système admet une unique solution."))
            else:
                display(Latex("C'est faux."))  
        interact_manual(correction, a=a, b=b, c=c, d=d, e=e)
    else:
        print('Veuillez entrer a,b,c ou d')
        
    return 


def Ex4Chapitre1_2():
    np.seterr(divide='ignore', invalid='ignore')

    A=[[1, -2], [-2, 1], [1, -1]] # we initialize the problem. The values of h and k are fixed
    b=[4, -4, 0]

    m=len(A)
    MatCoeff = [A[i]+[b[i]]for i in range(m)] #becomes augmented matrix
    MatCoeff=np.array(MatCoeff)
    data=[]
    x=np.linspace(-25,25,101)
    MatCoeff=np.array(MatCoeff)
    for i in range(len(MatCoeff)):
        if MatCoeff[i,1] == 0:
            MatCoeff[i,1] += 1e-3
        trace=go.Scatter(x=x,  y=(MatCoeff[i,2]-MatCoeff[i,0]*x)/MatCoeff[i,1], name='c) Droite %d'%(i+1))
        data.append(trace)

    f=go.FigureWidget(data=data,
        layout=go.Layout(xaxis=dict(
            range=[-25, 25]
        ),
        yaxis=dict(
            range=[-100, 100]
        ) )

    )

    def mat(h):
        MatCoeff= [[h, -2, 4], [-2, h, -4], [1, -1, h]]
        return MatCoeff

    def update_y(h):
        MatCoeff= mat(h)
        MatCoeff=np.array(MatCoeff)
        if MatCoeff[0,1] == 0:
            MatCoeff[0,1] += 1e-3
        f.data[0].y=(MatCoeff[0,2]-MatCoeff[0,0]*x)/MatCoeff[0,1]
        if MatCoeff[1,1] == 0:
            MatCoeff[1,1] += 1e-3
        f.data[1].y=(MatCoeff[1,2]-MatCoeff[1,0]*x)/MatCoeff[1,1]
        if MatCoeff[2,1] == 0:
            MatCoeff[2,1] += 1e-3
        f.data[2].y=(MatCoeff[2,2]-MatCoeff[2,0]*x)/MatCoeff[2,1]

    freq_slider = interactive(update_y, h=(-20, 20, 0.5)) 

    vb = VBox((f, freq_slider))
    vb.layout.align_items = 'center'
    return vb

def corr4Chapitre1_2():
    style = {'description_width': 'initial'}
    m = widgets.IntText(
        value=1,
        step=1,
        description='Une infinité de solutions pour h=',
        disabled=False,
        style=style
    )
    n = widgets.IntText(
        value=1,
        step=1,
        description='Une unique solution pour h=',
        disabled=False,
        style=style
    )
    display(m)
    display(n)

    def f():
        if m.value == 2 and n.value==-4:
            display(Latex("C'est correct! Si $k\\notin\{-4,2\}$ le système n'admet pas de solution."))
        else:
            print('Incorrect, entrez une nouvelle valeur')
    interact_manual(f) 
    return