''' the basis for this code was taken here: https://q4quanta.github.io/qcdocs/ '''


import numpy as np
import ipywidgets as widgets
import colorsys
import matplotlib.pyplot as plt
from qiskit import execute, Aer, BasicAer
from qiskit.visualization import plot_bloch_multivector
from qiskit.tools.jupyter import *
from qiskit.visualization import *
import seaborn as sns
sns.set()


#TODO make a function that automatically digests a circuit in the sequence and translates to plotPhaseCircle

'''========State Vector======='''

def getStateVector(qc):
    '''get state vector in row matrix form'''
    backend = BasicAer.get_backend('statevector_simulator')
    job = execute(qc,backend).result()
    vec = job.get_statevector(qc)
    return vec


def vec_in_braket(vec: np.ndarray) -> str:
    '''get bra-ket notation of vector'''
    nqubits = int(np.log2(len(vec)))
    state = ''
    for i in range(len(vec)):
        rounded = round(vec[i], 3)
        if rounded != 0:
            basis = format(i, 'b').zfill(nqubits)
            state += str(rounded).replace('-0j', '+0j')
            state += '|' + basis + '\\rangle + '
    state = state.replace("j", "i")
    return state[0:-2].strip()

def vec_in_text_braket(vec):
    return '$$\\text{{State:\n $|\\Psi\\rangle = $}}{}$$'\
                              .format(vec_in_braket(vec))

def writeStateVector(vec):
    return widgets.HTMLMath(vec_in_text_braket(vec))



'''==========Bloch Sphere ========='''

def getBlochSphere(qc):
    '''plot multi qubit bloch sphere'''
    vec = getStateVector(qc)
    return plot_bloch_multivector(vec)


'''=========Matrix================='''

def getMatrix(qc):
    '''get numpy matrix representing a circuit'''
    backend = BasicAer.get_backend('unitary_simulator')
    job = execute(qc, backend)
    ndArray = job.result().get_unitary(qc, decimals=3)
    Matrix = np.matrix(ndArray)
    return Matrix
    
    
def plotMatrix(M):
    '''visualize a matrix using seaborn heatmap'''
    MD = [["0" for i in range(M.shape[0])] for j in range(M.shape[1])]
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            r = M[i,j].real
            im = M[i,j].imag
            MD[i][j] =  str(r)[0:4]+ " , " +str(im)[0:4]
    plt.figure(figsize = [2*M.shape[1],M.shape[0]])
    sns.heatmap(np.abs(M),\
                annot = np.array(MD),\
                fmt = '',linewidths=.5,\
                cmap='Blues')
    return


'''========Phase============'''


def getPhaseCircle(vec):
    '''get phase color, angle and radious of phase circir'''
    Phase = []
    for i in range(len(vec)):
        angles = (np.angle(vec[i]) + (np.pi * 4)) % (np.pi * 2)
        rgb = colorsys.hls_to_rgb(angles / (np.pi * 2), 0.5, 0.5)
        mag = np.abs(vec[i])
        Phase.append({"rgb":rgb,"mag": mag,"ang":angles})
    return Phase
    
    
def getPhaseDict(QCs): 
    '''get a dictionary of state vector phase circles for 
       each quantum circuit and populate phaseDict list'''
    phaseDict = []
    for qc in QCs:
        vec = getStateVector(qc)
        Phase = getPhaseCircle(vec)
        phaseDict.append(Phase)  
    return phaseDict


def plotPhaseCircle(phaseDict,depth,path=None,show=False,save=False):
    #TODO add check for phaseDict to be a list otherwise convert ro dict\list 
    #TODO add auto setting depth 
    #TODO handle path better
    '''plot any quantum circuit phase circle diagram
       from provided phase Dictionary'''
    r = 0.30
    dx = 1.0
    nqubit = len(phaseDict[0])
    fig = plt.figure(figsize = [depth,nqubit])
    for i in range(depth):
        x0 = i
        for j in range(nqubit):
            y0 = j+1
            try:
                mag = phaseDict[i][j]['mag']
                ang = phaseDict[i][j]['ang']
                rgb = phaseDict[i][j]['rgb']
                ax=plt.gca()
                circle1= plt.Circle((dx+x0,y0), radius = r, color = 'white')
                ax.add_patch(circle1)
                circle2= plt.Circle((dx+x0,y0), radius= r*mag, color = rgb)
                ax.add_patch(circle2)
                line = plt.plot((dx+x0,dx+x0+(r*mag*np.cos(ang))),\
                            (y0,y0+(r*mag*np.sin(ang))),color = "black")
                
            except:
                ax=plt.gca()
                circle1= plt.Circle((dx+x0,y0), radius = r, color = 'white')
                ax.add_patch(circle1)
    plt.ylim(nqubit+1,0)
    plt.yticks([y+1 for y in range(nqubit)])
    plt.xticks([x for x in range(depth+2)])
    plt.xlabel("Circuit Depth")
    plt.ylabel("Basis States")
    if show:
        plt.show()
        plt.savefig(path+".png")
        plt.close(fig)
    if save:
        plt.savefig(path +".png")
        plt.close(fig)
    return 
