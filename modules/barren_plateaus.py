#### IN THIS CODE WE STUDY THE BARREN PLATEAUS PHENOMENON FOR AN EQUIVARIANT EIGENSOLVER
#### REPLICATING THE PAPER - EXPLOTIING EQUIVARIATN CIRCUITS 
from tfising import *
from qibo.derivative import parameter_shift
from multiprocessing import Pool
import sys
import json

#FILENUMBER = sys.argv[1]
rep = 10; p = 80
Nlist = range(4,9,2)
nN = len(Nlist)

def compute_gradients(arg):
    qibo.set_backend("qibojit")

    g = 1

    grad_inv = np.empty((nN, rep))
    grad_noinv = np.empty((nN, rep))

    custzz = qibo.hamiltonians.SymbolicHamiltonian(Z(0)*Z(1))

    for itn, N in enumerate(Nlist):
        print(N)
        hamiltonian = Hamiltonian(N, g)

        circuit_inv, circuit_noinv, params, params2, loss, loss2 = circs(N, p, custom_operator = custzz)
        for itt in range(rep):
            
            params = np.random.random((2*N*p,))*np.pi*2
            params2 = np.random.random((3*N*p,))*np.pi*2
            circuit_inv.set_parameters(params)
            circuit_noinv.set_parameters(params2)
            grad_inv[itn,itt] = parameter_shift(circuit_inv, hamiltonian, 0)
            grad_noinv[itn,itt] = parameter_shift(circuit_noinv, hamiltonian, 0)

    return grad_inv, grad_noinv

if __name__ == '__main__':
    count = 10
    with Pool(count) as p:
        res = p.map(compute_gradients, range(count))
    
    grad_inv = np.empty((nN,rep*count))
    grad_noinv = np.empty((nN,rep*count))

    for process in range(count):
        grad_inv[:,rep*process:rep*(process+1)] = res[process][0]
        grad_noinv[:,rep*process:rep*(process+1)] = res[process][1]

    with np.printoptions(threshold=np.inf):
        print(grad_inv)
        print(grad_noinv)