### In this python code the common qibo implementation of the transverse field ising hamiltonian can be found
# We try to solve it using the QOAO approximation

### Reminder: H = ∑ Z(i) · Z(i+1) + h·X(i)

import qibo
from qibo import gates, Circuit
from qibo.symbols import I, X, Y, Z
import numpy as np

# Hamiltonian for the TFIsing model
def Hamiltonian(N, g):
    ham = I(0)*0
    for q in range(N):
        ham += Z(q)*Z((q+1)%N) + g * X(q)
    return qibo.hamiltonians.SymbolicHamiltonian(-ham)

# Parmaterized Circuit: Equivariant ansatz

qibo.gates.RZZ.generator_eigenvalue = lambda self: 0.5
def circ_inv(N, p):
    params = np.random.random((2*N*p,))*4*np.pi
    circ_inv = Circuit(N)
    for q in range(N):
        circ_inv.add(gates.H(q))
    for itp in range(p):
        for itzz in range(N):
            circ_inv.add(gates.RZZ(itzz, (itzz+1)%N, params[2*N*itp + itzz]))
        for itx in range(N):
            circ_inv.add(gates.RX(itx, params[2*N*itp + N + itx]))
    return circ_inv, params

# Parametrized circuit: Non equivariant ansatz
def circ_noinv(N, p):
    params2 = np.random.random((3*N*p,))*4*np.pi
    circuit_noinv = Circuit(N)
    for q in range(N):
        circuit_noinv.add(gates.H(q))
    for itp in range(p):
        for itzz in range(N):
            circuit_noinv.add(gates.RZZ(itzz, (itzz+1)%N, params2[3*N*itp + itzz]))
        for itx in range(N):
            circuit_noinv.add(gates.RX(itx, params2[3*N*itp + N + itx]))
        for ity in range(N):
            circuit_noinv.add(gates.RY(itx, params2[3*N*itp + 2*N + ity]))
    return circuit_noinv, params2

# Optimization problem in terms of optimizing parameters of circuits to find minimum energy
def circs(N, p, g = 1, custom_operator=None):
    if custom_operator is None: 
        hamiltonian = Hamiltonian(N, g)
    else:
        hamiltonian = custom_operator
    
    circuit_inv, params = circ_inv(N,p)
    circuit_noinv, params2 = circ_noinv(N,p)

    def loss(params):
        circuit_inv.set_parameters(params)
        return hamiltonian.expectation(circuit_inv().state())

    def loss2(params):
        circuit_noinv.set_parameters(params)
        return hamiltonian.expectation(circuit_noinv().state())

    return circuit_inv, circuit_noinv, params, params2, loss, loss2


# Test the file works 
if __name__ == '__main__':
    N = 5; g = 1
    print(f"Building hamiltonian for N = {N}, g = {g}")
    
    ham = Hamiltonian(N, g)
    print("Finding exact minimum energy")
    print(" - Ground state found with diagonalisation: ", ham.eigenvalues(1)[0])

    p = 4
    print(f"Building ansatz circuits with p = {p} layers:")
    
    cinv, cninv, par1, par2, loss1, loss2 = circs(N, p, g)

    print("Optimizing equivariant ansatz:")
    minE, _, _ = qibo.optimizers.optimize(loss1, par1, method='BFGS')
    print(minE)

    print("Optimizing non equivariant ansatz:")
    minE, _, _ = qibo.optimizers.optimize(loss2, par2, method='BFGS')
    print(minE)

    

