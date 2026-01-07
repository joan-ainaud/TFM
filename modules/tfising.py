# -*- coding: utf-8 -*-
### In this python code a qibo implementation of the transverse field ising hamiltonian is given
### H = ∑ Z(i) · Z(i+1) + h·X(i)

import qibo
import qibo.noise
from qibo import gates, Circuit
from qibo.symbols import I, X, Y, Z
import numpy as np

from execution import evaluate, parameter_shift

ROUTING = True
IGNORE_QUBIT_3 = True
NOISY = True
bsc_parameters = {
    "t1": {"0": 24.44*1e-06, "1": 32.10*1e-06, "2": 25.60*1e-06, "3": 23.89*1e-06, "4": 15.48*1e-06},
    "t2": {"0": 1242*1e-09, "1": 3605*1e-09, "2": 1852*1e-09, "3": 3220*1e-09, "4": 6540*1e-09},
    "gate_times" : (40*1e-9, 50*1e-9),
    "excited_population" : 0,
    "depolarizing_one_qubit" : {"0": 44e-4, "1": 15e-4, "2": 25e-4, "3": 34e-4, "4": 17e-4},
    "depolarizing_two_qubit": {"0-1": 304e-4, "0-4": 499e-4, "2-1": 274e-4, "2-3": 168e-4, "2-4": 671e-4,
                               "1-0": 304e-4, "4-0": 499e-4, "1-2": 274e-4, "3-2": 168e-4, "4-2": 671e-4},
    "readout_one_qubit" : {"0": (0.165, 0.165), "1": (0.195, 0.195), "2": (0.154, 0.154), "3": (0.0161, 0.161), "4": (0.128, 0.128)},  # ( p(0|1), p(1|0) )
}
noise_model_bsc = qibo.noise.IBMQNoiseModel()
noise_model_bsc.from_dict(bsc_parameters)

# Hamiltonian for the TFIsing model
def hamiltonianTFI(N, g=1, J = 1):
    ham = I(0)*0
    for q in range(N):
        if not(IGNORE_QUBIT_3) or q != 3: ham += J*Z(q)*Z((q+1+1*(q == 2 and IGNORE_QUBIT_3))%N) + g * X(q)
    return qibo.hamiltonians.SymbolicHamiltonian(-ham)

# We have to substitute RZZ gates with  CNOT I·RZ CNOT, and CNOT maybe for I·H CZ I·H, but this will be optimzied anyways
# implcitly we only consider one to one connections
def RZZ_decomp(q0, q1, theta): 
    if ROUTING and (q0 in (3,4)) and (q1 in (3,4)): 
        return [gates.SWAP(2,4), gates.CNOT(2, 3), gates.RZ(3, theta), gates.CNOT(2,3), gates.SWAP(2,4)]
    else: return [gates.CNOT(q0, q1), gates.RZ(q1, theta), gates.CNOT(q0,q1)]
gates.RZZ.generator_eigenvalue = lambda self: 0.5


gate_dependence_inv = lambda N, p: {i: list(range(i*N, (i+1)*N)) for i in range(0,2*p)}
gate_dependence_noinv = lambda N, p: {i: list(range(i*N, (i+1)*N)) for i in range(0,3*p)}

# QAOA like structure
def circ_inv(N, p):
    params = np.random.random((2*p,))*4*np.pi
    circuit_inv = Circuit(N, density_matrix=True if NOISY else False)
    for q in range(N):
        circuit_inv.add(gates.H(q))  # WE START WITH |+> state
    for itp in range(p):
        for itzz in range(0,N,2):
            circuit_inv.add(RZZ_decomp(itzz, (itzz+1)%N, params[2*itp]))
        for itzz in range(1,N,2):
            circuit_inv.add(RZZ_decomp(itzz, (itzz+1)%N, params[2*itp]))
        for itx in range(N):
            circuit_inv.add(gates.RX(itx, params[2*itp + 1]))
    return circuit_inv, params

# Similar to equivariant structure, but in this case with extra nonequivariant Y rotations
def circ_noinv(N, p):
    params2 = np.random.random((3*p,))*4*np.pi
    circuit_noinv = Circuit(N, density_matrix=True if NOISY else False)
    for q in range(N):
        circuit_noinv.add(gates.H(q))
    for itp in range(p):
        for itzz in range(0,N,2):
            circuit_noinv.add(RZZ_decomp(itzz, (itzz+1)%N, params2[3*itp]))
        for itzz in range(1,N,2):
            circuit_noinv.add(RZZ_decomp(itzz, (itzz+1)%N, params2[3*itp]))
        for itx in range(N):
            circuit_noinv.add(gates.RX(itx, params2[3*itp + 1]))
        for ity in range(N):
            circuit_noinv.add(gates.RY(ity, params2[3*itp + 2]))
    return circuit_noinv, params2

# return abstract functions to compute loss (energy) and gradients
def circs_shots_noisy_jac(N, p, g = 1, custom_operator=None, nshots=100, noise_map=None, epsilon = 0.02, noise=True, platform = None, transp_config = None):
    if custom_operator is None: 
        hamiltonian = hamiltonianTFI(N, g)
    else:
        hamiltonian = custom_operator


    if IGNORE_QUBIT_3: 
        N = N-1
        global ROUTING
        ROUTING = False
    if noise_map is None:
        noise_map = {i: list(zip(["X", "Z"], [epsilon, epsilon])) for i in range(N)}

    
    print("Hamiltonian: ", hamiltonian._form)
    print("Hamiltonian ground state: ", hamiltonian.eigenvalues(1)[0])

    circuit_inv, params = circ_inv(N,p)
    circuit_noinv, params2 = circ_noinv(N,p)

    if noise:
        circuit_inv = circuit_inv.with_pauli_noise(noise_map)
        circuit_noinv = circuit_noinv.with_pauli_noise(noise_map)
    
    if IGNORE_QUBIT_3:
        c5_inv = Circuit(5, density_matrix=True if NOISY else False)
        c5_inv.add(circuit_inv.on_qubits(*[0,1,2,4]))
        c5_noinv = Circuit(5, density_matrix=True if NOISY else False)
        c5_noinv.add(circuit_noinv.on_qubits(*[0,1,2,4]))

        circuit_inv = c5_inv
        circuit_noinv = c5_noinv

    if NOISY:
        circuit_inv = noise_model_bsc.apply(circuit_inv)
        circuit_noinv = noise_model_bsc.apply(circuit_noinv)

    def loss(params):
        circuit_inv.set_parameters(np.repeat(params,N))
        return evaluate(circuit_inv, hamiltonian, nshots, platform=platform, transp_config=transp_config)

    def loss2(params):
        circuit_noinv.set_parameters(np.repeat(params,N))
        return evaluate(circuit_noinv, hamiltonian, nshots, platform=platform, transp_config=transp_config)
    
    gate_dependence_inv = {i: list(range(i*N, (i+1)*N)) for i in range(0,2*p)}
    gate_dependence_noinv = {i: list(range(i*N, (i+1)*N)) for i in range(0,3*p)}
    def jac_psr_inv(x, *args):
        circuit_inv.set_parameters(np.repeat(x,N))
        return np.array([parameter_shift(circuit_inv, hamiltonian, gate_dependence_inv[param], nshots=nshots, nruns=1, repeated=True, platform=platform, transp_config=transp_config) for param in range(len(params))])
    
    def jac_psr_noinv(x, *args):
        circuit_noinv.set_parameters(np.repeat(x,N))
        return np.array([parameter_shift(circuit_noinv, hamiltonian, gate_dependence_noinv[param], nshots=nshots, nruns=1, repeated=True, platform=platform, transp_config=transp_config) for param in range(len(params2))])
    
    return circuit_inv, circuit_noinv, params, params2, loss, loss2, jac_psr_inv, jac_psr_noinv


def tfisingMeasurement():
    # Define the TFIsing measurement:
    # Group Xi terms, and ZZ terms, to measure faster.
    # Provide expectations of errors of measurements? Interesting, to give error estimates based on sampling error
    pass

# Test the functions work, optimizing them
if __name__ == '__main__':
    N = 4; g = 1
    print( "Building hamiltonian for N = {0}, g = {1}".format(N,g) )
    
    ham = hamiltonianTFI(N, g)
    print("Finding exact minimum energy")
    print(" - Ground state found with diagonalisation: ", ham.eigenvalues(1)[0])

    p = 2
    print(f"Building ansatz circuits with p = {p} layers:")
    
    cinv, cninv, par1, par2, loss1, loss2, jac1, jac2 = circs_shots_noisy_jac(N, p, g, noise=False)

    print("Optimizing equivariant ansatz:")
    cinv.draw()
    print("depth", cinv.depth)
    minE, _, _ = qibo.optimizers.optimize(loss1, par1, method='BFGS')
    print(minE)


    print("Optimizing non equivariant ansatz:")
    cninv.draw()
    print("depth", cninv.depth)
    minE, _, _ = qibo.optimizers.optimize(loss2, par2, method='BFGS')
    print(minE)

    

