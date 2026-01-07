#### IN THIS CODE WE DEFINE A CLASS TO IMPLEMENT PARAMETRIZED QUANTUM CIRCUITS IN A STANDARIZED WAY
import qibo
import qibo.noise
from qibo import gates, Circuit, parameter
from qibo.symbols import I, X, Y, Z
import numpy as np
from qibo.config import raise_error
from optimizers import optimizer_gds
from functools import partial
import numpy as np

from qibochem.measurement.result import expectation_from_samples

import matplotlib.pyplot as plt

def evaluate(circuit, hamiltonian, nshots: int):
    # Computes circuit
    if nshots is None:
        return hamiltonian.expectation(circuit().state())
    return expectation_from_samples(circuit, hamiltonian, nshots, group_pauli_terms="qwc")

class quantumModel():
    """
    Class that defines a quantum model for solving a particular problem:
    - Number of qubits is variable, depending on model, instead the definition is the construction of the circuit
    - The definition should be without noise, noise is added later
    - Model has parameters and hyperparameters.
    - The relation of each actual parameter with each parametrized gate also needs to be passed
    """
    def __init__(self, circ_def, observable_def, parameters_def, hyperparameters=None) -> None:
        self.hyperparameters = hyperparameters
        self.circ_def = circ_def
        self.observable_def = observable_def
        self.parameters_def = parameters_def

    def instance(self, N, *args, **kwargs):
        return quantumModelInstance(self.circ_def(N, *args, **kwargs),
                                    self.observable_def(N, *args, **kwargs),
                                    *self.parameters_def)
    
    # Noise options
    # - With Pauli Noise
    # - Custom NoiseModel
    # 
    # - NOISE MITIGATION: Clifford etc CDR, Zero noise extrapolation ZNE


class quantumModelInstance():
    """
    Class that defines an instance of a particular quantum model:
    That is, with a particular nqubits, parametrized circuit, gate parameter relations etc
    :: parameter_to_gate :: indicates for each actual parameter the gates that repeat it

    The parameter initalization itself should be the gates which compose the circuit, sent in layers.

    [Layer1, Layer2, Layer3...] Where each layer is one of the sets of layers. Layers then should respect some symmetry, which
    will have some representation, and this representation should commute with the different layers in an equivariant circuit.
    """
    def __init__(self, circuit, observable, parameters, parameter_to_gate, func_loss = None, dfunc_loss = None) -> None:
        self.circuit = circuit
        self.observable = observable
        self.parameters = parameters
        self.parameter_to_gate = parameter_to_gate
        self.optimized_parameters = None

        
        if (func_loss is not None) and (dfunc_loss is not None):
            self.func_loss = func_loss
            self.dfunc_loss = dfunc_loss
            self.post_processing = True 
        else:
            self.func_loss = lambda x, output: x
            self.dfunc_loss = lambda x, output: 1
            self.post_processing = False

        

    def __call__(self, input = None, nshots = None) -> float:
        if nshots is None: 
            return self.observable.expectation(self.circuit().state())
        else:
            return expectation_from_samples(self.circuit, self.observable, nshots, group_pauli_terms="qwc")
    
    def set_parameters(self, new_parameters):
        if new_parameters.shape != self.parameters.shape:
            raise_error(ValueError, f"{len(new_parameters)} parameters were given, but \
                        the model has {len(self.parameters)}")
        
        self.parameters[:] = new_parameters[:]
        for param_index, param in enumerate(self.parameters):
            for dependent_gate in self.parameter_to_gate[param_index]:
                self.circuit.trainable_gates[dependent_gate].parameters = param

    execute_circuit = __call__ 

    def run_model(self, parameters, input = None, output = None, nshots = None, repeated = True) -> float:
        """Repeated: Indicates if parameter reusing gates are translationally inavariant among themselves"""
        self.set_parameters(parameters)
        if nshots is None: nshots = self.nshots
        # returns function and gradient
        loss = self.func_loss(self(input, nshots), output)
        grads = np.array([parameter_shift(self.circuit, self.observable, self.parameter_to_gate[param], nshots=nshots, repeated=repeated) for param in range(len(self.parameters))])
        grads *= self.dfunc_loss(loss, output)
        return self.func_loss(self(input, nshots)), grads
    
    def loss(self, parameters, input = None, output = None, nshots = None) -> float:
        """Repeated: Indicates if parameter reusing gates are translationally inavariant among themselves"""
        if nshots is None: nshots = self.nshots
        self.set_parameters(parameters)
        return self.func_loss(self(input, nshots), output)
    
    def loss_gradient(self, parameters, input = None, output = None, nshots = None, repeated = True) -> float:
        """Repeated: Indicates if parameter reusing gates are translationally inavariant among themselves"""
        if nshots is None: nshots = self.nshots
        self.set_parameters(parameters)
        grads = np.array([parameter_shift(self.circuit, self.observable, self.parameter_to_gate[param], nshots=nshots, repeated=repeated) for param in range(len(self.parameters))])
        if self.post_processing: grads *= self.dfunc_loss(self.loss(parameters, input, output, nshots), output) 
        return grads



    def optimize(self, optimizer, training_data = None, nshots=None):
        """
        Optimizer should take as input initial parameters and func that returns: 
            - f(params), gradients(params)
        And optimizer should return
            loss_list, 
        """
        self.nshots = nshots
        return optimizer(self.parameters, self.loss, self.loss_gradient)


    @property
    def max_variance(self):
        return np.sqrt(sum((term.coefficient**2) for term in self.observable.terms))
    
    def __repr__(self) -> str:
        out = f"""Params: {str(self.parameters)}\nObservable: {str(self.observable.form)}\n""" + str(self.circuit)
        return out
    
    


# CUSTOM PARAMETER SHIFT RULE: RECYCLING CODE FROM QIBO IMPLEMENTATION BUT INCLUDING MULTI PARAMETERS AND EQUIVARIANT OPTIMIZATION
def parameter_shift(
    circuit,
    hamiltonian,
    parameter_index,
    initial_state=None,
    scale_factor=1,
    nshots=None,
    nruns=1,
    repeated = False,
    noise_protected = False
):
    """Modified parameter shift rule (PSR) implementation, based on the Qibo.derivatives one. 
    
    Given a circuit U and an observable H, the PSR allows to calculate the derivative
    of the expected value of H on the final state with respect to a variational
    parameter of the circuit.
    There is also the possibility of setting a scale factor. It is useful when a
    circuit's parameter is obtained by combination of a variational
    parameter and an external object, such as a training variable in a Quantum
    Machine Learning problem. For example, performing a re-uploading strategy
    to embed some data into a circuit, we apply to the quantum state rotations
    whose angles are in the form: theta' = theta * x, where theta is a variational
    parameter and x an input variable. The PSR allows to calculate the derivative
    with respect of theta' but, if we want to optimize a system with respect its
    variational parameters we need to "free" this procedure from the x depencency.
    If the `scale_factor` is not provided, it is set equal to one and doesn't
    affect the calculation.
    Args:
        circuit (:class:`qibo.models.circuit.Circuit`): custom quantum circuit.
        hamiltonian (:class:`qibo.hamiltonians.Hamiltonian`): target observable.
        parameter_index (int): the index which identifies the target parameter in the circuit.get_parameters() list
        initial_state ((2**nqubits) vector): initial state on which the circuit acts (default None).
        scale_factor (float): parameter scale factor (default None).
        repeated (bool): indicates if the gates that use the parameter have a translational symmetry
    Returns:
        np.float value of the derivative of the expectation value of the hamiltonian
        with respect to the target variational parameter.
    """
    """We modify the Qibo implementation to allow multiple parameter_index. For the case where a parameter is reused in different gates"""
    gradient = 0
    try:
        iter(parameter_index) # check if it is iterable
    except:
        parameter_index = [parameter_index]
    if repeated:
        nparam = len(parameter_index)
        if noise_protected:
            nshots /= nparam
        else:
            parameter_index = parameter_index[:1]
    for parameter_index in parameter_index:
        # inheriting hamiltonian's backend
        backend = hamiltonian.backend

        # getting the gate's type
        gate = circuit.associate_gates_with_parameters()[parameter_index]

        # getting the generator_eigenvalue
        ###try: generator_eigenval = gate.generator_eigenvalue()
        ###except: 
        #generator_eigenval = 0.5   #### CAN'T DO SOMETHING LIKE THIS... ERROR GETS LOGGED ANYWAYS BY QIBO FUNCTION RAISE_ERROR, EVEN IF CAUGHT BY TRY
        # getting the generator_eigenvalue
        generator_eigenval = gate.generator_eigenvalue()

        # defining the shift according to the psr
        s = np.pi / (4 * generator_eigenval)

        # saving original parameters and making a copy
        original = np.asarray(circuit.get_parameters()).copy()
        shifted = original.copy()

        # forward shift and evaluation
        shifted[parameter_index] += s
        circuit.set_parameters(shifted)

        forward = 0
        backward = 0


        if nshots == None:
            forward = hamiltonian.expectation(
                backend.execute_circuit(circuit=circuit, initial_state=initial_state).state()
            )

            # backward shift and evaluation
            shifted[parameter_index] -= 2 * s
            circuit.set_parameters(shifted)

            backward = hamiltonian.expectation(
                backend.execute_circuit(circuit=circuit, initial_state=initial_state).state()
            )

        else:
            
            copied = shifted.copy()

            for _ in range(nruns):

                forward += evaluate(circuit, hamiltonian, nshots) #np.float64(hamiltonian.expectation_from_circuit(circuit, nshots=nshots).real)
                """backend.execute_circuit(
                    circuit=circuit, 
                    initial_state=initial_state, 
                    nshots=nshots
                    ).expectation_from_samples(hamiltonian)"""

                # backward shift and evaluation
                shifted[parameter_index] -= 2 * s
                circuit.set_parameters(shifted)

                backward += evaluate(circuit, hamiltonian, nshots)
                """backend.execute_circuit(
                    circuit=circuit, 
                    initial_state=initial_state, 
                    nshots=nshots
                    ).expectation_from_samples(hamiltonian)  """

                # restoring the original circuit
                shifted = copied.copy()
                circuit.set_parameters(copied)

            forward /= nruns
            backward /= nruns
                
        circuit.set_parameters(original)
        
        gradient += generator_eigenval * (forward - backward) * scale_factor

    if repeated and not noise_protected: gradient *= nparam
    return gradient

import math

iterations_count = round(1e6)

def complex_operation(input_index):
    print("Complex operation. Input index: {:2d}\n".format(input_index))
    [math.exp(i) * math.sinh(i) for i in [1] * iterations_count]
    return input_index


"""def run_complex_operations(operation, input):
   for i in input:
      operation(i) 
      
input = range(10)
run_complex_operations(complex_operation, input) """


processes_count = 10


if __name__ == '__main__':
    import time
    """with Pool(processes_count) as p:
       res = p.map(complex_operation, range(processes_count))"""
    from tfising import hamiltonianTFI, gate_dependence_inv, circ_inv

    #nqubits = 6
    #parameters = np.random.random((2,))
    #circ = Circuit(nqubits)
    #circ.add(gates.RX(q, 0) for q in range(nqubits))
    #circ.add(gates.CNOT(q, (q+1)%nqubits) for q in range(nqubits))
    #circ.add(gates.RZ(q, np.random.random()) for q in range(nqubits))
    #circ.draw()
    
    #observable = qibo.hamiltonians.SymbolicHamiltonian(sum(2*Z(q) for q in range(nqubits)))

    #param_to_gates = {0: range(nqubits), 1: range(nqubits, 2*nqubits)}

    N = 5
    p = 2

    for __ in range(1):
        np.random.seed(2005)
        circ, parameters = circ_inv(N,p, density_matrix=True)

        parameters = parameters[::N]

        copy_param = parameters.copy()

        observable = hamiltonianTFI(N, 1.)
        param_to_gates = gate_dependence_inv(N,p)


        model = quantumModelInstance(circ, observable, parameters, param_to_gates)

        bsc_parameters = {
    "t1": {"0": 24.44*1e-06, "1": 32.10*1e-06, "2": 25.60*1e-06, "3": 23.89*1e-06, "4": 15.48*1e-06},
    "t2": {"0": 1242*1e-09, "1": 3605*1e-09, "2": 1852*1e-09, "3": 3220*1e-09, "4": 6540*1e-09},
    "gate_times" : (40*1e-9, 3*40*1e-9),
    "excited_population" : 0,
    "depolarizing_one_qubit" : {"0": 44e-4, "1": 15e-4, "2": 25e-4, "3": 34e-4, "4": 17e-4},
    "depolarizing_two_qubit": {"0-1": 304e-4, "0-4": 499e-4, "2-1": 274e-4, "2-3": 168e-4, "2-4": 671e-4, "1-0": 304e-4, "4-0": 499e-4, "1-2": 274e-4, "3-2": 168e-4, "4-2": 671e-4},
    "readout_one_qubit" : {"0": (0.165, 0.165), "1": (0.195, 0.195), "2": (0.154, 0.154), "3": (0.0161, 0.161), "4": (0.128, 0.128)},  # ( p(0|1), p(1|0) )
    }
        noise_model = qibo.noise.IBMQNoiseModel()
        noise_model.from_dict(bsc_parameters)
        noisy_circ = noise_model.apply(circ)

        noisy_model = quantumModelInstance(noisy_circ, observable, copy_param, param_to_gates)

        #print(noisy_circ(nshots=100).frequencies())
        #print(model(10))
        #print(model.max_variance)
        print("\tNoiseless model: ")
        print(model)
        print("\tNoisy model")
        print(noisy_model)

        print("Init parameters:", copy_param)

        t0 = time.time()
        res = model.optimize(partial(optimizer_gds, Nepochs=100, epochs_print=10), nshots=None)
        print(f"Optimization took {time.time()-t0:.5f} s")

        plt.plot(res[1])
        plt.show(block=True)
        print(model.parameters)


        model.set_parameters(copy_param.copy())

        t0 = time.time()
        res = model.optimize(partial(optimizer_gds, Nepochs=100, epochs_print=10), nshots=40)
        print(f"Optimization took {time.time()-t0:.5f} s")

        plt.plot(res[1])
        plt.show(block=True)
        print(model.parameters)


        """t0 = time.time()
        res = noisy_model.optimize(partial(optimizer_gds, Nepochs=50, epochs_print=10), nshots=1000)
        print(f"Optimization took {time.time()-t0:.5f} s")

        plt.plot(res[1])
        plt.show(block=True)"""
        