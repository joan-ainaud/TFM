import numpy as np
import qibo
from qibo.hamiltonians import SymbolicHamiltonian
# # # # We don't have these in BSC environment
# from qibochem.measurement.result import expectation_from_samples 
# from qibochem.measurement.optimization import measurement_basis_rotations
# from qibochem.measurement.result import allocate_shots, pauli_term_measurement_expectation
from qibochem_local.optimization import measurement_basis_rotations
from qibochem_local.result import allocate_shots, pauli_term_measurement_expectation

from collections import Counter

# Modified method to obtain expectation values in a hardware efficient way:
#  - Measure simultaneously all commuting pauli strings that constitute an observable

def expectation_from_samples_qililab(
    circuit: qibo.models.Circuit,
    hamiltonian: SymbolicHamiltonian,
    n_shots: int = 1000,
    group_pauli_terms=None,
    n_shots_per_pauli_term: bool = True,
    shot_allocation=None,
    platform = None,
    transp_config = None,
) -> float:
    """
    Same as expectation_from_samples from qibochem.measurements.results, but circuit execution is performed with qililab.
    Used to optimize measurement by joining commuting terms qubit wise
    """
    # Group up Hamiltonian terms to reduce the measurement cost
    grouped_terms = measurement_basis_rotations(hamiltonian, grouping=group_pauli_terms)

    # Check shot_allocation argument if not using n_shots_per_pauli_term
    if not n_shots_per_pauli_term:
        if shot_allocation is None:
            shot_allocation = allocate_shots(grouped_terms, n_shots)
        assert len(shot_allocation) == len(
            grouped_terms
        ), f"shot_allocation list ({len(shot_allocation)}) doesn't match the number of grouped terms ({len(grouped_terms)})"

    total = 0.0
    for _i, (measurement_gates, terms) in enumerate(grouped_terms):
        if measurement_gates and terms:
            _circuit = circuit.copy()
            _circuit.add(measurement_gates)

            # Number of shots used to run the circuit depends on n_shots_per_pauli_term
            # ADAPTED TO RUN WITH QILILAB
            if platform is None:
                result = _circuit(nshots=n_shots if n_shots_per_pauli_term else shot_allocation[_i])
                frequencies = result.frequencies(binary=True)
            else:
                result = platform.execute(_circuit, num_avg=1, num_bins = n_shots if n_shots_per_pauli_term else shot_allocation[_i],
                                          transpilation_config=transp_config)
                frequencies = Counter(result.probabilities()) 

            qubit_map = sorted(qubit for gate in measurement_gates for qubit in gate.target_qubits)
            if frequencies:  # Needed because might have cases whereby no shots allocated to a group
                total += sum(pauli_term_measurement_expectation(term, frequencies, qubit_map) for term in terms)
    # Add the constant term if present. Note: Energies (in chemistry) are all real values
    total += hamiltonian.constant.real
    return total


def evaluate(circuit, hamiltonian, nshots, platform, transp_config):
    if nshots is None:
        if platform is None:
            return hamiltonian.expectation(circuit().state())
        else:
            # IMPORTANT, REMEMBER TO CATCH ERROR AND DISCONNECT PLATFORM OUTSIDE!!
            raise(ValueError("Can't perform no shot shot execution on hardware"))
    return expectation_from_samples_qililab(circuit, hamiltonian, nshots, group_pauli_terms="qwc",
                                            platform = platform, transp_config=transp_config)

def parameter_shift(
    circuit,
    hamiltonian,
    parameter_index,
    initial_state=None,
    scale_factor=1,
    nshots=None,
    nruns=1,
    repeated = False,
    platform = None,
    transp_config = None
):
    """We modify the Qibo implementation to allow multiple parameter_index. For the case where a parameter is reused in different gates"""
    gradient = 0
    try:
        iter(parameter_index) # check if it is iterable
    except:
        parameter_index = [parameter_index]
    if repeated:
        nparam = len(parameter_index)
        parameter_index = parameter_index[:1]
    for parameter_index in parameter_index:
        # inheriting hamiltonian's backend
        backend = hamiltonian.backend

        # getting the gate's type
        gate = circuit.associate_gates_with_parameters()[parameter_index]
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

                forward += evaluate(circuit, hamiltonian, nshots, platform=platform, transp_config=transp_config)

                # backward shift and evaluation
                shifted[parameter_index] -= 2 * s
                circuit.set_parameters(shifted)

                backward += evaluate(circuit, hamiltonian, nshots, platform=platform, transp_config=transp_config)

                # restoring the original circuit
                shifted = copied.copy()
                circuit.set_parameters(copied)

            forward /= nruns
            backward /= nruns
                
        circuit.set_parameters(original)
        
        gradient += generator_eigenval * (forward - backward) * scale_factor

    if repeated: gradient *= nparam
    return gradient
