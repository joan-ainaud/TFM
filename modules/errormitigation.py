# Executes routines to mitigate noise
# Also redefines expectation calculation to be able to implement this mitigation
from qibo import Circuit, gates
from qibo.config import raise_error
import numpy as np
import qibo

from functools import reduce

from qibo.symbols import Z



###############################
### READOUT ERROR MITGATION ###
###############################

readout_error_mitigation_coefficients = 10*[[0.5, 0.5]] # no noise, 10 qubits default

def set_readout_error_mitigation(platform=None, nqubits=None, readout_bitflip=None):
    """Estimates error parameters of each qubit (assumed independent)
    following Supplementary Information V.B: Kandala, A., Mezzacapo, A., Temme, K. et al. Hardware-efficient variational quantum eigensolver for small molecules and quantum magnets. Nature 549, 242–246 (2017). https://doi.org/10.1038/nature23879"""
    global readout_error_mitigation_coefficients
    circ0 = Circuit(nqubits)
    circ0.add(gates.M(q) for q in range(nqubits))
    circ1 = Circuit(nqubits)
    circ1.add(gates.X(q) for q in range(nqubits))
    circ1.add(gates.M(q) for q in range(nqubits))
    nshots = 2000

    if not(platform is None):
        freq0 = platform.execute(circ0, num_avg=1, num_bins=nshots).counts()
        freq1 = platform.execute(circ1, num_avg=1, num_bins=nshots).counts()
    elif not(readout_bitflip is None):
        freq0 = circ0(nshots=nshots)
        freq0 = get_noisy_frequencies(freq0, readout_bitflip)
        freq1 = circ1(nshots=nshots)
        freq1 = get_noisy_frequencies(freq1, readout_bitflip)
    else: raise(ValueError("Did not pass platform to run code nor noisy model"))

    p1given0 = np.array([0]*nqubits, dtype=float)
    for bitstring, count in freq0.items():
        for q in range(nqubits):
            if bitstring[q] == '1': p1given0[q] += count
    p1given0 /= nshots

    p0given1 = np.array([0]*nqubits, dtype=float)
    for bitstring, count in freq1.items():
        for q in range(nqubits):
            if bitstring[q] == '0': p0given1[q] += count
    p0given1 /= nshots

    # v0 = 1 - n0 - n1,  v1 = n0 - n1  Actually wrong in reference
    # 1-n0-n1 is probability of obtaining 0 with state being 1
    # n0-n1   is probability of obtaining 1 with state being 0
    readout_error_mitigation_coefficients = [[(1-v1+v0)/2, (1-v1-v0)/2] for v0, v1 in zip(p1given0, p0given1)]

    

# To apply
def expectation_from_samples_with_rem(observable, freq: dict, qubit_map: list = None, rem = True) -> float:
    """
    Calculate the expectation value from the samples applying, if desired, readout error mitigation.
    """
    if not rem: return observable.expectation_from_samples(freq, qubit_map)

    for term in observable.terms:
        # pylint: disable=E1101
        for factor in term.factors:
            if not isinstance(factor, Z):
                raise_error(
                    NotImplementedError, "Observable is not a Z Pauli string."
                )

    if qubit_map is None:
        qubit_map = list(range(observable.nqubits))

    keys = list(freq.keys())
    counts = observable.backend.cast(list(freq.values()), observable.backend.precision) / sum(
        freq.values()
    )
    expvals = []
    for term in observable.terms:
        qubits = {
            factor.target_qubit for factor in term.factors if factor.name[0] != "I"
        }
        expvals.extend(
            [
                (term.coefficient.real * \
                reduce(lambda x,y:x*y, 
                       [( (-1 if state[qubit_map.index(q)]=='1' else 1) - (1-2*readout_error_mitigation_coefficients[qubit_map.index(q)][0])) / (2 * readout_error_mitigation_coefficients[qubit_map.index(q)][1]) for q in qubits]) )
                for state in keys
            ]
        )
    expvals = observable.backend.cast(expvals, dtype=counts.dtype).reshape(
        len(observable.terms), len(freq)
    )
    return observable.backend.np.sum(expvals @ counts.T) + observable.constant.real

def get_noisy_frequencies(measurement_result, readout_bitflip):
    result_bitflips = measurement_result.apply_bitflips(p0={int(key): readout_bitflip[key][0] for key in readout_bitflip}, p1={int(key): readout_bitflip[key][1] for key in readout_bitflip})
    N = len(readout_bitflip)

    backend = qibo.get_backend()

    count_samples_int = result_bitflips.dot(2**np.arange(N)[::-1])
    res_freq = backend.calculate_frequencies(count_samples_int)
    freq_bitflips = qibo.measurements.frequencies_to_binary(res_freq,N)
    return freq_bitflips
    


############################################
### OPERATOR DECOHERENCE RENORMALIZATION ###
############################################


if __name__ == "__main__":
    import qibo
    from qibo.symbols import X, Y, Z, I
    from qibo.hamiltonians import SymbolicHamiltonian

    N = 4
    Nshots = 10000
    circ0 = Circuit(N)
    circ1 = Circuit(N)
    circ = Circuit(N)

    np.random.seed(2000)

    measurement_error_map = [{q: np.random.random()*0.2 for q in range(N)}, {q: np.random.random()*0.2 for q in range(N)}] # p0 and p1

    #print("Error probabilties: ")
    #print(measurement_error_map)

    # Simpler circuit, 0 state and 1 state
    circ1.add(gates.X(q) for q in range(N))

    # More complex circuit:
    circ.add(gates.H(q) for q in range(N))
    circ.add(gates.RZ(q, 2*np.pi*np.random.random()) for q in range(N))
    circ.add(gates.RXX(q0, q1, 2*np.pi*np.random.random()) for q0 in range(N-1) for q1 in range(q0+1, N))

    circ.add(gates.M(q) for q in range(N))
    circ0.add(gates.M(q) for q in range(N))
    circ1.add(gates.M(q) for q in range(N))

    exact_random = circ(nshots=Nshots)
    exact_0 = circ0(nshots=Nshots)
    exact_1 = circ1(nshots=Nshots)
    
    result_random = get_noisy_frequencies(exact_random, measurement_error_map)
    result_0 = get_noisy_frequencies(exact_0, measurement_error_map)
    result_1 = get_noisy_frequencies(exact_1, measurement_error_map)



    observable = SymbolicHamiltonian(sum((Z(q) for q in range(N)), start=I(0)*0))
    backend = qibo.get_backend()

    print("Measuring O = Z1 + Z2 + Z3 + Z4")
    print("--------------------------------")
    print("Exact random <O> =       ", observable.expectation_from_samples(exact_random.frequencies()) )
    print("Exact all 0 <O> =        ", observable.expectation_from_samples(exact_0.frequencies()) )
    print("Exact all 1 <O> =        ", observable.expectation_from_samples(exact_1.frequencies()) )

    print("Error random <O> =       ", observable.expectation_from_samples(result_random) )
    print("Error all 0 <O> =        ", observable.expectation_from_samples(result_0) )
    print("Error all 1 <O> =        ", observable.expectation_from_samples(result_1) )


    set_readout_error_mitigation(nqubits=N, readout_bitflip=measurement_error_map)
    print("Corrected random <O> =   ", expectation_from_samples_with_rem(observable, result_random))
    print("Corrected all 0 <O> =    ", expectation_from_samples_with_rem(observable, result_0))
    print("Corrected all 1 <O> =    ", expectation_from_samples_with_rem(observable, result_1))

    print(readout_error_mitigation_coefficients)
    
