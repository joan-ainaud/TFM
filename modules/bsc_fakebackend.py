# WE DEFINE FAKE BACKENDS TO USE WITH QISKIT TO PERFORM ROUTING / OPTIMIZATION

import qiskit
from qiskit import QuantumCircuit
from qiskit.providers.fake_provider import GenericBackendV2


import numpy as np
import rustworkx as rx
 
import matplotlib.pyplot as plt
from qiskit.providers import BackendV2, Options
from qiskit.transpiler import Target, InstructionProperties
from qiskit.circuit.library import XGate, SXGate, RZGate, CZGate, RXGate, UGate, U3Gate
from qiskit.circuit import Measure, Delay, Parameter, Reset, Gate
from qiskit import QuantumCircuit, transpile
from qiskit.visualization import plot_gate_map
import rustworkx
from qiskit.transpiler import generate_preset_pass_manager
from qiskit.circuit.equivalence_library import SessionEquivalenceLibrary
 
# QISKIT REFERENCE FOR CREATING CUSTOM BACKEND: https://quantum.cloud.ibm.com/docs/en/api/qiskit/providers#writing-a-new-backend

class DRAGGate(Gate):
    def __init__(self, theta, phi, label=None):
        super().__init__("drag", 1, [theta, phi], label=label)
        self.theta = theta
        self.phi = phi
 
    def _define(self):
        qc = QuantumCircuit(1)
        qc.rz(-self.phi,0)
        qc.rx(self.theta,0)
        qc.rz(+self.phi,0)
        self.definition = qc

# DEFINITION OF DRAG GATE (can simulate any 1body gate, up to virtual z rotation)
def drag(theta, phi):
    qc = qiskit.QuantumCircuit(1, name="drag")
    qc.rz(-phi,0)
    qc.rx(theta,0)
    qc.rz(+phi,0)
    return qc.to_gate()

#########################
#DEFINE GATE EQUIVALENCE:
# Qililab  DRAG(θ, φ) =                Rz( φ)Rx( θ)Rz(-φ)
# Qiskit   U(θ, φ, λ) = e^i(φ+λ)/2  ·  Rz( φ)Ry( θ)Rz( λ)      # https://quantum.cloud.ibm.com/docs/en/api/qiskit/qiskit.circuit.library.UGate
# Rx --> Ry  using φ --> φ+π/2
# U(θ, φ, λ) = DRAG(θ, φ+π/2) · Rz(λ+φ)  (1)
# And we can exploit commutation:  DRAG(θ, φ) Rz(λ) = Rz(λ) DRAG(θ, φ-λ) ,   which leads to
# U(θ, φ, λ) = Rz(λ+φ) · DRAG(θ,-λ+π/2)  (2)
# In fact this is shows DRAG is universal up to virtual Rz gates (which commute with CZ and can be ignored on measurement)
phi = Parameter("phi")
theta = Parameter("theta")
lamb = Parameter("lambda")
# Eq (1)
qc1 = qiskit.QuantumCircuit(1)
qc1.append(RZGate(phi+lamb), [0])
qc1.append(DRAGGate(theta, phi + np.pi/2), [0])
SessionEquivalenceLibrary.add_equivalence(UGate(theta,phi,lamb), qc1)
# Eq (2), implies swapping of DRAG and RZ
qc2 = qiskit.QuantumCircuit(1)
qc2.append(DRAGGate(theta, -lamb+np.pi/2), [0])
qc2.append(RZGate(phi+lamb), [0])
SessionEquivalenceLibrary.add_equivalence(UGate(theta,phi,lamb), qc2)


# QBLUE RUNCARD AT: /gpfs/apps/QUANTUM/CALIBRATION/qblue/qblue.yml
# LOCAL COPY AT testing/CALIBRATIONS/qblue/qblue.yml
class FakeQBlueBackend(BackendV2):
    """Fake Backend imitating BSC's 5 qubit Quantum Blue chip (as of 15/01/2026) https://www.bsc.es/supportkc/docs/Quantum/overview
    Based in example from: https://quantum.cloud.ibm.com/docs/en/guides/custom-backend"""
    NUM_QUBITS = 5

    # CALIBRATION PARAMETERS:
    # IDEALLY THEY WOULD NOT BE STATIC (CLASS), AS THEY DEPEND ON INSTANCE. EVERY DAY DIFFERENT CALIBRATION
    calibs_string = """Qubit	T1 (µs)	T2 (µs)	1Q Gate	Readout	2Q Gate
0	30.68	12.90	99.92	90.7	1_0: 91.07
1	28.75	13.07	99.90	90.4	1_0: 91.07 - 2_1: 93.07
2	24.57	15.46	99.90	86.8	2_1: 93.07 - 3_2: 95.56 - 4_2: 95.36
3	16.46	7.30	99.81	89.3	3_2: 95.56
4	32.75	7.11	99.92	86.8	4_2: 95.36"""

    DRAG_duration = [40e-9]*5
    M_duration = [1542.857142857143e-9, 1000.0e-9, 1542.857142857143e-9, 857.1428571428571e-9, 857.1428571428571e-9]
    topology = [[1,0], [1,2], [3,2], [4,2]]
    CZ_DURATION = [46e-9, 34e-9, 46e-9, 38e-9] # duration for corresponding connection (same element in topology)
    CZ_ERROR = [1-91.07/100, 1-93.07/100, 1-95.56/100, 1-95.36/100] # error for corresponding connection (same element in topology)
    
    # CHIP DATA COULD BE OBTAINED FROM : platform = ql.build_platform(runcard=PLATFORM_PATH)
    calibs = list(map(lambda line: line.split('\t'), calibs_string.split('\n')))
    #calibs_dict = {calibs[0][q]: calibs[1:][q] for q in range(5)}  # weird python technicality. class vars namespace is not accessible from scopes within clas (like list comprehension)
    calibs_dict = (lambda calibs=calibs: {calibs[0][q]: calibs[1:][q] for q in range(5)})()# lambda is irrelevant, just to get through this namespace technicality
 
    def __init__(self, manual=True):
        """Instantiate backend,
        arg: manual (bool):
            - If False: corresponds to actual true Backend, which can use only DRAG gates and CZ gates. It is enough, but Qiskit has problems to transpile directly
            - If True : uses U = RzRyRz gate instead of DRAG gate. To transpile, then, custom transpiler pass needed"""
        super().__init__(name="Fake QBlue backend")
        # Create a heavy-hex graph using the rustworkx library, then instantiate a new target
        self._graph = rustworkx.PyGraph()
        for q in range(5): self._graph.add_node(q)
        for connection in self.topology: self._graph.add_edge(connection[0], connection[1], None)  # could use *connection to unravel, but is less readable
        num_qubits = 5
        self._target = Target(
            "Fake QBlue backend", num_qubits=num_qubits
        )
 
        # Generate instruction properties for single qubit gates and a measurement, delay,
        #  and reset operation to every qubit in the backend.
        drag_props = {}
        rz_props = {}
        cz_props = {}
        measure_props = {}
        delay_props = {}
 
        # Add 1q gates
        for i in range(num_qubits):
            qarg = (i,)
            # TODO: Can't implement 1 qubit gate error, as transpilation then sometimes produces pairs of 1q gates, but should be 1
            NOT_IMPLEMENTED = 0
            drag_props[qarg] = InstructionProperties(
                error=float(FakeQBlueBackend.calibs_dict["1Q Gate"][i])*NOT_IMPLEMENTED,
                duration=FakeQBlueBackend.DRAG_duration[i],
            )
            measure_props[qarg] = InstructionProperties(
                error=float(FakeQBlueBackend.calibs_dict["Readout"][i]),
                duration=FakeQBlueBackend.M_duration[i],
            )  # duration ??
            delay_props[qarg] = None

        # TODO : Make the automatic transpiler work. RX are converted to Drag, and RY too, but when both appear in the circuit it breaks.
        if manual: self._target.add_instruction(U3Gate(Parameter("theta"), Parameter("phi"), Parameter("lambda")),drag_props)
        else: 
            self._target.add_instruction(DRAGGate(Parameter("theta"), Parameter("phi")), drag_props)
            self._target.add_instruction(RZGate(Parameter("lambda")), drag_props) # dangerous, don't need it
        self._target.add_instruction(Measure(), measure_props)
        self._target.add_instruction(Reset(), measure_props)
        self._target.add_instruction(Delay(Parameter("t")), delay_props)

        # Add chip local 2q gate which is CZ
        cz_props = {}
        for i in range(len(self.topology)):
            cz_props[(self.topology[i][0], self.topology[i][1])] = InstructionProperties(
                error=FakeQBlueBackend.CZ_DURATION[i],
                duration=FakeQBlueBackend.CZ_ERROR[i],
            )
            cz_props[(self.topology[i][1], self.topology[i][0])] = InstructionProperties(
                error=FakeQBlueBackend.CZ_DURATION[i],
                duration=FakeQBlueBackend.CZ_ERROR[i],
            )
        self._target.add_instruction(CZGate(), cz_props)
 
    @property
    def target(self):
        return self._target
    
    @property
    def max_circuits(self):
        return None
 
    @property
    def graph(self):
        return self._graph
 
    @classmethod
    def _default_options(cls):
        return Options(shots=1024)
 
    def run(self, circuit, **kwargs):
        raise NotImplementedError(
            "This backend does not contain a run method"
        )

if __name__ == "__main__":
    import qibo
    from qiskit import qasm2
    from tfising import circs_shots_noisy_jac
    MANUAL = True
    qc = QuantumCircuit(3)
    qc.rx(0.2, 0)
    qc.rx(0.2, 1)
    qc.rz(0.2, 2)
    qc.cz(0,1)
    qc.cz(1,2)
    qc.cz(2,0)
    qc.rx(0.2, 0)
    qc.ry(0.2, 1)
    qc.rx(0.2, 2)
    #qc.append(drag(Parameter("theta"),Parameter("phi")), [1])

    """qiskit.compiler.transpile(qc, basis_gates=['cz, rx rz'])"""

    circinv, circnoinv, *other = circs_shots_noisy_jac(N=5, p=1, noise=False)
    qc = qasm2.loads(circinv.to_qasm())


    print(qc.draw())

    qblue = FakeQBlueBackend(manual=MANUAL)

    # print(qblue.graph)
    # rustworkx.visualization.mpl_draw(qblue.graph)
    # plt.show()

    #qiskit.compiler.transpile(qc, target=qblue.target)
    pm = generate_preset_pass_manager(optimization_level=3, backend=qblue)
    transpiled_qc = pm.run(qc)
    print(transpiled_qc)

    print(" Transpiling with transpile() ")
    print(transpile(qc,basis_gates=['u3', 'cz'], coupling_map=FakeQBlueBackend.topology))



    cqibo = qibo.Circuit.from_qasm(qasm2.dumps(transpiled_qc))

    cqibo.draw()

    """# THE EFFECT OF TRANSPILATION WAS TO CHANGE CONNECTIONS:
    1-2,  3-4   -->   2-3,   1-4
    Was changed to:
    1-2,  4-3   -->  swap 2-4   and  1-4,  2-3"""

    

