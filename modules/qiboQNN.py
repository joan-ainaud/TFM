#### IN THIS CODE WE DEFINE A CLASS TO IMPLEMENT PARAMETRIZED QUANTUM CIRCUITS IN A STANDARIZED WAY
import qibo
from qibo import gates, Circuit, parameter
from qibo.symbols import I, X, Y, Z
import numpy as np
from multiprocessing import Pool

# class quantumModel():
#     """
#     Class that defines a quantum model for solving a particular problem:
#     - Number of qubits is variable, depending on model, instead the definition is the construction of the circuit"""
#     def __init__(observable, ):
#         self.parameter()

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
    with Pool(processes_count) as p:
       res = p.map(complex_operation, range(processes_count))
    
    res