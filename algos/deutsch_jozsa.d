module algos.deutsch_jozsa;

import core.stdc.stdlib : rand;

import std.stdio;
import std.format;
import std.complex;
import std.math;
import std.range;
import std.array;

import std.conv : to;

import quantum.pure_state.qc;

enum FunctionType {
    Balanced,
    Constant,
}

struct DeutschJozsa {
    int num_qubits;
    QuantumCircuit qc;

    this(int num_qubits) {
        this.num_qubits = num_qubits + 1;
        this.qc = QuantumCircuit(this.num_qubits);
    }

    /// A constant function f(x) = 0
    int f_constant_0(string _bit_str) {
        return 0;
    }
    /// A constant function f(x) = 1
    int f_constant_1(string _bit_str) {
        return 1;
    }

    /// A balanced function
    int f_balanced(string bit_str) {
        return bit_str[0];
    }

    // Implements the oracle gate which affects the quantum state differently based on
    // if the function provided is constant and has output 0 or 1, or balanced where their is an 
    // equal number of 0's and 1's. When the function is constant such that f(x) = 0 or f(x) = 1, 
    // the output will most of the time be all 0 (ideally). If the function is balanced the output
    // should be anything but all 0. 
    private void oracle_gate(int delegate(string) f, FunctionType type) {
        string bit_str = format("%0*b", this.num_qubits - 1, cast(int) rand() % (
                1 << (this.num_qubits - 1)));

        switch (type) {
        case FunctionType.Balanced:
            for (int i = 0; i < bit_str.length; i++) {
                if (bit_str[i] == '1') {
                    this.qc.pauli_x(i);
                }
            }

            for (int i = 0; i < this.num_qubits - 1; i++) {
                this.qc.cnot(i, this.num_qubits - 1);
            }

            for (int i = 0; i < bit_str.length; i++) {
                if (bit_str[i] == '1') {
                    this.qc.pauli_x(i);
                }
            }
            break;
        case FunctionType.Constant:
            int output = f(bit_str);
            if (output == 1) {
                this.qc.pauli_x(this.num_qubits - 1);
            }
            break;
        default:
            writeln("invalid function type");
            break;
        }
    }

    /**
    * Implements the deutsch-jozsa algorithm for a given fucntion type
    * 
    * params:
    * f = The function used in the algorithm
    *
    * type = The type of function being passed, this is required due to the limitations
    *        of classical simulation
    *
    * shots = The amount of times to run the algorithm (default = 2000)
    *
    * returns: An associative array of binary string of basis state to number of times measured
    */
    int[string] deutsch_jozsa(int delegate(string) f, FunctionType type, int shots = 2000) {
        this.qc.pauli_x(this.num_qubits - 1);
        this.qc.hadamard(this.num_qubits - 1);

        for (int i = 0; i < this.num_qubits - 1; i++) {
            this.qc.hadamard(i);
        }

        oracle_gate(f, type);

        for (int i = 0; i < this.num_qubits - 1; i++) {
            this.qc.hadamard(i);
        }

        int[string] counts = qc.measure_many(iota(0, this.num_qubits - 1, 1).array, shots);

        return counts;
    }
}
