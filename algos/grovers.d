module algos.grovers;

import std.complex;
import std.math;
import std.format;
import std.range;
import std.array;

import quantum.pure_state.qc;

enum OperatorType {
    Normal,
    Decomposition
}

struct Grovers {
    int num_qubits;
    QuantumCircuit qc;
    OperatorType ot;

    /**
    * The constructor for the grovers algorithm object
    *
    * params:
    * num_qubits = The number of qubits to use in the algorithms internal circuit
    */
    this(int num_qubits) {
        this.num_qubits = num_qubits;
        this.qc = QuantumCircuit(this.num_qubits);
        this.ot = OperatorType.Normal;
    }

    /**
    * Constructor overload for grovers algorithm object which allows for specification
    * of the type of oracle and diffusion operator to be used
    *
    * params:
    * num_qubits = The number of qubits to use in the algorithms internal circuit
    *
    * ot = The type of oracle and diffusion operator to be used. Choose between OperatorType.Normal
    *      and OperatorType.Decomposition, the Normal version does not use gates and is possibly faster 
    *      than the Decomposition version which uses gates in the implementation of both the oracle and 
    *      diffusion operator
    */
    this(int num_qubits, OperatorType ot) {
        this.num_qubits = num_qubits;
        this.qc = QuantumCircuit(this.num_qubits);
        this.ot = ot;
    }

    // The grovers algorithm oracle to be used
    private void oracle(int function(string) f) {
        for (int i = 0; i < this.qc.state.elems.length; i++) {
            if (f(format("%0*b", this.num_qubits, i)) == 1) {
                this.qc.state.elems[i] = this.qc.state.elems[i] * Complex!real(-1, 0);
            }
        }
    }

    // The gate decomposiiton of grovers oracle
    private void oracle_decomp(int function(string) f) {
        int[] qubit_indices;
        for (int i = 0; i < this.qc.state.elems.length; i++) {
            if (f(format("%0*b", this.num_qubits, i)) == 1) {
                for (int j = 0; j < this.num_qubits; j++) {
                    if ((i & (1 << j)) == 0) {
                        this.qc.pauli_x(j);
                        qubit_indices ~= [j];
                    }
                }

                this.qc.mcz(iota(0, this.num_qubits).array);

                foreach (qubit_idx; qubit_indices) {
                    this.qc.pauli_x(qubit_idx);
                }
            }
        }
    }

    // The diffusion operator to be used 
    private void diffusion() {
        Complex!real sum = Complex!real(0, 0);
        for (int i = 0; i < this.qc.state.elems.length; i++) {
            sum = sum + this.qc.state.elems[i];
        }

        Complex!real mean = sum / Complex!real(this.qc.state.elems.length, 0);

        for (int i = 0; i < this.qc.state.elems.length; i++) {
            this.qc.state.elems[i] = 2 * mean - this.qc.state.elems[i];
        }
    }

    // The gate decomposition of grovers diffusion operator
    private void diffusion_decomp() {
        for (int i = 0; i < this.num_qubits; i++) {
            this.qc.hadamard(i);
        }

        for (int i = 0; i < this.num_qubits; i++) {
            this.qc.pauli_x(i);
        }

        this.qc.mcz(iota(0, this.num_qubits).array);

        for (int i = 0; i < this.num_qubits; i++) {
            this.qc.pauli_x(i);
        }

        for (int i = 0; i < this.num_qubits; i++) {
            this.qc.hadamard(i);
        }
    }

    /**
    * The main grovers algorithm which searches for a solution in an unsorted 
    * search space.
    *
    * params:
    * f = The function which encodes the solution to the search
    *
    * shots = The amount of times to run measurement on the resulting state
    *
    * returns: An associative array of basis state to number of times measured
    */
    int[string] grovers(int function(string) f, int shots = 2000) {
        for (int i = 0; i < this.num_qubits; i++) {
            this.qc.hadamard(i);
        }

        real pi_over_four = PI / 4;
        int num_iterations = cast(int)(pi_over_four * sqrt(cast(real) pow(2, this.num_qubits)));

        for (int i = 0; i < num_iterations; i++) {
            switch (this.ot) {
            case OperatorType.Decomposition:
                oracle_decomp(f);
                diffusion_decomp();
                break;
            case OperatorType.Normal:
                oracle(f);
                diffusion();
                break;
            default:
                assert(false, "Unknown operator type");
            }
        }

        return this.qc.measure_all(shots);
    }
}
