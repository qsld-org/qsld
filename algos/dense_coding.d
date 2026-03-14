import std.stdio;
import std.math;
import std.format;
import std.algorithm;

import quantum.pure_state.qc;

struct DenseCoding {
    int alices_num;
    int num_qubits;
    int num_bell_pairs;
    int alice_max_num;

    this(int num_qubits, int alices_num) {
        this.alices_num = alices_num;
        this.num_qubits = 2 * num_qubits;
        this.num_bell_pairs = num_qubits;
        this.alice_max_num = pow(2, 2 * this.num_qubits) - 1;

        assert(this.alices_num <= this.alice_max_num,
            "The number alice will send is too large for the number of qubits");
    }

    void dense_coding() {
        QuantumCircuit qc = QuantumCircuit(this.num_qubits);

        // Make k disjoint bell pairs out of 2k qubits
        for (int i = 0; i < this.num_qubits / 2; i++) {
            qc.hadamard(i);
            qc.cnot(i, i + (this.num_qubits / 2));
        }

        // Get the binary representation of the number that alice is trying to send
        string alice_num_bin = format("%0*b", this.num_qubits, this.alices_num).dup.reverse;

        // Make pairs of bits out of the binary representation of alices number. The pairs
        // must match the entanglement pattern of the bell states.
        string[] pairs;
        string pair = "";
        for (int i = 0; i < this.num_qubits / 2; i++) {
            pair ~= alice_num_bin[i];
            pair ~= alice_num_bin[i + (this.num_qubits / 2)];
            pairs ~= pair;
            pair = "";
        }

        // Based on each pair, apply a specific pauli operator to alices qubits
        for (int i = 0; i < this.num_qubits / 2; i++) {
            switch (pairs[i]) {
            case "00":
                break;
            case "01":
                qc.pauli_x(i);
                break;
            case "10":
                qc.pauli_z(i);
                break;
            case "11":
                qc.pauli_x(i);
                qc.pauli_z(i);
                break;
            default:
                assert(false, "Unknown pair in the pairs array");
            }
        }

        // Bob now has the qubits that Alice has sent

        // Undo the entanglement of the Bell pairs to put the state back into the computational basis
        for (int i = 0; i < this.num_qubits / 2; i++) {
            qc.cnot(i, i + (this.num_qubits / 2));
            qc.hadamard(i);
        }

        // Measure all the qubits and hopefully get the number alice sent
        writeln(qc.measure_all(1000));
    }
}
