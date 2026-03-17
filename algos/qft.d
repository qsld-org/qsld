module algos.qft;

import std.stdio;

import quantum.pure_state.qc;

struct QFT {
    int num_qubits;
    int initial_state_idx;
    QuantumCircuit qc;

    this(int num_qubits) {
        this.num_qubits = num_qubits;
        this.qc = QuantumCircuit(this.num_qubits);
    }

    this(int num_qubits, int initial_state_idx) {
        this.num_qubits = num_qubits;
        this.initial_state_idx = initial_state_idx;
        this.qc = QuantumCircuit(this.num_qubits, this.initial_state_idx);
    }

    this(QuantumCircuit qc) {
        this.qc = qc;
        this.num_qubits = this.qc.num_qubits;
    }

    /** 
    * The Quantum Fourier Transform or QFT takes n qubits and their computational basis states
    * and maps them to superpositions with specific phases which affect the amplitudes
    */
    void qft() {
        for (int i = 0; i < this.num_qubits; i++) {
            for (int j = i + 1; j < this.num_qubits; j++) {
                this.qc.cr(i, j, j - i + 1);
            }
            this.qc.hadamard(i);
        }

        for (int i = 0; i < this.num_qubits / 2; i++) {
            this.qc.swap(i, this.num_qubits - (i + 1));
        }
    }

    /**
    * Overload of the qft function which only takes specific
    * qubit indices to act on
    */
    void qft(int[] qubit_idxs) {
        for (int i = 0; i < qubit_idxs.length; i++) {
            for (int j = i + 1; j < qubit_idxs.length; j++) {
                this.qc.cr(qubit_idxs[i], qubit_idxs[j], j - i + 1);
            }
            this.qc.hadamard(qubit_idxs[i]);
        }

        for (int i = 0; i < qubit_idxs.length / 2; i++) {
            this.qc.swap(qubit_idxs[i], qubit_idxs[cast(int) qubit_idxs.length - (i + 1)]);
        }
    }

    /**
    * Undoes the operation of the QFT
    */
    void qft_inverse() {
        for (int i = this.num_qubits - 1; i >= 0; i--) {
            for (int j = 0; j < i; j++) {
                this.qc.cr(j, i, i - j + 1, true);
            }
            this.qc.hadamard(i);
        }

        for (int i = 0; i < this.num_qubits / 2; i++) {
            this.qc.swap(i, this.num_qubits - (i + 1));
        }
    }

    /**
    * Overload of the inverse qft which only takes specific 
    * qubit indices to act on
    */
    void qft_inverse(int[] qubit_idxs) {
        for (int i = cast(int) qubit_idxs.length - 1; i >= 0; i--) {
            for (int j = 0; j < i; j++) {
                this.qc.cr(qubit_idxs[j], qubit_idxs[i], i - j + 1, true);
            }
            this.qc.hadamard(qubit_idxs[i]);
        }

        for (int i = 0; i < qubit_idxs.length / 2; i++) {
            this.qc.swap(qubit_idxs[i], qubit_idxs[cast(int) qubit_idxs.length - (i + 1)]);
        }
    }
}
