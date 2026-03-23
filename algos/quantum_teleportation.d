module algos.quantum_teleportation;

import std.stdio;
import std.format;
import std.conv;
import std.complex;
import std.math;

import quantum.pure_state.qc;

struct QuantumTeleportation {
    /**
    * Executes the quantum teleportation algorithm
    *
    * params:
    * randomize_q0 = A function which randomizes the state of the qubit at index 0.
    *                Takes a pointer to the QuantumCircuit object used during the 
    *                algorithm.
    */
    void quantum_teleportation(void function(QuantumCircuit* qc) randomize_q0) {
        // Initialize a quantum circuit with 3 qubits to represent 
        // Alice and Bob's EPR (Einstein Podolsky Rosen) pair.
        QuantumCircuit qc = QuantumCircuit(3);

        // Randomize the state of qubit 0 before the algorithm starts
        randomize_q0(&qc);

        // Alice puts qubit 1 in superposition
        qc.hadamard(1);

        // Alice entangles qubits 1 and 2
        qc.cnot(1, 2);

        // Entangle qubits 0 and 1 to prepare for telepotation
        qc.cnot(0, 1);

        // Put qubit 0 into superposition
        qc.hadamard(0);

        // Measure qubits 0 and 1 to get what Bob should do with 
        // his qubit
        string q0_measured = qc.measure(0, true);
        string q1_measured = qc.measure(1, true);

        int q0_state = to!int(q0_measured);
        int q1_state = to!int(q1_measured);

        // Check which combination of values Bob measured in order
        // to determine what Bob should do with his qubit.
        if (q0_state == 0 && q1_state == 1) {
            qc.pauli_x(2);
        } else if (q0_state == 1 && q1_state == 0) {
            qc.pauli_z(2);
        } else if (q0_state == 1 && q1_state == 1) {
            qc.pauli_x(2);
            qc.pauli_z(2);
        }

        // Draw the cicuit (commented because it is unnecessary in most cases)
        // qc.draw();

        writeln(format("Bob measured: %d%d", q1_state, q0_state));

        // Declare bob's final qubit as a cicuit
        QuantumCircuit bob_qc = QuantumCircuit(1);
        // Give the measurement as a single number between 0 and 3
        // which can be rperesented as a binary number
        int alice_measurement = (q1_state << 1) | q0_state;

        // Loop over each amplitude index in the original state vector
        for (int i = 0; i < 8; i++) {
            // Mask the index to get Alices bits
            int alice_bits = i & 0b11;
            // Compare the bits to the measured value
            if (alice_bits == alice_measurement) {
                // Get the value of Bob's qubit
                int bob_bit = (i >> 2) & 0b1;
                // Put the amplitudes corresponding to Bob's bit into 
                // Bob's state vector for his qubit
                bob_qc.state.elems[bob_bit] = qc.state.elems[i];
            }
        }

        // normalize bob's state vector
        real bob_qc_norm = 0;
        foreach (amp; bob_qc.state.elems) {
            bob_qc_norm += norm(amp);
        }

        bob_qc_norm = sqrt(bob_qc_norm);

        foreach (i, amp; bob_qc.state.elems) {
            if (amp.re != 0 && amp.im == 0) {
                bob_qc.state.elems[i] = Complex!real(amp.re / bob_qc_norm, 0);
            } else if (amp.re == 0 && amp.im != 0) {
                bob_qc.state.elems[i] = Complex!real(0, amp.im / bob_qc_norm);
            } else if (amp.re != 0 && amp.im != 0) {
                bob_qc.state.elems[i] = Complex!real(amp.re / bob_qc_norm, amp.im / bob_qc_norm);
            }
        }

        writeln("The full state vector is: ", qc.state.elems);
        writeln("Bob's final state vector is: ", bob_qc.state.elems);
    }
}
