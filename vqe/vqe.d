module vqe.vqe;

import std.stdio;
import std.complex;
import std.math;

import linalg.vector;

import quantum.pure_state.qc;
import quantum.pure_state.observable;

struct VQE {
    int iterations; // The number of times to minimize the energy of the given hamiltonian
    int num_qubits; // The number of qubits to use for the algorithm
    Observable hamiltonian; // The hamiltonian to find the minimum energy of
    real[] trainable_params; // The parameters which are modified to minimize the energy
    real learning_rate; // The rate at which to reach convergence

    QuantumCircuit function(QuantumCircuit, real[]) vqc; // The ansatz circuit

    this(
        int num_qubits,
        Observable hamiltonian,
        real[] trainable_params,
        int iterations,
        real learning_rate,
        QuantumCircuit function(QuantumCircuit, real[]) vqc) {

        this.num_qubits = num_qubits;
        this.iterations = iterations;
        this.hamiltonian = hamiltonian;
        this.trainable_params = trainable_params;
        this.learning_rate = learning_rate;

        this.vqc = vqc;
    }

    private real parameter_shift(ulong i) {
        // make two copies of the trainable parameters so that
        // so that they can be modified without modifying the 
        // original
        real[] train_params_minus = this.trainable_params.dup;
        real[] train_params_plus = this.trainable_params.dup;

        // add and subtract PI/2 from a given angle in each copy
        train_params_minus[i] -= PI / 2;
        train_params_plus[i] += PI / 2;

        // get the result of running the variational quantum circuit on the shifted parameters
        real minus_result = cost(train_params_minus);
        real plus_result = cost(train_params_plus);

        // get the difference between the results 
        real result_diff = plus_result - minus_result;
        result_diff = result_diff * 0.5;
        return result_diff;
    }

    private real cost(real[] trainable_params) {
        QuantumCircuit qc = QuantumCircuit(this.num_qubits);
        QuantumCircuit psi_theta = this.vqc(qc, trainable_params);
        real energy = psi_theta.expectation_value(this.hamiltonian);
        return energy;
    }

    void vqe() {
        foreach (iter; 0 .. this.iterations) {
            real cur_energy = cost(this.trainable_params);
            writefln("Iteration: %d | Energy/cost: %f", iter, cur_energy);
            foreach (i, param; this.trainable_params) {
                real gradient = parameter_shift(i);
                writefln("Param: %d | gradient: %f", i, gradient);
                this.trainable_params[i] = param - this.learning_rate * gradient;
            }
        }
    }
}
