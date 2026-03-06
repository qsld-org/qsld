module vqe.vqe;

import std.stdio;
import std.complex;
import std.math;
import std.algorithm;

import linalg.vector;
import linalg.matrix;

import quantum.pure_state.qc;
import quantum.pure_state.observable;

enum OptimizerType {
    ParameterShift,
    Cobyla
}

enum InterpolationMethod {
    BiasedRhoPositive,
    BiasedRhoNegative,
    Symmetric
}

struct CobylaConfig {
    real rho; // The size of the trust region
    real step_threshold; // The minimum difference between the current energy and the next energy in order for the step to be accepted
    real decay_factor; // The amount that rho will reduce when the step is rejected
    InterpolationMethod method; // The method that will be used to generate the interpolation point vectors

    /* 
    * The constructor for the CobylaConfig object which specifies the hyperparameters for
    * the COBYLA optimizer algorithm
    *
    * params:
    * rho = The radius or the trust region 
    * 
    * step_threshold = The amount that the energy or cost must decrese before the trainable
    *                  parameters are modified
    *
    * decay_factor = How much rho will be reduced on every iteration that the step is rejected
    * 
    * method = The method of interpolation to use in order to generate and solve the linear system
    */
    this(real rho, real step_threshold, real decay_factor, InterpolationMethod method) {
        this.rho = rho;
        this.step_threshold = step_threshold;
        this.decay_factor = decay_factor;
        this.method = method;
    }
}

struct VQE {
    int iterations; // The number of times to minimize the energy of the given hamiltonian
    int num_qubits; // The number of qubits to use for the algorithm
    Observable hamiltonian; // The hamiltonian to find the minimum energy of
    real[] trainable_params; // The parameters which are modified to minimize the energy
    real learning_rate; // The rate at which to reach convergence
    OptimizerType op_type; // The type of optimizer to use
    CobylaConfig cobyla_conf; // The configuration of the COBYLA optimizer if used as specified by op_type 
    real[][] interpolation_vectors; // Only used for COBYLA optimizer, specifies interpolation points 
    real[] b; // The energy/cost of each interpolation vector when using the COBYLA optimizer

    QuantumCircuit function(QuantumCircuit, real[]) vqc; // The ansatz circuit

    /*
    * The constructor which defines the configuraiton of the VQE algorithm
    *
    * params:
    * num_qubits = The number of qubits to use for the QuantumCircuit used in the vqc function
    * 
    * hamiltonian = The hamiltonian which defines the energy of the system and acts as a cost function
    *               in combination with the expectation value. Specified as a series of pauli strings with
    *               various coefficients
    * 
    * trainable_params = The parameters which are modified by the optimizer to efficiently reduce the energy 
    *                    after every iteration
    *
    * iterations = The amount of times to execute the cost function and optimizer in order to reach convergence
    * 
    * learning_rate = How fast the model learns when using the parameter shift optimizer. Does not apply to the  
    *                 COBYLA optimizer
    *
    * op_type = The optimizer algorithm to use, can be either OptimizerType.ParameterShift or OptimizerType.Cobyla
    * 
    * vqc = The variational quantum circuit used to map the trainable parameters to a specific quantum state
    * 
    * cobyla_conf = An optional parameter specifying the hyperparameters for the COBYLA optimizer
    */
    this(
        int num_qubits,
        Observable hamiltonian,
        real[] trainable_params,
        int iterations,
        real learning_rate,
        OptimizerType op_type,
        QuantumCircuit function(QuantumCircuit, real[]) vqc,
        CobylaConfig cobyla_conf = CobylaConfig(0.1, 0.01, 0.5, InterpolationMethod
            .BiasedRhoPositive)) {

        this.num_qubits = num_qubits;
        this.iterations = iterations;
        this.hamiltonian = hamiltonian;
        this.trainable_params = trainable_params;
        this.learning_rate = learning_rate;
        this.op_type = op_type;
        this.cobyla_conf = cobyla_conf;
        this.vqc = vqc;
    }

    // The optimizer function which uses parameter shift to
    // calculate the gradient of the cost function which in turn
    // minimizes the energy of the system
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

    // Used to get the coefficient vector c after performing gaussian elimination
    // on the interpolation matrix of interpolation vectors
    private real[] back_substitution(Matrix!real interpolation_mat, real[] b) {
        real[] c = new real[interpolation_mat.rows.length];
        c[] = 0;
        for (int i = cast(int) interpolation_mat.rows.length - 1; i >= 0; i--) {
            real sum = 0;
            for (int j = i + 1; j < interpolation_mat.rows.length; j++) {
                sum += interpolation_mat.rows[i].elems[j] * c[j];
            }

            c[i] = (b[i] - sum) / interpolation_mat.rows[i].elems[i];
        }

        return c;
    }

    // Puts the interpolation matrix generated by interpolation vectors into 
    // upper eschelon or upper triangular form
    private real[] gaussian_elimination(Matrix!real interpolation_mat, real[] b) {
        Vector!real[] mat_cols = interpolation_mat.get_cols();
        foreach (k, col; mat_cols) {
            ulong max_row = k;
            real pivot = interpolation_mat.rows[k].elems[k];
            for (int i = cast(int) k + 1; i < interpolation_mat.rows.length; i++) {
                real elem = interpolation_mat.rows[i].elems[k];
                if (abs(elem) > abs(pivot)) {
                    pivot = elem;
                    max_row = i;
                }
            }

            if (max_row != k) {
                Vector!real temp = interpolation_mat.rows[k];
                interpolation_mat.rows[k] = interpolation_mat.rows[max_row];
                interpolation_mat.rows[max_row] = temp;

                real temp_b = b[k];
                b[k] = b[max_row];
                b[max_row] = temp_b;
            }

            for (int i = cast(int) k + 1; i < interpolation_mat.rows.length; i++) {
                real elem = interpolation_mat.rows[i].elems[k];

                real m = elem / pivot;
                for (int j = 0; j < mat_cols.length; j++) {
                    interpolation_mat.rows[i].elems[j] = interpolation_mat.rows[i].elems[j] - m * interpolation_mat
                        .rows[k].elems[j];
                }
                b[i] = b[i] - m * b[k];
            }
        }

        real[] c = back_substitution(interpolation_mat, b);
        return c;
    }

    // The optimizer function which uses linear approximations
    // to minimize the energy of the system
    private real[] cobyla(real cur_energy) {
        Matrix!real interpolation_mat = Matrix!real(cast(int) this.interpolation_vectors.length, cast(
                int) this.interpolation_vectors[0].length + 1, []);

        real[] trainable_params_copy = this.trainable_params.dup;
        trainable_params_copy = [cast(real) 1] ~ trainable_params_copy;
        Vector!real trainable_params_vec = Vector!real(
            cast(int) this.trainable_params.length + 1, trainable_params_copy);
        interpolation_mat.append(trainable_params_vec);

        foreach (iv; this.interpolation_vectors) {
            iv = [cast(real) 1] ~ iv;
            Vector!real iv_vec = Vector!real(cast(int) iv.length, iv);
            interpolation_mat.append(iv_vec);
        }

        real[] c = gaussian_elimination(interpolation_mat, b);
        Vector!real g = Vector!real(cast(int) c.length - 1, c[1 .. c.length]);
        real magnitude = g.mag();

        foreach (i, elem; g.elems) {
            g[i] = elem / magnitude;
        }

        Vector!real s = g.mult(-this.cobyla_conf.rho);
        real[] tpc = this.trainable_params.dup;
        Vector!real tpv = Vector!real(cast(int) this.trainable_params.length, tpc);
        tpv = tpv.add(s);

        real energy = cost(tpv.elems);
        real r = cur_energy - energy;
        if (r > this.cobyla_conf.step_threshold) {
            real[] b_copy = this.b.dup[1 .. $];
            real worst_energy = b_copy.maxElement;
            auto worst_idx = b_copy.countUntil(worst_energy);
            this.interpolation_vectors[worst_idx] = tpv.elems;
            this.b[worst_idx] = energy;
            return tpv.elems;
        } else {
            this.cobyla_conf.rho = this.cobyla_conf.rho * this.cobyla_conf.decay_factor;
            return this.trainable_params;
        }
    }

    // Uses the ansatz to generate a state from the trainable parameters
    // and then takes the energy expectation value of the hamiltonian with
    // that state
    private real cost(real[] trainable_params) {
        QuantumCircuit qc = QuantumCircuit(this.num_qubits);
        QuantumCircuit psi_theta = this.vqc(qc, trainable_params);
        real energy = psi_theta.expectation_value(this.hamiltonian);
        return energy;
    }

    /*
    * The main VQE algorithm which minimizes the energy of a system given some hamiltonian
    */
    void vqe() {
        if (this.op_type == OptimizerType.ParameterShift) {
            foreach (iter; 0 .. this.iterations) {
                real cur_energy = cost(this.trainable_params);
                writefln("Iteration: %d | Energy/cost: %f", iter, cur_energy);
                foreach (i, param; this.trainable_params) {
                    real gradient = parameter_shift(i);
                    writefln("Param: %d | gradient: %f", i, gradient);
                    this.trainable_params[i] = param - this.learning_rate * gradient;
                }
            }
        } else if (this.op_type == OptimizerType.Cobyla) {
            foreach (iter; 0 .. this.iterations) {
                this.interpolation_vectors = [];
                switch (cobyla_conf.method) {
                case InterpolationMethod.BiasedRhoPositive:
                    for (int i = 0; i < this.trainable_params.length; i++) {
                        real[] interpolation_vector = new real[this.trainable_params.length];
                        for (int j = 0; j < this.trainable_params.length; j++) {
                            if (j == i) {
                                interpolation_vector[j] = this.trainable_params[j] + this
                                    .cobyla_conf.rho;
                            } else {
                                interpolation_vector[j] = this.trainable_params[j];
                            }
                        }
                        this.interpolation_vectors ~= interpolation_vector;
                    }
                    break;
                case InterpolationMethod.BiasedRhoNegative:
                    for (int i = 0; i < this.trainable_params.length; i++) {
                        real[] interpolation_vector = new real[this.trainable_params.length];
                        for (int j = 0; j < this.trainable_params.length; j++) {
                            if (j == i) {
                                interpolation_vector[i] = this.trainable_params[i] - this
                                    .cobyla_conf.rho;
                            } else {
                                interpolation_vector[i] = this.trainable_params[i];
                            }
                        }

                        this.interpolation_vectors ~= interpolation_vector;
                    }
                    break;
                case InterpolationMethod.Symmetric:
                    for (int i = 0; i < this.trainable_params.length; i++) {
                        real[] interpolation_vector = new real[this.trainable_params.length];

                        for (int j = 0; j < this.trainable_params.length; j++) {
                            if (j == i) {
                                interpolation_vector[j] = this.trainable_params[j] + this
                                    .cobyla_conf.rho;
                            } else {
                                interpolation_vector[j] = this.trainable_params[j] - (
                                    this.cobyla_conf.rho / 2);
                            }
                        }

                        this.interpolation_vectors ~= interpolation_vector;
                    }
                    break;
                default:
                    assert(false, "This should never ever happen");
                }

                this.b = new real[this.interpolation_vectors.length + 1];
                real cur_energy = cost(this.trainable_params);
                this.b[0] = cur_energy;
                int interpolation_vec_idx = 0;
                for (int i = 1; i < this.b.length; i++) {
                    real energy = cost(this.interpolation_vectors[interpolation_vec_idx]);
                    this.b[i] = energy;
                    interpolation_vec_idx++;
                }
                writefln("Iteration: %d | Energy/cost: %f", iter, cur_energy);
                this.trainable_params = cobyla(cur_energy);
            }
        }
    }
}
