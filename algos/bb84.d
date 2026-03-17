module algos.bb84;

import std.stdio;
import std.random;
import std.array;
import std.conv;
import std.range;
import std.format;

import std.math : round;
import std.algorithm : canFind, map;

import quantum.pure_state.qc;
import quantum.pure_state.gate_noise;

struct BB84 {
    QuantumCircuit qc;
    int num_qubits;
    bool check_compromised;
    float error_threshold;
    int[] a_bits;
    int[] a_bases;
    int[] b_bits;
    int[] b_bases;
    int[] e_bits;
    int[] e_bases;

    /**
    * Constructor for the BB84 protocol object, to be used when
    * calling the ideal implementaion of the protocol
    *
    * params:
    * num_qubits = The number of qubits to use with the protocol
    */
    this(int num_qubits) {
        this.num_qubits = num_qubits;

        this.qc = QuantumCircuit(this.num_qubits);

        this.error_threshold = 0.0;

        this.a_bits = new int[this.num_qubits];
        this.a_bases = new int[this.num_qubits];

        this.b_bits = new int[this.num_qubits];
        this.b_bases = new int[this.num_qubits];

        this.e_bits = new int[this.num_qubits];
        this.e_bases = new int[this.num_qubits];

        this.check_compromised = false;
    }

    /**
    * Constructor for the BB84 protocol object, to be used when
    * calling the intercept resend or any noisy implementaion of 
    * the protocol with Eve
    *
    * params:
    * num_qubits = The number of qubits to use with the protocol
    * 
    * error_threshold = The amount of permittable errors in the key after sampling
    */
    this(int num_qubits, float error_threshold) {
        this.num_qubits = num_qubits;

        this.qc = QuantumCircuit(this.num_qubits);

        this.error_threshold = error_threshold;

        this.a_bits = new int[this.num_qubits];
        this.a_bases = new int[this.num_qubits];

        this.b_bits = new int[this.num_qubits];
        this.b_bases = new int[this.num_qubits];

        this.e_bits = new int[this.num_qubits];
        this.e_bases = new int[this.num_qubits];

        this.check_compromised = true;
    }

    /**
    * Constructor for the BB84 protocol object, to be used when
    * calling the intercept resend or any noisy implementaion of 
    * the protocol with Eve. However, the key compromisation check is
    * configurable.
    *
    * params:
    * num_qubits = The number of qubits to use with the protocol
    * 
    * error_threshold = The amount of permittable errors in the key after sampling
    * 
    * check_compromised = Whether to check if the key generated has an error rate above the error 
    *                     threshold or not.
    */
    this(int num_qubits, float error_threshold, bool check_compromised) {
        this.num_qubits = num_qubits;

        this.qc = QuantumCircuit(this.num_qubits);

        this.error_threshold = error_threshold;

        this.a_bits = new int[this.num_qubits];
        this.a_bases = new int[this.num_qubits];

        this.b_bits = new int[this.num_qubits];
        this.b_bases = new int[this.num_qubits];

        this.e_bits = new int[this.num_qubits];
        this.e_bases = new int[this.num_qubits];

        this.check_compromised = check_compromised;
    }

    // Generates a random array of bits for all of the 
    // bit and bases arrays
    private int[] generate_rand_bits() {
        auto rng = Random(unpredictableSeed);
        int[] rand_bits = iota(this.num_qubits).map!(i => uniform(0, 2, rng)).array;
        return rand_bits;
    }

    // Generates the biased list of Eve's basis to use during a biased basis attack
    private void generate_eve_bases_biased(double basis_bias) {
        assert(basis_bias >= 0 && basis_bias <= 1,
            "The basis bias should be a decimal between 0 and 1, it is not");

        auto rng = Random(unpredictableSeed);
        for (int i = 0; i < this.num_qubits; i++) {
            double rand_double = uniform(0, 2, rng);
            if (rand_double < basis_bias) {
                this.e_bases[i] = 1;
            } else {
                this.e_bases[i] = 0;
            }
        }
    }

    // Represents Alice preparing the qubits to send to Bob
    private void alice_prepare() {
        foreach (i, bit; this.a_bits) {
            if (this.a_bases[i] == 0) { // z basis
                if (bit == 1) {
                    this.qc.pauli_x(cast(int) i);
                }
            } else if (this.a_bases[i] == 1) { // x basis
                if (bit == 0) {
                    this.qc.hadamard(cast(int) i);
                } else if (bit == 1) {
                    this.qc.pauli_x(cast(int) i);
                    this.qc.hadamard(cast(int) i);
                }
            }
        }
    }

    // Represents Eve measuring the qubits sent by Alice in her randomly
    // selected bases
    private void eve_measure() {
        for (int i = 0; i < this.num_qubits; i++) {
            string outcome;
            if (this.e_bases[i] == 0) {
                outcome = this.qc.measure(i);
            } else if (this.e_bases[i] == 1) {
                this.qc.hadamard(i);
                outcome = this.qc.measure(i);
            }
            int e_bit = to!int(outcome);
            this.e_bits[i] = e_bit;
        }
    }

    // Represents Bob measuring the qubits in a certain basis
    private void bob_measure() {
        foreach (i, bit; this.a_bits) {
            if (this.b_bases[i] == 1) { // x basis
                this.qc.hadamard(cast(int) i);
            }

            string outcome = this.qc.measure(cast(int) i);
            int b_bit = to!int(outcome);
            this.b_bits[i] = b_bit;
        }
    }

    // Represents Eve repreparing the state to send to Bob
    private void eve_reprepare() {
        for (int i = 0; i < this.num_qubits; i++) {
            if (this.e_bases[i] == 0) {
                if (this.e_bits[i] == 1) {
                    this.qc.pauli_x(i);
                }
            } else if (this.e_bases[i] == 1) {
                if (this.e_bits[i] == 0) {
                    this.qc.hadamard(i);
                } else if (this.e_bits[i] == 1) {
                    this.qc.pauli_x(i);
                    this.qc.hadamard(i);
                }
            }
        }
    }

    // This function is only used for partial intercept resend attacks,
    // it does not adhere to the physical reality of the attack because 
    // Eve or the simulator is implied to have acces to Alices choices when 
    // they technically shouldn't but it maintains the important details and 
    // information
    private void eve_reprepare_partial(ulong[] taken_indices) {
        for (int i = 0; i < this.num_qubits; i++) {
            if (taken_indices.canFind(cast(ulong) i)) {
                if (this.e_bases[i] == 0) {
                    if (this.e_bits[i] == 1) {
                        this.qc.pauli_x(i);
                    }
                } else if (this.e_bases[i] == 1) {
                    if (this.e_bits[i] == 0) {
                        this.qc.hadamard(i);
                    } else if (this.e_bits[i] == 1) {
                        this.qc.pauli_x(i);
                        this.qc.hadamard(i);
                    }
                }
            } else {
                if (this.a_bases[i] == 0) { // z basis
                    if (this.a_bits[i] == 1) {
                        this.qc.pauli_x(i);
                    }
                } else if (this.a_bases[i] == 1) { // x basis
                    if (this.a_bits[i] == 0) {
                        this.qc.hadamard(i);
                    } else if (this.a_bits[i] == 1) {
                        this.qc.pauli_x(i);
                        this.qc.hadamard(i);
                    }
                }
            }
        }
    }

    // samples bits from the key and compares them to get the error rate in order to 
    // compare it to the error threshold
    private void check_key_compromised(int[] bits, int[] sifted_a_bits, int[] sifted_b_bits) {
        if (this.check_compromised) {
            int sample_size = cast(int)(sifted_a_bits.length * 0.4);

            auto rng = Random(unpredictableSeed);
            auto all_indices = iota(sifted_a_bits.length).array;
            all_indices = randomShuffle(all_indices, rng);
            auto sampled_indices = all_indices[0 .. sample_size];

            int mismatches = 0;
            foreach (idx; sampled_indices) {
                if (sifted_a_bits[idx] != sifted_b_bits[idx]) {
                    mismatches++;
                }
            }

            double error_rate = mismatches / cast(double) sample_size;

            int[] non_sample_bits;
            foreach (i, bit; sifted_b_bits) {
                if (!(sampled_indices.canFind(i))) {
                    non_sample_bits ~= bit;
                }
            }

            if (error_rate > this.error_threshold) {
                writeln("The shared key is: ", bits);
                writeln("The key after sampling would be: ", non_sample_bits);
                writeln(format("The error rate is %f so the key is probably compromised, dont use it", error_rate));
            } else {

                writeln("The error rate is acceptable");
                writeln("The new key after sampling is: ", non_sample_bits);
            }
        } else {
            writeln("The shared key is: ", bits);
        }
    }

    /** 
    * Executes the BB84 protocol where Eve uses the intercept 
    * resend attack to spy on Alice and Bob. Eve will measure every
    * qubit sent by Alice.
    */
    void bb84_intercept_resend_full() {
        this.a_bits = generate_rand_bits();
        this.a_bases = generate_rand_bits();

        this.e_bases = generate_rand_bits();

        this.b_bases = generate_rand_bits();

        // Alice prepares each qubit to be sent to Bob
        alice_prepare();

        // Eve measures each qubit sent by alice
        eve_measure();

        // reset the quantum circuit to represent eve collapsing each qubit
        // sent by alice
        this.qc = QuantumCircuit(this.num_qubits);

        // Eve reprepares the qubit to be sent to bob based
        // on what she measured
        eve_reprepare();

        // Bob randomly measures the reprepared qubits eve sent
        // in either the x or z basis
        bob_measure();

        // Alice and Bob communicate over the classical channel
        // and compare their basis of measurement
        ulong[] keep_indices;
        foreach (i, basis; this.a_bases) {
            if (basis == this.b_bases[i]) {
                keep_indices ~= i;
            }
        }

        // Alice and Bob eliminate the measurement basis that 
        // are different
        int[] bits;
        int[] sifted_a_bits;
        int[] sifted_b_bits;
        foreach (idx; keep_indices) {
            bits ~= this.b_bits[idx];
            sifted_a_bits ~= this.a_bits[idx];
            sifted_b_bits ~= this.b_bits[idx];
        }

        // This will only happen if this.check_compromised is true
        check_key_compromised(bits, sifted_a_bits, sifted_b_bits);
    }

    /**
    * Executes the BB84 protocol where Eve attacks only a fraction of 
    * the qubits sent by Alice based on the parameter provided.
    *
    * params:
    * attack_fraction = The fraction, as a decimal of qubits for Eve to 
    *                   attack and resend
    */
    void bb84_intercept_resend_partial(double attack_fraction) {
        this.a_bits = generate_rand_bits();
        this.a_bases = generate_rand_bits();

        this.b_bases = generate_rand_bits();

        this.e_bases = generate_rand_bits();

        // Alice prepares her qubits to be sent to Bob
        alice_prepare();

        // Eve only attacks certain qubits, this represents her choosing
        // the qubits randomly which to attack based on the attack_fraction
        real k = round(attack_fraction * this.num_qubits);
        auto rng = Random(unpredictableSeed);
        int[] indices = randomShuffle(iota(this.num_qubits).array, rng);

        ulong[] taken_indices;
        for (ulong i = 0; i < k; i++) {
            taken_indices ~= cast(ulong) indices[i];
        }

        // Eve only measures the qubits selected above
        for (int i = 0; i < this.num_qubits; i++) {
            if (taken_indices.canFind(cast(ulong) i)) {
                string outcome;
                if (this.e_bases[i] == 0) {
                    outcome = this.qc.measure(i);
                } else if (this.e_bases[i] == 1) {
                    this.qc.hadamard(i);
                    outcome = this.qc.measure(i);
                }
                int e_bit = to!int(outcome);
                this.e_bits[i] = e_bit;
            }
        }

        this.qc = QuantumCircuit(this.num_qubits);

        // Eve prepares the qubits again to send to Bob
        eve_reprepare_partial(taken_indices);

        // Bob gets the qubits from Eve and measures in his bases
        bob_measure();

        // Alice and Bob communicate over the classical channel
        // and compare their basis of measurement
        ulong[] keep_indices;
        foreach (i, basis; this.a_bases) {
            if (basis == this.b_bases[i]) {
                keep_indices ~= i;
            }
        }

        int[] bits;
        int[] sifted_a_bits;
        int[] sifted_b_bits;
        foreach (idx; keep_indices) {
            bits ~= this.b_bits[idx];
            sifted_a_bits ~= this.a_bits[idx];
            sifted_b_bits ~= this.b_bits[idx];
        }

        check_key_compromised(bits, sifted_a_bits, sifted_b_bits);
    }

    /**
    * Executes the BB84 protocol where Eve uses one basis more than 
    * the other based on the parameter provided.
    * 
    * params;
    * basis_bias = The amount that Eve will use one basis over the other as a decimal
    */
    void bb84_biased_basis_eve(double basis_bias) {
        this.a_bits = generate_rand_bits();
        this.a_bases = generate_rand_bits();

        this.b_bases = generate_rand_bits();

        generate_eve_bases_biased(basis_bias);

        // Alice prepares her qubits to be sent to Bob
        alice_prepare();

        // Eve measures each qubit sent by alice
        eve_measure();

        this.qc = QuantumCircuit(this.num_qubits);

        // Eve reprepares the qubits based on her measurements to send to Bob
        eve_reprepare();

        // Bob measures the qubits sent by Eve
        bob_measure();

        // Alice and Bob communicate over the classical channel
        // and compare their basis of measurement
        ulong[] keep_indices;
        foreach (i, basis; this.a_bases) {
            if (basis == this.b_bases[i]) {
                keep_indices ~= i;
            }
        }

        // Alice and Bob eliminate the measurement bases that 
        // are different
        int[] bits;
        int[] sifted_a_bits;
        int[] sifted_b_bits;
        foreach (idx; keep_indices) {
            bits ~= this.b_bits[idx];
            sifted_a_bits ~= this.a_bits[idx];
            sifted_b_bits ~= this.b_bits[idx];
        }

        check_key_compromised(bits, sifted_a_bits, sifted_b_bits);
    }

    /**
    * Executes the BB84 protocol where Eve always measures qubits sent
    * by Alice in the same basis based on the parameter provided
    * 
    * params:
    * basis = An integer representing the basis Eve will always measure in, 0 = z, 1 = x
    */
    void bb84_fixed_basis_eve(int basis) {
        // define the basis Eve will always measure in
        if (basis == 0) { // z basis
            for (int i = 0; i < this.e_bases.length; i++) {
                this.e_bases[i] = 0;
            }
        } else if (basis == 1) { // x basis
            for (int i = 0; i < this.e_bases.length; i++) {
                this.e_bases[i] = 1;
            }
        }

        this.a_bits = generate_rand_bits();
        this.a_bases = generate_rand_bits();

        this.b_bases = generate_rand_bits();

        // Alice prepares her qubits to be sent to Bob
        alice_prepare();

        if (basis == 0) {
            // Eve measures each qubit sent by Alice in the fixed z basis
            for (int i = 0; i < this.num_qubits; i++) {
                string outcome = this.qc.measure(i);
                int e_bit = to!int(outcome);
                this.e_bits[i] = e_bit;
            }
        } else if (basis == 1) {
            // Eve measures each qubit sent by Alice in the fixed x basis
            for (int i = 0; i < this.num_qubits; i++) {
                this.qc.hadamard(i);
                string outcome = this.qc.measure(i);
                int e_bit = to!int(outcome);
                this.e_bits[i] = e_bit;
            }
        }

        this.qc = QuantumCircuit(this.num_qubits);

        // Eve reprepares the qubit to be sent to Bob based
        // on what she measured
        eve_reprepare();

        // Bob randomly measures the reprepared qubits eve sent
        // in either the x or z basis
        bob_measure();

        // Alice and Bob communicate over the classical channel
        // and compare their basis of measurement
        ulong[] keep_indices;
        foreach (i, a_basis; this.a_bases) {
            if (a_basis == this.b_bases[i]) {
                keep_indices ~= i;
            }
        }

        int[] bits;
        int[] sifted_a_bits;
        int[] sifted_b_bits;
        foreach (idx; keep_indices) {
            bits ~= this.b_bits[idx];
            sifted_a_bits ~= this.a_bits[idx];
            sifted_b_bits ~= this.b_bits[idx];
        }

        check_key_compromised(bits, sifted_a_bits, sifted_b_bits);
    }

    /**
    * Executes the BB84 protocol but Eve is simulated as noise instead of 
    * measuring Alices qubits in her own randomly generated bases
    *
    * params:
    * depolarizing_prob = The probability of depolarizing noise being applied to a qubit
    */
    void bb84_eve_noise(float depolarizing_prob) {
        this.a_bits = generate_rand_bits();
        this.a_bases = generate_rand_bits();

        this.b_bases = generate_rand_bits();

        // Alice prepares her qubits to send to Bob
        alice_prepare();

        // Depolarizing noise is applied to each qubit to represent
        // Eve
        GateNoise gn = GateNoise(&this.qc);
        for (int i = 0; i < this.num_qubits; i++) {
            gn.depolarizing_noise(i, depolarizing_prob);
        }

        // Bob measures the possibly noisy qubits
        bob_measure();

        // Alice and Bob communicate over the classical channel
        // and compare their basis of measurement
        ulong[] keep_indices;
        foreach (i, a_basis; this.a_bases) {
            if (a_basis == this.b_bases[i]) {
                keep_indices ~= i;
            }
        }

        int[] bits;
        int[] sifted_a_bits;
        int[] sifted_b_bits;
        foreach (idx; keep_indices) {
            bits ~= this.b_bits[idx];
            sifted_a_bits ~= this.a_bits[idx];
            sifted_b_bits ~= this.b_bits[idx];
        }

        check_key_compromised(bits, sifted_a_bits, sifted_b_bits);
    }

    /**
    * Runs the BB84 protocol in an ideal scenario without errors or eve
    */
    void bb84_ideal() {
        this.a_bits = generate_rand_bits();
        this.a_bases = generate_rand_bits();

        this.b_bases = generate_rand_bits();

        // Alice prepares each qubit to be sent to Bob
        alice_prepare();

        // Bob randomly measures the qubit in either the x or z basis
        bob_measure();

        // Alice and Bob communcate over the classical channel
        // and compare their basis of measurement
        ulong[] keep_indices;
        foreach (i, basis; this.a_bases) {
            if (basis == this.b_bases[i]) {
                keep_indices ~= i;
            }
        }

        int[] bits;
        foreach (idx; keep_indices) {
            bits ~= this.b_bits[idx];
        }

        writeln("The shared key is: ", bits);
    }
}
