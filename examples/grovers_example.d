module examples.grovers;

import std.stdio;

import algos.grovers;

// Grovers algorithm is a quantum algorithm which gives a substantial 
// speedup to unstructured seach problems. That is instead of requiring 
// N/2 queries to a function to find the answer where N is the number of 
// possible answers, it instead only requires sqrt(N). Therefore the algorithm
// can run in O(sqrt(N)).

/**
* The function that grovers algorithm will query to identify the
* correct answer
*
* params:
* guess = The bitstring which is being queried
*/
int f(string guess) {
    if (guess == "010") {
        return 1;
    }

    return 0;
}

void main() {
    // The two examples below should return the same state as the 
    // final answer

    // Example 1:

    // Intiialize a grovers circuit with 3 qubits
    Grovers g = Grovers(3);

    // Call grovers algorithm passing in the function to be queried
    // and print out the results
    writeln("Example 1 result: ", g.grovers(&f));

    //--------------------------------------------------------------

    // Example 2:

    // Initialize a grovers circuit with 3 qubits but 
    // instead of using the default oracle and diffusion
    // operator, instead use ones with the gate decomposition.
    // The default oracle and diffusion operator do not use gates
    // in their implementation.
    Grovers g2 = Grovers(3, OperatorType.Decomposition);

    // Call grovers algorithm passing in the function to be queried
    // and print out the results
    writeln("Example 2 result: ", g2.grovers(&f));
}
