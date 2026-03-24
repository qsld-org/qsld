module examples.deutsch_jozsa;

import algos.deutsch_jozsa;

import std.stdio;

void main() {
    // In these examples I count the number of times measurement gave a specific result because it
    // is easier to see the result of the algorithm due to the statistical invariance which occurs
    // due to noise in real quantum computing and floating point inaccuracies here. 

    // Example 1:

    // Initialize the Deutsch-Jozsa algorithm circuit with 2 qubits.
    // This will be increased wihtin the code of the circuit and used 
    // to know whether the function is constant or balanced
    DeutschJozsa dj = DeutschJozsa(2);

    // call the algorithm on a constant function where f(x) = 0.
    // It is neccessary to provide the type of function that will be used
    // in order to use the right oracle to modify the quantum state based
    // on various factors.
    int[string] counts = dj.deutsch_jozsa(&dj.f_constant_0, FunctionType.Constant);

    writeln("Counts for Example 1 (f(x) = 0): ", counts);

    //-------------------------------------------------------------------------

    // Example 2:

    // Initialize another Deutsch Jozsa quantum circuit with 2 qubits
    DeutschJozsa dj2 = DeutschJozsa(2);

    // call the algorithm with a constant function (f(x) = 1)
    int[string] counts2 = dj2.deutsch_jozsa(&dj2.f_constant_1, FunctionType.Constant);

    writeln("Counts for Example 2 (f(x) = 1): ", counts2);

    //---------------------------------------------------------------------------

    // Example 3:

    // Initialize another Deutsch Jozsa quantum circuit with 2 qubits
    DeutschJozsa dj3 = DeutschJozsa(2);

    // call the algorithm on a balanced function
    int[string] counts3 = dj3.deutsch_jozsa(&dj3.f_balanced, FunctionType.Balanced);

    writeln("Counts for Example 3 (balanced): ", counts3);

}
