module quantum.utils;

import std.complex;
import std.math;

import linalg.vector;
import linalg.matrix;

float state_fidelity(Vector!(Complex!real) sv1, Vector!(Complex!real) sv2) {
    Vector!(Complex!real) sv1_dagger = sv1.dagger().get_cols()[0];
    Complex!real inner_prod = sv1_dagger.dot(sv2);
    return norm(inner_prod);
}

float state_fidelity(Vector!(Complex!real) sv, Matrix!(Complex!real) dm) {
    Vector!(Complex!real) result = dm.mult_vec(sv);
    Vector!(Complex!real) sv_dagger = sv.dagger().get_cols()[0];
    real inner_prod = sv_dagger.dot(result).re;
    return inner_prod;
}

float purity(Matrix!(Complex!real) dm) {
    Matrix!(Complex!real) dm_squared = dm.mult_mat(dm);
    return dm_squared.trace();
}
