module linalg.vector;

import std.stdio;
import std.math;
import std.complex;

import linalg.matrix;

struct Vector(T) {
    int dim;
    T[] elems;

    /**
    * Constructor for the Vector type
    *
    * params:
    * dimension = the amount of elements to be in the vector
    * args = the elements in the vector as an array
    */
    this(int dimension, T[] args) {
        this.dim = dimension;
        this.elems = args;
    }

    /**
    * Adds two vector objects together
    *
    * params:
    * v2 = the vector to add
    *
    * returns: A new vector
    */
    Vector add(Vector v2) {
        assert(this.dim == v2.dim, "Vectors must be equal in dimension to add");

        T[] elems = new T[this.dim];
        Vector!T v3 = Vector!T(this.dim, elems);

        foreach (i, elem; this.elems) {
            T result = elem + v2.elems[i];
            v3.elems[i] = result;
        }

        return v3;
    }

    /**
    * Operator overload for add() function, for operator +
    *
    * params:
    * rhs = the vector on the right hand side of the operator
    *
    * returns: A new vector
    */
    Vector opBinary(string s : "+")(Vector rhs) {
        return this.add(rhs);
    }

    /**
    * Subtracts two vectors
    *
    * params:
    * v2 = The vector to subtract
    * 
    * returns: A new vector
    */
    Vector sub(Vector v2) {
        assert(this.dim == v2.dim, "Vectors must be equal in dimension to subtract");

        T[] elems = new T[this.dim];
        Vector!T v3 = Vector!T(this.dim, elems);

        foreach (i, elem; this.elems) {
            T result = elem - v2.elems[i];
            v3.elems[i] = result;
        }

        return v3;
    }

    /**
    * Operator overload for sub() function, for operator -
    * 
    * params:
    * rhs = vector on the right hand side of the operator
    *
    * returns: A new vector
    */
    Vector opBinary(string s : "-")(Vector rhs) {
        return this.sub(rhs);
    }

    /**
    * Takes the dot product of two vectors
    *
    * params:
    * v2 = the vector to dot product with
    *
    * returns: A singular value of type T
    */
    T dot(Vector v2) {
        assert(this.dim == v2.dim, "Vectors must be equal in dimension to take the dot product");

        T sum = 0;

        foreach (i, elem; this.elems) {
            T result = elem * v2.elems[i];
            sum = sum + result;
        }

        return sum;
    }

    /**
    * Operator overload for dot() function, for operator *
    * 
    * params:
    * rhs = the vector to dot product with
    *
    * returns: A singular value of type T
    */
    T opBinary(string s : "*")(Vector rhs) {
        return this.dot(rhs);
    }

    /**
    * Multiplies a scalar and a vector together 
    *
    * params:
    * scalar = A scalar value of type T
    *
    * returns: A new vector
    */
    Vector mult(T scalar) {
        T[] elems = new T[this.dim];
        Vector!T v = Vector!T(this.dim, elems);

        foreach (i, elem; this.elems) {
            v.elems[i] = elem * scalar;
        }

        return v;
    }

    /**
    * Operator overload for mult() function with scalar on left hand side
    * 
    * params:
    * lhs = the scalar value on the left hand side
    *
    * returns: A new vector
    */
    Vector opBinaryRight(string s : "*")(T lhs) {
        return this.mult(lhs);
    }

    /**
    * Operator overload for mult() function with scalar on the right hand side
    *
    * params:
    * rhs = the scalar value on the right hand side
    * 
    * returns: A new vector
    */
    Vector opBinary(string s : "*")(T rhs) {
        return this.mult(rhs);
    }

    /**
    * Append a value to a vector 
    *'
    *  params:
    * elem = A value to append to the vector of type T
    */
    void append(T elem) {
        this.elems[this.elems.length++] = elem;
        this.dim = cast(int) this.elems.length;
    }

    /**
    * Clear a vector of all its elements
    */
    void clear() {
        this.elems = [];
        this.elems.length = 0;
        this.dim = 0;
    }

    /**
    * Takes the transpose of the vector
    * 
    * returns: A matrix of nx1 dimensions where n is the dimensions of the original vector
    */
    Matrix!T transpose() {
        Matrix!T mat = Matrix!T(this.dim, 1, []);

        foreach (elem; this.elems) {
            mat.append(Vector!T(1, [elem]));
        }

        return mat;
    }

    /**
    * Takes the complex conjugate transpose of the vector
    *
    * returns: A matrix of nx1 dimensions with the signs of complex values inverted
    */
    Matrix!T dagger()() if (is(T == Complex!real)) {
        Vector!T v = Vector!T(this.dim, this.elems.dup);
        foreach (i, elem; v.elems) {
            v[i] = conj(elem);
        }

        return v.transpose();
    }

    /**
    * Takes the magnitude of a complex valued vector
    *
    * returns: A real number representing the magnitude
    */
    real mag()() if (is(T == Complex!real) || is(T == real)) {
        real sum = 0;
        foreach (elem; this.elems) {
            sum = sum + pow(abs(elem), 2);
        }
        return sqrt(sum);
    }

    /**
    * Takes the outer product of two vectors. The column vector must be represented as a matrix here
    * for clarity.
    *
    * params:
    * col_vec = Some column vector as a Matrix!T type
    *
    * returns: A matrix
    */
    Matrix!T outer_prod(Matrix!T col_vec) {
        assert(col_vec.col_num == 1, "The input matrix must have one column, to represent a column vector");

        Vector!T row = Vector!T(cast(int) this.elems.length, []);
        Matrix!T result = Matrix!T(cast(int) this.elems.length, col_vec.row_num, [
            ]);

        foreach (elem; this.elems) {
            foreach (col_vec_row; col_vec.rows) {
                T res = elem * col_vec_row[0];
                row.append(res);
            }
            result.append(row);
            row.clear();
        }
        return result;
    }

    // Array operator overloading begin
    T opIndex(size_t i) const {
        return this.elems[i];
    }

    void opIndexAssign(T value, size_t i) {
        this.elems[i] = value;
    }

    size_t length() const {
        return this.dim;
    }
    // Array operator overloading end
}
