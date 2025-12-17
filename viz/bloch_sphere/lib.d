module viz.bloch_sphere.lib;

import std.stdio;
import std.format;
import std.complex;
import std.math;
import std.process;
import std.file;
import std.array;

import std.typecons : Tuple;

import core.stdc.stdlib : exit;

import linalg.vector;
import linalg.matrix;

package void normalize_rho(Matrix!(Complex!real) rho) {
    real trace = rho.trace();
    foreach (i, row; rho.rows) {
        foreach (j, elem; row.elems) {
            rho.rows[i].elems[j] = elem / trace;
        }
    }
}

package Matrix!(Complex!real) get_reduced_density_matrix(Matrix!(Complex!real) rho, int qubit_idx) {
    ulong num_bases = rho.rows.length / (cast(ulong) 2);

    Vector!(Complex!real)[] rows = new Vector!(Complex!real)[num_bases];
    foreach (i; 0 .. num_bases) {
        rows[i] = Vector!(Complex!real)((cast(int) num_bases), new Complex!real[num_bases]);

        for (int j = 0; j < rows[i].elems.length; j++) {
            rows[i].elems[j] = Complex!real(0, 0);
        }
    }

    Matrix!(Complex!real) rho_prime = Matrix!(Complex!real)((cast(int) num_bases), (
            cast(int) num_bases), rows);

    for (int row_idx = 0; row_idx < rho.rows.length; row_idx++) {
        for (int col_idx = 0; col_idx < rho.rows[row_idx].elems.length; col_idx++) {
            for (int k = 0; k < 2; k++) {
                int low_mask_row = (1 << qubit_idx) - 1;
                int low_bits_row = row_idx & low_mask_row;
                int high_bits_row = row_idx >> (qubit_idx + 1);
                int a = (high_bits_row << qubit_idx) | low_bits_row;

                int low_mask_col = (1 << qubit_idx) - 1;
                int low_bits_col = col_idx & low_mask_col;
                int high_bits_col = col_idx >> (qubit_idx + 1);
                int b = (high_bits_col << qubit_idx) | low_bits_col;

                int row_idx_updated = (high_bits_row << (qubit_idx + 1)) | (
                    k << qubit_idx) | low_bits_row;
                int col_idx_updated = (high_bits_col << (qubit_idx + 1)) | (
                    k << qubit_idx) | low_bits_col;

                rho_prime.rows[a].elems[b] += rho.rows[row_idx_updated].elems[col_idx_updated];
            }
        }
    }

    return rho_prime;
}

package Tuple!(real, real) get_bloch_vector_angles(Matrix!(Complex!real) rho) {
    real x = 2 * rho.rows[0].elems[1].re;
    real y = -2 * rho.rows[0].elems[1].im;
    real z = (rho.rows[0].elems[0] - rho.rows[1].elems[1]).re;

    real theta = acos(z) * (180 / PI);
    real phi = atan2(y, x) * (180 / PI);

    return Tuple!(real, real)(theta, phi);
}

package void compile_tex_and_cleanup(string compiler, string filename) {
    auto output_file = File("/dev/null", "w");
    auto tex_compilation_pid = spawnProcess([compiler, filename], std.stdio.stdin, output_file, output_file);
    if (wait(tex_compilation_pid) != 0) {
        writeln("The compilation of the the latex file with name ", filename, " failed");
        exit(1);
    }

    string[] filename_split = filename.split(".");
    string filename_no_ext = filename_split[0];

    remove(filename);
    remove(format("./%s.aux", filename_no_ext));
    remove(format("./%s.log", filename_no_ext));
}

package void convert_pdf_to_png(string input_filename, string output_filename) {
    assert(input_filename != "", "The filename of the pdf file to convert to png must be specified");
    assert(output_filename != "", "The filename of the converted file from pdf must be specified");

    input_filename ~= ".pdf";
    output_filename ~= ".png";

    string[] command = [
        "magick", "-density", "300", input_filename, "-background", "white",
        "-alpha", "remove", "-alpha", "off", "-resize", "1600x",
        output_filename
    ];

    auto pdf_conv_pid = spawnProcess(command);
    if (wait(pdf_conv_pid) != 0) {
        writeln("The conversion of the pdf file with name: ", input_filename, " to png failed");
        exit(1);
    }
}
