module viz.bloch_sphere.pure_state.bloch_sphere;

import std.stdio;
import std.complex;
import std.format;
import std.file;
import std.array;

import std.typecons : Tuple;

import linalg.vector;
import linalg.matrix;

import quantum.pure_state.qc;

import viz.bloch_sphere.lib;

struct BlochSphere {
    QuantumCircuit* qc;

    this(QuantumCircuit* qc) {
        this.qc = qc;
    }

    private Matrix!(Complex!real) get_qubit_rho(int qubit_idx) {
        Matrix!(Complex!real) rho = qc.get_rho();
        Matrix!(Complex!real) rho_prime = rho;
        for (int i = qc.num_qubits - 1; i > qubit_idx; i--) {
            rho_prime = get_reduced_density_matrix(rho_prime, i);
        }

        for (int i = qubit_idx - 1; i >= 0; i--) {
            rho_prime = get_reduced_density_matrix(rho_prime, i);
        }

        return rho_prime;
    }

    void draw_bloch_sphere(int qubit_idx, string compiler = "pdflatex", bool remove_pdf_file = true) {
        Matrix!(Complex!real) rho_reduced = get_qubit_rho(qubit_idx);
        normalize_rho(rho_reduced);

        string filename = format("qubit%d_bloch.tex", qubit_idx);

        string file_content = "
            \\documentclass{standalone}

            \\usepackage{blochsphere}
            \\usepackage{braket}

            \\begin{document}

            \\begin{blochsphere}[radius=1.5 cm,tilt=15,rotation=-20,opacity=0,statecolor=yellow]
                \\drawBallGrid[style={opacity=0.1}]{30}{30}
                \\labelLatLon{up}{90}{0};
                \\labelLatLon{down}{-90}{90};
                \\node[above] at (up) {\\tiny $\\ket{0}$};
                \\node[below] at (down) {\\tiny $\\ket{1}$};

                \\drawAxis{0}{0}

                \\drawAxis{90}{90}
                \\labelPolar{ylabel}{90}{90}
                \\node[above] at (ylabel) {\\tiny $y$};

                \\drawAxis{90}{0}
                \\labelPolar{xlabel}{90}{0}
                \\node[above] at (xlabel) {\\tiny $x$};

                \\drawStatePolar{psivec}{%f}{%f}
            \\end{blochsphere}
        \\end{document}
        ";

        Tuple!(real, real) angles = get_bloch_vector_angles(rho_reduced);
        string formatted_content = format(file_content, angles[0], angles[1]);
        append(filename, formatted_content);

        compile_tex_and_cleanup(compiler, filename);

        string[] filename_split = filename.split(".");
        string filename_prefix = filename_split[0];
        convert_pdf_to_png(filename_prefix, filename_prefix);

        string pdf_fname = filename_prefix ~ ".pdf";
        if (remove_pdf_file) {
            if (exists(pdf_fname)) {
                remove(pdf_fname);
            }
        }
    }
}
