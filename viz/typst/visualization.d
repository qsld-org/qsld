module viz.typst.visualization;

import std.stdio;
import std.file;
import std.format;
import std.process;

import std.conv : to;
import std.typecons : Tuple;

import core.stdc.stdlib : exit;

struct Visualization {
    Tuple!(string, int[], int)[] vis_arr;
    int num_qubits;
    int initial_state_idx;

    /**
    * The constructor for the type that allows for drawing the circuit
    * 
    * params:
    * vis_arr = The array of gates that the user calls as functions in the main program.
    *           This is generated internally by the quantum.qc module.
    * 
    * num_qubits = The number of qubits in the system 
    */
    this(Tuple!(string, int[], int)[] vis_arr, int num_qubits, int initial_state_idx) {
        this.vis_arr = vis_arr;
        this.num_qubits = num_qubits;
        this.initial_state_idx = initial_state_idx;
    }

    // Converts the qubit indices provided in the gate functions to usable strings
    // in typst
    private string qubit_idxs_to_string(int[] qubit_idxs) {
        assert(qubit_idxs.length >= 1, "You must specify qubit indices to gates");
        if (qubit_idxs.length == 1) {
            return to!string(qubit_idxs[0]);
        }

        string qubit_idxs_string = "(";
        for (int i = 0; i < qubit_idxs.length; i++) {
            if (i < cast(int) qubit_idxs.length - 1) {
                qubit_idxs_string ~= format("%d, ", qubit_idxs[i]);
            } else {
                qubit_idxs_string ~= format("%d", qubit_idxs[i]);
            }
        }
        qubit_idxs_string ~= ")";

        return qubit_idxs_string;
    }

    /**
    * Parses the entire vis_arr and writes the typst format to a file
    *
    * params:
    * filename = The name of the file to write the latex output to
    */
    void parse_and_write_vis_arr(string filename) {
        append(filename, "#import \"@preview/quill:0.7.2\" as quill: tequila as tq\n");
        append(filename, "#import \"@preview/physica:0.9.8\" as phys\n");
        append(filename, "#set page(width: auto, height: auto, margin: 0.5cm)\n");
        append(filename, "#quill.quantum-circuit(\n");

        for (int i = 0; i < this.num_qubits; i++) {
            int qubit_val = this.initial_state_idx & (1 << i);
            string qubit_state = format("quill.lstick(phys.ket($%d$), x: %d, y: %d),\n",
                (qubit_val >> i), 0, i);
            append(filename, qubit_state);
        }

        append(filename, "..tq.build(\n");

        foreach (i, item; this.vis_arr) {
            string gate_name = item[0];
            int[] qubit_idxs = item[1];
            int _timestep = item[2];

            switch (gate_name) {
            case "H":
                string qubit_idxs_string = qubit_idxs_to_string(qubit_idxs);
                append(filename, format("tq.h(%s),\n", qubit_idxs_string));
                break;
            case "X":
                string qubit_idxs_string = qubit_idxs_to_string(qubit_idxs);
                append(filename, format("tq.x(%s),\n", qubit_idxs_string));
                break;
            case "Y":
                string qubit_idxs_string = qubit_idxs_to_string(qubit_idxs);
                append(filename, format("tq.y(%s),\n", qubit_idxs_string));
                break;
            case "Z":
                string qubit_idxs_string = qubit_idxs_to_string(qubit_idxs);
                append(filename, format("tq.z(%s),\n", qubit_idxs_string));
                break;
            case "S":
                string qubit_idxs_string = qubit_idxs_to_string(qubit_idxs);
                append(filename, format("tq.s(%s),\n", qubit_idxs_string));
                break;
            case "T":
                string qubit_idxs_string = qubit_idxs_to_string(qubit_idxs);
                append(filename, format("tq.s(%s),\n", qubit_idxs_string));
                break;
            case "CX":
                int[] control = [qubit_idxs[0]];
                int[] target = [qubit_idxs[1]];
                string control_qubit = qubit_idxs_to_string(control);
                string target_qubit = qubit_idxs_to_string(target);
                append(filename, format("tq.cx(%s, %s),\n", control_qubit, target_qubit));
                break;
            case "CH":
                int[] control = [qubit_idxs[0]];
                int[] target = [qubit_idxs[1]];
                string control_qubit = qubit_idxs_to_string(control);
                string target_qubit = qubit_idxs_to_string(target);
                string gate_str = format(
                    "tq.multi-controlled-gate((%s,), %s, (x: 0, y: 0) => quill.gate($H$, x: x, y: y)),\n",
                    control_qubit,
                    target_qubit
                );
                append(filename, gate_str);
                break;
            case "CZ":
                int[] control = [qubit_idxs[0]];
                int[] target = [qubit_idxs[1]];
                string control_qubit = qubit_idxs_to_string(control);
                string target_qubit = qubit_idxs_to_string(target);
                append(filename, format("tq.cz(%s, %s),\n", control_qubit, target_qubit));
                break;
            case "SWAP":
                int[] qubit1 = [qubit_idxs[0]];
                int[] qubit2 = [qubit_idxs[1]];
                string qubit1_str = qubit_idxs_to_string(qubit1);
                string qubit2_str = qubit_idxs_to_string(qubit2);
                append(filename, format("tq.swap(%s, %s),\n", qubit1_str, qubit2_str));
                break;
            case "iSWAP":
                int[] qubit1 = [qubit_idxs[0]];
                int[] qubit2 = [qubit_idxs[1]];
                string qubit1_str = qubit_idxs_to_string(qubit1);
                string qubit2_str = qubit_idxs_to_string(qubit2);
                string gate_str = format(
                    "tq.multi-controlled-gate((%s,), %s, (x: 0, y: 0) => quill.gate($i\"SWAP\"$, x: x, y: y)),\n",
                    qubit1_str,
                    qubit2_str
                );
                append(filename, gate_str);
                break;
            case "R_X":
                string qubit_idxs_str = qubit_idxs_to_string(qubit_idxs);
                append(filename, format("tq.rx($θ$, %s),\n", qubit_idxs_str));
                break;
            case "R_Y":
                string qubit_idxs_str = qubit_idxs_to_string(qubit_idxs);
                append(filename, format("tq.ry($θ$, %s),\n", qubit_idxs_str));
                break;
            case "R_Z":
                string qubit_idxs_str = qubit_idxs_to_string(qubit_idxs);
                append(filename, format("tq.rz($θ$, %s),\n", qubit_idxs_str));
                break;
            case "MCZ":
                int[] controls = qubit_idxs[0 .. ($ - 1)];
                int[] target = [qubit_idxs[cast(int) qubit_idxs.length - 1]];
                string control_qubits = qubit_idxs_to_string(controls);
                string target_qubit = qubit_idxs_to_string(target);
                if (controls.length == 1) {
                    string gate_str = format(
                        "tq.multi-controlled-gate((%s,), %s, (x: 0, y: 0) => quill.gate($Z$, x: x, y: y)),\n",
                        control_qubits,
                        target_qubit
                    );
                    append(filename, gate_str);
                } else {
                    string gate_str = format(
                        "tq.multi-controlled-gate(%s, %s, (x: 0, y: 0) => quill.gate($Z$, x: x, y: y)),\n",
                        control_qubits,
                        target_qubit
                    );
                    append(filename, gate_str);
                }
                break;
            case "BAR":
                append(filename, "tq.barrier(),\n");
                break;
            case "M":
                string qubit_idxs_str = qubit_idxs_to_string(qubit_idxs);
                append(filename, format("tq.meter(%s),\n", qubit_idxs_str));
                break;
            case "MA":
                string qubit_idxs_str = qubit_idxs_to_string(qubit_idxs);
                append(filename, format("tq.meter(%s),\n", qubit_idxs_str));
                break;
            default:
                assert(false, format("Unrecognized or unimplemented gate: %s", gate_name));
            }
        }

        append(filename, "),\n");
        append(filename, ")\n");
    }

    /**
    * Compiles the typst file output by the parse_and_write_vis_arr() function
    *
    * params:
    * filename = The name of the file to compile ending with an extension .typ
    */
    void compile_typst(string filename) {
        auto output_file = File("/dev/null", "w");
        auto typst_compilation_pid = spawnProcess(["typst", "compile", filename], std.stdio.stdin, output_file, output_file);

        if (wait(typst_compilation_pid) != 0) {
            writeln("The compilation of the typst file with name ", filename, " failed");
            exit(1);
        }

        remove(filename);
    }

    /**
    * Convert the pdf generated by compile_tex_and_cleanup to a png image
    * 
    * params: 
    * input_filename = The name of the pdf file to convert with no extension
    * 
    * output_filename = The name of the png file to be outputted with no extension
    */
    void convert_pdf_to_png(string input_filename, string output_filename) {
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
}
