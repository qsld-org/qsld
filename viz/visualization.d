module viz.visualization;

import std.stdio;
import std.typecons;
import std.algorithm.searching;
import std.algorithm.sorting;
import std.process;
import std.format;
import std.array;
import std.file;

import core.stdc.stdlib : exit;
import std.algorithm : canFind;

struct Visualization {
    Tuple!(string, int[], int)[] vis_arr;
    int num_qubits;
    int initial_state_idx;
    string[][] lines;

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
        this.lines = [];
    }

    private void make_lines_equal_length() {
        for (int j = 0; j < this.lines.length; j++) {
            for (int k = 1; k < this.lines.length; k++) {
                if (j == k) {
                    break;
                }

                if (this.lines[j].length - 1 < this.lines[k].length - 1) {
                    while (this.lines[j].length - 1 < this.lines[k].length - 1) {
                        this.lines[j][this.lines[j].length++] = " \\qw &";
                    }
                } else if (this.lines[k].length - 1 < this.lines[j].length - 1) {
                    while (this.lines[k].length - 1 < this.lines[j].length - 1) {
                        this.lines[k][this.lines[k].length++] = " \\qw &";
                    }
                }
            }
        }
    }

    private void pad_other_gates(int[] qubit_idxs) {
        foreach (k; 0 .. this.num_qubits) {
            if (!qubit_idxs.canFind(k)) {
                this.lines[k][this.lines[k].length++] = " \\qw &";
            }
        }
    }

    /**
    * Parses the entire vis_arr and writes the latex format to a file
    *
    * params:
    * filename = The name of the file to write the latex output to
    */
    void parse_and_write_vis_arr(string filename) {
        // write beginning of file boilerplate
        append(filename, "\\documentclass{standalone}\n");
        append(filename, "\\usepackage{quantikz}\n");
        append(filename, "\\begin{document}\n");
        append(filename, "\\scalebox{1.8} {%\n");
        append(filename, "\\begin{quantikz}\n");

        for (int i = 0; i < this.num_qubits; i++) {
            int qubit_val = this.initial_state_idx & (1 << i);
            this.lines[lines.length++] = [
                format("\\lstick{\\ket{%d}} &", (qubit_val >> i))
            ];
        }

        foreach (i, item; this.vis_arr) {
            string gate_name = item[0];
            int[] qubit_idxs = item[1];
            int timestep = item[2];

            if (!gate_name.startsWith("C") && gate_name != "SWAP" && gate_name != "iSWAP" && gate_name != "TF") {
                if (gate_name != "M" && gate_name != "MA") {
                    if (gate_name == "R_X" || gate_name == "R_Y" || gate_name == "R_Z") {
                        this.lines[qubit_idxs[0]][this.lines[qubit_idxs[0]].length++] = format(" \\gate{%s(\\theta)} &", gate_name);
                    } else if (gate_name.startsWith("U")) {
                        qubit_idxs.sort();

                        int[] group = [qubit_idxs[0]];
                        int[][] groups;
                        if (qubit_idxs.length == 1) {
                            groups ~= [qubit_idxs[0]];
                        } else {
                            for (int j = 1; j < qubit_idxs.length; j++) {
                                if (qubit_idxs[j] == qubit_idxs[j - (cast(ulong) 1)] + 1) {
                                    group ~= qubit_idxs[j];
                                } else {
                                    groups ~= group;
                                    group = [qubit_idxs[j]];
                                }
                            }
                            groups ~= group;
                        }

                        make_lines_equal_length();

                        for (int j = 0; j < groups.length; j++) {
                            if (j != groups.length - (cast(ulong) 1)) {
                                if (groups[j].length > 1 && groups.length > 1) {

                                    ulong group_offset = groups[j].length;

                                    this.lines[groups[j][0]][this.lines[groups[j][0]].length++] = format(
                                        " \\gate[%d]{%s} &", group_offset, gate_name);
                                } else if (groups[j].length == 1 && groups.length > 1) {
                                    this.lines[groups[j][0]][this.lines[groups[j][0]].length++] = format(
                                        " \\gate{%s} &", gate_name);
                                }
                            } else {
                                if (groups[j].length > 1) {
                                    ulong group_offset = groups[j].length;

                                    this.lines[groups[j][0]][this.lines[groups[j][0]].length++] = format(
                                        " \\gate[%d]{%s} &", group_offset, gate_name);
                                } else {
                                    this.lines[groups[j][0]][this.lines[groups[j][0]].length++] = format(
                                        " \\gate{%s} &", gate_name);
                                }
                            }
                        }
                    } else if (gate_name.startsWith("slice")) {
                        string slice_tex = format(" \\%s &", gate_name);
                        this.lines[0][this.lines[0].length++] = slice_tex;
                    } else {
                        this.lines[qubit_idxs[0]][this.lines[qubit_idxs[0]].length++] = format(" \\gate{%s} &", gate_name);
                    }
                } else {
                    if (gate_name == "M") {
                        this.lines[qubit_idxs[0]][this.lines[qubit_idxs[0]].length++] = " \\meter{} &";
                    } else if (gate_name == "MA") {
                        foreach (idx; qubit_idxs) {
                            this.lines[qubit_idxs[idx]][this.lines[qubit_idxs[idx]].length++] = " \\meter{} &";
                        }
                    }
                }
            } else {
                make_lines_equal_length();

                switch (gate_name) {
                case "CX":
                    pad_other_gates(qubit_idxs);

                    this.lines[qubit_idxs[0]][this.lines[qubit_idxs[0]].length++] = format(" \\ctrl{%d} &", qubit_idxs[1] - qubit_idxs[0]);
                    this.lines[qubit_idxs[1]][this.lines[qubit_idxs[1]].length++] = " \\targ{} &";
                    break;
                case "TF":
                    int target_qubit = qubit_idxs[qubit_idxs.length - 1];

                    for (int j = 0; j < qubit_idxs.length - 1; j++) {
                        this.lines[qubit_idxs[j]][this.lines[qubit_idxs[j]].length++] = format(" \\ctrl{%d} &", target_qubit - qubit_idxs[j]);
                    }

                    pad_other_gates(qubit_idxs);

                    this.lines[target_qubit][this.lines[target_qubit].length++] = " \\targ{} &";
                    break;
                case "SWAP":
                    pad_other_gates(qubit_idxs);

                    this.lines[qubit_idxs[0]][this.lines[qubit_idxs[0]].length++] = format(" \\swap{%d} &", qubit_idxs[1] - qubit_idxs[0]);
                    this.lines[qubit_idxs[1]][this.lines[qubit_idxs[1]].length++] = " \\targX{} &";
                    break;
                default:
                    pad_other_gates(qubit_idxs);

                    this.lines[qubit_idxs[0]][this.lines[qubit_idxs[0]].length++] = format(" \\ctrl{%d} &", qubit_idxs[1] - qubit_idxs[0]);
                    this.lines[qubit_idxs[1]][this.lines[qubit_idxs[1]].length++] = format(" \\gate{%s} &", gate_name);
                }
            }
        }

        ulong max_len = 0;
        foreach (line; this.lines) {
            if (line.length > max_len)
                max_len = line.length;
        }

        foreach (idx, line; this.lines) {
            while (line.length < max_len) {
                line[line.length++] = " \\qw &";
            }

            if (idx != lines.length - cast(ulong) 1) {
                line[line.length++] = " \\qw \\\\\n";
            }
            string full_line = line.join();
            append(filename, full_line);
        }

        //write end of file boilerplate
        append(filename, "\\end{quantikz}\n");
        append(filename, "}\n");
        append(filename, "\\end{document}\n");
    }

    /**
    * Compiles the latex file output by the parse_and_write_vis_arr() function
    *
    * params:
    * filename = The name of the file to compile ending with an extension .tex
    */
    void compile_tex_and_cleanup(string compiler, string filename) {
        auto tex_compilation_pid = spawnShell(format("%s %s > /dev/null 2&>1", compiler, filename));

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
