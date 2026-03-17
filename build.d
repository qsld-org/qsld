#!/usr/bin/rdmd

import std.stdio;
import std.getopt;
import std.format;
import std.file;
import std.process;
import std.array;
import std.algorithm;

import core.stdc.stdlib : exit;

import std.algorithm.searching : canFind;

string compiler = "dmd";
string[] examples = [];
bool force = false;
bool cleanup = false;

string[] library_files = [
    "linalg/vector.d",
    "linalg/matrix.d",
    "quantum/pure_state/qc.d",
    "quantum/pure_state/observable.d",
    "quantum/pure_state/decoherence.d",
    "quantum/pure_state/gate_noise.d",
    "quantum/impure_state/qc.d",
    "quantum/impure_state/observable.d",
    "quantum/impure_state/decoherence.d",
    "quantum/impure_state/gate_noise.d",
    "algos/qft.d",
    "algos/deutsch_jozsa.d",
    "algos/grovers.d",
    "algos/shors.d",
    "algos/bb84.d",
    "algos/quantum_teleportation.d",
    "algos/dense_coding.d",
    "viz/visualization.d",
    "viz/bloch_sphere/pure_state/bloch_sphere.d",
    "viz/bloch_sphere/impure_state/bloch_sphere.d",
    "viz/bloch_sphere/lib.d",
    "qec/stabilizer.d",
    "qec/decoder.d",
    "qec/lib.d",
    "topological/lib.d",
    "topological/pure_state/tqc.d",
    "vqe/vqe.d"
];

void compile(string[] compile_cmd, bool library = true, string example_name = "") {
    auto compilation_pid = spawnProcess(compile_cmd);
    if (wait(compilation_pid) != 0) {
        if (library) {
            writeln("Compilation of libqsld.a failed");
        } else {
            writeln("Compilation of example with name ", example_name, " failed");
        }
        exit(1);
    }
}

// overload to pass const string[]
void compile(const string[] compile_cmd, bool library = true, string example_name = "") {
    auto compilation_pid = spawnProcess(compile_cmd);
    if (wait(compilation_pid) != 0) {
        if (library) {
            writeln("Compilation of libqsld.a failed");
        } else {
            writeln("Compilation of example with name ", example_name, " failed");
        }
        exit(1);
    }
}

void remove_o_files() {
    foreach (entry; dirEntries(".", SpanMode.shallow)) {
        if (entry.isDir()) {
            continue;
        }

        if (entry.name.endsWith(".o")) {
            remove(entry.name);
        }
    }
}

void main(string[] args) {
    auto help_info = getopt(args,
        "c|compiler", "specify the compiler to use (default: dmd)", &compiler,
        "e|example", "specify an example or examples to build", &examples,
        "f|force", "force recompile", &force,
        "d|delete", "cleanup any binary, object and library files in the root of the project or examples/", &cleanup,
    );

    if (help_info.helpWanted) {
        defaultGetoptPrinter("QSLD build script", help_info.options);
        exit(0);
    }

    if (cleanup) {
        chdir("./examples");
        foreach (entry; dirEntries(".", SpanMode.shallow)) {
            if (entry.isDir()) {
                continue;
            }

            if (entry.name.endsWith(".o") || entry.name.canFind("_example") && !entry.name.endsWith(
                    ".d")) {
                remove(entry.name);
            }
        }
        writeln("cleanup finished!");
        exit(0);
    }

    string[] flags;
    if (compiler == "ldc2") {
        flags = ["-lib", "-O", "-of=libqsld.a", "--oq"];
    } else {
        flags = ["-lib", "-O", "-of=libqsld.a"];
    }

    const string[] LIBRARY_BUILD_COMMAND = [compiler] ~ flags ~ library_files;
    string f = "libqsld.a";
    if (!force && !f.exists) {
        compile(LIBRARY_BUILD_COMMAND);
        if (compiler == "ldc2") {
            remove_o_files();
        }
    } else if (force && f.exists) {
        compile(LIBRARY_BUILD_COMMAND);
        if (compiler == "ldc2") {
            remove_o_files();
        }
    } else if (!force && f.exists) {
        writeln("The library file libqsld.a already exists...skipping, please use -f or --force or remove the existing file to recompile it");
    } else {
        compile(LIBRARY_BUILD_COMMAND);
        if (compiler == "ldc2") {
            remove_o_files();
        }
    }

    foreach (example; examples) {
        string bin_name_path = format("./examples/%s", example);
        string src_name_path = format("./examples/%s.d", example);

        string[] compile_example_cmd = [
            compiler, "-O", "-L-L.", "-L=-lqsld",
            "-of=" ~ bin_name_path,
            src_name_path,
        ];

        if (!src_name_path.exists) {
            writeln("The example source file with the name ", src_name_path, " does not exist...skipping");
            continue;
        }

        if (!force && !bin_name_path.exists) {
            compile(compile_example_cmd, false, src_name_path);
        } else if (force && f.exists) {
            compile(compile_example_cmd, false, src_name_path);
        } else if (!force && f.exists) {
            writeln("The example binary with name ", bin_name_path, " already exists...skipping, use -f or --force or remove the existing file to recompile it");
            continue;
        } else {
            compile(compile_example_cmd, false, src_name_path);
        }
    }
}
