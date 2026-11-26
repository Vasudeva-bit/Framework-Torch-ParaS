#include "pjrt_types.hpp"

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace paras_pjrt_api {
namespace {

bool is_mlir_bytecode(const char* code, std::size_t size) {
    return size >= 4 &&
           static_cast<unsigned char>(code[0]) == 'M' &&
           static_cast<unsigned char>(code[1]) == 'L' &&
           static_cast<unsigned char>(code[2]) == 0xEF &&
           static_cast<unsigned char>(code[3]) == 'R';
}

bool run_tool_to_text(const std::string& cmd, std::string& out_text) {
    FILE* pipe = popen(cmd.c_str(), "r");
    if (!pipe) return false;
    char buf[4096];
    out_text.clear();
    while (std::size_t n = std::fread(buf, 1, sizeof(buf), pipe)) {
        out_text.append(buf, n);
    }
    int rc = pclose(pipe);
    return rc == 0 && !out_text.empty();
}

bool deserialize_mlir_bytecode(const char* code, std::size_t size,
                               std::string& out_text, std::string& err_msg) {
    char tmp_path[] = "/tmp/paras_pjrt_mlir_XXXXXX";
    int fd = mkstemp(tmp_path);
    if (fd < 0) {
        err_msg = "mkstemp failed for MLIR bytecode";
        return false;
    }
    std::size_t written = 0;
    while (written < size) {
        ssize_t n = write(fd, code + written, size - written);
        if (n <= 0) {
            close(fd);
            unlink(tmp_path);
            err_msg = "failed writing MLIR bytecode temp file";
            return false;
        }
        written += static_cast<std::size_t>(n);
    }
    close(fd);

    const std::string py_cmd =
        std::string("python -c \"import sys; from jax._src.interpreters import mlir; "
                    "ctx = mlir.make_ir_context(); "
                    "data = open(sys.argv[1], 'rb').read(); "
                    "print(mlir.ir.Module.parse(data, ctx))\" ") +
        tmp_path + " 2>/dev/null";

    const std::vector<std::string> tools = {
        std::string("stablehlo-translate --deserialize ") + tmp_path + " 2>/dev/null",
        py_cmd,
        std::string("mlir-opt --allow-unregistered-dialect ") + tmp_path + " 2>/dev/null",
    };

    bool ok = false;
    for (const auto& cmd : tools) {
        if (run_tool_to_text(cmd, out_text)) {
            ok = true;
            break;
        }
    }
    unlink(tmp_path);

    if (!ok) {
        err_msg = "Received MLIR bytecode (ML\\xefR) but neither stablehlo-translate nor "
                  "mlir-opt could deserialize it on PATH.";
    }
    return ok;
}

bool parse_stablehlo_matmul(const std::string& mlir, MatMulPlan& plan, std::string& err_msg) {
    // Extract all 2D f32 tensor types from the function signature or dot_general op.
    std::vector<std::pair<int64_t, int64_t>> sig_tensors;
    {
        // Prefer func.func @main(...) -> ... line if present
        std::size_t fn_pos = mlir.find("func.func");
        std::string header = (fn_pos != std::string::npos)
            ? mlir.substr(fn_pos, mlir.find('{', fn_pos) - fn_pos)
            : mlir;

        static const std::regex re(R"(tensor<(\d+)x(\d+)xf32>)");
        auto begin = std::sregex_iterator(header.begin(), header.end(), re);
        auto end = std::sregex_iterator();
        for (auto it = begin; it != end; ++it) {
            sig_tensors.emplace_back(
                std::stoll((*it)[1].str()),
                std::stoll((*it)[2].str()));
        }
    }

    if (sig_tensors.size() < 3) {
        // Fallback: scan entire MLIR for 2D f32 tensor types
        sig_tensors.clear();
        static const std::regex re(R"(tensor<(\d+)x(\d+)xf32>)");
        auto begin = std::sregex_iterator(mlir.begin(), mlir.end(), re);
        auto end = std::sregex_iterator();
        for (auto it = begin; it != end; ++it) {
            sig_tensors.emplace_back(
                std::stoll((*it)[1].str()),
                std::stoll((*it)[2].str()));
        }
    }

    if (sig_tensors.size() < 3) {
        err_msg = "Expected at least 3 2D f32 tensor types (A, B, C) in StableHLO module";
        return false;
    }

    const auto a_shape = sig_tensors[0];
    const auto b_shape = sig_tensors[1];
    const auto c_shape = sig_tensors.back();

    // Check for dot_general or dot
    const bool has_dot =
        (mlir.find("dot_general") != std::string::npos) ||
        (mlir.find("stablehlo.dot") != std::string::npos);
    if (!has_dot) {
        err_msg = "StableHLO module does not contain dot_general / dot";
        return false;
    }

    // Default: standard A[M, K] @ B[K, N] -> C[M, N]
    int lhs_contract = 1;
    int rhs_contract = 0;

    // Parse pretty syntax: contracting_dims = [1] x [0]
    {
        static const std::regex re_pretty(
            R"(contracting_dims\s*=\s*\[(\d+)\]\s*x\s*\[(\d+)\])");
        std::smatch m;
        if (std::regex_search(mlir, m, re_pretty)) {
            lhs_contract = std::stoi(m[1].str());
            rhs_contract = std::stoi(m[2].str());
        }
    }
    // Parse generic syntax: lhs_contracting_dimensions = [1], rhs_contracting_dimensions = [0]
    {
        static const std::regex re_lhs(R"(lhs_contracting_dimensions\s*=\s*\[(\d+)\])");
        static const std::regex re_rhs(R"(rhs_contracting_dimensions\s*=\s*\[(\d+)\])");
        std::smatch m;
        if (std::regex_search(mlir, m, re_lhs)) lhs_contract = std::stoi(m[1].str());
        if (std::regex_search(mlir, m, re_rhs)) rhs_contract = std::stoi(m[1].str());
    }

    // Check if JAX emitted a standalone stablehlo.transpose on %arg0 or %arg1
    bool arg0_transposed = false;
    bool arg1_transposed = false;
    {
        static const std::regex re_t0(R"(stablehlo\.transpose\s+%arg0\b)");
        static const std::regex re_t1(R"(stablehlo\.transpose\s+%arg1\b)");
        arg0_transposed = std::regex_search(mlir, re_t0);
        arg1_transposed = std::regex_search(mlir, re_t1);
    }

    // Determine effective trans_a / trans_b relative to input buffers A (a_shape) and B (b_shape):
    // In dot_general, if lhs_contract == 0 (without prior transpose) -> A is transposed.
    // If arg0 was transposed before dot_general, flip the flag.
    bool trans_a = (lhs_contract == 0) ^ arg0_transposed;
    bool trans_b = (rhs_contract == 1) ^ arg1_transposed;

    const int64_t m = trans_a ? a_shape.second : a_shape.first;
    const int64_t k_a = trans_a ? a_shape.first : a_shape.second;
    const int64_t k_b = trans_b ? b_shape.second : b_shape.first;
    const int64_t n = trans_b ? b_shape.first : b_shape.second;

    if (k_a != k_b || c_shape.first != m || c_shape.second != n) {
        std::ostringstream os;
        os << "MatMul shape mismatch: A=(" << a_shape.first << "," << a_shape.second
           << ") B=(" << b_shape.first << "," << b_shape.second
           << ") C=(" << c_shape.first << "," << c_shape.second
           << ") trans_a=" << trans_a << " trans_b=" << trans_b;
        err_msg = os.str();
        return false;
    }

    plan.m = m;
    plan.n = n;
    plan.k = k_a;
    plan.trans_a = trans_a;
    plan.trans_b = trans_b;
    plan.out_dims = {m, n};
    return true;
}

} // namespace

PJRT_Error* client_compile(PJRT_Client_Compile_Args* args) {
    if (!args || !args->client || !args->program) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "invalid Compile args");
    }
    ParasClient* client = as_impl(args->client);
    if (client->device_ptrs.empty()) {
        return make_pjrt_error(PJRT_Error_Code_FAILED_PRECONDITION, "no devices available");
    }

    const PJRT_Program* prog = args->program;
    if (!prog->code || prog->code_size == 0) {
        return make_pjrt_error(PJRT_Error_Code_INVALID_ARGUMENT, "empty program code");
    }

    std::string mlir_text;
    std::string err_msg;
    if (is_mlir_bytecode(prog->code, prog->code_size)) {
        if (!deserialize_mlir_bytecode(prog->code, prog->code_size, mlir_text, err_msg)) {
            return make_pjrt_error(PJRT_Error_Code_UNIMPLEMENTED, err_msg);
        }
    } else {
        mlir_text.assign(prog->code, prog->code_size);
    }

    if (const char* dump = std::getenv("PARAS_PJRT_DUMP_MLIR")) {
        if (std::string(dump) == "1") {
            std::cerr << "[pjrt_paras] StableHLO MLIR:\n" << mlir_text << "\n";
        }
    }

    MatMulPlan plan;
    if (!parse_stablehlo_matmul(mlir_text, plan, err_msg)) {
        return make_pjrt_error(PJRT_Error_Code_UNIMPLEMENTED, err_msg);
    }

    auto exec = std::make_unique<ParasExecutable>();
    exec->plan = plan;
    exec->output_types = {PJRT_Buffer_Type_F32};
    exec->output_dims = plan.out_dims;
    exec->output_dim_sizes = {plan.out_dims.size()};

    auto* loaded = new ParasLoadedExecutable();
    loaded->client = client;
    loaded->executable = std::move(exec);
    loaded->addressable_devices = {client->device_ptrs[0]};

    args->executable = to_pjrt(loaded);
    return nullptr;
}

} // namespace paras_pjrt_api
