#include "kernel.hpp"

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

struct CliOptions {
    std::string mode;
    std::filesystem::path fixture_dir;
    std::filesystem::path dataset;
    std::string sample_name = "anna";
    std::string preset = "small";
    int num_steps = -1;
    std::uint32_t seed = 42;
};

struct BenchmarkPreset {
    ModelConfig config;
    int steps = 0;
};

const std::unordered_map<std::string, BenchmarkPreset> kBenchmarkPresets = {
    {"small", {ModelConfig{1, 64, 64, 4, 0}, 200}},
    {"medium", {ModelConfig{2, 128, 64, 8, 0}, 200}},
    {"large", {ModelConfig{4, 256, 128, 8, 0}, 100}},
};

std::string require_value(int argc, char** argv, int* index) {
    if (*index + 1 >= argc) {
        throw std::runtime_error(std::string("missing value for option ") + argv[*index]);
    }
    ++(*index);
    return argv[*index];
}

CliOptions parse_cli(int argc, char** argv) {
    CliOptions options;
    for (int idx = 1; idx < argc; ++idx) {
        const std::string arg = argv[idx];
        if (arg == "--mode") {
            options.mode = require_value(argc, argv, &idx);
        } else if (arg == "--fixture-dir") {
            options.fixture_dir = require_value(argc, argv, &idx);
        } else if (arg == "--dataset") {
            options.dataset = require_value(argc, argv, &idx);
        } else if (arg == "--sample-name") {
            options.sample_name = require_value(argc, argv, &idx);
        } else if (arg == "--preset") {
            options.preset = require_value(argc, argv, &idx);
        } else if (arg == "--num-steps") {
            options.num_steps = std::stoi(require_value(argc, argv, &idx));
        } else if (arg == "--seed") {
            options.seed = static_cast<std::uint32_t>(std::stoul(require_value(argc, argv, &idx)));
        } else {
            throw std::runtime_error("unknown option: " + arg);
        }
    }
    if (options.mode.empty()) {
        throw std::runtime_error("--mode is required");
    }
    return options;
}

std::vector<std::string> load_docs(const std::filesystem::path& dataset_path) {
    std::ifstream input(dataset_path);
    if (!input) {
        throw std::runtime_error("failed to open dataset: " + dataset_path.string());
    }
    std::vector<std::string> docs;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty()) {
            docs.push_back(line);
        }
    }
    return docs;
}

std::pair<std::string, std::unordered_map<char, int>> build_vocab(const std::vector<std::string>& docs) {
    std::string chars;
    for (const std::string& doc : docs) {
        chars += doc;
    }
    std::sort(chars.begin(), chars.end());
    chars.erase(std::unique(chars.begin(), chars.end()), chars.end());
    std::unordered_map<char, int> vocab;
    for (int idx = 0; idx < static_cast<int>(chars.size()); ++idx) {
        vocab[chars[idx]] = idx;
    }
    return {chars, vocab};
}

std::vector<int> encode_doc(const std::string& doc, const std::unordered_map<char, int>& vocab, int bos_token_id) {
    std::vector<int> tokens;
    tokens.reserve(doc.size() + 2);
    tokens.push_back(bos_token_id);
    for (char ch : doc) {
        const auto it = vocab.find(ch);
        if (it == vocab.end()) {
            throw std::runtime_error(std::string("sample name contains character not in dataset: ") + ch);
        }
        tokens.push_back(it->second);
    }
    tokens.push_back(bos_token_id);
    return tokens;
}

std::unordered_map<std::string, std::string> parse_manifest(const std::filesystem::path& manifest_path) {
    std::ifstream input(manifest_path);
    if (!input) {
        throw std::runtime_error("failed to open manifest: " + manifest_path.string());
    }
    std::unordered_map<std::string, std::string> manifest;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty()) {
            continue;
        }
        const std::size_t pos = line.find('=');
        if (pos == std::string::npos) {
            throw std::runtime_error("invalid manifest line: " + line);
        }
        manifest[line.substr(0, pos)] = line.substr(pos + 1);
    }
    return manifest;
}

std::vector<int> parse_int_list(const std::string& text) {
    std::vector<int> values;
    std::stringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ',')) {
        if (!token.empty()) {
            values.push_back(std::stoi(token));
        }
    }
    return values;
}

std::vector<float> read_f32_file(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("failed to open binary file: " + path.string());
    }
    input.seekg(0, std::ios::end);
    const std::streamsize file_size = input.tellg();
    input.seekg(0, std::ios::beg);
    if (file_size % static_cast<std::streamsize>(sizeof(float)) != 0) {
        throw std::runtime_error("binary file is not float32-aligned: " + path.string());
    }
    std::vector<float> values(static_cast<std::size_t>(file_size / static_cast<std::streamsize>(sizeof(float))));
    input.read(reinterpret_cast<char*>(values.data()), file_size);
    return values;
}

double compare_arrays(const std::string& label, const std::vector<double>& actual, const std::vector<float>& expected, double epsilon) {
    if (actual.size() != expected.size()) {
        throw std::runtime_error(label + " size mismatch");
    }
    double max_abs_error = 0.0;
    std::size_t max_idx = 0;
    for (std::size_t idx = 0; idx < actual.size(); ++idx) {
        const double abs_error = std::abs(actual[idx] - static_cast<double>(expected[idx]));
        if (abs_error > max_abs_error) {
            max_abs_error = abs_error;
            max_idx = idx;
        }
    }
    std::cout << label << " max_abs_error=" << std::setprecision(8) << max_abs_error;
    if (!actual.empty()) {
        std::cout << " at_index=" << max_idx;
    }
    std::cout << '\n';
    if (max_abs_error > epsilon) {
        throw std::runtime_error(label + " exceeded validation epsilon");
    }
    return max_abs_error;
}

int run_validate(const CliOptions& options) {
    if (options.fixture_dir.empty()) {
        throw std::runtime_error("--fixture-dir is required for validate mode");
    }

    const std::filesystem::path manifest_path = options.fixture_dir / "manifest.txt";
    const auto manifest = parse_manifest(manifest_path);
    ModelConfig config;
    config.n_layer = std::stoi(manifest.at("n_layer"));
    config.n_embd = std::stoi(manifest.at("n_embd"));
    config.block_size = std::stoi(manifest.at("block_size"));
    config.n_head = std::stoi(manifest.at("n_head"));
    config.vocab_size = std::stoi(manifest.at("vocab_size"));

    const std::vector<int> tokens = parse_int_list(manifest.at("token_ids"));
    const double epsilon = std::stod(manifest.at("validation_epsilon"));

    Model model = make_empty_model(config);
    load_model_from_f32(model, read_f32_file(options.fixture_dir / manifest.at("weights_init_file")));
    const KernelResult result = run_forward_backward(model, tokens);

    compare_arrays("logits", result.logits, read_f32_file(options.fixture_dir / manifest.at("expected_logits_file")), epsilon);
    compare_arrays("loss", {result.loss}, read_f32_file(options.fixture_dir / manifest.at("expected_loss_file")), epsilon);
    compare_arrays(
        "grads",
        flatten_model_values(result.grads),
        read_f32_file(options.fixture_dir / manifest.at("expected_grads_file")),
        epsilon
    );
    std::cout << "validation=pass\n";
    return 0;
}

int run_benchmark(const CliOptions& options) {
    if (options.dataset.empty()) {
        throw std::runtime_error("--dataset is required for benchmark mode");
    }
    const auto preset_it = kBenchmarkPresets.find(options.preset);
    if (preset_it == kBenchmarkPresets.end()) {
        throw std::runtime_error("unknown preset: " + options.preset);
    }

    const std::vector<std::string> docs = load_docs(options.dataset);
    if (docs.empty()) {
        throw std::runtime_error("dataset is empty");
    }
    const auto [uchars, vocab] = build_vocab(docs);
    BenchmarkPreset preset = preset_it->second;
    preset.config.vocab_size = static_cast<int>(uchars.size()) + 1;
    const int requested_steps = options.num_steps >= 0 ? options.num_steps : preset.steps;
    const int steps = std::min(requested_steps, static_cast<int>(docs.size()));
    const Model model = initialize_model(preset.config, options.seed);

    double last_loss = 0.0;
    std::string last_doc;
    for (int step = 0; step < steps; ++step) {
        last_doc = docs[step];
        const std::vector<int> tokens = encode_doc(last_doc, vocab, static_cast<int>(uchars.size()));
        last_loss = run_forward_backward(model, tokens).loss;
    }

    std::cout << "mode=benchmark "
              << "preset=" << options.preset << ' '
              << "requested_steps=" << requested_steps << ' '
              << "steps=" << steps << ' '
              << "last_doc=" << last_doc << ' '
              << "loss=" << std::setprecision(8) << last_loss << '\n';
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const CliOptions options = parse_cli(argc, argv);
        if (options.mode == "validate") {
            return run_validate(options);
        }
        if (options.mode == "benchmark") {
            return run_benchmark(options);
        }
        throw std::runtime_error("unsupported mode: " + options.mode);
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << '\n';
        return 1;
    }
}
