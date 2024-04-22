#include <iostream>
#include <random>
#include <unistd.h>
#include <sys/wait.h>
#include <pthread.h>
#include <mutex>

using namespace std;

const int batch_size = 4;
const int neurons_per_layer = 16;
const int input_size = 10;
const int output_size = 2;
const int num_hidden_layers = 3;
const double learning_rate = 0.01;
const int num_epochs = 5;

// Global mutexes for synchronization
mutex pipe_mutex;
mutex layer_mutex;

random_device rd;
mt19937 gen(rd());
uniform_real_distribution<double> distribution(0.0, 1.0);

// Forward declaration
class Layer;

// Neuron structure definition
struct Neuron {
    double weight;
    double bias;
    double output;
    double error;

    Neuron() : weight(distribution(gen)), bias(distribution(gen)), output(0.0), error(0.0) {}

    double* compute(double* input, int batch_size) {
        double* res = new double[batch_size];
        for (int i = 0; i < batch_size; i++)
            res[i] = input[i] * weight + bias;
        return res;
    }

    void print() {
        cout << "Weight: " << weight << "\tBias: " << bias << endl;
    }
};

struct NeuronComputeArgs {
    Layer* layer;
    Neuron* neuron;
    double* input;
    int batchSize;
    int index;
};

// Layer class definition
class Layer {
public:
    Neuron* neurons;
    double** res;
    double** deltas;  // For storing intermediate values during backpropagation
    int in_features;
    int out_features;

    // Default constructor
    Layer() : neurons(nullptr), res(nullptr), deltas(nullptr), in_features(0), out_features(0) {}

    // Parameterized constructor
    Layer(int in_features, int out_features) : in_features(in_features), out_features(out_features) {
        neurons = new Neuron[out_features];
        res = new double*[out_features];
        deltas = new double*[out_features];

        for (int i = 0; i < out_features; ++i) {
            res[i] = new double[batch_size];
            deltas[i] = new double[batch_size];
        }
    }
    
    static void* neuronCompute(void* args) {
        auto* arg = static_cast<NeuronComputeArgs*>(args);
        arg->layer->res[arg->index] = arg->neuron->compute(arg->input, arg->batchSize);
        return nullptr;
    }
    
    // Compute the layer output
    void compute(double* input, int read_fd, int write_fd) {
        NeuronComputeArgs args[out_features];
        pthread_t threads[out_features];
        
        for (int i = 0; i < out_features; i++) {
            NeuronComputeArgs arg = {this, &neurons[i], input, batch_size, i};
            args[i] = arg;
            pthread_create(&threads[i], nullptr, neuronCompute, &args[i]);
        }
        
        for (int i = 0; i < out_features; i++)
            pthread_join(threads[i], nullptr);

        // Protect the write to the pipe with a mutex
        pipe_mutex.lock();
        write(write_fd, &res[0][0], out_features * batch_size * sizeof(double));
        pipe_mutex.unlock();

        // Read the deltas from the previous layer through the pipe
        if (read_fd != -1)
            read(read_fd, &deltas[0][0], out_features * batch_size * sizeof(double));
    }
    

    // Calculate the deltas for backpropagation
    void calculateDeltas(double* target) {
        for (int i = 0; i < out_features; i++) {
            for (int j = 0; j < batch_size; j++) {
                deltas[i][j] = res[i][j] - target[i * batch_size + j];
            }
        }
    }

    // Update weights and biases using backpropagation
    void updateWeightsAndBiases(Layer* prev_layer) {
        for (int i = 0; i < out_features; i++) {
            double weightUpdateSum = 0.0;
            double biasUpdateSum = 0.0;

            for (int j = 0; j < batch_size; j++) {
                weightUpdateSum += deltas[i][j] * prev_layer->res[i][j];
                biasUpdateSum += deltas[i][j];
            }

            // Protect the update with a mutex
            layer_mutex.lock();
            // Update weights and biases after the inner loop
            neurons[i].weight -= learning_rate * weightUpdateSum;
            neurons[i].bias -= learning_rate * biasUpdateSum;
            layer_mutex.unlock();
        }
    }
};

// Generate random input and target data
void generateData(double**& input_data, double**& target_data) {
    input_data = new double*[input_size];
    for (int i = 0; i < input_size; ++i) {
        input_data[i] = new double[batch_size];
        for (int j = 0; j < batch_size; ++j) {
            input_data[i][j] = distribution(gen);
        }
    }

    target_data = new double*[output_size];
    for (int i = 0; i < output_size; ++i) {
        target_data[i] = new double[batch_size];
        for (int j = 0; j < batch_size; ++j) {
            target_data[i][j] = distribution(gen);
        }
    }
}

// Print the results
void printResults(double** result, int rows, int cols, const string& label) {
    cout << label << ":" << endl;
    for (int i = 0; i < rows; ++i) {
        cout << result[i][0] << "\t";
    }
    cout << endl;
}

int main() {
    // Generate random input data and target
    double** input_data, ** target_data;
    generateData(input_data, target_data);

    // Define the neural network structure
    Layer input_layer(input_size, neurons_per_layer);
    Layer* hidden_layers = new Layer[num_hidden_layers];
    for (int i = 0; i < num_hidden_layers; ++i)
        hidden_layers[i] = Layer(neurons_per_layer, neurons_per_layer);
    Layer output_layer(neurons_per_layer, output_size);

    // Pipes for interprocess communication
    int pipe_fd[num_hidden_layers][2]; // One pipe per hidden layer

    for (int i = 0; i < num_hidden_layers; ++i) {
        if (pipe(pipe_fd[i]) == -1) {
            cerr << "Pipe creation failed." << endl;
            exit(EXIT_FAILURE);
        }
    }

    // Main training loop
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        // Forward pass
        input_layer.compute(input_data[0], -1, pipe_fd[0][1]);

        // Fork for hidden layers
        for (int i = 0; i < num_hidden_layers; i++) {
            pid_t pid = fork();

            if (pid == -1) {
                cerr << "Fork failed." << endl;
                exit(EXIT_FAILURE);
            }

            if (pid == 0) {
                // Child process
                if (i > 0) {
                    close(pipe_fd[i - 1][1]); // Close write end of the previous pipe
                    close(pipe_fd[i][0]);     // Close read end of the current pipe
                }

                // Compute results for the current hidden layer
                hidden_layers[i].compute(i == 0 ? input_layer.res[0] : hidden_layers[i - 1].res[0], (i == num_hidden_layers - 1) ? -1 : pipe_fd[i][1], -1);

                if (i < num_hidden_layers - 1)
                    close(pipe_fd[i][1]); // Close write end of the current pipe

                exit(EXIT_SUCCESS);
            }
        }

        // Close pipes in the parent process
        for (int i = 0; i < num_hidden_layers; i++) {
            close(pipe_fd[i][1]); // Close write end of the pipe in the parent process
        }

        // Wait for all child processes to complete
        for (int i = 0; i < num_hidden_layers; i++) {
            wait(NULL);
        }

        // Forward pass in the output layer
        output_layer.compute(hidden_layers[num_hidden_layers - 1].res[0], -1, -1);
        printResults(output_layer.res, output_size, batch_size, "Results After Forward Pass of Output");

        // Backward pass (Backpropagation)
        output_layer.calculateDeltas(target_data[0]);
        output_layer.updateWeightsAndBiases(&output_layer);

        for (int i = num_hidden_layers - 1; i >= 0; i--) {
            Layer* prev_layer = (i == 0) ? &input_layer : &hidden_layers[i - 1];
            hidden_layers[i].calculateDeltas((i == num_hidden_layers - 1) ? output_layer.deltas[0] : hidden_layers[i + 1].deltas[0]);
            hidden_layers[i].updateWeightsAndBiases(prev_layer);
        }
    }

    // Forward Pass 2
    input_layer.compute(input_data[0], -1, pipe_fd[0][1]);

    // Fork for hidden layers
    for (int i = 0; i < num_hidden_layers; i++) {
        pid_t pid = fork();

        if (pid == -1) {
            cerr << "Fork failed." << endl;
            exit(EXIT_FAILURE);
        }

        if (pid == 0) {
            // Child process
            if (i > 0) {
                close(pipe_fd[i - 1][1]); // Close write end of the previous pipe
                close(pipe_fd[i][0]);     // Close read end of the current pipe
            }

            // Compute results for the current hidden layer
            hidden_layers[i].compute(i == 0 ? input_layer.res[0] : hidden_layers[i - 1].res[0], (i == num_hidden_layers - 1) ? -1 : pipe_fd[i][1], -1);

            if (i < num_hidden_layers - 1)
                close(pipe_fd[i][1]); // Close write end of the current pipe

            exit(EXIT_SUCCESS);
        }
    }

    // Close pipes in the parent process
    for (int i = 0; i < num_hidden_layers; i++) {
        close(pipe_fd[i][1]); // Close write end of the pipe in the parent process
    }

    // Wait for all child processes to complete
    for (int i = 0; i < num_hidden_layers; i++) {
        wait(NULL);
    }

    // Forward pass in the output layer
    output_layer.compute(hidden_layers[num_hidden_layers - 1].res[0], -1, -1);
    printResults(output_layer.res, output_size, batch_size, "Results After Forward Pass 2 of Output");

    // Print Target
    printResults(target_data, output_size, batch_size, "Target");

    // Clean up memory
    for (int i = 0; i < input_size; ++i) {
        delete[] input_data[i];
    }
    delete[] input_data;

    for (int i = 0; i < output_size; ++i) {
        delete[] target_data[i];
    }
    delete[] target_data;

    return 0;
}
