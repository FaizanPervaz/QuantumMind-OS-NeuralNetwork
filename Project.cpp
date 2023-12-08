#include <iostream>
#include <pthread.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <semaphore.h>
#include <sys/wait.h>
#include <string>
#include <iomanip>
#include <sstream>
#include <stack>
#include <cmath>

using namespace std;

const int TotalLayers = 5;
float InputLayer[2][1][8];
float HiddenLayer[TotalLayers][8][8];
float OutputLayer[1][8];
float layersAns[1][8]{0}; // Storing Answer of Each Layer
stack<float> st; //Building Stack to Keep Track of Values
sem_t sem;

float inputvalues[2] = {0.1, 0.2};
float valuesArr[2]; 
float finalOutput = 0.0;
float learning_rate = 0.0001;

float sigmoid(float x);
void GenerateRandomWeights();
void *critical_Section(void *id_ptr);

struct NN_Layers
{
    int num_layers; // total number of layers

    NN_Layers(int num)
    {

        num_layers = num; // total numbers layers

        // Writing initial answer of input layers on pipe
        int fd = open("my_pipe", O_WRONLY); // for writing only

        for (int k = 0; k < 2; k++)
        { // input layer neurons

            for (int j = 0; j < 8; j++)
            { // 8 values of neurons

                for (int i = 0; i < 2; i++)
                {                                                                   // 2 values of input
                    layersAns[0][j] += inputvalues[i] * InputLayer[k][0][j]; // generating input layers
                }
            }
        }

        for (int i = 0; i < 8; i++)
        { // inserting values of input layer in stack
            st.push(layersAns[0][i]);
        }

        for (int i = 0; i < 1; i++)
        {
            write(fd, layersAns[i], sizeof(layersAns[i]));
        }
        // close named pipe
        // num_layers--;
        close(fd);
    }
    float Back_Prop_First(float x)
    {
        float value = ((x * x) + x + 1) / 2;
        float result = std::floor(value * 1000000) / 1000000; // Rounding to 6 decimal places
        float temp = sigmoid(result);
        return temp;
    }
    float Back_Prop_Second(float x)
    {
        float value = ((x * x) - x) / 2;
        float result = std::floor(value * 1000000) / 1000000; // Rounding to 6 decimal places
        float temp = sigmoid(result);
        return temp;
    }

    void FP()
    {
        cout << "Forward Propagation start processing\n";
        sleep(2);
        int ids[num_layers];
        pid_t pid;
        pthread_t processes[num_layers];
        for (int i = 0; i < num_layers; i++)
        {
            ids[i] = i;
            pthread_create(&processes[i], NULL, critical_Section, &ids[i]);
        }
        for (int i = 0; i < num_layers; i++)
        {
            pthread_join(processes[i], NULL);
        }

        // output layer multiplying
        for (int j = 0; j < 8; j++)
            for (int i = 0; i < 8; i++)
            {
                layersAns[0][j] += layersAns[0][i] * OutputLayer[0][j];
            }
        for (int i = 0; i < 8; i++)
            st.push(layersAns[0][i]);
    }

    void Critical_Section()
    {

        sem_wait(&sem);
        int counter = 0; 

        while (counter != 8 && !st.empty())
        {
            cout << st.top() << " ";
            st.pop();
            counter++;
        }
        cout << endl;

        counter = 0;
        int fd = open("values", O_RDONLY); // creating pipe for reading and writing
        read(fd, valuesArr, sizeof(valuesArr));
        close(fd);
        valuesArr[0] = Back_Prop_First(valuesArr[0]);
        valuesArr[1] = Back_Prop_Second(valuesArr[1]);
        cout << "The value of x1 = " << valuesArr[0] << endl;
        cout << "The value of x2 = " << valuesArr[1] << endl;
        int fd2 = open("values", O_WRONLY);
        write(fd2, valuesArr, sizeof(valuesArr));
        close(fd2);
        sem_post(&sem);
    }

    void backPropagation()
    {

        for (int i = 0; i < 8; i++)
        { // size of answer array [1][8]
            finalOutput += layersAns[0][i];
        }
        cout << "Final ouput I get in End after forward propagation :\n";

        cout << finalOutput << endl;
        cout << "--------------------------" << endl;
        cout << "Back Propagation start processing\n";
        sleep(2);

        // writing values in pipe
        valuesArr[0] = Back_Prop_First(finalOutput);
        valuesArr[1] = Back_Prop_Second(finalOutput);
        int fd = open("values", O_WRONLY);
        write(fd, valuesArr, sizeof(valuesArr));
        cout << "The value of x1 on output : " << valuesArr[0] << endl;
        cout << "The value of x2 on output : " << valuesArr[1] << endl;
        close(fd);
        int totalLayers = num_layers + 1 + 1; // input layer + remaining layers + output layer

        for (int i = totalLayers; i > 0; i--)
        {
            cout << "----Process " << i << " Layer values----" << endl;
            Critical_Section();
        }
        // last values after multiplying with input values
        fd = open("values", O_RDONLY); // creating pipe for reading and writing
        read(fd, valuesArr, sizeof(valuesArr));
        close(fd);
        valuesArr[0] = Back_Prop_First(valuesArr[0]);
        valuesArr[1] = Back_Prop_Second(valuesArr[1]);
        cout << "\nThe value of x1 : " << valuesArr[0] << endl;
        cout << "The value of x2 : " << valuesArr[1] << endl;
        inputvalues[0] = valuesArr[0];
        inputvalues[1] = valuesArr[1];

        float gradientsOutputLayer[1][8]; // Gradients for the output layer
        for (int i = 0; i < 8; ++i) {
        // Compute error derivative (assume some loss function)
        float error_derivative = layersAns[0][i] - OutputLayer[0][i];

        // Multiply by derivative of sigmoid (assuming sigmoid activation in output layer)
        gradientsOutputLayer[0][i] = error_derivative * sigmoid(layersAns[0][i]);
    }

    // Propagate gradients backward through hidden layers
    for (int i = TotalLayers - 1; i >= 0; --i) {
        float gradientsHiddenLayer[8][8] = {}; // Gradients for Hidden Layer i

        // Compute gradients for Hidden Layer i
        for (int j = 0; j < 8; ++j) {
            for (int k = 0; k < 8; ++k) {
                float error = 0.0;
                for (int l = 0; l < 8; ++l) {
                    error += gradientsOutputLayer[0][l] * HiddenLayer[i][l][k];
                }
                gradientsHiddenLayer[j][k] += sigmoid(layersAns[0][j]) * error;
            }
        }

        // Update weights for Hidden Layer i
        for (int j = 0; j < 8; ++j) {
            for (int k = 0; k < 8; ++k) {
                HiddenLayer[i][j][k] -= learning_rate * gradientsHiddenLayer[j][k];
            }
        }

        // Update gradients for the next backward pass
        for (int j = 0; j < 8; ++j) {
            for (int k = 0; k < 8; ++k) {
                gradientsOutputLayer[0][j] = gradientsHiddenLayer[j][k];
            }
        }
    }
    }

    void startProcessing()
    {
        sem_t processsem;
        sem_init(&processsem, 0, 1);

        for (int i = 0; i < 2; i++)
        {
            pid_t id = fork(); // Multiprocessing
            sem_wait(&processsem);

            if (id < 0)
            {
                cerr << "Error in creating process";
                exit(1);
            }
            else if (id == 0)
            {
                FP();
                backPropagation();
            }
            else if (id == 1)
            {
                wait(NULL);
            }

            sem_post(&processsem);
        }
    }
};

int main()
{
    GenerateRandomWeights();
    sem_init(&sem, 0, 1);
    NN_Layers obj(TotalLayers);
    obj.startProcessing();
}


float sigmoid(float x)
{
    return 1 / (1 + exp(-x));
}

void GenerateRandomWeights()
{
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 1; ++j)
        {
            for (int k = 0; k < 8; ++k)
            {
                InputLayer[i][j][k] = ((float)rand() / RAND_MAX) * 2.0 - 1.0;
            }
        }
    }

    cout << "-------------- Input Layer Weights --------------\n";
    for (int i = 0; i < 2; ++i)
    {
        for (int j = 0; j < 1; ++j)
        {
            for (int k = 0; k < 8; ++k)
            {
                cout << InputLayer[i][j][k] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }

    for (int i = 0; i < TotalLayers; ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            for (int k = 0; k < 8; ++k)
            {
                HiddenLayer[i][j][k] = ((float)rand() / RAND_MAX) * 2.0 - 1.0;
            }
        }
    }

    cout << "-------------- Layer Weights --------------\n";
    for (int i = 0; i < TotalLayers; ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            for (int k = 0; k < 8; ++k)
            {
                cout << HiddenLayer[i][j][k] << " ";
            }
            cout << endl;
        }
        cout << endl;
    }

    for (int i = 0; i < 1; ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            OutputLayer[i][j] = ((float)rand() / RAND_MAX) * 2.0 - 1.0;
        }
    }

    cout << "-------------- Output Layer Weights --------------\n";
    for (int i = 0; i < 1; ++i)
    {
        for (int j = 0; j < 8; ++j)
        {
            cout << OutputLayer[i][j] << " ";
        }
        cout << endl;
    }
}

void *critical_Section(void *id_ptr)
{
    int id = *((int *)id_ptr);
    int fd = open("my_pipe", O_RDONLY); // creating pipe for reading and writing
    sem_wait(&sem);
    for (int i = 0; i < 1; i++) // rows of answer array [1][8 ]
    {
        read(fd, layersAns[i], sizeof(layersAns[i]));
    }
    close(fd);

    float val = 0.0;
    float dummy[1][8];
    for (int i = 0; i < 8; i++) // layer rows
    {
        for (int k = 0; k < 8; k++) // ans values
        {
            for (int j = 0; j < 8; j++) // layer col
            {
                val += layersAns[0][k] * HiddenLayer[id][i][j];
            }
        }
        dummy[0][i] = val;
        val = 0;
    }
    for (int i = 0; i < 8; i++)
    {
        layersAns[0][i] = dummy[0][i];
        st.push(layersAns[0][i]);
    }

    fd = open("my_pipe", O_WRONLY);
    for (int i = 0; i < 1; i++)
    {
        write(fd, layersAns[i], sizeof(layersAns[i]));
    }
    close(fd);
    sem_post(&sem);

    return NULL;
}

