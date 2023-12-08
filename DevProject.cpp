#include <iostream>
#include <unistd.h>
#include <pthread.h>
#include <sys/wait.h>
#include <sys/types.h>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <cstring>
#include <string.h>

using namespace std;

#define Number_of_Layers 7
#define Hidden_Layer_Neurons 8
#define Input_Neurons 2
#define Output_Neurons 1

//Configuration
// 2 - 8 - 8 - 8 - 8 - 8 - 8 - 1

//Defining Global Variables

//Mutex Locks
pthread_mutex_t First_Layer[Hidden_Layer_Neurons];
pthread_mutex_t Hidden_Layer[Number_of_Layers][Hidden_Layer_Neurons];
pthread_mutex_t Outer_Layer[Hidden_Layer_Neurons];
pthread_mutex_t Back_Layer[Number_of_Layers];

//Arrays For Functions


//Defining Pipes
//2 Neurons each giving Weights to every 8 Neurons in the First Layer
int Input_Pipes[2][2];
//8 Neurons each giving Weights to every 8 Neurons in the 7 Hidden Layers
int Hidden_Pipes[Number_of_Layers][Hidden_Layer_Neurons][2];
//
int Output_Pipes[2];

int Layer_Pipes[Number_of_Layers][2];

 double Input_Layer_Weights[2][8] = {
        {0.1, -0.2, 0.3, 0.1, -0.2, 0.3, 0.1, -0.2},
        {-0.4, 0.5, 0.6, -0.4, 0.5, 0.6, -0.4, 0.5}
    };

double Inputs[2] = {0.1, 0.2};

double Hidden_Layer_Weights[5][8][8] = {
        {
        {-0.2, 0.3, -0.4, 0.5, 0.6, -0.7, 0.8, -0.9},
        {0.1, 0.9, -0.3, 0.2, -0.5, 0.4, 0.6, -0.8},
        {-0.7, 0.5, 0.8, -0.2, -0.3, -0.6, 0.1, 0.4},
        {-0.2, 0.3, -0.4, 0.5, 0.6, -0.7, 0.8, -0.9},
        {0.1, 0.9, -0.3, 0.2, -0.5, 0.4, 0.6, -0.8},
        {-0.7, 0.5, 0.8, -0.2, -0.3, -0.6, 0.1, 0.4},
        {-0.2, 0.3, -0.4, 0.5, 0.6, -0.7, 0.8, -0.9},
        {0.1, 0.9, -0.3, 0.2, -0.5, 0.4, 0.6, -0.8}
    },
    {
        {0.2, -0.3, 0.4, -0.5, -0.6, 0.7, -0.8, 0.9},
        {-0.1, -0.9, 0.3, -0.2, 0.5, -0.4, -0.6, 0.8},
        {0.7, -0.5, -0.8, 0.2, 0.3, 0.6, -0.1, -0.4},
        {0.2, -0.3, 0.4, -0.5, -0.6, 0.7, -0.8, 0.9},
        {-0.1, -0.9, 0.3, -0.2, 0.5, -0.4, -0.6, 0.8},
        {0.7, -0.5, -0.8, 0.2, 0.3, 0.6, -0.1, -0.4},
        {0.2, -0.3, 0.4, -0.5, -0.6, 0.7, -0.8, 0.9},
        {-0.1, -0.9, 0.3, -0.2, 0.5, -0.4, -0.6, 0.8}
    },
    {
        {0.3, -0.4, 0.5 ,-0.6, -0.7, 0.8, -0.9, 0.1},
        {-0.2, -0.9, 0.4, -0.3, 0.5, -0.6, -0.8, 0.1},
        {0.6 ,-0.5, -0.7, 0.2, 0.4, 0.8, -0.1 ,-0.3 },
        {0.3 ,-0.4 ,0.5 ,-0.6 ,-0.7 ,0.8 ,-0.9   ,0.1 },
        {-0.2, -0.9 ,0.4 ,-0.3, 0.5, -0.6, -0.8  ,0.1}, 
        {0.6, -0.5, -0.7, 0.2 ,0.4 ,0.8 ,-0.1 ,-0.3 },
        {0.3 ,-0.4, 0.5, -0.6, -0.7, 0.8, -0.9, 0.1 },
        {-0.2 ,-0.9 ,0.4 ,-0.3 ,0.5 ,-0.6 ,-0.8 ,0.1} 
    },
    {
       {0.4, -0.5 ,0.6 ,-0.7 ,-0.8, 0.9 ,-0.1 ,0.2 },
       {-0.3 ,-0.8 ,0.5 ,-0.4 ,0.6 ,-0.7, -0.9 ,0.2 },
       {0.5 ,-0.4 ,-0.6, 0.3, 0.2, 0.8 ,-0.2, -0.1 },
       {0.4, -0.5, 0.6, -0.7, -0.8 ,0.9 ,-0.1 ,0.2 },
       {-0.3 ,-0.8 ,0.5 ,-0.4 ,0.6 ,-0.7, -0.9, 0.2 },
       {0.5, -0.4, -0.6, 0.3, 0.2 ,0.8 ,-0.2 ,-0.1 },
       {0.4, -0.5, 0.6, -0.7, -0.8, 0.9 ,-0.1, 0.2 },
       {-0.3, -0.8, 0.5, -0.4, 0.6, -0.7, -0.9, 0.2 }
    },
    {
        {0.5 ,-0.6, 0.7 -0.8 ,-0.9 ,0.1 ,-0.2, 0.3 },
        {-0.4 ,-0.7 ,0.6 ,-0.5, 0.8 -0.6 ,-0.2, 0.1 },
        {0.4 ,-0.3 ,-0.5, 0.1, 0.6, 0.7, -0.3, -0.2 },
        {0.5 ,-0.6, 0.7 ,-0.8 ,-0.9 ,0.1 ,-0.2 ,0.3 },
        {-0.4 ,-0.7, 0.6 ,-0.5, 0.8 ,-0.6, -0.2, 0.1 },
        {0.4 ,-0.3 ,-0.5, 0.1, 0.6 ,0.7 ,-0.3 ,-0.2 },
        {0.5, -0.6 ,0.7, -0.8 ,-0.9, 0.1, -0.2, 0.3 },
        {-0.4, -0.7, 0.6, -0.5, 0.8, -0.6, -0.2, 0.1 }
    }
    };

double Output_Layer_Weights[8] = {-0.1, 0.2, 0.3, 0.4, 0.5, -0.6, -0.7, 0.8};

double sigmoid(double x) {
        return 1 / (1 + exp(-x));
}
    
void* First_Layer_Thread(void*)
{
    for(int i = 0; i < 8; i++)
    {
        pthread_mutex_lock(&First_Layer[i]);
        double Calc_1 = Inputs[0] * Input_Layer_Weights[0][i];
        double Calc_2 = Inputs[1] * Input_Layer_Weights[1][i];

        double result = Calc_1 + Calc_2;
        write(Hidden_Pipes[0][i][1], &result, sizeof(result));
        pthread_mutex_unlock(&First_Layer[i]);
    }    
    return nullptr;
}

void* Hidden_Layer_Thread(void* arg)
{
    double* inputs = static_cast<double*>(arg);
    double Calc;
    double result = 0;

    for(int i = 1; i < Number_of_Layers - 1; i++)
    {
        for(int j = 0; j < Hidden_Layer_Neurons; j++)
        {
            pthread_mutex_lock(&Hidden_Layer[i][j]);
            for(int k = 0; k < Hidden_Layer_Neurons; k++)
            {
                Calc = inputs[k] * Hidden_Layer_Weights[i - 1][j][k];
                result += Calc;
                pthread_mutex_unlock(&Hidden_Layer[i][j]);
            }
            write(Hidden_Pipes[i][j][1], &result, sizeof(result));
        }
    }
    return nullptr;
}

void* Output_Layer_Thread(void* arg)
{
    // Hardcoded output layer weights

    double* inputs = static_cast<double*>(arg);
    double Calc;
    double result = 0;

    for(int i = 0; i < 8; i++)
    {
        Calc = inputs[i] * Output_Layer_Weights[i];
        result += Calc;
    }
    write(Output_Pipes[1], &result, sizeof(result));

    return nullptr;
}

void Calculate_Inputs(double* inputs)
{
    double F_result;
    read(Output_Pipes[0],&F_result,sizeof(F_result));


    double New_Inputs[2];
    for(int i = 0; i<2; i++)
    {
        New_Inputs[i] = (inputs[i] - F_result) * F_result * (1.0 - F_result);
    }
    //Storing New Values in Layer Pipes
   write(Layer_Pipes[Number_of_Layers-1][1], &New_Inputs, sizeof(New_Inputs));

}

void forwardPropagation(double input[][Input_Neurons], double weights[][Hidden_Layer_Neurons][Hidden_Layer_Neurons], double output[][Output_Neurons]) {
    double layerOutput[Hidden_Layer_Neurons] = {0};
    
    // First layer computation
    for (int j = 0; j < Hidden_Layer_Neurons; ++j) {
        for (int i = 0; i < Input_Neurons; ++i) {
            layerOutput[j] += input[0][i] * weights[0][i][j]; // Adjusted weight indexing
        }
        layerOutput[j] = sigmoid(layerOutput[j]);
    }
    
    // Hidden layers computation
    for (int l = 1; l < Number_of_Layers - 1; ++l) { // Excluding output layer
        double nextLayerOutput[Hidden_Layer_Neurons] = {0};
        for (int j = 0; j < Hidden_Layer_Neurons; ++j) {
            for (int i = 0; i < Hidden_Layer_Neurons; ++i) {
                nextLayerOutput[j] += layerOutput[i] * weights[l][i][j]; // Adjusted weight indexing
            }
            nextLayerOutput[j] = sigmoid(nextLayerOutput[j]);
        }
        copy(begin(nextLayerOutput), end(nextLayerOutput), begin(layerOutput));
    }
    
    // Output layer computation
    for (int i = 0; i < Hidden_Layer_Neurons; ++i) {
        output[0][0] += layerOutput[i] * weights[Number_of_Layers - 1][i][0]; // Adjusted for output layer
    }
    output[0][0] = sigmoid(output[0][0]);
}

void* Back_Propagation(void * x)
{
    int* ptr = static_cast<int*>(x);
    int Index = *ptr;

    double* B_Inputs = new double[2];

    pthread_mutex_lock(&Back_Layer[Index]);
    cout<<"\nInputs in Layer: "<<Index<<endl;
    read(Layer_Pipes[Index][0], &B_Inputs,sizeof(B_Inputs));
    write(Layer_Pipes[Index-1][0], &B_Inputs,sizeof(B_Inputs));
    cout<<"\nInput 1: "<<B_Inputs[0]<<" Input 2: "<<B_Inputs[1]<<endl;
    pthread_mutex_unlock(&Back_Layer[Index]);

    return nullptr;
}

int main()
{
    //First Layer Pipes
    for (int i = 0; i<2; i++)
    {
        if(pipe(Input_Pipes[i])<0)
        {
            cout<<"\nError Creating Pipe for Neuron "<< i <<"\n"; 
        }
    }

    //Second Layer Pipes
    for(int i = 0; i<Number_of_Layers-1; i++)
    {
        for(int j = 0; j<Hidden_Layer_Neurons; j++)
        {
                if(pipe(Hidden_Pipes[i][j])<0)
                {
                    cout<<"\nError Creating " << j << " Pipe for Neuron "<< i <<"\n"; 
                }
        }
    }

    //Layer Pipes
    for(int i = 0; i<Number_of_Layers; i++)
    {
        if(pipe(Layer_Pipes[i])<0)
        {
             cout<<"\nError Creating Pipe for Layer "<< i <<"\n"; 
        }
    }

    //outer Layer Pipes
    if(pipe(Output_Pipes)<0)
    {
        cout<<"\nError Creating Pipe for Outer Layer\n"; 
    }

    //Run Forward Propagation:
    pthread_t Back_Prob[Number_of_Layers];
     
    //create Processes
    pid_t Layers[Number_of_Layers];

    pthread_t Input_Threads[2];
    pthread_t Hidden_Threads[2];
    pthread_t Output_Thread;

    double inputs[1][Input_Neurons] = { {Inputs[0], Inputs[1]} };
    double weights[Number_of_Layers][Hidden_Layer_Neurons][Hidden_Layer_Neurons];
    double outputs[1][Output_Neurons] = {0};  // Initialize to zero


    for(int i = 0; i<Number_of_Layers; i++)
    {
        Layers[i] = fork();
        if(Layers[i]==-1)
        {
            cout<<"Error Creating Layer: "<<i<<endl;
        }
        else if(Layers[i]==0)
        {
            if(i==0)
            {
                for(int i = 0; i<2; i++)
                {
                    pthread_create(&Input_Threads[i],NULL,First_Layer_Thread,NULL);
                    forwardPropagation(inputs, weights, outputs);

                }
            }
            if(i>0 && i<6)
            {
                for(int i = 0; i<6; i++)
                {
                    pthread_create(&Hidden_Threads[i],NULL,Hidden_Layer_Thread,NULL);
                }
            }
            if(i==7)
            {
                pthread_create(&Output_Thread,NULL,Output_Layer_Thread,NULL);
            }
        }
    }

    //write inputs to first layer
    write(Layer_Pipes[0][1],&Inputs,sizeof(Inputs));
    forwardPropagation(inputs, weights, outputs);

    for(int i = 0; i<8; i++)
    {   
        //Forward Propogation For Output Calculation
        cout<<"\n\nIteration---"<<i<<endl;
        forwardPropagation(inputs, weights, outputs);
        
       //Performing Back Propagation for Returning New Inputs
       for(int j = Number_of_Layers-1; j>0; j--)
        {
            pthread_create(&Back_Prob[j], NULL,Back_Propagation,&j);
        }

        for(int k = Number_of_Layers-1; k>0; k--)
        {
            //Join the threads
            pthread_join(Back_Prob[k], NULL);
        }
    }
        
}