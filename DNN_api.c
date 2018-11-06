//
//  DNN_api.c
//  DNN_PredictCall
//
//  Created by gulian on 2018/10/26.
//  Copyright © 2018 gulian. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "DNN_api.h"

#define OUT

/*
 *   sigmoid function
 */
static double sigmoid(double in)
{
    return 1.0 / (1.0 + exp(-1.0 * in));
}

/* get number of all ws in the model*/
static int AllNum_W(S_DNN_Model *model)
{
    int sum = 0;
    S_Model_Parameters *p_mp = &model->model_parameters;
    sum += p_mp->input_num * p_mp->numOfeveryLayer[0];
    for (int i=1; i<p_mp->layers_num; i++)
    {
        sum += p_mp->numOfeveryLayer[i] * p_mp->numOfeveryLayer[i-1];
    }
    sum += p_mp->numOfeveryLayer[p_mp->layers_num - 1] * p_mp->output_num;

    return sum;
}

/* get number of all bs in the model*/
static int AllNum_b(S_DNN_Model *model)
{
    int sum = 0;
    int *p_num = model->model_parameters.numOfeveryLayer;
    for (int i=0; i<model->model_parameters.layers_num; i++)
    {
        sum += p_num[i];
    }
    
    sum += model->model_parameters.output_num;
    
    return sum;
}

/* generate gauss random number for initializing w */
double gaussrand()
{
    static double U, V;
    static int phase = 0;
    double Z;
    if(phase == 0)
    {
        U = rand() / (RAND_MAX + 1.0);
        V = rand() / (RAND_MAX + 1.0);
        Z = sqrt(-2.0 * log(U))* sin(2.0 * 3.141592654 * V);
    }
    else
    {
        Z = sqrt(-2.0 * log(U)) * cos(2.0 * 3.141592654 * V);
    }
    phase = 1 - phase;
    return Z;
}
/*
 *   create DNN(Deep Neuron Network) model
 */
S_DNN_Model *CreateDnnModel(int input_num,
                            int output_num,
                            int layers_num,
                            int *layers_Array,
                            char activeFunc[])
{
    S_DNN_Model *p_model = malloc(sizeof(S_DNN_Model));
    /* step 1: set up model parameters */
    p_model->model_parameters.input_num = input_num;
    p_model->model_parameters.output_num = output_num;
    p_model->model_parameters.layers_num = layers_num;
    p_model->model_parameters.numOfeveryLayer = layers_Array;
    strcpy(p_model->model_parameters.activeFunction, activeFunc);
    
    /* step 2: initialize model's w and b, index of w and dw start with 0, other with 1. */
    int num_w = AllNum_W(p_model);
    int num_b = AllNum_b(p_model);
    p_model->w = malloc(sizeof(double) * num_w);
    p_model->dw = malloc(sizeof(double) * num_w);
    p_model->b = malloc(sizeof(double) * (num_b + 1));
    p_model->db = malloc(sizeof(double) * (num_b + 1));
    //initialize b
    memset(p_model->b, 0, sizeof(sizeof(double) * (num_b + 1)));
    //initialize w
    double *w = p_model->w;
    for(int i = 0; i < num_w; i++)
        w[i] = 0.1 * gaussrand();
    
    return p_model;
}

S_Train_Parameters *GenerateTrainPara(double lr,
                                      char method[],
                                      int batch_size,
                                      int iteration)
{
    S_Train_Parameters *p_TrainPara = malloc(sizeof(S_Train_Parameters));
    p_TrainPara->learn_rate = lr;
    strcpy(p_TrainPara->optimizer_method, method);
    p_TrainPara->batch_size = batch_size;
    p_TrainPara->iteration = iteration;
    
    return p_TrainPara;
}

/* get neuron number of fronter layer */
static int GetformerLayerNeuNum(int layer, S_Model_Parameters *para)
{
    if (layer == 0)
        return 0;
    if (layer == 1)
        return para->input_num;
    return para->numOfeveryLayer[layer - 2];
}

/* get neuron number of next layer */
static int GetnextLayerNeuNum(int layer, S_Model_Parameters *para)
{
    if (layer == 0)
        return para->numOfeveryLayer[0];
    else if (layer == para->layers_num)
        return para->output_num;
    else
        return para->numOfeveryLayer[layer];
    
}

/* get the neuron number of current layer */
static int GetcurrLayerNeuNum(int layer, S_Model_Parameters *para)
{
    if (layer == 0)
        return para->input_num;
    else if (layer <= para->layers_num)
        return para->numOfeveryLayer[layer - 1];
    else
        return para->output_num;
}

static void updateSum(double *sum_dw, double *dw, int Num)
{
    for (int i = 0; i < Num; i++)
    {
        sum_dw[i] += dw[i];
    }
}

/*
 *  forward propagation
 */
static double *forward_propagation(double *X,
                         S_DNN_Model *model,
                         int sample_i,
                         double *z_output,
                         double *a_output)
{
    double *currInput = NULL;  //pointer which point to current Input
    double *hide_units_output = NULL;  //指向隐藏神经元输出数组的指针,将结果返回到函数外
    double *avtive_value_output = NULL;  //指向隐藏神经元激活函数输出数组的指针，将结果返回到函数外
    
    double *curr_hide_units = NULL;  //指向当前要进行计算的隐藏神经元指针
    double *curr_hide_active_units = NULL;  //指向当前要进行计算的隐藏神经元指针
    double *curr_active_value = NULL;  //指向当前用于计算的隐藏神经元激活值指针
    double *w = model->w;
    double *b = model->b;
    double *curr_w = NULL;
    double *curr_b = NULL;
    int input_num = 0;
    int DnnlayerNum = 0;
    int neuron_num = 0;  //all neurons's number in hide layers
    int NextLayer_neuronNum = 0;
    int currLayer_neuronNum = 0;
    
    input_num = model->model_parameters.input_num;
    DnnlayerNum = model->model_parameters.layers_num;
    neuron_num = AllNum_b(model);  //the number of hide layer's b is equal to neuron
    hide_units_output = z_output;
    avtive_value_output = a_output;
    currInput = X + sample_i * input_num;
    
    //forward propagate from input layer to 1st hide layer
    int layer1_NeuNum = GetnextLayerNeuNum(0, &model->model_parameters);
    double tmp_sum;
    for (int neuron = 1; neuron <= layer1_NeuNum; neuron++)
    {
        tmp_sum = 0.0;
        for (int in = 0; in < input_num; in++)
        {
            int tmp = input_num * (neuron - 1) + in;
            tmp_sum += currInput[in] * w[tmp];
        }
        tmp_sum += b[neuron];
        hide_units_output[neuron] = tmp_sum;
        //compute activation value with sigmoid
        avtive_value_output[neuron] = sigmoid(tmp_sum);
    }
    
    curr_w = w + input_num * layer1_NeuNum;
    curr_b = b + layer1_NeuNum + 1;
    curr_hide_units = hide_units_output + layer1_NeuNum + 1;
    curr_hide_active_units = avtive_value_output + layer1_NeuNum + 1;
    curr_active_value = avtive_value_output + 1;
    for (int layer = 1; layer <= DnnlayerNum; layer++) //layer starts with 1, and 1 represents the 1st layer of hide layers
    {
        currLayer_neuronNum = GetcurrLayerNeuNum(layer, &model->model_parameters);
        NextLayer_neuronNum = GetnextLayerNeuNum(layer, &model->model_parameters);
        
        for (int neuron = 1; neuron <= NextLayer_neuronNum; neuron++)
        {
            tmp_sum = 0.0;
            for (int in = 0; in < currLayer_neuronNum; in++)
            {
                int tmp = currLayer_neuronNum * (neuron - 1) + in;
                tmp_sum += curr_active_value[in] * curr_w[tmp];
            }
            tmp_sum += curr_b[neuron - 1];
            curr_hide_units[neuron - 1] = tmp_sum;
            //compute activation value with sigmoid
            curr_hide_active_units[neuron - 1] = sigmoid(tmp_sum);
        }
        
        //update pointers for compution of next layer
        if(layer != DnnlayerNum)
        {
            curr_w += currLayer_neuronNum * NextLayer_neuronNum;
            curr_b += NextLayer_neuronNum;
            curr_active_value += currLayer_neuronNum;
            curr_hide_units += NextLayer_neuronNum;
            curr_hide_active_units += NextLayer_neuronNum;
        }
    }
    
    //return pointer which points to first output neuron
    int first_output_num = neuron_num - model->model_parameters.output_num + 1;
    return &hide_units_output[first_output_num];
}

/*
 *  back propagation
 */
void back_propagation(S_DNN_Model *model,
                      double x,
                      double *dz,
                      double *active_unit_output)
{
    int layers_num = model->model_parameters.layers_num;  //number of hide layers
    S_Model_Parameters *p_mp = &model->model_parameters;
    double *w = NULL;
    double *dw = NULL;
    double *db = NULL;
    double *curr_dz = NULL;  //pointer which points to current dz
    double *curr_w = NULL;  //pointer which points to current w
    double *curr_dw = NULL;  //pointer which points to current dw
    double *curr_db = NULL;  //pointer which points to current db
    double *curr_active_unit_output = NULL;
    double *last_dz = NULL;
    int neuron_num = 0;
    int w_num = 0;
    int formerLayer_neuNum = 0;
    int currLayer_NeuNum = 0;
    
    dw = model->dw;
    db = model->db;
    w = model->w;
    neuron_num = AllNum_b(model);  //the number of hide layer's b is equal to neuron
    w_num = AllNum_W(model);
    
    curr_dz = dz + neuron_num - p_mp->output_num + 1;
    curr_db = db + neuron_num + 1;
    curr_active_unit_output = active_unit_output + neuron_num - p_mp->output_num + 1;
    curr_w = w + w_num;
    curr_dw = dw + w_num;
    
    /* update db of current layer and dz of fronter layer in each loop, and update dw between them. From output layer to 2nd layer of hide layers （每次循环更新当前层的神经元的db和前一层神经元的dz，并且更新他们之间的全连接dw。从输出层遍历到隐藏层第二层。）*/
    for (int layer = layers_num + 1; layer >= 2; layer--)
    {
        //compute neuron numbers of current and fronter layer
        currLayer_NeuNum = GetcurrLayerNeuNum(layer, p_mp);
        formerLayer_neuNum = GetformerLayerNeuNum(layer, p_mp);
        
        //update pointers
        last_dz = curr_dz;
        curr_dz -= formerLayer_neuNum;
        curr_active_unit_output -= formerLayer_neuNum;
        
        curr_db -= currLayer_NeuNum;
        
        curr_w -= currLayer_NeuNum * formerLayer_neuNum;
        curr_dw -= currLayer_NeuNum * formerLayer_neuNum;
        
        //compute db of current layer and dw of fronter layer
        for (int c_neuron = 1; c_neuron <= currLayer_NeuNum; c_neuron++)
        {
            for (int neuron = 1; neuron <= formerLayer_neuNum; neuron++)
            {
                int tmp = formerLayer_neuNum * (c_neuron - 1) + (neuron - 1);
                curr_dw[tmp] = last_dz[c_neuron - 1] * curr_active_unit_output[neuron - 1];
            }
            curr_db[c_neuron - 1] = last_dz[c_neuron - 1];
        }

        //compute dz of every neuron in fronter layer
        for (int neuron = 1; neuron <= formerLayer_neuNum; neuron++)
        {
            curr_dz[neuron - 1] = 0;
            //compute dz
            for (int c_neuron = 1; c_neuron <= currLayer_NeuNum; c_neuron++)
            {
                int tmp = neuron - 1 + formerLayer_neuNum * (c_neuron - 1);
                curr_dz[neuron - 1] += last_dz[c_neuron - 1] * curr_w[tmp];
            }
            double tmp = curr_active_unit_output[neuron - 1];
            double a = tmp * (1 - tmp);
            curr_dz[neuron - 1] *= a;
        }
    }
    
    //compute the dw and db from hide layer 1 to input layer
    currLayer_NeuNum = GetcurrLayerNeuNum(1, p_mp);
    formerLayer_neuNum = p_mp->input_num;
    last_dz = curr_dz;
    curr_db -= currLayer_NeuNum;
    curr_dw -= currLayer_NeuNum * formerLayer_neuNum;
    //compute db and dw
    for (int c_neuron = 1; c_neuron <= currLayer_NeuNum; c_neuron++)
    {
        for (int neuron = 1; neuron <= formerLayer_neuNum; neuron++)
        {
            int tmp = formerLayer_neuNum * (c_neuron - 1) + (neuron - 1);
            curr_dw[tmp] = last_dz[c_neuron - 1] * x;
        }
        curr_db[c_neuron - 1] = last_dz[c_neuron - 1];
    }
    
    return ;
}

void Train_DnnModel(double *X,
                    double *Y,
                    int sample_size,  //number of sample
                    S_DNN_Model *model,
                    S_Train_Parameters *hyperPara)
{
    int iterationNum = 0;
    double once_loss;
    int neuron_num = 0;
    double loss;
    double *hide_unit_output = NULL;
    double *active_unit_output = NULL;
    double *predictValue = NULL;
    double *w = NULL;
    double *b = NULL;
    double *dw = NULL;
    double *db = NULL;
    double *dz = NULL;
    double *sum_dw = NULL;
    double *sum_db = NULL;
    double *curr_Y = NULL; //pointer which point to current output
    double *curr_dz = NULL;
    double lr;
    
    int output_num = model->model_parameters.output_num;
    int WNum = AllNum_W(model);
    neuron_num = AllNum_b(model);  //the number of hide layer's b is equal to neuron
    hide_unit_output = malloc(sizeof(double) * (neuron_num + 1));
    active_unit_output = malloc(sizeof(double) * (neuron_num + 1));
    sum_dw = malloc(sizeof(double) * WNum);
    sum_db = malloc(sizeof(double) * (neuron_num + 1));
    dz = malloc(sizeof(double) * (neuron_num + 1));
    w = model->w;
    b = model->b;
    iterationNum = hyperPara->iteration;
    lr = hyperPara->learn_rate;
    
    //begin iteration
    for (int iter = 1; iter <= iterationNum; iter++)
    {
        /* new round iteration */
        
        loss = 0;
        memset(sum_dw, 0, sizeof(double) * WNum);
        memset(sum_db, 0, sizeof(double) * (neuron_num + 1));
        for (int i = 0; i < sample_size; i++)
        {
            memset(dz, 0, sizeof(double) * (neuron_num + 1));
            memset(hide_unit_output, 0, sizeof(double) * (neuron_num + 1));
            memset(active_unit_output, 0, sizeof(double) * (neuron_num + 1));
            curr_Y = Y + i * output_num;
            curr_dz = dz + neuron_num - output_num + 1;
            predictValue = forward_propagation(X, model, i, OUT hide_unit_output, OUT active_unit_output);
            //compute loss
            once_loss = 0;
            for (int neuron = 0; neuron < output_num; neuron++)
            {
                once_loss += 0.5 * pow((predictValue[neuron] - curr_Y[neuron]), 2);
                curr_dz[neuron] = predictValue[neuron] - curr_Y[neuron];
            }
            loss += once_loss;
            
            //back propagation
            back_propagation(model, X[i], dz, active_unit_output);
            dw = model->dw;
            db = model->db;
            updateSum(sum_dw, dw, WNum);
            updateSum(sum_db, db, neuron_num + 1);
        }
        
        //output the average loss each time
        loss /= sample_size;
        printf("iteration %d--loss: %f\n", iter, loss);
        //update dw
        for (int i = 0; i < WNum; i++)
        {
            w[i] -= lr * (sum_dw[i] / sample_size);
        }
        //update db
        for (int i = 1; i <= neuron_num; i++)
        {
            b[i] -= lr * (sum_db[i] / sample_size);
        }
    }
    
    //free tmp memory
    free(hide_unit_output);
    free(active_unit_output);
    free(sum_db);
    free(sum_dw);
    
    return ;
}

/*
 *  predict with the model
 */
double *Predict(double *X, S_DNN_Model *model)
{
    int neuron_num = AllNum_b(model);  //the number of hide layer's b is equal to neuron
    double *hide_unit_output = malloc(sizeof(double) * (neuron_num + 1));
    double *active_unit_output = malloc(sizeof(double) * (neuron_num + 1));
    double *result = forward_propagation(X, model, 0, hide_unit_output, active_unit_output);
    return result;
}


/*
 *  print model's w and b
 */
void PrintW(S_DNN_Model *model)
{
    //print w
    double *w = model->w;
    int layers = model->model_parameters.layers_num + 1;
    for (int layer = 1; layer <= layers; layer++)
    {
        printf("layer %d:\n", layer);
        int curr_num = GetcurrLayerNeuNum(layer, &model->model_parameters);
        int former_num = GetformerLayerNeuNum(layer, &model->model_parameters);
        int length = curr_num * former_num;
        for(int i = 1; i <= length; i++)
        {
            printf("%f ", *w);
            w ++;
            if (i % former_num == 0)
                printf("\n");
        }
        printf("\n");
    }
    
    return ;
}

void PrintB(S_DNN_Model *model)
{
    double *b = model->b;
    int layers = model->model_parameters.layers_num;
    
    b++;
    for (int layer = 1; layer <= layers + 1; layer++)
    {
        printf("hide_layer %d:\n", layer);
        int neuron_num = GetcurrLayerNeuNum(layer, &model->model_parameters);
        for (int i = 0; i < neuron_num; i++)
        {
            printf("%f", *b);
            b++;
            printf(" ");
        }
        printf("\n");
    }
    
    return ;
}

void freeModel(S_DNN_Model *model)
{
    if (model->b != NULL)
    {
        free(model->b);
    }
    if (model->db != NULL)
    {
        free(model->db);
    }
    if (model->w != NULL)
    {
        free(model->w);
    }
    if (model->dw != NULL)
    {
        free(model->dw);
    }

    return ;
}
