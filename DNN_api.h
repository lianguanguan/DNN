//
//  DNN_api.h
//  DNN_PredictCall
//
//  Created by gulian on 2018/10/26.
//  Copyright © 2018 gulian. All rights reserved.
//

#ifndef DNN_api_h
#define DNN_api_h


//模型参数
typedef struct Model_parameters{
    int input_num;
    int layers_num;  //不包括input_num和output_num
    int output_num;
    int *numOfeveryLayer;  //指向一个整形数组，长度为layers_num
    char activeFunction[10];
}S_Model_Parameters;

//训练超参数
typedef struct Train_parameters{
    int batch_size;  //批量个数
    int iteration;  //迭代次数
    double learn_rate;  //训练过程学习率
    char optimizer_method[6];  //优化方法（梯度下降、随机梯度下降、动量、adam）
}S_Train_Parameters;

//神经网络模型
typedef struct DNN_Model{
    S_Model_Parameters model_parameters;
    double *w;
    double *b;
    double *dw;  //用于计算梯度时使用
    double *db;  //用于计算梯度时使用
}S_DNN_Model;

S_DNN_Model *CreateDnnModel(int input_num,
                            int output_num,
                            int layers_num,
                            int *layers_Array,
                            char activeFunc[]);

S_Train_Parameters *GenerateTrainPara(double lr,
                                      char *method,
                                      int batch_size,
                                      int iteration);

void Train_DnnModel(double *X,
                    double *Y,
                    int sample_size,  //样本个数
                    S_DNN_Model *model,
                    S_Train_Parameters *hyperPara);

/*
 *  用模型进行预测
 */
double *Predict(double *X, S_DNN_Model *model);

double gaussrand();

void PrintW(S_DNN_Model *model);

void PrintB(S_DNN_Model *model);

#endif /* DNN_api_h */
