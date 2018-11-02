//
//  main.c
//  DNN_PredictCall
//
//  Created by gulian on 2018/10/26.
//  Copyright © 2018 gulian. All rights reserved.
//

#include <stdio.h>
#include <stdlib.h>
#include "DNN_api.h"

int main(int argc, const char * argv[]) {
    double X[] = {1,2,3,4,5,6,7,8,9};
    double Y[] = {2,4,6,8,10,12,14,16,18};
    
    int layers[] = {50,30};
    S_DNN_Model *dnn_Model = CreateDnnModel(1, 1, 2, layers, "sigmoid");
    S_Train_Parameters *train_para = GenerateTrainPara(0.01, "GD", 100, 20000);
    Train_DnnModel(X, Y, 9, dnn_Model, train_para);
    
    //输出w和b
    PrintW(dnn_Model);
    PrintB(dnn_Model);
    double x = 3;
    double result = *Predict(&x, dnn_Model);
    printf("x = 3, predict result: %f\n", result);
    x = 4;
    result = *Predict(&x, dnn_Model);
    printf("x = 4, predict result: %f\n", result);
    x = 10;
    result = *Predict(&x, dnn_Model);
    printf("x = 10, predict result: %f\n", result);
    
    return 0;
}