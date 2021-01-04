from torchvision import datasets, transforms
from FLAlgorithms.trainmodel.models import *

def get_model(args):
    if(args.model == "mclr"):
        if(args.dataset == "human_activity"):
            model = Mclr_Logistic(561,6).to(args.device)
        elif(args.dataset == "gleam"):
            model = Mclr_Logistic(561,6).to(args.device)
        elif(args.dataset == "vehicle_sensor"):
            model = Mclr_Logistic(100,2).to(args.device)
        elif(args.dataset == "Synthetic"):
            model = Mclr_Logistic(60,10).to(args.device)
        else:#(dataset == "Mnist"):
            model = Mclr_Logistic().to(args.device)

    elif(args.model == "dnn"):
        if(args.dataset == "human_activity"):
            model = DNN2(561,100,100,12).to(args.device)
        elif(args.dataset == "gleam"):
            model = DNN(561,20,6).to(args.device)
        elif(args.dataset == "vehicle_sensor"):
            model = DNN(100,20,2).to(args.device)
        elif(args.dataset == "Synthetic"):
            model = DNN(60,20,10).to(args.device)
        else:#(dataset == "Mnist"):
            model = DNN2().to(args.device)
        
    elif(args.model == "cnn"):
        if(args.dataset == "Cifar10"):
            model = CNNCifar().to(args.device)
    else:
        pasexit('Error: unrecognized model')
    return model