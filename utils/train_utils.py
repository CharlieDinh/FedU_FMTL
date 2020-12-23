from torchvision import datasets, transforms
from models.Nets import *

def get_model(args):
    if args.model == 'cnn' and args.dataset in ['Cifar10', 'Cifar100']:
        net_glob = CNNCifar(args=args).to(args.device)
    elif args.model == 'cnn' and args.dataset == 'Mnist':
        net_glob = CNNMnist(args=args).to(args.device)
    elif args.model == 'mlp' and args.dataset == 'Mnist':
        net_glob = MLP(dim_in=784, dim_hidden=256, dim_out=args.num_classes).to(args.device)
    elif args.model == 'mclr' and args.dataset == 'Mnist':
        net_glob = Mclr_Logistic().to(args.device)
    elif args.model == 'mclr' and args.dataset == 'human_activity':
        net_glob = Mclr_Logistic(561,6).to(args.device)
    else:
        exit('Error: unrecognized model')
    print(net_glob)

    return net_glob