from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

from utils import Net, train_robust, test_standard_adv, test_cert

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: %(default)s)')
    parser.add_argument('--test-batch-size', type=int, default=100, metavar='N',
                        help='input batch size for testing (default: %(default)s)')
    parser.add_argument('--epochs', type=int, default=30, metavar='N',
                        help='number of epochs to train (default: %(default)s)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: %(default)s)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: %(default)s)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status (default: %(default)s)')
    parser.add_argument('--dataset', choices=['mnist', 'fashion-mnist'], default='mnist', metavar='D',
                        help='mnist/fashion-mnist (default: %(default)s)') 
    parser.add_argument('--nonlin', choices=['softplus', 'sigmoid', 'tanh'], 
                        default='softplus', metavar='D',
                        help='softplus/sigmoid/tanh (default: %(default)s)') 
    parser.add_argument('--num-layers', choices=['2', '3', '4'], 
                        default=2, metavar='N',
                        help='2/3/4 (default: %(default)s)') 
    parser.add_argument('--epsilon', type=float, default=1.58, 
                        metavar='E', help='ball radius (default: %(default)s)')
    parser.add_argument('--test-epsilon', type=float, default=1.58, 
                        metavar='E', help='ball radius (default: %(default)s)')
    parser.add_argument('--step-size', type=float, default=0.005, 
                        metavar='L', help='step size for finding adversarial example (default: %(default)s)')
    parser.add_argument('--num-steps', type=int, default=200, 
                        metavar='L', help='number of steps for finding adversarial example (default: %(default)s)')
    parser.add_argument('--beta', type=float, default=0.005, 
                        metavar='L', help='regularization coefficient for Lipschitz (default: %(default)s)')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if args.dataset=='mnist':
        dataset = datasets.MNIST
    elif args.dataset=='fashion-mnist': 
        dataset = datasets.FashionMNIST
    else:
        raise ValueError('Unknown dataset %s', args.dataset)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        dataset('./' + args.dataset, train=True, download=True,
                transform=transforms.Compose([
                transforms.ToTensor()])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        dataset('./' + args.dataset, train=False, 
                transform=transforms.Compose([
                transforms.ToTensor()])),
                batch_size=args.test_batch_size, 
                shuffle=False, **kwargs)

    model = Net(int(args.num_layers), args.nonlin).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    model_name = 'saved_models/' + args.dataset + '_' + str(args.num_layers) + '_' + args.nonlin + '_L2_' + str(args.epsilon) + '_EIGEN_' + str(args.beta)

    print(args)
    print(model_name)

    acc, empirical_acc = test_standard_adv(args, model, device, test_loader)
    certified_acc = test_cert(args, model, device, test_loader)

    best_acc = 0.
    best_empirical_acc = 0.
    best_certified_acc = 0.
    for epoch in range(1, args.epochs + 1):
        train_robust(args, model, device, train_loader, optimizer, epoch)
        acc, empirical_acc = test_standard_adv(args, model, device, test_loader)
        certified_acc = test_cert(args, model, device, test_loader)

        if acc > best_acc:
            best_acc = acc
            best_empirical_acc = empirical_acc
            best_certified_acc = certified_acc
            torch.save(model.state_dict(), model_name)
        print('Saved model: Accuracy: {:.4f}, Empirical Robust Accuracy: {:.4f}, Certified Robust Accuracy: {:.4f}\n'.\
            format(best_acc, best_empirical_acc, best_certified_acc))

if __name__ == '__main__':
    main()
