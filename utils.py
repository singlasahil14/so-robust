from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

class Net(nn.Module):
    def __init__(self, num_layers, nonlin='softplus'):
        super(Net, self).__init__()
        self.num_classes = 10
        self.num_layers = num_layers
        self.input_shape = (1, 28, 28)
        self.nonlin = nonlin
        if num_layers == 2:
            self.fc1 = nn.Linear(784, 1024)
            self.fc2 = nn.Linear(1024, 10)
        elif num_layers == 3:
            self.fc1 = nn.Linear(784, 1024)
            self.fc2 = nn.Linear(1024, 1024)
            self.fc3 = nn.Linear(1024, 10)
        elif num_layers == 4:
            self.fc1 = nn.Linear(784, 1024)
            self.fc2 = nn.Linear(1024, 1024)
            self.fc3 = nn.Linear(1024, 1024)
            self.fc4 = nn.Linear(1024, 10)
        else:
            raise ValueError('Invalid number of layers')

    def forward(self, x):
        num_layers = self.num_layers
        if self.nonlin == 'softplus':
            fn = F.softplus
        if self.nonlin == 'sigmoid':
            fn = torch.sigmoid
        elif self.nonlin == 'tanh':
            fn = torch.tanh
        x = x.view(x.size(0), -1)
        z = self.fc1(x)
        x = fn(z)
        if num_layers==2:
            z = self.fc2(x)
        elif num_layers==3:
            z = self.fc2(x)
            x = fn(z)
            z = self.fc3(x)
        elif num_layers==4:
            z = self.fc2(x)
            x = fn(z)
            z = self.fc3(x)
            x = fn(z)
            z = self.fc4(x)
        return z

def power_iteration(W1, W2=None, num_iters=50, return_vectors=False, verbose=False):
    if W2 is None:
        W2 = torch.ones(1, W1.shape[0], device='cuda')
    x = torch.randn((1, W1.shape[1]), device='cuda')
    x_norm = torch.norm(x, dim=1, keepdim=True)
    x_n = x/x_norm
    for i in range(num_iters):
        x = torch.matmul(torch.matmul(x_n, W1.t())*W2, W1)
        x_norm = torch.norm(x, dim=1, keepdim=True)
        x_n = x/x_norm
    if return_vectors:
        y = torch.matmul(x_n, W1.t())
        y_norm = torch.norm(y, dim=1, keepdim=True)
        y_n = y/y_norm
        return x_norm[0, 0], y_n, x_n
    return x_norm

class MatrixNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight):
        sigma_squared, u, v = power_iteration(weight, return_vectors=True, verbose=True)
        sigma = torch.sqrt(sigma_squared)
        ctx.save_for_backward(weight, u, v)
        return sigma

    @staticmethod
    def backward(ctx, grad_output):
        weight, u, v = ctx.saved_tensors
        grad_weight = grad_output.clone()
        return grad_weight*((u.t()).mm(v))

def lipschitz_bound(model, true_label, false_target):
    if model.nonlin == 'softplus':
        g = 1.
        h = 0.25
    elif model.nonlin == 'sigmoid':
        g = 0.25
        h = 0.09623
    elif model.nonlin == 'tanh':
        g = 1.
        h = 0.7699
    params = list(model.parameters())
    if model.num_layers == 2:
        W1 = params[0]
        W2 = params[2]

        W2_label = W2[true_label]
        W2_target = W2[false_target]
        W2_eff = (W2_label - W2_target)

        W1_sigma = MatrixNorm.apply(W1)
        W2_diag = torch.abs(W2_eff)
        L = h*W1_sigma*W1_sigma*torch.max(W2_diag, dim=1)[0]
    elif model.num_layers == 3:
        W1 = params[0]
        W2 = params[2]
        W3 = params[4]
    
        W3_label = W3[true_label]
        W3_target = W3[false_target]
        W3_eff = (W3_label - W3_target)
    
        W1_sigma = MatrixNorm.apply(W1)
        W2_sigma = MatrixNorm.apply(W2)
        W1_W2_sigma = W1_sigma*W2_sigma*g
    
        W3_diag = torch.abs(W3_eff)
        left_L = (W1_W2_sigma*W1_W2_sigma)*torch.max(W3_diag, dim=1)[0]
    
        W2_diag = g*torch.abs(W3_eff).mm(torch.abs(W2))
        right_L = W1_sigma*W1_sigma*torch.max(W2_diag, dim=1)[0]
    
        L = h*(right_L + left_L)
    elif model.num_layers == 4:
        W1 = params[0]
        W2 = params[2]
        W3 = params[4]
        W4 = params[6]
    
        W4_label = W4[true_label]
        W4_target = W4[false_target]
        W4_eff = (W4_label - W4_target)
    
        W1_sigma = MatrixNorm.apply(W1)
        W2_sigma = MatrixNorm.apply(W2)
        W3_sigma = MatrixNorm.apply(W3)
        W1_W2_sigma = W1_sigma*g*W2_sigma
        W1_W2_W3_sigma = W1_sigma*g*W2_sigma*g*W3_sigma
    
        W4_diag = torch.abs(W4_eff)
        W3_diag = g*(W4_diag.mm(torch.abs(W3)))
        W2_diag = g*(W3_diag.mm(torch.abs(W2)))
    
        left_L = (W1_W2_W3_sigma*W1_W2_W3_sigma)*torch.max(W4_diag, dim=1)[0]
        middle_L = (W1_W2_sigma*W1_W2_sigma)*torch.max(W3_diag, dim=1)[0]
        right_L = (W1_sigma*W1_sigma)*torch.max(W2_diag, dim=1)[0]
    
        L = h*(left_L + middle_L + right_L)
    return L

def curvature_bound(model, true_label, false_target):
    if model.nonlin == 'softplus':
        g = 1.
        h = 0.25
    elif model.nonlin == 'sigmoid':
        g = 0.25
        h = 0.09623
    elif model.nonlin == 'tanh':
        g = 1.
        h = 0.7699
    with torch.no_grad():
        if model.num_layers == 2:
            params = list(model.parameters())
            W1 = params[0]
            W2 = params[2]

            W2_label = W2[true_label]
            W2_target = W2[false_target]
            W2_eff = (W2_label - W2_target)

            if model.nonlin == 'softplus':
                W2_pos = ((W2_eff > 0).float()*W2_eff)
                W2_neg = ((W2_eff < 0).float()*W2_eff)
            elif (model.nonlin == 'sigmoid') or (model.nonlin == 'tanh'):
                W2_pos = torch.abs(W2_eff)
                W2_neg = -torch.abs(W2_eff)

            m = h*power_iteration(W1, W2_neg)
            M = h*power_iteration(W1, W2_pos)
        elif model.num_layers == 3:
            params = list(model.parameters())
            W1 = params[0]
            W2 = params[2]
            W3 = params[4]

            W3_label = W3[true_label]
            W3_target = W3[false_target]
            W3_eff = (W3_label - W3_target)

            W1_sigma = MatrixNorm.apply(W1)*g
            if model.nonlin == 'softplus':
                left_m = power_iteration(W2, W3_eff*(W3_eff<0).float())*W1_sigma*W1_sigma
                left_M = power_iteration(W2, W3_eff*(W3_eff>0).float())*W1_sigma*W1_sigma
            elif (model.nonlin == 'sigmoid') or (model.nonlin == 'tanh'):
                W3_diag = torch.abs(W3_eff)
                left_m = power_iteration(W2, W3_diag)*W1_sigma*W1_sigma
                left_M = left_m.clone()

            W2_tensor = g*W3_eff.unsqueeze(2)*W2.unsqueeze(0)
            W2_neg = ((W2_tensor < 0).float()*W2_tensor).sum(1)
            W2_pos = ((W2_tensor > 0).float()*W2_tensor).sum(1)
            W2_diag = torch.max(W2_neg.abs(), W2_pos.abs())
            if model.nonlin == 'softplus':
                right_m = power_iteration(W1, W2_neg)
                right_M = power_iteration(W1, W2_pos)
            elif (model.nonlin == 'sigmoid') or (model.nonlin == 'tanh'):
                right_m = power_iteration(W1, W2_diag)
                right_M = right_m.clone()

            m = h*(left_m + right_m)
            M = h*(left_M + right_M)
        elif model.num_layers == 4:
            params = list(model.parameters())
            W1 = params[0]
            W2 = params[2]
            W3 = params[4]
            W4 = params[6]

            W4_label = W4[true_label]
            W4_target = W4[false_target]
            W4_eff = (W4_label - W4_target)
    
            W3_tensor = g*W4_eff.unsqueeze(2)*W3.unsqueeze(0)
            W3_neg = ((W3_tensor < 0).float()*W3_tensor).sum(1)
            W3_pos = ((W3_tensor > 0).float()*W3_tensor).sum(1)
            W3_diag = torch.max(W3_neg.abs(), W3_pos.abs())
            W2_diag = g*(W3_diag.mm(torch.abs(W2)))
    
            left_m = power_iteration(W1, W2_diag)
            left_M = left_m.clone()
    
            W1_sigma = MatrixNorm.apply(W1)*g
            if model.nonlin == 'softplus':
                middle_m = power_iteration(W2, W3_neg)*W1_sigma*W1_sigma
                middle_M = power_iteration(W2, W3_pos)*W1_sigma*W1_sigma
            elif (model.nonlin == 'sigmoid') or (model.nonlin == 'tanh'):
                middle_m = power_iteration(W2, W3_diag)*W1_sigma*W1_sigma
                middle_M = middle_m.clone()
    
            W1_W2_sigma = W1_sigma*MatrixNorm.apply(W2)*g
            if model.nonlin == 'softplus':
                right_m = power_iteration(W3, W4_eff*(W4_eff<0).float())*W1_W2_sigma*W1_W2_sigma
                right_M = power_iteration(W3, W4_eff*(W4_eff>0).float())*W1_W2_sigma*W1_W2_sigma
            elif (model.nonlin == 'sigmoid') or (model.nonlin == 'tanh'):
                W4_diag = torch.abs(W4_eff)
                right_m = power_iteration(W3, W4_diag)*W1_W2_sigma*W1_W2_sigma
                right_M = right_m.clone()
    
            m = h*(left_m + middle_m + right_m)
            M = h*(left_M + middle_M + right_M)
    return m, M

def gradient_x(x, model, true_label, false_target):
    x_var = x.clone()
    x_var.requires_grad = True
    batch_size = x_var.shape[0]
    with torch.enable_grad():
        logits = model(x_var)
        logits_diff = logits[torch.arange(batch_size), true_label] - logits[torch.arange(batch_size), false_target]
    grad_x = torch.autograd.grad(logits_diff.sum(), x_var)[0]
    return grad_x

def newton_step(x0, true_label, false_target, model, eps, verbose=True):
    batch_size = x0.shape[0]
    x0 = x0.view(batch_size, -1)
    input_dim = x0.shape[1]

    m, M = curvature_bound(model, true_label, false_target)

    eta = torch.zeros((batch_size, 1)).cuda()
    eta_min = torch.zeros((batch_size, 1)).cuda()
    eta_max = torch.zeros((batch_size, 1)).cuda()

    eta_min = m.clone()
    eta_max = 1000 + m.clone()
    eta = (eta_min + eta_max)/2.

    x = x0.clone()

    outer_iters = 30
    inner_iters = 10
    for i in range(outer_iters):
        for j in range(inner_iters):
            g_batch = gradient_x(x, model, true_label, false_target)

            dual_grad = g_batch - M*x - eta*x0
            dual_hess = M + eta
            x = -torch.reciprocal(dual_hess)*dual_grad

        norm = torch.norm(x - x0, dim=1)
        ge_indicator = (norm > eps)
        eta_min[ge_indicator] = eta[ge_indicator]
        eta_max[~ge_indicator] = eta[~ge_indicator]
        eta = (eta_min + eta_max)/2.

    if verbose:
        grad_norm = torch.norm(g_batch + eta*(x - x0), dim=1).abs().max().item()
        eta_m, eta_M = eta.abs().min().item(), eta.abs().max().item()
        min_norm = torch.norm((x - x0), dim=1).min().item()
        max_norm = torch.norm((x - x0), dim=1).max().item()
        print(grad_norm, m.min().item(), m.max().item(), eta_m,  eta_M, min_norm, max_norm)
    return x

def newton_step_cert(x0, true_label, false_target, model, eps, verbose=True):
    batch_size = x0.shape[0]
    x0 = x0.view(batch_size, -1)
    input_dim = x0.shape[1]

    m, M = curvature_bound(model, true_label, false_target)

    eta = torch.zeros((batch_size, 1)).cuda()
    eta_min = -1/M*torch.zeros((batch_size, 1)).cuda()
    eta_max = 1/m*torch.ones((batch_size, 1)).cuda()
    eta = (eta_min + eta_max)/2.

    x = x0.clone()
    outer_iters = 30
    inner_iters = 20
    for i in range(outer_iters):
        for j in range(inner_iters):
            g_batch = gradient_x(x, model, true_label, false_target)

            dual_grad = eta*g_batch - eta*M*x - x0
            dual_hess = 1 + eta*M
            x = -torch.reciprocal(dual_hess)*dual_grad

        if i < outer_iters:
            logits = model(x)
            logits_diff = logits[torch.arange(batch_size), true_label] - logits[torch.arange(batch_size), false_target]
            ge_indicator = (logits_diff > 0)
            eta_min[ge_indicator] = eta[ge_indicator]
            eta_max[~ge_indicator] = eta[~ge_indicator]
            eta = (eta_min + eta_max)/2.

    dist_sqrd = ((x - x0)*(x - x0)).sum(dim=1) + 2*eta[:, 0]*logits_diff
    lower_bound = torch.sqrt((dist_sqrd>0).float()*dist_sqrd)
    grad_norm = torch.norm(eta*g_batch + (x - x0), dim=1)

    if verbose:
        max_grad_norm = grad_norm.max().item()
        eta_m, eta_M = eta.abs().min().item(), eta.abs().max().item()
        min_norm = torch.norm((x - x0), dim=1).min().item()
        max_norm = torch.norm((x - x0), dim=1).max().item()
        print(max_grad_norm, (1./M).min().item(), (1./M).max().item(), eta_m, eta_M, min_norm, max_norm)
    return lower_bound, grad_norm

def test_cert(args, model, device, test_loader):
    model.eval()

    correct_sum = 0
    correct_cert_sum = 0
    sum_dists = 0
    eps = args.test_epsilon
    for data, label in test_loader:
        data, label = data.to(device), label.to(device)
        targets = adversarial_targets(model, data, label, target_method='runner-up')
        lower_bound, grad_norm = newton_step_cert(data, label, targets, model, eps, verbose=False)
        logits = model(data)
        pred = logits.argmax(dim=1)  # get the index of the max log-probability

        correct_ind = (pred.eq(label.view_as(pred)))
        correct_cert_ind = (pred.eq(label.view_as(pred))*(lower_bound > eps)*(grad_norm < 1e-5))

        correct_sum += correct_ind.sum()
        correct_cert_sum += correct_cert_ind.sum()
        sum_dists += (correct_ind*lower_bound).sum()

    cert_acc = float(correct_cert_sum.item())/10000
    mean_dists = float(sum_dists.item())/correct_sum.item()
    print('Certified Robust Accuracy: {}, Mean Distances: {}'.format(cert_acc, mean_dists))
    return cert_acc 

def test_standard_adv(args, model, device, test_loader):
    model.eval()
    correct_sum = 0
    correct_attack_sum = 0
    attack_success = 0
    attack_sum = 0

    eps = args.test_epsilon
    num_steps = int(eps*200)
    for data, label in test_loader:
        data, label = data.to(device), label.to(device)
        batch_size = data.shape[0]
        output = model(data)
        pred = output.argmax(dim=1)  # get the index of the max log-probability
        correct = pred.eq(label.view_as(pred))
        correct_sum += correct.sum().item()

        f = adversarial_targets(model, data, label, target_method='runner-up')
        data_attack = adversarial_attack(model, data, label, 0.01, eps, num_steps, 'L2')
        output_attack = model(data_attack)
        pred_attack = output_attack.argmax(dim=1)  # get the index of the max log-probability
        correct_attack = (pred_attack.eq(f.view_as(pred_attack)))

        dists = torch.norm((data_attack - data).view(data.shape[0], -1) , dim=1)

        correct_attack_sum += (correct*(~correct_attack)).sum().item()
        attack_success += (correct*correct_attack).sum()
        attack_sum += (correct*correct_attack*dists).sum()

    acc = correct_sum/len(test_loader.dataset)
    robust_acc = correct_attack_sum/len(test_loader.dataset)

    mean_dists = (attack_sum/attack_success).item()
    print('Standard Accuracy: {}'.format(acc))
    print('Empirical Robust Accuracy: {}, Mean Distances: {}'.format(robust_acc, mean_dists))
    return acc, robust_acc

def train_robust(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, label) in enumerate(train_loader):
        optimizer.zero_grad()
        data, label = data.to(device), label.to(device)
        target = adversarial_targets(model, data, label, 'runner-up')
        if args.epsilon > 0:
            data = newton_step(data, label, target, model, args.epsilon)
        lipschitz_loss = (lipschitz_bound(model, label, target)).mean()

        output = model(data)
        ce_loss = F.cross_entropy(output, label)
        loss = ce_loss + (args.beta)*(lipschitz_loss)
        loss.backward()
        optimizer.step()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLipschitz: {:.6f}\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), lipschitz_loss.item(), loss.item()))

def adversarial_targets(model, data, label, target_method):
    logits = model(data)
    batch_size, num_classes = logits.shape
    if target_method == 'runner-up':
        preds = torch.zeros_like(logits).cuda()
        preds[torch.arange(data.shape[0]), label] = 1e10
        target_index = (logits - preds).argmax(dim=1)
    elif target_method == 'random':
        target_index = torch.randint(num_classes - 1, (batch_size,)).cuda()
        target_index = target_index*(target_index < label).long() + (1 + target_index)*(target_index >= label).long()
    else:
        preds = torch.zeros_like(logits).cuda()
        preds[torch.arange(data.shape[0]), label] = 1e10
        target_index = (logits + preds).argmin(dim=1)
    return target_index

def adversarial_attack(model, x_natural, y, step_size, epsilon, perturb_steps, distance):
    batch_size = len(x_natural)
    f = adversarial_targets(model, x_natural, y, target_method='runner-up')
    x_adv = x_natural.detach() + 0.01 * torch.randn(x_natural.shape).cuda().detach()
    x_adv = x_adv.view(batch_size, -1)
    if distance == 'Linf':
       for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                logits = model(x_adv)
                loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural.view(batch_size, -1) - epsilon), x_natural.view(batch_size, -1) + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    elif distance == 'L2':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                logits = model(x_adv)
                logits_diff = logits[torch.arange(batch_size), f] - logits[torch.arange(batch_size), y]
                loss = logits_diff.sum()
            grad = torch.autograd.grad(loss, [x_adv])[0]
            grad_norm = torch.norm(grad, dim=1, keepdim=True)
            grad_normalized = grad/(grad_norm + 1e-10)
            x_adv = x_adv.detach() + step_size * grad_normalized
            diff_x_adv = x_adv - x_natural.view(batch_size, -1)
            norm_diff = torch.norm(diff_x_adv, dim=1)
            normalized_diff = ((diff_x_adv * epsilon)/norm_diff.unsqueeze(1))
            diff_x_adv[norm_diff > epsilon] = normalized_diff[norm_diff > epsilon]
            x_adv = x_natural.view(batch_size, -1) + diff_x_adv
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    x_adv = x_adv.view_as(x_natural)
    return x_adv

