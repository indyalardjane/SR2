import torch
from torch.optim.optimizer import Optimizer, required
from copy import deepcopy
import numpy as np


class SR2optim(Optimizer):
    """Implementation of SR2 algorithm
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
    """

    def __init__(self, params, nu1=1e-4, nu2=0.9, g1=1.5, g2=1.25, g3=0.5, lmbda=0.001, sigma=0.75,
                 weight_decay=0.2):
        if not 0.0 <= nu1 < 1:
            raise ValueError("Invalid nu1 parameter: {}".format(nu1))
        if not 0.0 <= nu2 < 1.0:
            raise ValueError("Invalid nu1 parameter: {}".format(nu2))
        if not nu1 <= nu2:
            raise ValueError("nu1 should be lower than nu2")
        if not g1 > 1.0:
            raise ValueError("Invalid g1 parameter: {}".format(g1))
        if not g2 <= g1:
            raise ValueError("Invalid g2 value: {}".format(g2))
        if not 0 < g3 <= 1:
            raise ValueError("Invalid g3 value: {}".format(g3))

        self.successful_steps = 0
        self.failed_steps = 0
        self.stop_counter = 0
        defaults = dict(nu1=nu1, nu2=nu2, g1=g1, g2=g2, g3=g3, lmbda=lmbda, sigma=sigma,
                        weight_decay=weight_decay)
        super(SR2optim, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SR2optim, self).__setstate__(state)

    def _copy_params(self):
        current_params = []
        for param in self.param_groups[0]['params']:
            current_params.append(deepcopy(param.data))
        return current_params

    def _load_params(self, current_params):
        i = 0
        for param in self.param_groups[0]['params']:
            param.data[:] = current_params[i]
            i += 1

    def get_step(self, x, grad, sigma, lmbda):
        step = torch.zeros_like(x.data)
        return step

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        # load parameters
        group = self.param_groups[0]
        sigma = group['sigma']
        lmbda = group['lmbda']

        loss = None
        h_x = None
        if closure is not None:
            loss, h_x = closure()

        loss.backward()
        f_x = loss.item()
        h_x *= lmbda
        current_obj = f_x + h_x
        l1 = h_x

        # saving the parameters in case the step is rejected
        current_params = self._copy_params()

        norm_s = 0
        phi_x = f_x
        gts = 0
        stop = False

        for x in group['params']:
            if x.grad is None:
                continue

            # Perform weight-decay
            x.data.mul_(1 - 0.001 * group['weight_decay'])

            grad = x.grad.data

            state = self.state[x]
            if len(state) == 0:
                state['s'] = torch.zeros_like(x.data)

            # Compute the step s
            state['s'].data = self.get_step(x, grad, sigma, lmbda)
            norm_s += torch.sum(torch.square(state['s'])).item()

            # phi(x+s) ~= f(x) + grad^T * s
            flat_g = grad.view(-1)
            flat_s = state['s'].view(-1)
            gts += torch.dot(flat_g, flat_s).item()
            phi_x += torch.dot(flat_g, flat_s).item()

            # Update the weights
            x.data = x.data.add_(state['s'].data)


        # f(x+s), h(x+s)
        fxs, hxs = closure()
        hxs *= lmbda

        # Compute model
        m_s = phi_x + hxs + 0.5 * sigma * norm_s ** 2
        xi = current_obj - m_s

        # Rho
        rho = current_obj - (fxs.item() + hxs)
        try:
            rho /= current_obj - (phi_x + hxs)
            self.stop_counter = 0
        except ZeroDivisionError:
            rho = 0
            self.stop_counter += 1

        if self.stop_counter > 30:
            stop = True

        criteria = (torch.abs(f_x - fxs - gts) / norm_s ** 2).item()

        # Updates
        if rho >= self.param_groups[0]['nu1']:
            loss = fxs
            l1 = hxs
            loss.backward()
            self.successful_steps += 1
        else:
            # Reject the step
            self._load_params(current_params)
            group['sigma'] *= group['g1']
            self.failed_steps += 1
            print('> Failed step')

        if rho >= self.param_groups[0]['nu2']:
            group['sigma'] *= group['g3']

        return loss, l1, norm_s, xi, group['sigma'], rho, criteria, stop


class SR2optiml1(SR2optim):
    def __init__(self, params, nu1=1e-4, nu2=0.9, g1=1.5, g2=1.25, g3=0.5, lmbda=0.001, sigma=0.75,
                 weight_decay=0.2):
        super().__init__(params, nu1=nu1, nu2=nu2, g1=g1, g2=g2, g3=g3, lmbda=lmbda, sigma=sigma,
                         weight_decay=weight_decay)

    def get_step(self, x, grad, sigma, lmbda):
        step = torch.max(x.data - grad / sigma - (lmbda / sigma), torch.zeros_like(x.data)) - \
               torch.max(-x.data + grad / sigma - (lmbda / sigma), torch.zeros_like(x.data)) - x.data
        return step


class SR2optiml0(SR2optim):
    def __init__(self, params, nu1=1e-4, nu2=0.9, g1=1.5, g2=1.25, g3=0.5, lmbda=0.001, sigma=0.75,
                 weight_decay=0.2):
        super().__init__(params, nu1=nu1, nu2=nu2, g1=g1, g2=g2, g3=g3, lmbda=lmbda, sigma=sigma,
                         weight_decay=weight_decay)

    def get_step(self, x, grad, sigma, lmbda):
        step = torch.where(torch.abs(x.data - grad / sigma) >= np.sqrt(2 * lmbda / sigma),
                           -grad / sigma, -x.data)
        return step
