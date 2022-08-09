import torch
from torch.optim.optimizer import Optimizer, required
from copy import deepcopy
import numpy as np
import logging


class SR2optim(Optimizer):
    """Implementation of SR2 algorithm
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
    """

    def __init__(self, params, nu1=1e-4, nu2=0.9, g1=1.5, g2=1.25, g3=0.5, lmbda=0.001, sigma=0.75,
                 weight_decay=0.2):
        if not 0.0 <= nu1 <= nu2 < 1:
            raise ValueError("Invalid parameter: 0 <= {} <= {} < 1".format(nu1, nu2))
        if not g1 > 1.0:
            raise ValueError("Invalid g1 parameter: {}".format(g1))
        if not 0 < g3 <= 1:
            raise ValueError("Invalid g3 value: {}".format(g3))

        self.successful_steps = 0
        self.failed_steps = 0
        self.stop_counter = 0
        
        logging.basicConfig(level=logging.DEBUG)
        defaults = dict(nu1=nu1, nu2=nu2, g1=g1, g2=g2, g3=g3, lmbda=lmbda, sigma=sigma, weight_decay=weight_decay)
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

    def update_weights(self, x, step, grad, sigma):
        res = torch.zeros_like(x.data)
        return res

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
        l = h_x

        # saving the parameters in case the step is rejected
        current_params = self._copy_params()

        norm_s = 0
        phi_x = f_x
        gts = 0
        stop = False
        do_updates = True

        for x in group['params']:
            if x.grad is None:
                continue

            # Perform weight-decay
            x.data.mul_(1 - 0.001 * group['weight_decay'])

            grad = x.grad.data

            # Direction with momentum

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

            # Update the weights
            self.update_weights(x, state['s'], grad, sigma)

        
        phi_x += gts
        # f(x+s), h(x+s)
        fxs, hxs = closure()
        hxs *= lmbda

        delta_model= current_obj - (phi_x + hxs)
        
        if delta_model  < -1e-4:
            logging.error('denominator is negatif {}: '.format(delta_model))
            logging.debug('current_objectif = {}: '.format(current_obj))
            logging.debug('phi = {}: '.format(phi_x))
            logging.debug('h(x+s) = {}: '.format(hxs))
            sys.exit()

        elif -1e-4 <= delta_model <= 0:
            rho = 0
            self.stop_counter += 1
            logging.info('denominator of rho is slightly negatif  {}:'.format(delta_model))
            do_updates = False
        else:
            rho = (current_obj - fxs - hxs) / delta_model
            self.stop_counter = 0
            
        if self.stop_counter > 30:
            stop = True

        # Updates
        if do_updates:
            if rho >= self.param_groups[0]['nu1']:
                logging.debug('step accepted')
                loss = fxs
                l = hxs
                loss.backward()
                self.successful_steps += 1
            else:
                # Reject the step
                logging.debug('step rejected')
                self._load_params(current_params)
                group['sigma'] *= group['g1']
                self.failed_steps += 1
                logging.debug('>> sigma = {} '.format(group['sigma']))

            if rho >= self.param_groups[0]['nu2']:
                group['sigma'] *= group['g3']
                logging.debug('>> sigma = {} '.format(group['sigma']))

        return loss, l, norm_s, group['sigma'], rho, stop


class SR2optiml23(SR2optim):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_step(self, x, grad, sigma, lmbda):
        X = x.data - grad / sigma
        L = 2* lmbda/sigma
        phi = torch.arccosh(27/16 * (X**2) * (L**(-3/2)))
        A = 2/np.sqrt(3) * L**(1/4) * (torch.cosh(phi/3))**(1/2)
        cond = 2/3 * (3 * L**3)**(1/4)
        s = ((A + ((2 * torch.abs(X))/A - A**2)**(1/2)) / 2)**3

        step = torch.where(X > cond, s - x.data,
                          torch.where(X <  -cond, -s - x.data, -x.data))

#         eff_lam = 2 * lmbda / sigma
#         threshold = (2/3) * (3 * eff_lam ** 3) ** (1/4)

#         X = x.data - grad / sigma

#         mask = X.abs() > threshold
#         mask = mask.float()
        # logging.debug('> Mask ')
        # logging.debug(mask)

#         zero_mask = X.abs() <= threshold
#         zero_mask = zero_mask.float() * 100
        # logging.debug('> zero mask')
        # logging.debug(zero_mask)

#         X.mul_(mask)
#         angle = torch.arccosh((27/16) * (X ** 2 + zero_mask) * (eff_lam ** (-1.5)))
        # logging.debug('> angle before mask')
        # logging.debug(angle)
        # angle = angle * mask
        # logging.debug('> angle after mask')
        # logging.debug(angle)

        # absA = (2/np.sqrt(3)) * (eff_lam ** (1/4)) * (torch.cosh(angle/3) ** (1/2))
        # value = ((absA + torch.sqrt(2 * (X.abs() + zero_mask) / absA - absA ** 2)) / 2) ** 3

        # step = X.sign() * value * mask - x.data

        return step

    def update_weights(self, x, step, grad, sigma):
        # if len(x.data.shape) == 2 or len(x.data.shape) == 4:
        x.data = x.data.add_(step.data)
        # else:
        #     logging.debug('layer of shape != 2, 4')
        #     logging.debug(x.data.shape)
        #    x.data.add_(- grad / sigma)
        return x

