#!/usr/bin/env python

# multi_exit_kan.py
# Jim Bagrow

# Based on PyKAN (c) 2024 Ziming Liu - MIT License
# https://github.com/KindXiaoming/pykan


import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from kan import MultKAN, KANLayer, Symbolic_KANLayer
from kan.spline import curve2coef


class MultiExitKAN(MultKAN):
    """
    Neural network with multiple exits using KANLayers.
    Each layer is connected to both the next layer and an output (exit).
    """
    # Modifications to __init__ method
    def __init__(self, width=None, grid=3, k=3, mult_arity = 2, noise_scale=0.3, scale_base_mu=0.0, 
            scale_base_sigma=1.0, base_fun='silu', symbolic_enabled=True, affine_trainable=False, 
            grid_eps=0.02, grid_range=[-1, 1], sp_trainable=True, sb_trainable=True, seed=1, 
            save_act=True, sparse_init=False, auto_save=True, first_init=True, ckpt_path='./model', 
            state_id=0, round=0, device='cpu'):
        super(MultKAN, self).__init__()

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        self.act_fun = []
        self.depth = len(width) - 1

        for i in range(len(width)):
            if type(width[i]) == int or type(width[i]) == np.int64:
                width[i] = [width[i],0]

        self.width = width

        # if mult_arity is just a scalar, we extend it to a list of lists
        if isinstance(mult_arity, int):
            self.mult_homo = True # when homo is True, parallelization is possible
        else:
            self.mult_homo = False # when home if False, for loop is required. 
        self.mult_arity = mult_arity

        width_in = self.width_in
        width_out = self.width_out

        self.base_fun_name = base_fun
        if base_fun == 'silu':
            base_fun = torch.nn.SiLU()
        elif base_fun == 'identity':
            base_fun = torch.nn.Identity()
        elif base_fun == 'zero':
            base_fun = lambda x: x*0.

        self.grid_eps = grid_eps
        self.grid_range = grid_range

        # create main network layers
        for l in range(self.depth):
            if isinstance(grid, list):
                grid_l = grid[l]
            else:
                grid_l = grid

            if isinstance(k, list):
                k_l = k[l]
            else:
                k_l = k


            sp_batch = KANLayer(in_dim=width_in[l], out_dim=width_out[l+1], num=grid_l, k=k_l, 
                            noise_scale=noise_scale, scale_base_mu=scale_base_mu, 
                            scale_base_sigma=scale_base_sigma, scale_sp=1., base_fun=base_fun, 
                            grid_eps=grid_eps, grid_range=grid_range, sp_trainable=sp_trainable, 
                            sb_trainable=sb_trainable, sparse_init=sparse_init)
            self.act_fun.append(sp_batch)

        # create exit layers
        self.exit_layers = []
        final_width = width[-1]
        for l in range(self.depth-1):
            # Each exit layer takes the current layer's width and maps to the final network output size
            exit_layer = KANLayer(in_dim=width_in[l], out_dim=final_width[0], num=grid_l, k=k_l,
                                noise_scale=noise_scale, scale_base_mu=scale_base_mu,
                                scale_base_sigma=scale_base_sigma, scale_sp=1., base_fun=base_fun,
                                grid_eps=grid_eps, grid_range=grid_range, sp_trainable=sp_trainable,
                                sb_trainable=sb_trainable, sparse_init=sparse_init)
            self.exit_layers.append(exit_layer)
        self.exit_layers = nn.ModuleList(self.exit_layers)

        # Rest of the original init code
        self.node_bias = []
        self.node_scale = []
        self.subnode_bias = []
        self.subnode_scale = []

        globals()['self.node_bias_0'] = torch.nn.Parameter(torch.zeros(3,1)).requires_grad_(False)
        exec('self.node_bias_0' + " = torch.nn.Parameter(torch.zeros(3,1)).requires_grad_(False)")

        for l in range(self.depth):
            exec(f'self.node_bias_{l} = torch.nn.Parameter(torch.zeros(width_in[l+1])).requires_grad_(affine_trainable)')
            exec(f'self.node_scale_{l} = torch.nn.Parameter(torch.ones(width_in[l+1])).requires_grad_(affine_trainable)')
            exec(f'self.subnode_bias_{l} = torch.nn.Parameter(torch.zeros(width_out[l+1])).requires_grad_(affine_trainable)')
            exec(f'self.subnode_scale_{l} = torch.nn.Parameter(torch.ones(width_out[l+1])).requires_grad_(affine_trainable)')
            exec(f'self.node_bias.append(self.node_bias_{l})')
            exec(f'self.node_scale.append(self.node_scale_{l})')
            exec(f'self.subnode_bias.append(self.subnode_bias_{l})')
            exec(f'self.subnode_scale.append(self.subnode_scale_{l})')

        # create node_bias and node_scale parameters for exit layers
        self.exit_node_bias = []
        self.exit_node_scale = []
        for l in range(self.depth-1):
            exit_bias = torch.nn.Parameter(torch.zeros(final_width[0])).requires_grad_(affine_trainable)
            exit_scale = torch.nn.Parameter(torch.ones(final_width[0])).requires_grad_(affine_trainable)
            self.exit_node_bias.append(exit_bias)
            self.exit_node_scale.append(exit_scale)
        self.exit_node_bias = nn.ParameterList(self.exit_node_bias)
        self.exit_node_scale = nn.ParameterList(self.exit_node_scale)

        self.act_fun = nn.ModuleList(self.act_fun)

        self.grid = grid
        self.k = k
        self.base_fun = base_fun

        ### initializing the symbolic front ###
        self.symbolic_fun = []
        for l in range(self.depth):
            sb_batch = Symbolic_KANLayer(in_dim=width_in[l], out_dim=width_out[l+1])
            self.symbolic_fun.append(sb_batch)
        self.symbolic_fun = nn.ModuleList(self.symbolic_fun)

        # create symbolic front for exit layers
        self.exit_symbolic_fun = []
        for l in range(self.depth-1):
            exit_sb = Symbolic_KANLayer(in_dim=width_in[l], out_dim=final_width[0])
            self.exit_symbolic_fun.append(exit_sb)
        self.exit_symbolic_fun = nn.ModuleList(self.exit_symbolic_fun)

        self.symbolic_enabled = symbolic_enabled
        self.affine_trainable = affine_trainable
        self.sp_trainable = sp_trainable
        self.sb_trainable = sb_trainable

        self.save_act = save_act

        self.node_scores = None
        self.edge_scores = None
        self.subnode_scores = None

        self.cache_data = None
        self.acts = None

        self.auto_save = auto_save
        self.state_id = 0
        self.ckpt_path = ckpt_path
        self.round = round

        self.device = device
        self.to(device)

        if auto_save:
            if first_init:
                if not os.path.exists(ckpt_path):
                    os.makedirs(ckpt_path)
                print(f"checkpoint directory created: {ckpt_path}")
                print('saving model version 0.0')

                history_path = self.ckpt_path+'/history.txt'
                with open(history_path, 'w') as file:
                    file.write(f'### Round {self.round} ###' + '\n')
                    file.write('init => 0.0' + '\n')
                self.saveckpt(path=self.ckpt_path+'/'+'0.0')
            else:
                self.state_id = state_id

        self.input_id = torch.arange(self.width_in[0],)

    def forward(self, x, singularity_avoiding=False, y_th=10.):
        x = x[:,self.input_id.long()]
        assert x.shape[1] == self.width_in[0]

        # cache data
        self.cache_data = x

        self.acts = []  # shape ([batch, n0], [batch, n1], ..., [batch, n_L])
        self.acts_premult = []
        self.spline_preacts = []
        self.spline_postsplines = []
        self.spline_postacts = []
        self.acts_scale = []
        self.acts_scale_spline = []
        self.subnode_actscale = []
        self.edge_actscale = []

        # store exit activation data
        self.exit_activations = []
        self.exit_preacts = []
        self.exit_spline_postacts = []
        self.exit_spline_postsplines = []

        self.acts.append(x)  # acts shape: (batch, width[l])

        # create list to store all outputs
        all_outputs = []

        for l in range(self.depth):
            # process this layer's exit if it's not the final layer
            if l < self.depth-1:
                x_exit_numerical, preacts_exit, postacts_exit, postspline_exit = self.exit_layers[l](x)

                if self.symbolic_enabled:
                    x_exit_symbolic, postacts_symbolic_exit = self.exit_symbolic_fun[l](x, singularity_avoiding=singularity_avoiding, y_th=y_th)
                else:
                    x_exit_symbolic = 0.
                    postacts_symbolic_exit = 0.

                x_exit = x_exit_numerical + x_exit_symbolic

                # apply scaling and bias
                x_exit = self.exit_node_scale[l][None,:] * x_exit + self.exit_node_bias[l][None,:]

                # save exit activations for visualization
                self.exit_activations.append(x_exit.detach())
                self.exit_preacts.append(preacts_exit.detach())
                self.exit_spline_postacts.append(postacts_exit.detach())
                self.exit_spline_postsplines.append(postspline_exit.detach())

                all_outputs.append(x_exit)


            x_numerical, preacts, postacts_numerical, postspline = self.act_fun[l](x)

            if self.symbolic_enabled == True:
                x_symbolic, postacts_symbolic = self.symbolic_fun[l](x, singularity_avoiding=singularity_avoiding, y_th=y_th)
            else:
                x_symbolic = 0.
                postacts_symbolic = 0.

            x = x_numerical + x_symbolic

            if self.save_act: # save subnode_scale
                self.subnode_actscale.append(torch.std(x, dim=0).detach())

            # subnode affine transform
            x = self.subnode_scale[l][None,:] * x + self.subnode_bias[l][None,:]

            if self.save_act:
                postacts = postacts_numerical + postacts_symbolic
                input_range = torch.std(preacts, dim=0) + 0.1
                output_range_spline = torch.std(postacts_numerical, dim=0)
                output_range = torch.std(postacts, dim=0)
                self.edge_actscale.append(output_range)

                self.acts_scale.append((output_range / input_range).detach())
                self.acts_scale_spline.append(output_range_spline / input_range)
                self.spline_preacts.append(preacts.detach())
                self.spline_postacts.append(postacts.detach())
                self.spline_postsplines.append(postspline.detach())

                self.acts_premult.append(x.detach())

            # multiplication
            dim_sum = self.width[l+1][0]
            dim_mult = self.width[l+1][1]

            if self.mult_homo == True:
                for i in range(self.mult_arity-1):
                    if i == 0:
                        x_mult = x[:,dim_sum::self.mult_arity] * x[:,dim_sum+1::self.mult_arity]
                    else:
                        x_mult = x_mult * x[:,dim_sum+i+1::self.mult_arity]

            else:
                for j in range(dim_mult):
                    acml_id = dim_sum + np.sum(self.mult_arity[l+1][:j])
                    for i in range(self.mult_arity[l+1][j]-1):
                        if i == 0:
                            x_mult_j = x[:,[acml_id]] * x[:,[acml_id+1]]
                        else:
                            x_mult_j = x_mult_j * x[:,[acml_id+i+1]]

                    if j == 0:
                        x_mult = x_mult_j
                    else:
                        x_mult = torch.cat([x_mult, x_mult_j], dim=1)

            if self.width[l+1][1] > 0:
                x = torch.cat([x[:,:dim_sum], x_mult], dim=1)

            # node affine transform
            x = self.node_scale[l][None,:] * x + self.node_bias[l][None,:]

            self.acts.append(x.detach())


        # Add the main network output as the final exit
        all_outputs.append(x)

        return all_outputs

    def update_grid_from_samples(self, x):
        '''
        update grid from samples

        Args:
        -----
            x : 2D torch.tensor
                inputs

        Returns:
        --------
            None
        '''
        for l in range(self.depth):
            self.get_act(x)
            self.act_fun[l].update_grid_from_samples(self.acts[l])
            if l < self.depth-1: # update exit grids
                self.exit_layers[l].update_grid_from_samples(self.acts[l])

    def initialize_grid_from_another_model(self, model, x):
        '''
        initialize grid from another model

        Args:
        -----
            model : MultKAN
                parent model
            x : 2D torch.tensor
                inputs

        Returns:
        --------
            None
        '''
        model(x)
        for l in range(self.depth):
            self.act_fun[l].initialize_grid_from_parent(model.act_fun[l], model.acts[l])

        for l in range(self.depth-1):
            self.exit_layers[l].initialize_grid_from_parent(model.exit_layers[l], model.acts[l])

    def initialize_from_another_model(self, another_model, x):
        '''
        initialize from another model of the same width, but their 'grid' parameter can be different. 
        Note this is equivalent to refine() when we don't want to keep another_model

        Args:
        -----
            another_model : MultKAN
            x : 2D torch.float

        Returns:
        --------
            self
        '''
        another_model(x)  # get activations

        self.initialize_grid_from_another_model(another_model, x)

        for l in range(self.depth):
            spb = self.act_fun[l]
            #spb_parent = another_model.act_fun[l]

            # spb = spb_parent
            preacts = another_model.spline_preacts[l]
            postsplines = another_model.spline_postsplines[l]
            self.act_fun[l].coef.data = curve2coef(preacts[:,0,:], postsplines.permute(0,2,1), spb.grid, k=spb.k)
            self.act_fun[l].scale_base.data = another_model.act_fun[l].scale_base.data
            self.act_fun[l].scale_sp.data = another_model.act_fun[l].scale_sp.data
            self.act_fun[l].mask.data = another_model.act_fun[l].mask.data

        for l in range(self.depth):
            self.node_bias[l].data = another_model.node_bias[l].data
            self.node_scale[l].data = another_model.node_scale[l].data

            self.subnode_bias[l].data = another_model.subnode_bias[l].data
            self.subnode_scale[l].data = another_model.subnode_scale[l].data

        for l in range(self.depth):
            self.symbolic_fun[l] = another_model.symbolic_fun[l]

        # update exit grids
        for l in range(self.depth-1):
            spb = self.exit_layers[l]
            exit_preacts = another_model.exit_preacts[l]
            exit_postsplines = another_model.exit_spline_postsplines[l]
            self.exit_layers[l].coef.data = curve2coef(
                exit_preacts[:,0,:],
                exit_postsplines.permute(0,2,1),
                spb.grid, k=spb.k)
            self.exit_layers[l].scale_base.data = another_model.exit_layers[l].scale_base.data
            self.exit_layers[l].scale_sp.data = another_model.exit_layers[l].scale_sp.data
            self.exit_layers[l].mask.data = another_model.exit_layers[l].mask.data

        for l in range(self.depth-1):
            self.exit_node_bias[l].data = another_model.exit_node_bias[l].data
            self.exit_node_scale[l].data = another_model.exit_node_scale[l].data

        for l in range(self.depth-1):
            self.exit_symbolic_fun[l] = another_model.exit_symbolic_fun[l]


        return self.to(self.device)



    def refine(self, new_grid):
        '''
        grid refinement

        Args:
        -----
            new_grid : init
                the number of grid intervals after refinement

        Returns:
        --------
            a refined model : MultKAN
        '''

        model_new = MultiExitKAN(width=self.width, 
                     grid=new_grid, 
                     k=self.k, 
                     mult_arity=self.mult_arity, 
                     base_fun=self.base_fun_name, 
                     symbolic_enabled=self.symbolic_enabled, 
                     affine_trainable=self.affine_trainable, 
                     grid_eps=self.grid_eps, 
                     grid_range=self.grid_range, 
                     sp_trainable=self.sp_trainable,
                     sb_trainable=self.sb_trainable,
                     ckpt_path=self.ckpt_path,
                     auto_save=True,
                     first_init=False,
                     state_id=self.state_id,
                     round=self.round,
                     device=self.device)

        model_new.initialize_from_another_model(self, self.cache_data)
        model_new.cache_data = self.cache_data
        model_new.grid = new_grid

        self.log_history('refine')
        model_new.state_id += 1

        return model_new.to(self.device)



def average_exit_loss(outputs, target, criterion=None, weights=None):
    """ Compute the average loss across all exits.
    """
    if criterion is None:
        criterion = nn.MSELoss()

    losses = torch.stack([criterion(output, target) for output in outputs])

    if weights is None:
        mean_loss = torch.mean(losses)
        return mean_loss
    weighted_loss = (losses * weights).mean()
    return weighted_loss


