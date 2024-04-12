from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
import jax.tree_util as jax_tree


def compute_metrics(state, grads_true, grads_est, mode, lam):
    """
    Compute alignment metrics of current epoch for given state of the model and current gradients. Metrics computed are:
    - alignment of forward anf feedback weights per layer (only for fa and kp mode)
    - alignment of bias gradients of current vs. comparison model per layer
    - alignment of weight & bias gradients of current vs. comparison model per layer
    - alignment of total gradient of current vs. comparison model
    - relative norm of gradients of current vs. comparison model
    Alignment metric is the cosine similiarity of the flattened vectors of the respective gradients i.e. cos(theta) = (a.dot(b))/(||a||*||b||
    ...
    Parameters
    ----------
    state : flax.training.train_state.TrainState
        current state of the model
    grads_ : dict
        gradients of comparison model in bp mode
    grads : dict
        gradients of current model in respecitve mode
    mode : str
        mode of current model
    """

    if mode == 'fa' or mode == 'kp':
        weight_al_per_layer, norms_kernels_per_layer, norms_Bs_per_layer = compute_weight_alignment(state.params)
    elif mode == 'interpolate_fa_bp':
        weight_al_per_layer, norms_kernels_per_layer, norms_Bs_per_layer = compute_interpolate_weight_alignment(state.params, lam)
    else:
        weight_al_per_layer, norms_kernels_per_layer, norms_Bs_per_layer = 0

    bias_true, _ = jax_tree.tree_flatten(
        flatten_matrices_in_tree(remove_keys(grads_true, ['kernel', 'B' ])))
    bias_est, _ = jax_tree.tree_flatten(
        flatten_matrices_in_tree(remove_keys(grads_est, ['kernel', 'B'])))
    kernel_true, _ = jax_tree.tree_flatten(
        flatten_matrices_in_tree(remove_keys(grads_true, ['bias', 'B'])))
    kernel_est, _ = jax_tree.tree_flatten(
        flatten_matrices_in_tree(remove_keys(grads_est, ['bias', 'B'])))

    bias_al_per_layer = compute_bias_grad_al_layerwise(bias_true, bias_est)
    wandb_grad_al_per_layer = compute_wandb_grad_al_layerwise(
        bias_true, bias_est, kernel_true, kernel_est)
    wandb_grad_al_total, norm_true, norm_est, rel_norm_grads, norm_proj_grad  = compute_wandb_al_total(bias_true, bias_est, kernel_true, kernel_est)
    #rel_norm_grads = compute_rel_norm(bias_true, bias_est, kernel_true, kernel_est)
    return bias_al_per_layer, wandb_grad_al_per_layer, wandb_grad_al_total, weight_al_per_layer, rel_norm_grads, norm_true, norm_est, norms_kernels_per_layer, norms_Bs_per_layer, norm_proj_grad

def summarize_metrics_epoch(bias_als_per_layer, wandb_grad_als_per_layer, wandb_grad_als_total, weight_als_per_layer, rel_norms_grads, norms_true, norms_est, norms_kernels_per_layer, norms_Bs_per_layer, norms_proj_grad,mode):
    """
    Summarizes all metrics for an epoch - averaging over collected entries.
    ...
    Parameters
    ----------
    bias_als_per_layer : list
        list of bias gradient alignments per layer over epochs
    wandb_grad_als_per_layer : list
        list of weight & bias gradient alignments per layer over epochs
    wandb_grad_als_total : list
        list of total gradient alignments over epochs
    weight_als_per_layer : list
        list of weight alignments per layer over epochs
    rel_norms_grads : list
        list of relative norms of gradients over epochs
    mode : str
        mode of current model
    """
    if mode == 'fa' or mode == 'kp' or mode == 'interpolate_fa_bp':
        avg_weight_al_per_layer = entrywise_average(weight_als_per_layer)
    else:
        avg_weight_al_per_layer = 0
    avg_bias_al_per_layer = entrywise_average(bias_als_per_layer)
    avg_wandb_grad_al_per_layer = entrywise_average(wandb_grad_als_per_layer)
    avg_wandb_grad_al_total = jnp.mean(jnp.array(wandb_grad_als_total))
    avg_weight_al_per_layer = entrywise_average(weight_als_per_layer)
    avg_rel_norm_grads = jnp.mean(jnp.array(rel_norms_grads))
    avg_norm_true = jnp.mean(jnp.array(norms_true))
    avg_norm_est = jnp.mean(jnp.array(norms_est))
    avg_norm_kernel_per_layer = entrywise_average(norms_kernels_per_layer)
    avg_norm_B_per_layer = entrywise_average(norms_Bs_per_layer)
    avg_norm_proj_grad = jnp.mean(jnp.array(norms_proj_grad))
    return avg_bias_al_per_layer, avg_wandb_grad_al_per_layer, avg_wandb_grad_al_total, avg_weight_al_per_layer, avg_rel_norm_grads, avg_norm_true, avg_norm_est, avg_norm_kernel_per_layer, avg_norm_B_per_layer, avg_norm_proj_grad

@jax.jit
def compute_interpolate_weight_alignment(params, lam):
    """
    Computes the alignment of feedforward and feedback matrices per layer.
    ...
    Parameters
    ----------
    params : dict
        dictionary of model parameters
    """
    kernels, _ = jax_tree.tree_flatten(
        flatten_matrices_in_tree(remove_keys(params, ['bias', 'B'])))
    Bs, _ = jax_tree.tree_flatten(flatten_matrices_in_tree(
        remove_keys(params, ['bias', 'kernel'])))
    Bs = jax.tree_map((lambda a, b : lam * a + (1-lam)*b), Bs, kernels)
    dot_prods = jax_tree.tree_map(jnp.dot, kernels, Bs)
    layerwise_norm_kern = jax_tree.tree_map(jnp.linalg.norm, kernels)
    layerwise_norm_B = jax_tree.tree_map(jnp.linalg.norm, Bs)
    layerwise_alignments = jax.tree_map(
        (lambda x, y, z: x/(y*z)), dot_prods, layerwise_norm_kern, layerwise_norm_B)
    return layerwise_alignments, layerwise_norm_kern, layerwise_norm_B

@jax.jit
def compute_weight_alignment(params):
    """
    Computes the alignment of feedforward and feedback matrices per layer.
    ...
    Parameters
    ----------
    params : dict
        dictionary of model parameters
    """
    kernels, _ = jax_tree.tree_flatten(
        flatten_matrices_in_tree(remove_keys(params, ['bias', 'B'])))
    Bs, _ = jax_tree.tree_flatten(flatten_matrices_in_tree(
        remove_keys(params, ['bias', 'kernel'])))
    dot_prods = jax_tree.tree_map(jnp.dot, kernels, Bs)
    layerwise_norm_kern = jax_tree.tree_map(jnp.linalg.norm, kernels)
    layerwise_norm_B = jax_tree.tree_map(jnp.linalg.norm, Bs)
    layerwise_alignments = jax.tree_map(
        (lambda x, y, z: x/(y*z)), dot_prods, layerwise_norm_kern, layerwise_norm_B)
    return layerwise_alignments, layerwise_norm_kern, layerwise_norm_B


@jax.jit
def compute_bias_grad_al_layerwise(bias_true, bias_est):
    """
    Computes the alignment of bias gradients per layer: cos(theta) = (bias_.dot(bias))/(||bias_||*||bias||)
    ...
    Parameters
    ----------
    bias_ : dict
        list of bias gradients of comparison model
    bias : dict
        list of bias gradients of current model
    """
    layerwise_alignments = jax.tree_map((lambda a, b: jnp.dot(
        a, b)/(jnp.linalg.norm(a)*jnp.linalg.norm(b))), bias_true, bias_est)
    return layerwise_alignments


@jax.jit
def compute_wandb_grad_al_layerwise(bias_true, bias_est, kernel_true, kernel_est):
    """
    Computes the alignment of weight & bias gradients per layer: 
    cos(theta) = ((bias_, kernel_).dot(bias, kernel)/(||(bias_, kernel_)|| * ||(bias, kernel)||) 
    - where the biases and kernels are layerwise 
    ...
    Parameters
    ----------
    bias_ : list
        list of bias gradients of comparison model
    bias : list
        list of bias gradients of current model
    kernel_ : list
        list of weight gradients of comparison model
    kernel : list
        list of weight gradients of current model
    """
    layerwise_alignments = jax.tree_map((lambda a, b, c, d: (jnp.dot(a, b) + jnp.dot(c, d)) /
                                         (jnp.sqrt(jnp.sum(jnp.multiply(a, a)) + jnp.sum(jnp.multiply(c, c)))
                                          * jnp.sqrt(jnp.sum(jnp.multiply(b, b)) + jnp.sum(jnp.multiply(d, d))))), bias_true, bias_est, kernel_true, kernel_est)
    return layerwise_alignments


@jax.jit
def compute_wandb_al_total(bias_true, bias_est, kernel_true, kernel_est):
    """
    Computes the alignment of total gradients:
    cos(theta) = ((bias_, kernel_).dot(bias, kernel)/(||(bias_, kernel_)|| * ||(bias, kernel)||)
    - where the biases and weights are concatenated across layers
    ...
    Parameters
    ----------
    bias_ : list
        list of bias gradients of comparison model
    bias : list
        list of bias gradients of current model
    kernel_ : list
        list of weight gradients of comparison model
    kernel : list
        list of weight gradients of current model
    """
    layerwise_alignments = jax.tree_map((lambda a, b, c, d: (
        jnp.dot(a, b) + jnp.dot(c, d))), bias_true, bias_est, kernel_true, kernel_est)
    squared_norms_true = squared_norm(bias_true, kernel_true)
    squared_norms_est = squared_norm(bias_est, kernel_est)
    norm_true = jnp.sqrt(jnp.sum(jnp.array(squared_norms_true)))
    norm_est = jnp.sqrt(jnp.sum(jnp.array(squared_norms_est)))
    rel_norm_grads = norm_est/norm_true
    proj = jnp.sum(jnp.array(layerwise_alignments))/(norm_true * norm_true)
    projected_biases = jax.tree_map((lambda a, b: a - proj * b), bias_est, bias_true)
    projected_kernels = jax.tree_map((lambda a, b: a - proj * b), kernel_est, kernel_true)
    norm_proj_grad = jnp.sqrt(jnp.sum(jnp.array(squared_norm(projected_biases, projected_kernels))))

    res = jnp.sum(jnp.array(layerwise_alignments))/(norm_true * norm_est)
    return res, norm_true, norm_est, rel_norm_grads, norm_proj_grad


# @jax.jit
# def compute_rel_norm(bias_true, bias_est, kernel_true, kernel_est):
#     """
#     Computes the relative norm of gradient of current model compared to gradient of comparison model:
#     ||(bias_, kernel_)||/||(bias, kernel)||
#     - where the biases and weights are concatenated across layers
#     ...
#     Parameters
#     ----------
#     bias_ : list
#         list of bias gradients of comparison model
#     bias : list
#         list of bias gradients of current model
#     kernel_ : list
#         list of weight gradients of comparison model
#     kernel : list
#         list of weight gradients of current model
#     """
#     squared_norms_true = squared_norm(bias_true, kernel_true)
#     squared_norms_est = squared_norm(bias_est, kernel_est)
#     return jnp.sqrt(jnp.sum(jnp.array(squared_norms_est)))/jnp.sqrt(jnp.sum(jnp.array(squared_norms_true)))


def squared_norm(a, b):
    """
    Computes squared norm of concatenated a and b.
    ...
    Parameters
    ----------
    a : list
    b : list
    """
    return jax.tree_map((lambda a, b: jnp.sum(jnp.multiply(a, a)) + jnp.sum(jnp.multiply(b, b))), a, b)


def entrywise_average(array_of_arrays):
    """
    Computes entrywise average of list of arrays.
    ...
    Parameters
    ----------
    array_of_arrays : list
    """
    average_array = np.zeros_like(array_of_arrays[0])
    for arr in array_of_arrays:
        average_array += arr

    average_array = average_array/len(array_of_arrays)

    return average_array


def reorganize_dict(input_dict):
    """
    Takes as input a dictionary describing parameters for one of the biologically plausible models and reorganizes it such that the parameters
    can be apllied to a standard backpropagation model.
    ...
    Parameters
    ----------
    input_dict : dict
        dictionary describing parameters for one of the biologically plausible models
    """
    new_dict = {'params': {}}
    dense_counter = 0
    for _, layer_group_val in input_dict['params'].items():
        for layer_key, layer_val in layer_group_val.items():
            if layer_key.startswith('Dense'):
                new_key = f'Dense_{dense_counter}'
                dense_counter += 1
                new_dict['params'][new_key] = layer_val
    return new_dict


def remove_keys(pytree, key_list):
    """
    Removes leaves from the pytree whose path contains any key from key_list.
    ...
    Parameters
    ----------
    pytree : pytree
        pytree to remove leaves from
    key_list : list
        list of keys to remove from pytree
    """

    def filter_fn(path, value):
        if not any(p.key in key_list for p in path):
            return value
    filtered_pytree = jax_tree.tree_map_with_path(filter_fn, pytree)
    return filtered_pytree


def flatten_array(arr):
    """
    Flattens an array to a 1D array.
    ...
    Parameters
    ----------
    arr : array
        array to flatten
    """
    flattened = arr.reshape(-1)
    return flattened


def flatten_matrices_in_tree(pytree):
    """
    Flattens all matrices in a pytree to 1D arrays.
    ...
    Parameters
    ----------
    pytree : pytree
        pytree to flatten matrices in
    """
    return jax_tree.tree_map(flatten_array, pytree)
