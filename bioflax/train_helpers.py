import jax
import jax.numpy as jnp
import optax
import matplotlib.pyplot as plt
import torch
import numpy as np
import flax.linen as nn
import wandb
import flax
import jax.tree_util as jax_tree
from typing import Any
from jax.nn import one_hot
from tqdm import tqdm
from jax.flatten_util import ravel_pytree
from flax.training import train_state
from functools import partial
from .metric_computation import compute_metrics, summarize_metrics_epoch, reorganize_dict
from dataclasses import replace


def create_train_state(model, rng, lr, momentum, weight_decay_1, weight_decay_2, in_dim, batch_size, seq_len, optimizer, epochs, steps_per_epoch):
    """
    Initializes the training state using optax
    ...
    Parameters
    __________
    model : Any
        model to be trained
    rng : jnp.PRNGKey
        key for randomness
    lr : float
        learning rate for optimizer
    momentum : float
        momentum for optimizer
    in_dim : int
        input dimension of model
    batch_size : int
        batch size used when running model
    seq_len : int
        sequence length used when running model
    """
    dummy_input = jnp.ones((batch_size, in_dim, seq_len))
    params = model.init(rng, dummy_input)["params"]
    #optax_transformation_mask = create_mask_dict(params)
    mask1 = create_mask_dict_layerwise(params, None, 0)#{'RandomDenseLinearInterpolateFABP_0': {'B': False, 'Dense_0': {'bias': True, 'kernel': True}}, 'RandomDenseLinearInterpolateFABP_1': {'B': False, 'Dense_0': {'bias': True, 'kernel': True}}, 'RandomDenseLinearInterpolateFABP_2': {'B': False, 'Dense_0': {'bias': True, 'kernel': True}}, 'RandomDenseLinearInterpolateFABP_1': {'B': False, 'Dense_0': {'bias': False, 'kernel': False}}}
    mask2 = create_mask_dict_layerwise(params, None, 1)#{'RandomDenseLinearInterpolateFABP_0': {'B': False, 'Dense_0': {'bias': False, 'kernel': False}}, 'RandomDenseLinearInterpolateFABP_1': {'B': False, 'Dense_0': {'bias': False, 'kernel': False}}, 'RandomDenseLinearInterpolateFABP_2': {'B': False, 'Dense_0': {'bias': False, 'kernel': False}}, 'RandomDenseLinearInterpolateFABP_1': {'B': False, 'Dense_0': {'bias': True, 'kernel': True}}}
    mask3 = create_mask_dict_layerwise(params, None, 2)
    print(mask1)

    cosine_fn = optax.cosine_decay_schedule(init_value=lr, decay_steps=epochs * steps_per_epoch)
    sgd_optimizer = optax.sgd(learning_rate=lr, momentum=momentum)
    adam = optax.adam(learning_rate=lr)

    if (optimizer == 'sgd'):
        tx = optax.chain(
            sgd_optimizer,
            optax.add_decayed_weights(weight_decay_1, mask = mask1),
            optax.add_decayed_weights(weight_decay_1, mask = mask2),
            optax.add_decayed_weights(weight_decay_2, mask = mask3)
        )
    elif (optimizer == 'adam'):
        tx = optax.chain(
            adam,
            optax.add_decayed_weights(weight_decay, mask = optax_transformation_mask)
        )
    else:
        print("Optimzer not supported, fallback sgd was used")
        tx = optax.chain(
            sgd_optimizer,
            optax.add_decayed_weights(weight_decay)
        )
    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)


@partial(jax.jit, static_argnums=(4, 5))
def compute_bp_grads(state, state_bp, inputs, labels, loss_function, alpha):

    def loss_comp(params):
                    logits = state_bp.apply_fn({"params": params}, inputs)
                    loss = get_loss(loss_function, logits, labels, alpha)
                    return loss
    
    _, grads_ = jax.value_and_grad(loss_comp)(
                        reorganize_dict({"params": state.params})["params"]
                    )
    return grads_
#this one allows layerwise control of the mask but might only work for a randomdenseinterpolate one
# index muss von aussen mit None aufgerufen werden
def create_mask_dict_layerwise(input_dict, index, active_index):
    """
    Recursively creates a mask dictionary that matches the structure of the input dictionary.
    Only the 'bias' and 'kernel' entries for the specific 'RandomDenseInterpolateFA' layer
    indicated by 'active_index' will be set to True; all other entries are set to False.

    Parameters:
        input_dict (dict): The input dictionary with nested structure containing 'B', 'bias', 'kernel'.
        index (int or None): Current index being processed, used to match against 'active_index'.
        active_index (int): The index of the layer that should have True for 'bias' and 'kernel'.

    Returns:
        dict: A mask dictionary with structured True or False values as described.
    """
    mask_dict = {}
    for key, value in input_dict.items():
        if isinstance(value, dict):  # If the value is another dictionary, recurse
            # Determine if this dictionary key matches the pattern of interest and extract its index
            if key.startswith('RandomDenseLinearInterpolateFABP_'):
                current_index = int(key.split('_')[-1])
            else:
                current_index = index  # Continue with the current index if not a specific layer key

            # Recurse with the current_index updated if this is a layer key
            mask_dict[key] = create_mask_dict_layerwise(value, current_index, active_index)
        else:  # It's a leaf node
            if key == 'B':
                mask_dict[key] = False
            elif key in ['bias', 'kernel']:
                # Set to True only if the current layer's index matches the active_index
                mask_dict[key] = (index == active_index)

    return mask_dict

def create_mask_dict(input_dict):
    """
    Recursively creates a mask dictionary that matches the structure of the input dictionary.
    Each 'B' entry will have a mask value of False, and 'bias' or 'kernel' entries will have True.

    Parameters:
        input_dict (dict): The input dictionary with nested structure containing 'B', 'bias', and 'kernel'.

    Returns:
        dict: A mask dictionary with the same structure where each 'B' is marked False and 'bias'/'kernel' True.
    """
    mask_dict = {}
    for key, value in input_dict.items():
        if isinstance(value, dict):  # If the value is another dictionary, recurse
            mask_dict[key] = create_mask_dict(value)
        else:  # It's a leaf node, decide the mask based on the key
            if key == 'B':
                mask_dict[key] = False
            elif key in ['bias', 'kernel']:
                mask_dict[key] = True

    return mask_dict

def train_epoch(model, state, state_bp, trainloader, loss_function, n, mode, compute_alignments, lam, reset, p, key_mask, use_wandb, prev_loss, key, steps, full_batch, grads_minus_mode, alpha):
    """
    Training function for an epoch that loops over batches.
    ...
    Parameters
    __________
    state : TrainState
        current train state of the model
    model : BioBeuralNetwork
        model that is trained
    seq_len : int
        length of a single input
    in_dim
        dimensionality of a single input
    loss_function: str
        identifier to select loss function
    n : int
        number of batches over which to compute alignments
    mode : str
        identifier for training mode
    compute_alignments : bool
        flag to compute alignments

    """
    batch_losses = []
    bias_als_per_layer = []
    wandb_grad_als_per_layer = []
    wandb_grad_als_total = []
    rel_norms_grads = []
    weight_als_per_layer = []
    conv_rates = []
    norms_true = []
    norms_est = []
    sharpness_collected = []
    norms_kernels_per_layer = []
    norms_Bs_per_layer = []
    norms_proj_grad = []

    

    for i, batch in enumerate(tqdm(trainloader)):
        if full_batch  and i == 0 or not(full_batch):
            key, key_power_it, key_random_labels = jax.random.split(key, num=3)
            inputs, labels = prep_batch(batch)
            if loss_function == "MSE_with_random_labels":
                labels = jax.random.normal(key_random_labels, jax.nn.one_hot(labels, num_classes=10).shape)
            elif loss_function == "CE_with_random_labels_0_pred":
                labels = jax.random.randint(key_random_labels, labels.shape, 0, 10)

            true_loss_fn = loss_function

            if loss_function in ["CE_interpolate_loss_alignment", "CE_with_control_alignment"]:
                true_loss_fn = "CE"

            if i % 6 == 0:
                if compute_alignments:
                    
                    if mode != "bp":
                        grads_true = compute_bp_grads(
                            state, state_bp, inputs, labels, true_loss_fn, alpha)
                if loss_function == "MSE_interpolate_loss_alignment" or loss_function == "MSE_with_control_alignment":
                    loss_true, grads_true = loss_comp(state, inputs, labels,"MSE_with_integer_labels")
                    loss_align = loss_comp(state, inputs, labels, "MSE_with_zero_pred_correlated_labels") 
                    
                elif loss_function in ["CE_interpolate_loss_alignment", "CE_with_control_alignment"]:
                    loss_true, grads_true_fa = loss_comp(state, inputs, labels, true_loss_fn)
                    loss_align = loss_comp(state, inputs, labels, "CE_with_random_labels_0_pred")
            
            state, loss, grads_est = train_step(state, inputs, labels, loss_function, grads_true, grads_minus_mode, alpha)
                
            
            batch_losses.append(loss)
            
            #atm tracking only in the first epoch and second last
            if i % 6 == 0:
                if loss_function in ["CE_interpolate_loss_alignment"]:
                    grads_est = grads_true_fa
                if compute_alignments:
                    (
                        bias_al_per_layer,
                        wandb_grad_al_per_layer,
                        wandb_grad_al_total,
                        weight_al_per_layer,
                        rel_norm_grads,
                        norm_true,
                        norm_est, 
                        norm_kernels_per_layer,
                        norm_Bs_per_layer,
                        norm_proj_grad
                    ) = compute_metrics(state, grads_true, grads_est, mode, lam)
                    bias_als_per_layer.append(bias_al_per_layer)
                    wandb_grad_als_per_layer.append(wandb_grad_al_per_layer)
                    wandb_grad_als_total.append(wandb_grad_al_total)
                    weight_als_per_layer.append(weight_al_per_layer)
                    rel_norms_grads.append(rel_norm_grads)
                    norms_true.append(norm_true)
                    norms_est.append(norm_est)
                    norms_kernels_per_layer.append(norm_kernels_per_layer)
                    norms_Bs_per_layer.append(norm_Bs_per_layer)
                    norms_proj_grad.append(norm_proj_grad)


                    rng, key_power_it = jax.random.split(key_power_it, num=2)
                    #if i == 0:
                    #    sharpness = power_iteration(state, state_bp, inputs, labels, loss_function, rng, steps, alpha)
                    #
                    #    sharpness_collected.append(sharpness)

                if i == 0:
                    curr_rate=batch_losses[-1]/prev_loss
                if i > 0:
                    curr_rate = batch_losses[-1]/batch_losses[-2]
                    conv_rates.append(curr_rate)
                neg_rate = 1-curr_rate
                metrics={
                    "Training loss": loss,
                    "Conv_Rate": curr_rate,
                    "1-Conv_Rate": neg_rate,
                    "Rel_norm_grads": rel_norm_grads,
                    "Gradient alignment": wandb_grad_al_total,
                    "Norm true gradient": norm_true,
                    "Norm est. gradient": norm_est,
                    "Norm of est_gradient projected on plane orthogonal to true gradient": norm_proj_grad
                }
                if loss_function in ["MSE_interpolate_loss_alignment", "MSE_with_control_alignment", "CE_with_control_alignment", "CE_interpolate_loss_alignment"]:
                    metrics["True loss"] = loss_true
                    metrics["Loss aligning force"] = loss_align
                #if i == 0:
                #    metrics["Sharpness"] = sharpness
                for i, al in enumerate(bias_al_per_layer):
                        metrics[f"Alignment bias gradient layer {i}"] = al
                for i, al in enumerate(wandb_grad_al_per_layer):
                    metrics[f"Alignment gradient layer {i}"] = al
                if mode == "fa" or mode == "kp" or mode == "interpolate_fa_bp":
                    for i, al in enumerate(weight_al_per_layer):
                        metrics[f"Alignment layer {i}"] = al
                    for i, norm in enumerate(norm_kernels_per_layer):
                        metrics[f"Norm layer {i}"] = norm
                    for i, norm in enumerate(norm_Bs_per_layer):
                        metrics[f"Norm B layer {i}"] = norm
                if use_wandb: 
                    wandb.log(metrics)
        
            if full_batch:
                return state, jnp.mean(jnp.array(batch_losses)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    if compute_alignments:
        (
            avg_bias_al_per_layer,
            avg_wandb_grad_al_per_layer,
            avg_wandb_grad_al_total,
            avg_weight_al_per_layer,
            avg_rel_norm_grads,
            avg_norm_true,
            avg_norm_est,
            avg_norm_kernel_per_layer,
            avg_norm_B_per_layer, 
            avg_norm_proj_grad
        ) = summarize_metrics_epoch(
            bias_als_per_layer,
            wandb_grad_als_per_layer,
            wandb_grad_als_total,
            weight_als_per_layer,
            rel_norms_grads,
            norms_true,
            norms_est,
            norms_kernels_per_layer,
            norms_Bs_per_layer,
            norms_proj_grad,
            mode,
        )
        return (
            state,
            batch_losses[-1],#jnp.mean(jnp.array(batch_losses)),
            avg_bias_al_per_layer,
            avg_wandb_grad_al_per_layer,
            avg_wandb_grad_al_total,
            avg_weight_al_per_layer,
            avg_rel_norm_grads,
            jnp.mean(jnp.array(conv_rates)),
            avg_norm_true,
            avg_norm_est,
            avg_norm_kernel_per_layer,
            avg_norm_B_per_layer,
            avg_norm_proj_grad,
        )
    else:
        return state, jnp.mean(jnp.array(batch_losses)), 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

@partial(jax.jit, static_argnames=("loss_function"))
def loss_comp(state, inputs, labels, loss_function):
    def loss_fn(params):
        logits = state.apply_fn({"params": params}, inputs)
        loss = get_loss(loss_function, logits, labels, 0)
        return loss
    
    loss, grad_est = jax.value_and_grad(loss_fn)(state.params)
    return loss, grad_est

# the lambda used here is in double use with interpolation which needs to be corrected
def interpolate_B_with_kernel(d, lam, p, key):
    """Replace 'B' with 'kernel' in each 'RandomDenseLinearFA_i' layer."""
    new_dict = {}
    for layer, subdict in d.items():
        key_1, key = jax.random.split(key, num=2)
        mask = jax.random.choice(key_1, jnp.array([0,1]), jnp.shape(subdict['Dense_0']['kernel']), p = jnp.array([1-p,p]))
        if layer.startswith('RandomDenseLinearFA_'):
            new_subdict = {key: (lam * value + (1-lam)* jnp.multiply(mask,subdict['Dense_0']['kernel']) if key == 'B' else value) 
                           for key, value in subdict.items()}
            new_dict[layer] = new_subdict
        else:
            new_dict[layer] = subdict
    return new_dict

@partial(jax.jit, static_argnames=("steps", "loss_function"))
def power_iteration(state, state_bp, inputs, labels, loss_function, rng, steps, alpha):
    params = reorganize_dict({"params": state.params})["params"]
    #print(params)
    #print(state.params)
    #print(state_bp.params)
    p, unravel = ravel_pytree(params)

    safe_unravel = lambda p: jax.tree_util.tree_map(lambda x: x.astype(p.dtype), unravel(p))

    def loss_fn(p):
        logits = state_bp.apply_fn({"params": safe_unravel(p)}, inputs)
        loss = get_loss(loss_function, logits, labels, alpha)
        return loss
    

    def hvp(p, v):
        return jax.jvp(jax.grad(loss_fn), (p,), (v,))[1]

    # Compute biggest eigval and eigvect of Hessian with power iteration
    v = jax.random.normal(rng, (p.size,)).astype(p.dtype)
    v = v / jnp.linalg.norm(v)
    def loop_body(v, _):
        v_new = hvp(p, v)
        v_new_normalized = v_new / jnp.linalg.norm(v_new)
        return v_new_normalized, v_new_normalized

    v, vs = jax.lax.scan(loop_body, v, xs=jnp.arange(steps))
    v = vs[-1]
    sharpness = v @ hvp(p,v)
    return sharpness

def reset_to_fa(d):
    """Replace keys in the dictionary."""
    new_dict = {}
    for key, value in d.items():
        # Check if the key starts with 'ResetLayer_' and replace it
        if key.startswith('ResetLayer_'):
            new_key = 'RandomDenseLinearFA_' + key[len('ResetLayer_'):]
            new_dict[new_key] = value
        else:
            new_dict[key] = value
    return new_dict

def fa_to_reset(d):
    """Revert keys in the dictionary from 'RandomDenseLinearFA_i' to 'ResetLayer_i'."""
    new_dict = {}
    for key, value in d.items():
        # Check if the key starts with 'RandomDenseLinearFA_' and replace it
        if key.startswith('RandomDenseLinearFA_'):
            new_key = 'ResetLayer_' + key[len('RandomDenseLinearFA_'):]
            new_dict[new_key] = value
        else:
            new_dict[key] = value
    return new_dict

@partial(jax.jit, static_argnums=(3, 5, 6))
def train_step(state, inputs, labels, loss_function, grads_true, grad_minus_mode, alpha):
    """
    Performs a single training step given a batch of data
    ...
    Parameters
    __________
    state : TrainState
        current train state of the model
    inputs : float32
        inputs for the batch
    labels : int32
        labels for the batch
    loss_function: str
        identifier to select loss function
    """

    def loss_fn(params):
        logits = state.apply_fn({"params": params}, inputs)
        loss = get_loss(loss_function, logits, labels, alpha)
        return loss

    

    loss, grads_est = jax.value_and_grad(loss_fn)(state.params)
    if grad_minus_mode:
        #print(f"Grads true: {grads_true}")
        #print(f"Grads est: {grads_est}")
        grads_est = subtract_grads(grads_true, grads_est)
        #print(f"Grads est after subtraction: {grads_est}")

    state = state.apply_gradients(grads=grads_est)
    return state, loss, grads_est

def subtract_grads(grads_true, grads_est):
    """
    Subtracts the gradients of 'bias' and 'kernel' in layers from grads_true from those
    in grads_est. Specifically, it subtracts the 'Dense_i' gradients in grads_true from 
    'Dense_0' within 'RandomDenseLinearInterpolateFABP_i' in grads_est. It also ensures
    that the matrix 'B' in 'RandomDenseLinearInterpolateFABP_i' is preserved in the output.

    Parameters:
        grads_true (dict): Dictionary containing true gradients with keys like 'Dense_i'.
        grads_est (dict): Dictionary containing estimated gradients with keys structured
                          as 'RandomDenseLinearInterpolateFABP_i', which further contains
                          'Dense_0' with 'bias' and 'kernel', and a matrix 'B'.

    Returns:
        dict: A new dictionary containing the subtracted gradients and preserved matrices structured
              similarly to grads_est.
    """
    result_grads = {}

    # Iterate over the estimated gradients
    for est_key, est_value in grads_est.items():
        if 'RandomDenseLinearInterpolateFABP_' in est_key:
            # Find the corresponding index
            index = est_key.split('_')[-1]
            true_key = f'Dense_{index}'
            result_grads[est_key] = {}

            # Copy the 'B' matrix if present
            if 'B' in est_value:
                result_grads[est_key]['B'] = est_value['B']

            # Check for the 'Dense_0' key and corresponding 'Dense_i' in grads_true
            if 'Dense_0' in est_value and true_key in grads_true:
                result_grads[est_key]['Dense_0'] = {}

                # Perform subtraction for 'bias' and 'kernel'
                for param in ['bias', 'kernel']:
                    if param in est_value['Dense_0'] and param in grads_true[true_key]:
                        result_grads[est_key]['Dense_0'][param] = grads_est[est_key]['Dense_0'][param] - grads_true[true_key][param]

    return result_grads

def get_loss(loss_function, logits, labels, alpha, num_classes=10):
    """
    Returns the loss for network outputs and labels
    ...
    Parameters
    __________
    loss_function : str
        identifier that slects the correct loss function
    logits : float32
        outputs of the network for the batch
    labels : int32
        labels for the batch
    """
    if loss_function == "CE":
        return optax.softmax_cross_entropy_with_integer_labels(
            logits=jnp.squeeze(logits), labels=labels
        ).mean()
    elif loss_function == "MSE":
        return optax.l2_loss(jnp.squeeze(logits), jnp.squeeze(labels)).mean()
    elif loss_function == "MSE_with_integer_labels":
        if num_classes is None:
            raise ValueError("num_classes must be provided for MSE_with_integer_labels")
        # Convert integer labels to one-hot encoded labels
        one_hot_labels = jax.nn.one_hot(labels, num_classes)
        # Compute MSE loss using the one-hot encoded labels
        return optax.l2_loss(jnp.squeeze(logits), one_hot_labels).mean()
    elif loss_function == "MSE_with_random_labels":
        return optax.l2_loss(jnp.squeeze(logits), jnp.squeeze(labels)).mean()
    elif loss_function == "MSE_with_predictions":
        return optax.l2_loss(jnp.squeeze(logits), jnp.squeeze(jnp.zeros_like(logits))).mean()
    elif loss_function == "MSE_with_zero_pred_correlated_targets":
        return optax.l2_loss(jnp.squeeze(logits) - jax.lax.stop_gradient(jnp.squeeze(logits)), jnp.squeeze(labels)).mean()
    elif loss_function == "MSE_with_zero_pred_correlated_labels":
        return optax.l2_loss(jnp.squeeze(logits) - jax.lax.stop_gradient(jnp.squeeze(logits)), jax.nn.one_hot(labels, num_classes)).mean()
    elif loss_function == "MSE_interpolate_loss_alignment":
        return alpha * optax.l2_loss(jnp.squeeze(logits), jax.nn.one_hot(labels, 10)).mean() + \
            (1-alpha)*optax.l2_loss(jnp.squeeze(logits) - jax.lax.stop_gradient(jnp.squeeze(logits)), jax.nn.one_hot(labels, num_classes)).mean()
    elif loss_function == "CE_interpolate_loss_alignment":
        return alpha * optax.softmax_cross_entropy_with_integer_labels(
            logits=jnp.squeeze(logits), labels=labels).mean() + \
            (1-alpha)*optax.softmax_cross_entropy_with_integer_labels(
            logits=jnp.squeeze(logits-jax.lax.stop_gradient(logits)), labels=labels).mean()
    elif loss_function == "CE_with_control_alignment":
        return optax.softmax_cross_entropy_with_integer_labels(jnp.squeeze(logits), labels).mean() + \
            alpha*optax.softmax_cross_entropy_with_integer_labels(jnp.squeeze(logits) - jax.lax.stop_gradient(jnp.squeeze(logits)), labels).mean()
    elif loss_function == "MSE_with_control_alignment":
        return optax.l2_loss(jnp.squeeze(logits), jax.nn.one_hot(labels, 10)).mean() + \
            alpha*optax.l2_loss(jnp.squeeze(logits) - jax.lax.stop_gradient(jnp.squeeze(logits)), jax.nn.one_hot(labels, num_classes)).mean()
    elif loss_function == "CE_with_random_labels_0_pred":
        return optax.softmax_cross_entropy_with_integer_labels(
            logits=jnp.squeeze(logits-jax.lax.stop_gradient(logits)), labels=labels).mean()
    elif loss_function == "MSE_with_zero_targets":
        return optax.l2_loss(jnp.squeeze(logits), jnp.squeeze(jnp.zeros_like(logits))).mean()
    elif loss_function == "CE_with_label_zero":
        return optax.softmax_cross_entropy_with_integer_labels(
            logits=jnp.squeeze(logits), labels=jnp.zeros_like(labels, dtype=jnp.int32)).mean()



def prep_batch(batch):
    """
    Prepares a batch of data for training.
    ...
    Parameters
    __________
    batch : tuple
        batch of data
    seq_len : int
        sequence length used when running model
    in_dim : int
        input dimension of model
    """
    inputs, labels = batch
    inputs = jnp.array(inputs.numpy()).astype(float)
    labels = jnp.array(labels)
    return inputs, labels


@partial(jnp.vectorize, signature="(c),()->()")
def compute_accuracy(logits, label):
    """
    Computes the accuracy of the network outputs for a batch foe a classification task
    ...
    Parameters
    __________
    logits : float32
        outputs of the network for the batch
    label : int32
        labels for the batch
    """
    return jnp.mean(jnp.argmax(logits) == label)


@partial(jax.jit, static_argnums=(3, 4))
def eval_step(inputs, labels, state, loss_function, alpha):
    """
    Performs a single evaluation step given a batch of data
    ...
    Parameters
    __________
    inputs : float32
        inputs for the batch
    labels : int32
        labels for the batch
    state : TrainState
        current train state of the model
    loss_function: str
        identifier to select loss function
    """
    logits = state.apply_fn({"params": state.params}, inputs)
    losses = get_loss(loss_function, logits, labels, alpha)
    accs = 0
    if loss_function in ["CE", "MSE_with_integer_labels", "MSE_interpolate_loss_alignment", "MSE_with_control_alignment", "CE_interpolate_loss_alignment", "CE_with_control_alignment", "CE_with_random_labels_0_pred"]:
        accs = compute_accuracy((jnp.squeeze(logits)), labels)
    return jnp.mean(losses), accs, logits


def validate(state, testloader, seq_len, in_dim, loss_function, alpha):
    """
    Validation function that loops over batches
    ...
    Parameters
    __________
    state : TrainState
        current train state of the model
    testloader : torch.utils.data.DataLoader
        pytorch dataloader for the test set
    seq_len : int
        sequence length used when running model
    in_dim : int
        input dimension of model
    loss_function: str
        identifier to select loss function
    """
    losses, accuracies = [], []

    for batch in tqdm(testloader):
        inputs, labels = prep_batch(batch)
        loss, acc, _ = eval_step(inputs, labels, state, loss_function, alpha)
        losses.append(loss)
        if loss_function in ["CE", "MSE_with_integer_labels", "MSE_interpolate_loss_alignment", "MSE_with_control_alignment", "CE_interpolate_loss_alignment", "CE_with_control_alignment", "CE_with_random_labels_0_pred"]:
            accuracies.append(jnp.mean(acc))

    acc_mean = 10000000.0
    if loss_function in ["CE", "MSE_with_integer_labels", "MSE_interpolate_loss_alignment", "MSE_with_control_alignment", "CE_interpolate_loss_alignment", "CE_with_control_alignment", "CE_with_random_labels_0_pred"]:
        acc_mean = jnp.mean(jnp.array([accuracies]))
    return jnp.mean(jnp.array([losses])), acc_mean


def pred_step(state, batch, task):
    """
    Prediction function to obtain outputs for a batch of data
    ...
    Parameters
    __________
    state : TrainState
        current train state of the model
    batch : tuple
        batch of data
    task : str
        identifier for task
    """
    inputs, _ = prep_batch(batch)
    logits = state.apply_fn({"params": state.params}, inputs)
    if task == "classification":
        return jnp.squeeze(logits).argmax(axis=1)
    elif task == "regression":
        return logits
    else:
        print("Task not supported")
        return None


def plot_sample(testloader, state, seq_len, in_dim, task, output_features):
    """
    Plots a sample of the test set
    ...
    Parameters
    __________
    testloader : torch.utils.data.DataLoader
        pytorch dataloader for the test set
    state : TrainState
        current train state of the model
    seq_len : int
        sequence length used when running model
    in_dim : int
        input dimension of model
    task : str
        identifier for task
    """
    if task == "classification":
        return plot_mnist_sample(testloader, state, task)
    elif task == "regression":
        return plot_regression_sample(testloader, state, seq_len, in_dim, task, output_features)
    else:
        print("Task not supported")
        return None


def plot_mnist_sample(testloader, state, task):
    """
    Plots a sample of the test set for the MNIST dataset
    ...
    Parameters
    __________
    testloader : torch.utils.data.DataLoader
        pytorch dataloader for the test set
    state : TrainState
        current train state of the model
    task : str
        identifier for task
    """
    test_batch = next(iter(testloader))
    pred = pred_step(state, test_batch, task)

    _, axs = plt.subplots(5, 5, figsize=(12, 12))
    inputs, _ = test_batch
    inputs = torch.reshape(inputs, (inputs.shape[0], 28, 28, 1))
    print(inputs.shape)
    for i, ax in enumerate(axs.flatten()):
        ax.imshow(inputs[i, ..., 0], cmap="gray")
        ax.set_title(f"label={pred[i]}")
        ax.axis("off")
    plt.show()


def plot_regression_sample(testloader, state, seq_len, in_dim, task, output_features):
    """
    Plots a sample of the test set for the regression task
    ...
    Parameters
    __________
    testloader : torch.utils.data.DataLoader
        pytorch dataloader for the test set
    state : TrainState
        current train state of the model
    seq_len : int
        sequence length used when running model
    in_dim : int
        input dimension of model
    task : str
        identifier for task
    """
    inputs_array = np.array([])
    labels_array = np.array([])
    pred_array = np.array([])
    if in_dim != 1 or seq_len != 1 or output_features != 1:
        print("Regression sample plotting only possible for 1D inputs")
        return
    for i in range(5):
        test_batch = next(iter(testloader))
        pred = pred_step(state, test_batch, task)
        inputs, labels = test_batch
        labels = labels + 1
        inputs_array = np.append(inputs_array, inputs)
        labels_array = np.append(labels_array, labels)
        pred_array = np.append(pred_array, pred)
    plt.scatter(inputs_array.flatten(), labels_array.flatten(), label="True")
    plt.scatter(inputs_array.flatten(), pred_array.flatten(), label="Pred")
    plt.show()


def select_initializer(initializer, scale):
    if (initializer == 'lecun'):
        return nn.initializers.lecun_normal()
    elif (initializer == 'uniform'):
        return uniform_init(-scale, scale)
    elif (initializer == 'variance_scaling'):
        return nn.initializers.variance_scaling(scale=scale, mode="fan_in", distribution="normal")
    else:
        print("Initializer not supported. Lecun used as fallback")
        return nn.initializers.lecun_normal()


def uniform_init(a, b):
    def init_func(rng, shape, dtype=jnp.float32):
        return jax.random.uniform(rng, shape, dtype, minval=a, maxval=b)
    return init_func

def update_freezing(state, model, unfreeze_layer, lr, momentum):
    label_fn = flattened_traversal(
        lambda path, _: 'sgd' if path[0] == unfreeze_layer else 'none'
    )
    tx = optax.multi_transform({'sgd': optax.sgd(
        learning_rate=lr, momentum=momentum), 'none': optax.set_to_zero()}, label_fn)
    return train_state.TrainState.create(apply_fn=model.apply, params=state.params, tx=tx)

def flattened_traversal(fn):
    """Returns function that is called with `(path, param)` instead of pytree."""
    def mask(tree):
        flat = flax.traverse_util.flatten_dict(tree)
        return flax.traverse_util.unflatten_dict(
            {k: fn(k, v) for k, v in flat.items()})
    return mask