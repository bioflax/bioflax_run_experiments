import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.core.frozen_dict import freeze, unfreeze
from typing import Any
import copy


class RandomDenseLinearFA(nn.Module):
    """
    Creates a linear layer which uses feedback alignment for the backward pass.
    For detailed information please refer to https://github.com/yschimpf/bioflax/blob/main/docs/README.md#feedback-alignment-4
    ...
    Attributes
    __________
    features : int
        number of output features
    """

    features: int
    initializer_kernel: Any = nn.initializers.lecun_normal()
    initializer_B: Any = nn.initializers.lecun_normal()

    @nn.compact
    def __call__(self, x):
        B = self.param("B", self.initializer_B,
                       (jnp.shape(x)[-1], self.features))

        def f(module, x, B):
            return module(x)

        def fwd(module, x, B):
            primals_out, vjp_fun = nn.vjp(f, module, x, B)
            return primals_out, (vjp_fun, B)

        def bwd(vjp_fn_B, delta):
            vjp_fn, B = vjp_fn_B
            delta_params, _, _ = vjp_fn(delta)
            delta_x = (B @ delta.transpose()).transpose()
            return (delta_params, delta_x, jnp.zeros_like(B))

        forward_module = nn.Dense(
            self.features, kernel_init=self.initializer_kernel)
        custom_f = nn.custom_vjp(fn=f, forward_fn=fwd, backward_fn=bwd)
        return custom_f(forward_module, x, B)


class RandomDenseLinearKP(nn.Module):
    """
    Creates a linear layer which uses feedback alignment for the backward pass and uses Kolen-Pollack 
    updates to align forward and backward matrices.
    For more detailed information please refer to https://github.com/yschimpf/bioflax/blob/main/docs/README.md#kolen-pollack-6
    ...
    Attributes
    __________
    features : int
        number of output features
    initializer : Any
        initializer for matrices
    """

    features: int
    initializer_kernel: Any = nn.initializers.lecun_normal()
    initializer_B: Any = nn.initializers.lecun_normal()

    @nn.compact
    def __call__(self, x):
        B = self.param("B", self.initializer_B,
                       (jnp.shape(x)[-1], self.features))

        def f(module, x, B):
            return module(x)

        def fwd(module, x, B):
            primals_out, vjp_fun = nn.vjp(f, module, x, B)
            return primals_out, (vjp_fun, B)

        def bwd(vjp_fn_B, delta):
            vjp_fn, B = vjp_fn_B
            delta_params, _, _ = vjp_fn(delta)
            delta_x = (B @ delta.transpose()).transpose()
            return (delta_params, delta_x, delta_params["params"]["kernel"])

        forward_module = nn.Dense(
            self.features, kernel_init=self.initializer_kernel)
        custom_f = nn.custom_vjp(fn=f, forward_fn=fwd, backward_fn=bwd)
        return custom_f(forward_module, x, B)


class RandomDenseLinearDFAOutput(nn.Module):
    """
    Creates an output linear layer which uses direct feedback alignment for the backward pass computations.
    For more detailed information please refer to https://github.com/yschimpf/bioflax/blob/main/docs/README.md#direct-feedback-alignment-5
    ...
    Attributes
    __________
    features : int
        number of output features
    initializer : Any
        initializer for matrices
    """

    features: int
    initializer_kernel: Any = nn.initializers.lecun_normal()

    @nn.compact
    def __call__(self, x):

        def f(module, x):
            return module(x)

        def fwd(module, x):
            return nn.vjp(f, module, x)

        def bwd(vjp_fn, delta):
            delta_params, _ = vjp_fn(delta)
            return (delta_params, delta)

        forward_module = nn.Dense(
            self.features, kernel_init=self.initializer_kernel)
        custom_f = nn.custom_vjp(fn=f, forward_fn=fwd, backward_fn=bwd)
        return custom_f(forward_module, x)


class RandomDenseLinearDFAHidden(nn.Module):
    """
    Creates a hidden linear layer which uses direct feedback alignment for the backward pass.
    For more detailed information please refer to https://github.com/yschimpf/bioflax/blob/main/docs/README.md#direct-feedback-alignment-5
    ...
    Attributes
    __________
    features : int
        number of output features
    final_output_dim : int
        number of output features of the output layer
    activation : Any = nn.relu
        activation function applied to the weighted inputs of the layer
    initializer : Any   
        initializer for matrices
    """
    features: int
    final_output_dim: int
    initializer_kernel: Any = nn.initializers.lecun_normal()
    initializer_B: Any = nn.initializers.lecun_normal()
    activation: Any = nn.relu

    @nn.compact
    def __call__(self, x):
        B = self.param("B", self.initializer_B,
                       (self.features, self.final_output_dim))

        def f(module, x, B):
            return self.activation(module(x))

        def fwd(module, x, B):
            primals_out, vjp_fun = nn.vjp(f, module, x, B)
            return primals_out, (vjp_fun, B)

        def bwd(vjp_fn_B, delta):
            vjp_fn, B = vjp_fn_B
            delta_params, _, _ = vjp_fn((B @ delta.transpose()).transpose())
            return (delta_params, delta, jnp.zeros_like(B))

        forward_module = nn.Dense(
            self.features, kernel_init=self.initializer_kernel)
        custom_f = nn.custom_vjp(fn=f, forward_fn=fwd, backward_fn=bwd)
        return custom_f(forward_module, x, B)

class RandomDenseLinearInterpolateFABP(nn.Module):
    """
    Creates a linear layer which interpolates between feedback alignment and backpropagation.
    ...
    Attributes
    __________
    features : int
        number of output features
    interpolation_factor : float
        interpolation factor for linear interpolation between fa and bp. A value of 1 represents fa and a value of 0 bp.
    """

    features: int
    interpolation_factor: float
    initializer_kernel: Any = nn.initializers.lecun_normal()
    initializer_B: Any = nn.initializers.lecun_normal()

    @nn.compact
    def __call__(self, x):
        B = self.param("B", self.initializer_B,
                       (jnp.shape(x)[-1], self.features))

        def f(module, x, B):
            return module(x)

        def fwd(module, x, B):
            primals_out, vjp_fun = nn.vjp(f, module, x, B)
            inter = self.interpolation_factor * B + \
                (1-self.interpolation_factor) * \
                module.variables['params']['kernel']
            return primals_out, (vjp_fun, B, inter)

        def bwd(vjp_fn_B, delta):
            vjp_fn, B, inter = vjp_fn_B
            delta_params, _, _ = vjp_fn(delta)
            delta_x = (inter @ delta.transpose()).transpose()
            return (delta_params, delta_x, jnp.zeros_like(B))

        forward_module = nn.Dense(
            self.features, kernel_init=self.initializer_kernel)
        custom_f = nn.custom_vjp(fn=f, forward_fn=fwd, backward_fn=bwd)
        return custom_f(forward_module, x, B)
    

class ResetLayer(nn.Module):
    """
    Creates a linear layer which interpolates between feedback alignment and backpropagation.
    ...
    Attributes
    __________
    features : int
        number of output features
    interpolation_factor : float
        interpolation factor for linear interpolation between fa and bp. A value of 1 represents fa and a value of 0 bp.
    """

    features: int
    interpolation_factor: float
    initializer_kernel: Any = nn.initializers.lecun_normal()
    initializer_B: Any = nn.initializers.lecun_normal()

    @nn.compact
    def __call__(self, x):
        B = self.param("B", self.initializer_B,
                       (jnp.shape(x)[-1], self.features))

        def f(module, x, B):
            return module(x)

        def fwd(module, x, B):
            primals_out, vjp_fun = nn.vjp(f, module, x, B)

            inter = self.interpolation_factor * B + \
                (1-self.interpolation_factor) * \
                module.variables['params']['kernel']
            return primals_out, (vjp_fun, B, inter)

        def bwd(vjp_fn_B, delta):
            vjp_fn, B, inter = vjp_fn_B
            delta_params, _, _ = vjp_fn(delta)
            delta_x = (inter @ delta.transpose()).transpose()
            return (delta_params, delta_x, B - inter)

        forward_module = nn.Dense(
            self.features, kernel_init=self.initializer_kernel)
        custom_f = nn.custom_vjp(fn=f, forward_fn=fwd, backward_fn=bwd)
        return custom_f(forward_module, x, B)


"""
B = self.param("B", self.initializer_B,
                       (jnp.shape(x)[-1], self.features))

        def f(module, x, B):
            return module(x)

        def fwd(module, x, B):
            primals_out, vjp_fun = nn.vjp(f, module, x, B)
            inter = self.interpolation_factor * B + (1-self.interpolation_factor) * module.variables['params']['kernel']
            return primals_out, (vjp_fun, inter)

        def bwd(vjp_fn_B, delta):
            vjp_fn, inter = vjp_fn_B
            delta_params, _, _ = vjp_fn(delta)
            delta_x = (inter @ delta.transpose()).transpose()
            return (delta_params, delta_x, jnp.zeros_like(B))

        forward_module = nn.Dense(
            self.features, kernel_init=self.initializer_kernel)
        custom_f = nn.custom_vjp(fn=f, forward_fn=fwd, backward_fn=bwd)
        return custom_f(forward_module, x, B, inter)
"""


class BioNeuralNetwork(nn.Module):
    """
    Creates a neural network with bio-inspired layers according to selected mode.
    ...
    Attributes
    __________
    hidden_layers : [int]
        list of hidden layer dimensions
    activations : [str]
    features : int 
        number of output features
        list of activation functions for hidden layers
    mode : str
        mode of the network, either "bp" for backpropagation, "fa" for feedback alignment, 
        "dfa" for direct feedback alignment, or "kp" for kollen-pollack
    """

    hidden_layers: [int]
    activations: [str]
    interpolation_factor: float = 0
    initializer_kernel: Any = nn.initializers.lecun_normal()
    initializer_B: Any = nn.initializers.lecun_normal()
    features: int = 4
    mode: str = "bp"

    @nn.jit
    @nn.compact
    def __call__(self, x):
        for features, activation in zip(self.hidden_layers, self.activations):
            if self.mode == "bp":
                x = nn.Dense(features, kernel_init=self.initializer_kernel)(x)
                if activation != "identity":
                    x = getattr(nn, activation)(x)
            elif self.mode == "fa":
                x = RandomDenseLinearFA(
                    features, self.initializer_kernel, self.initializer_B)(x)
                if activation != "identity":
                    x = getattr(nn, activation)(x)
            elif self.mode == "dfa":
                x = RandomDenseLinearDFAHidden(
                    features, self.features, self.initializer_kernel, self.initializer_B, getattr(nn, activation))(x)
            elif self.mode == "kp":
                x = RandomDenseLinearKP(
                    features, self.initializer_kernel, self.initializer_B)(x)
                if activation != "identity":
                    x = getattr(nn, activation)(x)
            elif self.mode == "interpolate_fa_bp":
                x = RandomDenseLinearInterpolateFABP(
                    features, self.interpolation_factor, self.initializer_kernel, self.initializer_B)(x)
                if activation != "identity":
                    x = getattr(nn, activation)(x)
            elif self.mode == "reset":
                x = ResetLayer(
                    features, self.interpolation_factor, self.initializer_kernel, self.initializer_B)(x)
                if activation != "identity":
                    x = getattr(nn, activation)(x)
        if self.mode == "bp":
            x = nn.Dense(self.features, kernel_init=self.initializer_kernel)(x)
        elif self.mode == "fa":
            x = RandomDenseLinearFA(
                self.features, self.initializer_kernel, self.initializer_B)(x)
        elif self.mode == "dfa":
            x = RandomDenseLinearDFAOutput(
                self.features, self.initializer_kernel)(x)
        elif self.mode == "kp":
            x = RandomDenseLinearKP(
                self.features, self.initializer_kernel, self.initializer_B)(x)
        elif self.mode == "interpolate_fa_bp":
            x = RandomDenseLinearInterpolateFABP(
                self.features, self.interpolation_factor, self.initializer_kernel, self.initializer_B)(x)
        elif self.mode == "reset":
            x = ResetLayer(
                self.features, self.interpolation_factor, self.initializer_kernel, self.initializer_B)(x)
        return x


class TeacherNetwork(nn.Module):
    """
    Creates a simple Teacher Network to train a student network.
    """
    features: int = 1
    activation: str = "sigmoid"

    @nn.compact
    def __call__(self, x):
        # x = nn.Dense(64)(x)
        # if self.activation != "identity":
        #    x = getattr(nn, self.activation)(x)
        x = nn.Dense(2)(x)
        if self.activation != "identity":
            x = getattr(nn, self.activation)(x)
        x = nn.Dense(self.features)(x)
        return x

class FreezeNeuralNetwork(nn.Module):
    features: int = 10

    @nn.compact
    def __call__(self,x):
        x = nn.Dense(500)(x)
        x = nn.relu(x)
        x = nn.Dense(500)(x)
        x = nn.relu(x)
        x = RandomDenseLinearFA(self.features)(x)
        return x

BatchFreezeNeuralNetwork = nn.vmap(
    FreezeNeuralNetwork,
    in_axes=0,
    out_axes=0,
    variable_axes={'params': None},
    split_rngs={'params': False},
    axis_name='batch',
)

BatchTeacher = nn.vmap(
    TeacherNetwork,
    in_axes=0,
    out_axes=0,
    variable_axes={'params': None},
    split_rngs={'params': False},
    axis_name='batch',
)

BatchBioNeuralNetwork = nn.vmap(
    BioNeuralNetwork,
    in_axes=0,
    out_axes=0,
    variable_axes={'params': None},
    split_rngs={'params': False},
    axis_name='batch',
)
