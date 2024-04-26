import wandb
import optax
from jax import random
import jax.numpy as jnp
from jax import config
from typing import Any
from .model import BatchBioNeuralNetwork, BioNeuralNetwork
from .train_helpers import create_train_state, train_epoch, validate, plot_sample, select_initializer, update_freezing
from .dataloading import create_dataset


def train(args):
    """
    Main function for training and eveluating a biomodel. Training and evaluation set up by
    arguments passed in args.

    a.) Selecting the respective dataset is chosen various forms of random error stuff can be done here. 

    b.) Can run in gradient estimator - true gradient mode

    Moreover sharpness is tracked now. args.steps computes the number of steps in the power iteration.

    With full_batch true one uses only the first batch and that way can emulate full batch training.
    """
    config.update("jax_enable_x64", True)

    best_test_loss = 100000000
    best_test_acc = -10000.0

    batch_size = args.batch_size
    val_split = args.val_split
    epochs = args.epochs
    mode = args.mode
    activations = args.activations
    hidden_layers = args.hidden_layers
    seed = args.jax_seed
    dataset = args.dataset
    in_dim = args.in_dim
    seq_len = args.seq_len
    output_features = args.output_features
    train_set_size = args.train_set_size
    test_set_size = args.test_set_size
    teacher_act = args.teacher_act
    lr = args.lr
    momentum = args.momentum
    weight_decay_1 = args.weight_decay_1
    weight_decay_2 = args.weight_decay_2
    plot = args.plot
    compute_alignments = args.compute_alignments
    project = args.wandb_project
    use_wandb = args.use_wandb
    n = args.n
    entity = args.wandb_entity
    optimizer = args.optimizer
    initializer = args.initializer
    scale_w = args.scale_w
    scale_b = args.scale_b
    lam = args.lam
    architecture = args.architecture
    tune_for_lr = args.tune_for_lr
    period = args.period
    p = args.probability
    periodically = args.periodically
    freeze = args.freeze
    steps = args.steps
    full_batch = args.full_batch
    grads_minus_mode = args.grads_minus_mode
    alpha = args.alpha
    period_alpha = args.period_alpha

    if mode == 'bp':
        compute_alignments = False

    if dataset == "mnist":
        task = "classification"
        loss_fn = "CE"
        output_features = 10
    elif dataset == "mnnist_with_targets": # is MSE
        task = "classification"
        loss_fn = "MSE_with_zero_pred_correlated_targets"
        output_features = 10
    elif dataset == "mnist_mse_with_labels":
        dataset = "mnist"
        task = "classification"
        loss_fn = "MSE_with_zero_pred_correlated_labels"
        output_features = 10
    elif dataset == "mnist_with_mse":
        task = "classification"
        dataset = "mnist"
        loss_fn = "MSE_with_integer_labels"
        output_features = 10
    elif dataset == "mnist_mse_with_random_labels":
        task = "classification"
        dataset = "mnist"
        loss_fn = "MSE_with_random_labels"
        output_features = 10
    elif dataset == "mnist_mse_with_predictions":
        task = "classification"
        dataset = "mnist"
        loss_fn = "MSE_with_predictions"
        output_features = 10
    elif dataset == "mnist_ce_with_random_labels_0_pred":
        task = "classification"
        dataset = "mnist"
        loss_fn = "CE_with_random_labels_0_pred"
        output_features = 10
    elif dataset == "mnist_mse_with_loss_interpolation":
        task = "classification"
        dataset = "mnist"
        loss_fn = "MSE_interpolate_loss_alignment"
        output_features = 10
    elif dataset == "mnist_mse_with_control_alignment":
        task = "classification"
        dataset = "mnist"
        loss_fn = "MSE_with_control_alignment"
        output_features = 10
    elif dataset == "mnist_ce_with_loss_interpolation":
        task = "classification"
        dataset = "mnist"
        loss_fn = "CE_interpolate_loss_alignment"
        output_features = 10
    elif dataset == "mnist_ce_with_control_alignment":
        task = "classification"
        dataset = "mnist"
        loss_fn = "CE_with_control_alignment"
        output_features = 10
    elif dataset == "mnist_mse_zero_target":
        task = "classification"
        dataset = "mnist"
        loss_fn = "MSE_with_zero_targets"
        output_features = 10
    elif dataset == "mnist_ce_with_label_zero":
        task = "classification"
        dataset = "mnist"
        loss_fn = "CE_with_label_zero"
        output_features = 10
    elif dataset == "autoencoder":
        task = "regression"
        loss_fn = "MSE"
        output_features = seq_len
    else:
        task = "regression"
        loss_fn = "MSE"

    if dataset == "sinreg":
        output_features = 1
    
    if use_wandb:
        # Make wandb config dictionary
        wandb.init(
            project=project,
            job_type="model_training",
            config=vars(args),
            entity=entity,
        )
    
    #standard
    if architecture == 5:
        hidden_layers = [1000]
        args.hidden_layers = hidden_layers
        activations = ['relu']
        args.activations = activations
    elif architecture == 2:
        hidden_layers = [500, 500]
        args.hidden_layers = hidden_layers
        activations = ['relu', 'relu']
        args.activations = activations
    elif architecture == 3:
        hidden_layers = [5, 5]
        args.hidden_layers = hidden_layers
        activations = ['relu', 'relu']
        args.activations = activations
    elif architecture == 1:
        hidden_layers = [500, 500]
        args.hidden_layers = hidden_layers
        activations = ['identity', 'identity']
        args.activations = activations
    elif architecture == 6:
        hidden_layers = [10]
        args.hidden_layers = hidden_layers
        activations = ['identity']
        args.activations = activations
    elif architecture == 7:
        hidden_layers = [100]
        args.hidden_layers = hidden_layers
        activations = ['identity']
        args.activations = activations
    elif architecture == 8:
        hidden_layers = [500]
        args.hidden_layers = hidden_layers
        activations = ['identity']
        args.activations = activations
    elif architecture == 9:
        hidden_layers = [500, 10, 500]
        args.hidden_layers = hidden_layers
        activations = ['identity', 'identity', 'identity']
        args.activations = activations
    # elif architecture == 1:
    #     hidden_layers = [500, 500]
    #     args.hidden_layers = hidden_layers
    #     activations = ['sigmoid', 'sigmoid']
    #     args.activations = activations
    # elif architecture == 3:
    #     hidden_layers = [500, 500]
    #     args.hidden_layers = hidden_layers
    #     activations = ['leaky_relu', 'leaky_relu']
    #     args.activations = activations
    
    """
    #upon testing activations.yml uncomment this
    if architecture == 1:
            hidden_layers = [500, 500]
            args.hidden_layers = hidden_layers
            activations = ['sigmoid', 'sigmoid']
            args.activations = activations
    elif architecture == 2:
        hidden_layers = [500, 500]
        args.hidden_layers = hidden_layers
        activations = ['relu', 'relu']
        args.activations = activations
    elif architecture == 3:
        hidden_layers = [500, 500]
        args.hidden_layers = hidden_layers
        activations = ['leaky_relu', 'leaky_relu']
        args.activations = activations
    """
    """
    #upon testing architecture.yml uncomment this
    if architecture == 1:
        hidden_layers = [1000]
        args.hidden_layers = hidden_layers
        activations = ['relu']
        args.activations = activations
    elif architecture == 2:
        hidden_layers = [1000, 1000, 1000]
        args.hidden_layers = hidden_layers
        activations = ['relu', 'relu', 'relu']
        args.activations = activations
    elif architecture == 3:
        hidden_layers = [1000, 1000, 1000, 1000, 1000]
        args.hidden_layers = hidden_layers
        activations = ['relu', 'relu', 'relu', 'relu', 'relu']
        args.activations = activations
    elif architecture == 4:
        hidden_layers = [1000, 500, 10, 500, 1000]
        args.hidden_layers = hidden_layers
        activations = ['relu', 'relu', 'relu', 'relu', 'relu']
        args.activations = activations
    elif architecture == 5:
        hidden_layers = [500, 500]
        args.hidden_layers = hidden_layers
        activations = ['relu', 'relu']
        args.activations = activations
    elif architecture == 6:
        hidden_layers = [5, 5]
        args.hidden_layers = hidden_layers
        activations = ['relu', 'relu']
        args.activations = activations
    """
    # Set seed for randomness
    print("[*] Setting Randomness...")
    key = random.PRNGKey(seed)
    key_data, key_model, key_model_bp, key_power_it = random.split(key, num=4)

    (
        trainloader,
        valloader,
        testloader,
        output_features,
        seq_len,
        in_dim,
        train_size
    ) = create_dataset(
        seed=key_data[0].item(),
        batch_size=batch_size,
        dataset=dataset,
        val_split=val_split,
        input_dim=in_dim,
        output_dim=output_features,
        L=seq_len,
        train_set_size=train_set_size,
        test_set_size=test_set_size,
        teacher_act=teacher_act,
    )

    model = BatchBioNeuralNetwork(
        hidden_layers=hidden_layers,
        activations=activations,
        interpolation_factor=lam,
        features=output_features,
        mode=mode,
        initializer_kernel=select_initializer(initializer, scale_w),
        initializer_B=select_initializer(initializer, scale_b),
    )

    # Backpropagation model to compute alignments
    bp_model = BatchBioNeuralNetwork(
        hidden_layers=hidden_layers,
        activations=activations,
        interpolation_factor=lam,
        features=output_features,
        mode="bp",
        initializer_kernel=select_initializer(initializer, scale_w),
        initializer_B=select_initializer(initializer, scale_b),
    )

    key, key_model = random.split(key, num=2)
    if tune_for_lr:
        best_loss_rate = 10000.
        for rate in [0.01, 0.0316, 0.07, 0.1, 0.15, 0.2, 0.25, 0.316, 0.4, 0.5, 0.6, 0.7, 0.8]:
            iter_state = create_train_state(
                model=model,
                rng=key_model,
                lr=rate,  # scheduler, #relevant change at the moment
                momentum=momentum,
                weight_decay_1=weight_decay_1,
                weight_decay_2=weight_decay_2,
                in_dim=in_dim,
                batch_size=batch_size,
                seq_len=seq_len,
                optimizer=optimizer
            )
            for epoch in range(300):
                (
                    iter_state,
                    train_loss_,
                    avg_bias_al_per_layer,
                    avg_wandb_grad_al_per_layer,
                    avg_wandb_grad_al_total,
                    avg_weight_al_per_layer,
                    avg_rel_norm_grads,
                    avg_conv_rate,
                    avg_norm_,
                    avg_norm
                ) = train_epoch(iter_state, bp_model, trainloader, loss_fn, n, mode, False, lam)
            print(train_loss_)
            if train_loss_ < best_loss_rate:
                best_loss_rate = train_loss_
                lr = rate
                # print(rate)

        args.lr = lr
        print(lr)

    # Model to run experiments with
    state = create_train_state(
        model=model,
        rng=key_model,
        lr=lr,  # scheduler, #relevant change at the moment
        momentum=momentum,
        weight_decay_1=weight_decay_1,
        weight_decay_2=weight_decay_2,
        in_dim=in_dim,
        batch_size=batch_size,
        seq_len=seq_len,
        optimizer=optimizer,
        epochs=epochs,
        steps_per_epoch=train_size,
    )

    state_bp = create_train_state(
        model=bp_model,
        rng=key_model_bp,
        lr=lr,  # scheduler, #relevant change at the moment
        momentum=momentum,
        weight_decay_1=weight_decay_1,
        weight_decay_2=weight_decay_2,
        in_dim=in_dim,
        batch_size=batch_size,
        seq_len=seq_len,
        optimizer=optimizer,
        epochs=epochs,
        steps_per_epoch=train_size,
    )

    init_params = state.params

    if freeze:
        unfreeze_layer = 'RandomDenseLinearInterpolateFABP_0'
        state = update_freezing(
                        state, model, unfreeze_layer, lr, momentum)

        

    reset = False
    #HACK: Must be removed
    compute_sharpness = False
    # Training loop over epochs
    best_loss, best_acc, best_epoch = 100000000, - \
        100000000.0, 0  # This best loss is val_loss
    prev_loss=2.
    decay = alpha
    step = alpha
    for i, epoch in enumerate(range(epochs)):  # (args.epochs):
        # periodic switching
        # if i % period_alpha == 0:
        #     inter = alpha
        #     alpha = alpha_
        #     alpha_ = inter
        #if i >= 20:
        #    alpha = 1.0
        alpha = 1 - (.4 * decay)
        decay = decay * step

        
        #print(f"[*] Starting training epoch {epoch + 1}...")
        key_mask, key, key_epoch = random.split(key, num=3)
        # print(state.step)
        # lr_ = scheduler(state.step)
        # HACK: is not used can be used to control how often to compute the sharpness
        if i % 10 == 0:
            compute_sharpness = True
        else:
            compute_sharpness = False
        if(period == -1): #-1 is FA pure (in frist periodic run it was 0)
            reset = False
        else:
            if(periodically and i % period == 0):
                reset = True
            elif( (not periodically) and i == period-1):
                reset = True
            else:
                reset = False
        (
            state,
            train_loss,
            avg_bias_al_per_layer,
            avg_wandb_grad_al_per_layer,
            avg_wandb_grad_al_total,
            avg_weight_al_per_layer,
            avg_rel_norm_grads,
            avg_conv_rate,
            avg_norm_true,
            avg_norm_est,
            avg_norm_kernel_per_layer,
            avg_norm_B_per_layer,
            avg_norm_proj_grad,

        ) = train_epoch(model, state, state_bp, trainloader, 
                        loss_fn, n, mode, compute_alignments, lam, reset, p, key_mask, use_wandb, prev_loss, key_epoch, steps, full_batch, grads_minus_mode, alpha, init_params)
                        #, state_reset, trainloader, loss_fn, n, mode, compute_alignments, lam, reset)
        if (i > 0):
            avg_conv_rate = train_loss/prev_loss
        prev_loss = train_loss
        convergence_metric = avg_wandb_grad_al_total * avg_rel_norm_grads
        cos_squared = avg_wandb_grad_al_total*avg_wandb_grad_al_total

        #HACK: Comment back in if you want the accuracies
        if valloader is not None:
            #print(f"[*] Running Epoch {epoch + 1} Validation...")
            val_loss, val_acc = validate(
                state, valloader, seq_len, in_dim, loss_fn, alpha)

            #print(f"[*] Running Epoch {epoch + 1} Test...")
            test_loss, test_acc = validate(
                state, testloader, seq_len, in_dim, loss_fn, alpha)

            #print(f"\n=>> Epoch {epoch + 1} Metrics ===")
            #print(
            #    f"\tTrain Loss: {train_loss:.5f} "
            #   f"-- Val Loss: {val_loss:.5f} "
            #    f"-- avg_conv_rate: {avg_conv_rate:.5f} "
            #    f"-- Test Loss: {test_loss:.5f}"
            #)
            #if task == "classification":
            #    print(
            #        f"\tVal Accuracy: {val_acc:.4f} " f"-- Test Accuracy: {test_acc:.4f} ")
        else:
            # Use test set as validation set
            #print(f"[*] Running Epoch {epoch + 1} Test...")
            val_loss, val_acc = validate(
                state, testloader, seq_len, in_dim, loss_fn, alpha)

            #print(f"\n=>> Epoch {epoch + 1} Metrics ===")
            #print(
            #    f"\tTrain loss: {train_loss:.5f}" f"-- Test loss: {val_loss:.5f}")
            #if task == "classification":
            #   print(f"-- Test accuracy: {val_acc:.4f}\n")

        #if compute_alignments:
            #print(
            #    f"\tRelative norm gradients: {avg_rel_norm_grads:.4f} "
            #    f"-- Gradient alignment: {avg_wandb_grad_al_total:.4f}"
            #)

        if val_acc > best_acc:
            # Increment counters etc.
            best_loss, best_acc, best_epoch = val_loss, val_acc, epoch
            if valloader is not None:
                best_test_loss, best_test_acc = test_loss, test_acc
            else:
                best_test_loss, best_test_acc = best_loss, best_acc

        # Print best accuracy & loss so far...
        #if task == "regression":
        #    print(
        #        f"\tBest val loss: {best_loss:.5f} at epoch {best_epoch + 1}\n"
        #       f"\tBest test loss: {best_test_loss:.5f} at epoch {best_epoch + 1}\n"
        #    )
        #elif task == "classification":
        #    print(
        #        f"\tBest val loss: {best_loss:.5f} -- Best val accuracy:"
        #        f" {best_acc:.4f} at epoch {best_epoch + 1}\n"
        #        f"\tBest test loss: {best_test_loss:.5f} -- Best test accuracy:"
        #        f" {best_test_acc:.4f} at epoch {best_epoch + 1}\n"
        #    )

        metrics = {}
        #     "Training loss": train_loss,
        #     "Val loss": val_loss,
        #     # "lr" : lr_
        # }

        if task == "classification":
            metrics["Val accuracy"]: val_acc
        if valloader is not None:
            metrics["Test loss"] = test_loss
            if task == "classification":
                metrics["Test accuracy"] = test_acc
            metrics["Validation loss"] = val_loss
            if task == "classification":
                metrics["Validation accuracy"] = val_acc
        if use_wandb:
            wandb.log(metrics)
        # if compute_alignments:
        #     metrics["lambda"] = lam
        #     metrics["Relative norms gradients"] = avg_rel_norm_grads
        #     metrics["Gradient alignment"] = avg_wandb_grad_al_total
        #     #metrics["Convergence metric"] = convergence_metric
        #     metrics["Cos_Squared"] = cos_squared
        #     #metrics["Conv_Rate"] = avg_conv_rate
        #     #metrics["1-Conv_Rate"] = 1. - avg_conv_rate
        #     metrics["Norm true gradient"] = avg_norm
        #     metrics["Norm of est. gradient"] = avg_norm_
        #     #metrics["Approx const. mu/l"] = (1.-avg_conv_rate)/cos_squared
        #     metrics["lr_final"] = lr
        #     for i, al in enumerate(avg_bias_al_per_layer):
        #         metrics[f"Alignment bias gradient layer {i}"] = al
        #     for i, al in enumerate(avg_wandb_grad_al_per_layer):
        #         metrics[f"Alignment gradient layer {i}"] = al
        #     if mode == "fa" or mode == "kp" or mode == "interpolate_fa_bp":
        #         for i, al in enumerate(avg_weight_al_per_layer):
        #             metrics[f"Alignment layer {i}"] = al
        # if use_wandb:
        #     wandb.log(metrics)
        #     wandb.run.summary["Best val loss"] = best_loss
        #     wandb.run.summary["Best epoch"] = best_epoch
        #     wandb.run.summary["Best test loss"] = best_test_loss
        #     if task == "classification":
        #         wandb.run.summary["Best test accuracy"] = best_test_acc
        #         wandb.run.summary["Best val accuracy"] = best_acc

    # print(lr)
    if plot:
        plot_sample(testloader, state, seq_len, in_dim, task, output_features)
