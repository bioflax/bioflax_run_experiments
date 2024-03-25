import wandb
import optax
from jax import random
from jax import config
from typing import Any
from .model import BatchBioNeuralNetwork, BioNeuralNetwork
from .train_helpers import create_train_state, train_epoch, validate, plot_sample, select_initializer
from .dataloading import create_dataset


def train(args):
    """
    Main function for training and eveluating a biomodel. Training and evaluation set up by
    arguments passed in args.
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
    weight_decay = args.weight_decay
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
    loss_func = args.loss_func
    eval_ = args.eval

    """
    Now contains possibility to run with a random loss as follows: For L_CE sample label
    y randomly and fix the output to be constant zero. The random sampling is done via setting 
    loss_func to 'rand' normal computation is done via 'arch'. To have constant zero output the resective 
    line in compute_bp_grads and train_step have to be outcomment and incommented respectively. 
    """

    if mode == 'bp':
        compute_alignments = False

    if dataset == "mnist":
        task = "classification"
        if loss_func == 'arch':
            loss_fn = "CE"
        elif loss_func == "rand":
            loss_fn = "random"
        output_features = 10
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
    """
    standard
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
    elif architecture == 6:
        hidden_layers = [10, 10]
        args.hidden_layers = hidden_layers
        activations = ['identity', 'identity']
        args.activations = activations
    elif architecture == 4:
        hidden_layers = [2]
        args.hidden_layers = hidden_layers
        activations = ['identity']
        args.activations = activations
    elif architecture == 1:
        hidden_layers = [500, 500]
        args.hidden_layers = hidden_layers
        activations = ['sigmoid', 'sigmoid']
        args.activations = activations
    elif architecture == 3:
        hidden_layers = [500, 500]
        args.hidden_layers = hidden_layers
        activations = ['leaky_relu', 'leaky_relu']
        args.activations = activations
    """
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
        activations = ['relu', 'relu'] #standard is relu
        args.activations = activations
    elif architecture == 6:
        hidden_layers = [500, 500]
        args.hidden_layers = hidden_layers
        activations = ['sigmoid', 'sigmoid'] #standard is relu
        args.activations = activations

    # Set seed for randomness
    print("[*] Setting Randomness...")
    key = random.PRNGKey(seed)
    key_data, key_model, key_model_bp = random.split(key, num=3)
    print(key_data[0].item())

    # Create dataset
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
    print(f"[*] Starting training on '{dataset}'...")

    # Learning rate scheduler
    # print(total_steps)
    # scheduler = optax.warmup_cosine_decay_schedule(init_value=lr, peak_value=5*lr, warmup_steps=0.1*total_steps, decay_steps=total_steps)

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
                weight_decay=weight_decay,
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
        weight_decay=weight_decay,
        in_dim=in_dim,
        batch_size=batch_size,
        seq_len=seq_len,
        optimizer=optimizer
    )

    state_bp = create_train_state(
        model=bp_model,
        rng=key_model_bp,
        lr=lr,  # scheduler, #relevant change at the moment
        momentum=momentum,
        weight_decay=weight_decay,
        in_dim=in_dim,
        batch_size=batch_size,
        seq_len=seq_len,
        optimizer=optimizer
    )

    # Training loop over epochs
    best_loss, best_acc, best_epoch = 100000000, - \
        100000000.0, 0  # This best loss is val_loss
    for i, epoch in enumerate(range(epochs)):  # (args.epochs):
        #print(f"[*] Starting training epoch {epoch + 1}...")
        print(i)
        # print(state.step)
        # lr_ = scheduler(state.step)
        key, key_curr = random.split(key, num=2)
        (
            state,
            train_loss,
            avg_bias_al_per_layer,
            avg_wandb_grad_al_per_layer,
            avg_wandb_grad_al_total,
            avg_weight_al_per_layer,
            avg_rel_norm_grads,
            avg_conv_rate,
            avg_norm_,
            avg_norm
        ) = train_epoch(state, state_bp, trainloader, loss_fn, n, mode, compute_alignments, lam, key_curr)
        if (i > 0):
            avg_conv_rate = train_loss/prev_loss
        prev_loss = train_loss
        convergence_metric = avg_wandb_grad_al_total * avg_rel_norm_grads
        cos_squared = avg_wandb_grad_al_total*avg_wandb_grad_al_total

        if valloader is not None:
            #print(f"[*] Running Epoch {epoch + 1} Validation...")
            if eval_:
                val_loss, val_acc = validate(
                    state, valloader, seq_len, in_dim, loss_fn)
            else:
                val_loss = 0
                val_acc = 0

            #print(f"[*] Running Epoch {epoch + 1} Test...")
            if eval_:
                test_loss, test_acc = validate(
                    state, testloader, seq_len, in_dim, loss_fn)
            else:
                test_loss = 0
                test_acc = 0

            print(f"\n=>> Epoch {epoch + 1} Metrics ===")
            print(
               f"\tTrain Loss: {train_loss:.5f} "
              f"-- Val Loss: {val_loss:.5f} "
               f"-- avg_conv_rate: {avg_conv_rate:.5f} "
               f"-- Test Loss: {test_loss:.5f}"
            )
            #if task == "classification":
            #    print(
            #        f"\tVal Accuracy: {val_acc:.4f} " f"-- Test Accuracy: {test_acc:.4f} ")
        else:
            # Use test set as validation set
            #print(f"[*] Running Epoch {epoch + 1} Test...")
            if eval_:
                val_loss, val_acc = validate(
                    state, testloader, seq_len, in_dim, loss_fn)
            else:
                val_loss = 0
                val_acc = 0

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

        metrics = {
            "Training loss": train_loss,
            "Val loss": val_loss,
            # "lr" : lr_
        }

        if task == "classification":
            metrics["Val accuracy"]: val_acc
        if valloader is not None:
            metrics["Test loss"] = test_loss
            if task == "classification":
                metrics["Test accuracy"] = test_acc

        if compute_alignments:
            metrics["lambda"] = lam
            metrics["Relative norms gradients"] = avg_rel_norm_grads
            metrics["Gradient alignment"] = avg_wandb_grad_al_total
            metrics["Convergence metric"] = convergence_metric
            metrics["Cos_Squared"] = cos_squared
            metrics["Conv_Rate"] = avg_conv_rate
            metrics["1-Conv_Rate"] = 1. - avg_conv_rate
            metrics["Norm true gradient"] = avg_norm
            metrics["Norm of est. gradient"] = avg_norm_
            metrics["Approx const. mu/l"] = (1.-avg_conv_rate)/cos_squared
            metrics["lr_final"] = lr
            for i, al in enumerate(avg_bias_al_per_layer):
                metrics[f"Alignment bias gradient layer {i}"] = al
            for i, al in enumerate(avg_wandb_grad_al_per_layer):
                metrics[f"Alignment gradient layer {i}"] = al
            if mode == "fa" or mode == "kp" or mode == "interpolate_fa_bp":
                for i, al in enumerate(avg_weight_al_per_layer):
                    metrics[f"Alignment layer {i}"] = al

        if use_wandb:
            wandb.log(metrics)
            wandb.run.summary["Best val loss"] = best_loss
            wandb.run.summary["Best epoch"] = best_epoch
            wandb.run.summary["Best test loss"] = best_test_loss
            if task == "classification":
                wandb.run.summary["Best test accuracy"] = best_test_acc
                wandb.run.summary["Best val accuracy"] = best_acc

    # print(lr)
    if plot:
        plot_sample(testloader, state, seq_len, in_dim, task, output_features)
