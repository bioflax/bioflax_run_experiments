import wandb
import optax
import jax
import jax.numpy as jnp
import flax.core.frozen_dict as froz_dict
from jax import random
from jax import config
from tqdm import tqdm
from typing import Any
from .metric_computation import compute_metrics
from .model import BatchBioNeuralNetwork, BatchTeacher, BioNeuralNetwork
from .train_helpers import alignment_step, get_loss, reorganize_dict, random_initialize_grad, prep_batch, create_train_state, train_epoch, train_step, validate, plot_sample, select_initializer
from .dataloading import create_dataset


def train(args):
    """
    Main function for training and eveluating a biomodel. Training and evaluation set up by
    arguments passed in args.
    """
    config.update("jax_enable_x64", True)

    best_test_loss = 100000000
    best_test_acc = -10000.0
    # halloo
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
    samples = args.samples
    use_bias = args.use_bias
    hid_lay_size = args.hid_lay_size
    angle = args.angle
    use_fixed_direction = args.use_fixed_direction

    if dataset == "mnist":
        task = "classification"
        loss_fn = "CE"
        output_features = 10
    else:
        task = "regression"
        loss_fn = "MSE"

    if dataset == "sinreg":
        output_features = 1

    if architecture == 1:
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
        hidden_layers = [50, 50]
        args.hidden_layers = hidden_layers
        activations = ['identity', 'identity']
        args.activations = activations
    elif architecture == 4:
        hidden_layers = [100, 100]
        args.hidden_layers = hidden_layers
        activations = ['identity', 'identity']
        args.activations = activations
    elif architecture == 5:
        hidden_layers = [hid_lay_size]
        args.hidden_layers = hidden_layers
        activations = ['identity']
        args.activations = activations
    elif architecture == 6:
        hidden_layers = [15, 15]
        args.hidden_layers = hidden_layers
        activations = ['identity', 'identity']
        args.activations = activations

    if use_wandb:
        # Make wandb config dictionary
        wandb.init(
            project=project,
            job_type="model_training",
            config=vars(args),
            entity=entity,
        )

    # Set seed for randomness
    print("[*] Setting Randomness...")
    key = random.PRNGKey(seed)
    # das war evtl davor ganz falsch
    key, key_data, key_model, key_model_lr, key_model_bp, key_model_teacher, key_dataset, key_rand = random.split(
        key, num=8)

    # Create dataset
    if test_set_size > 0 or train_set_size > 0:
        (
            trainloader,
            valloader,
            testloader,
            output_features,
            seq_len,
            in_dim,
            train_size
        ) = create_dataset(
            seed=key_dataset[0].item(),
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
        use_bias=use_bias
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
        use_bias=use_bias
    )

    if tune_for_lr:
        best_loss_rate = 10000.
        for rate in [0.01, 0.0316, 0.07, 0.1, 0.15, 0.2, 0.25, 0.316, 0.4, 0.5, 0.6, 0.7, 0.8]:
            iter_state = create_train_state(
                model=model,
                rng=key_model_lr,
                lr=rate,  # scheduler, #relevant change at the moment
                momentum=momentum,
                weight_decay=weight_decay,
                in_dim=in_dim,
                batch_size=batch_size,
                seq_len=seq_len,
                optimizer=optimizer,
                use_bias=use_bias
            )
            for epoch in range(5):
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
                ) = train_epoch(iter_state, bp_model, trainloader, loss_fn, n, mode, False, lam, angle)
            print(train_loss_)
            if train_loss_ < best_loss_rate:
                best_loss_rate = train_loss_
                lr = rate
                # print(rate)

        args.lr = lr

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
        lr=lr,  # scheduler, relevant change at the moment
        momentum=momentum,
        weight_decay=weight_decay,
        in_dim=in_dim,
        batch_size=batch_size,
        seq_len=seq_len,
        optimizer=optimizer
    )

    jited_compute_metrics = jax.jit(compute_metrics, static_argnums=[3, 4, 5])

    teacher_model = BatchTeacher(
        features=output_features, activation=teacher_act, hid_lay_size=hid_lay_size)
    teacher_params = teacher_model.init(key_model_teacher, jnp.ones(
        (batch_size, in_dim, seq_len)))['params']

    key, key_data = jax.random.split(key, num=2)
    initializer = jax.nn.initializers.normal(1.0)
    inputs = initializer(key_data, shape=(batch_size, in_dim, seq_len))
    # x = nn.initializers.uniform(scale=2)(
    #    key, shape=(batch_size, d_input, L)) - 1.
    labels = teacher_model.apply({'params': teacher_params}, inputs)
    # das mal weglassen sonst unmöglich die teacher weights zu recovern labels -= jnp.mean(labels, axis=0)

    def loss_comp(params):
        logits = bp_model.apply({"params": params}, inputs)
        loss = get_loss(loss_fn, logits, labels)
        return loss
    _, grads_ = jax.value_and_grad(loss_comp)(
        reorganize_dict({"params": state.params})["params"]
    )

    n = random_initialize_grad(grads_, key_rand)

    # Training loop over epochs
    conv_rate = 0
    best_loss, best_acc, best_epoch = 100000000, - \
        100000000.0, 0  # This best loss is val_loss
    for i, epoch in enumerate(range(epochs)):  # (args.epochs):
        print(f"[*] Starting training epoch {epoch + 1}...")

        # print(state.step)
        # lr_ = scheduler(state.step)

        # , batch in enumerate(tqdm(trainloader)):
        for i in tqdm(range(samples)):
            key, key_data = jax.random.split(key, num=2)
            initializer = jax.nn.initializers.normal(1.0)
            inputs = initializer(key_data, shape=(batch_size, in_dim, seq_len))
            # x = nn.initializers.uniform(scale=2)(
            #    key, shape=(batch_size, d_input, L)) - 1.
            labels = teacher_model.apply({'params': teacher_params}, inputs)
            # das mal weglassen sonst unmöglich teacher weights zu recovern labels -= jnp.mean(labels, axis=0)
            # inputs, labels = prep_batch(batch)

            if compute_alignments:
                if mode != "bp":
                    grads_ = alignment_step(state, state_bp, inputs, labels, loss_fn)
                else:
                    grads_ = grads
            state, loss, grads, norm_p = train_step(
                state, inputs, labels, loss_fn, grads_, angle, use_fixed_direction, n)
            if compute_alignments:
                (
                    bias_al_per_layer,
                    wandb_grad_al_per_layer,
                    wandb_grad_al_total,
                    weight_al_per_layer,
                    rel_norm_grads,
                    norm_,
                    norm
                ) = jited_compute_metrics(state, grads_, grads, mode, lam, use_bias)
                if (i > 0):
                    conv_rate = loss/prev_loss
                prev_loss = loss
                convergence_metric = wandb_grad_al_total * rel_norm_grads
                cos_squared = wandb_grad_al_total*wandb_grad_al_total
                theoret_scale_lr = wandb_grad_al_total * 1./(rel_norm_grads)
                # state.opt_state.hyperparams["learning_rate"] = jnp.array(
                #    theoret_scale_lr, dtype=jnp.float64
                # )
            metrics = {
                "Training loss epoch": loss,
                # "lr" : lr_
            }
            if compute_alignments:
                metrics["norm_p"] = norm_p
                metrics["lambda"] = lam
                metrics["Relative norms gradients"] = rel_norm_grads
                metrics["Gradient alignment"] = wandb_grad_al_total
                metrics["Convergence metric"] = convergence_metric
                metrics["Cos_Squared"] = cos_squared
                metrics["Conv_Rate"] = conv_rate
                metrics["1-Conv_Rate"] = 1. - conv_rate
                metrics["Norm true gradient"] = norm
                metrics["Norm of est. gradient"] = norm_
                metrics["Approx const. mu/l"] = (1.-conv_rate)/cos_squared
                metrics["lr_final"] = lr
                metrics["theoret_scale_lr"] = theoret_scale_lr
                for i, al in enumerate(bias_al_per_layer):
                    metrics[f"Alignment bias gradient layer {i}"] = al
                for i, al in enumerate(wandb_grad_al_per_layer):
                    metrics[f"Alignment gradient layer {i}"] = al
                if mode == "fa" or mode == "kp" or mode == "interpolate_fa_bp":
                    for i, al in enumerate(weight_al_per_layer):
                        metrics[f"Alignment layer {i}"] = al
            if use_wandb:
                wandb.log(metrics)
        """
        if valloader is not None:
            print(f"[*] Running Epoch {epoch + 1} Validation...")
            val_loss, val_acc = validate(
                state, valloader, seq_len, in_dim, loss_fn)

            print(f"[*] Running Epoch {epoch + 1} Test...")
            test_loss, test_acc = validate(
                state, testloader, seq_len, in_dim, loss_fn)

            print(f"\n=>> Epoch {epoch + 1} Metrics ===")
            print(
                f"\tTrain Loss: {loss:.5f} "
                f"-- Val Loss: {val_loss:.5f} "
                f"-- conv_rate: {conv_rate:.5f} "
                f"-- Test Loss: {test_loss:.5f}"
            )
            if task == "classification":
                print(
                    f"\tVal Accuracy: {val_acc:.4f} " f"-- Test Accuracy: {test_acc:.4f} ")
        else:
            # Use test set as validation set
            print(f"[*] Running Epoch {epoch + 1} Test...")
            val_loss, val_acc = validate(
                state, testloader, seq_len, in_dim, loss_fn)

            print(f"\n=>> Epoch {epoch + 1} Metrics ===")
            print(
                f"\tTrain loss: {loss:.5f}" f"-- Test loss: {val_loss:.5f}")
            if task == "classification":
                print(f"-- Test accuracy: {val_acc:.4f}\n")

        if compute_alignments:
            print(
                f"\tRelative norm gradients: {rel_norm_grads:.4f} "
                f"-- Gradient alignment: {wandb_grad_al_total:.4f}"
            )

        if val_acc > best_acc:
            # Increment counters etc.
            best_loss, best_acc, best_epoch = val_loss, val_acc, epoch
            if valloader is not None:
                best_test_loss, best_test_acc = test_loss, test_acc
            else:
                best_test_loss, best_test_acc = best_loss, best_acc

        # Print best accuracy & loss so far...
        if task == "regression":
            print(
                f"\tBest val loss: {best_loss:.5f} at epoch {best_epoch + 1}\n"
                f"\tBest test loss: {best_test_loss:.5f} at epoch {best_epoch + 1}\n"
            )
        elif task == "classification":
            print(
                f"\tBest val loss: {best_loss:.5f} -- Best val accuracy:"
                f" {best_acc:.4f} at epoch {best_epoch + 1}\n"
                f"\tBest test loss: {best_test_loss:.5f} -- Best test accuracy:"
                f" {best_test_acc:.4f} at epoch {best_epoch + 1}\n"
            )

        epoch_metrics = {
            "Training loss epoch": loss,
            "Val loss epoch": val_loss,
        }

        if task == "classification":
            epoch_metrics["Val accuracy"]: val_acc
        if valloader is not None:
            epoch_metrics["Test loss"] = test_loss
            if task == "classification":
                epoch_metrics["Test accuracy"] = test_acc

        if use_wandb:
            wandb.log(epoch_metrics)
            wandb.run.summary["Best val loss"] = best_loss
            wandb.run.summary["Best epoch"] = best_epoch
            wandb.run.summary["Best test loss"] = best_test_loss
            if task == "classification":
                wandb.run.summary["Best test accuracy"] = best_test_acc
                wandb.run.summary["Best val accuracy"] = best_acc
        """

    # print(state)

    # print(lr)
    if plot:
        plot_sample(testloader, state, seq_len, in_dim, task, output_features)
