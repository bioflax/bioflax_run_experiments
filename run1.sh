for DEVICE_NUM in 2 3
do
  CUDA_VISIBLE_DEVICES=$DEVICE_NUM wandb agent bioflax/bioflax_run_experiments/7tpm8hsw &
  sleep 3
done