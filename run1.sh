for DEVICE_NUM in 1 2 3
do
  CUDA_VISIBLE_DEVICES=$DEVICE_NUM wandb agent bioflax/bioflax_run_experiments/2qvmdajy &
  sleep 3
done