for DEVICE_NUM in 2 3
do
  CUDA_VISIBLE_DEVICES=$DEVICE_NUM wandb agent bioflax/bioflax_run_experiments/z6gk0oqf &
  sleep 3
done

