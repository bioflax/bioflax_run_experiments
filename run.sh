for DEVICE_NUM in 0 1
do
  CUDA_VISIBLE_DEVICES=$DEVICE_NUM wandb agent bioflax/bioflax_run_experiments/3jfhzhoe &
  sleep 3
done

