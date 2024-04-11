for DEVICE_NUM in 0 1 2
do
  CUDA_VISIBLE_DEVICES=$DEVICE_NUM wandb agent bioflax/bioflax_run_experiments/jiygavj7 &
  sleep 3
done

