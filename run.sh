for DEVICE_NUM in 2 3
do
  CUDA_VISIBLE_DEVICES=$DEVICE_NUM wandb agent bioflax/bioflax_run_experiments/t6q9f1nv &
  sleep 3
done

