for DEVICE_NUM in 0
do
  CUDA_VISIBLE_DEVICES=$DEVICE_NUM wandb agent bioflax/bioflax_run_experiments/pbya336x &
  sleep 3
done

