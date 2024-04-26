for DEVICE_NUM in 2 3
do
  CUDA_VISIBLE_DEVICES=$DEVICE_NUM wandb agent bioflax/bioflax_run_experiments/kgwipsp3 &
  sleep 3
done