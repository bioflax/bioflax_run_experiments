for DEVICE_NUM in 1 3
do
  CUDA_VISIBLE_DEVICES=$DEVICE_NUM wandb agent bioflax/bioflax_run_experiments/6gv8ejyz &
  sleep 3
done

