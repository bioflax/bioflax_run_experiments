for DEVICE_NUM in 2
do
  CUDA_VISIBLE_DEVICES=$DEVICE_NUM  wandb agent bioflax/bioflax_run_experiments/1eync3wg &
  sleep 3
done

