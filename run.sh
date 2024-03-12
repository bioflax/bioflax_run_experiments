for DEVICE_NUM in 3
do
  CUDA_VISIBLE_DEVICES=$DEVICE_NUM  wandb agent bioflax/bioflax_run_experiments/g9iu7ryy &
  sleep 3
done

