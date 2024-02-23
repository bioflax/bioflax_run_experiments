for DEVICE_NUM in 2 3
do
  CUDA_VISIBLE_DEVICES=$DEVICE_NUM wandb agent bioflax/bioflax_run_experiments/2s3rzz5f &
  sleep 3
done

