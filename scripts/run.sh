python scripts/main.py --config-file configs/USL_MTCluster.yml \
  GPU_Device [0,1,2,3] \
  OUTPUT_DIR "/mnt/data1/shenleqi/log/USLUnReID/MTCluster_GPU4"
# python scripts/main.py --config-file configs/USL_MTCluster.yml \
#   GPU_Device [0,1,2,3] \
#   RESUME '/mnt/data1/shenleqi/log/USLUnReID/MTCluster_GPU4/model_mAP_best.pth.tar' \
#   TEST.EVAL_ONLY True
