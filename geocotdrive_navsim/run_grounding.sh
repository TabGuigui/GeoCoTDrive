python3 /home/zhouxubin/navsim/navsimvladrive/navsim/planning/script/run_generate_plangrounding_withgemni.py \
    --navsim_log_path /mnt/tf-mdriver-jfs/sdagent-shard-bj-baiducloud/openscene-v1.1/meta_datas  \
    --sensor_blobs_path /mnt/tf-mdriver-jfs/sdagent-shard-bj-baiducloud/openscene-v1.1/sensor_blobs  \
    --output_dir /data/geocotdrive_data/PlanGrounding_v2 \
    --api_key sk-5unacqt1ajdDWlOeacYIVb7tZQ4Ap2r9nmzCf5dl2HTCqHU8 \
    --backend gemini \
    --model gemini-3.1-flash-lite-preview