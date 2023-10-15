
import os

#extract the tensorboard file and config file to target path with same floder structure
source_path = '/home/wangwenxuan/gops_idp/gops/results/mujoco/dsac2w2mean_mlp'
target_path = '/home/wangwenxuan/gops_idp/gops/results/mujoco/test'

if not os.path.exists(target_path):
    os.makedirs(target_path)

for floders in os.listdir(source_path):
    if not os.path.exists(os.path.join(target_path, floders)):
        os.makedirs(os.path.join(target_path, floders))
    for files in os.listdir(os.path.join(source_path, floders)):
        if files.startswith("events.out.tfevents"):
            os.system("cp " + os.path.join(source_path, floders, files) + " " + os.path.join(target_path, floders, files))
        if files == "config.json":
            os.system("cp " + os.path.join(source_path, floders, files) + " " + os.path.join(target_path, floders, files))