import sys
import shutil

name = sys.argv[1]

results_folder: str = "./results/"+name
tensorboard_log_name = './log/snr_test/'+name

print(name)
shutil.rmtree(tensorboard_log_name)
shutil.rmtree(results_folder)
