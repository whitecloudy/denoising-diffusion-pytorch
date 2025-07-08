import sys
import shutil

name_list = sys.argv[1:]

for name in name_list:
    results_folder: str = "./results/"+name
    tensorboard_log_name = './log/'+name

    print(name)
    shutil.rmtree(tensorboard_log_name)
    shutil.rmtree(results_folder)
