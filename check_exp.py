import os
dir = "experiments"



subfolders = [f.path for f in os.scandir(dir) if f.is_dir() ]
print(subfolders, len(subfolders))


initial_count = 0
for folder in subfolders:
    path, dirs, files = next(os.walk(folder))
    initial_count += len(files)
print('Experiments Done' , int(initial_count / 3))
print('Experiments Missing', 720 - int(initial_count / 3))
