import os
import shutil
from tqdm import tqdm
import fnmatch

if __name__ == "__main__":

    # set the source and destination folder paths
    src_folder = '/ds2/remote_sensing/ben-ge/ben-ge/sentinel-2/'
    dst_folder = '/ds2/remote_sensing/ben-ge/ben-ge/s2_npy/'

    # get a list of all files in the source folder
    folders = os.listdir(src_folder)

    # iterate over the files and check if they are NumPy files
    i = 0
    for folder in tqdm(folders):
        file_folder = os.listdir(src_folder+folder)
        for file in file_folder:
            if file.endswith('_all_bands.npy'):
                # if the file is a NumPy file, copy it to the destination folder
                src_file = os.path.join(src_folder, folder, file)
                dst_file = os.path.join(dst_folder, file)
                shutil.copy(src_file, dst_file)