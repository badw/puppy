import os 
import gzip
def file_unzip(files:list)->None:
    for file in files:
        if os.path.exists(file):
            with gzip.open(file, 'rb') as f_in, open(file.replace('.gz', ''), 'wb') as f_out:
                f_out.writelines(f_in)