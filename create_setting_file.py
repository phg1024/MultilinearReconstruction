import os
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('directory', type=str)
parser.add_argument('--output_file', type=str, default=None)
parser.add_argument('--recursive', action='store_true')

args = parser.parse_args()

root_dir = args.directory

img_exts = ['.jpg', '.png']
img_paths = []

if args.recursive:
    for subdir, dirs, files in os.walk(root_dir):
        for file in files:
            basename, ext = os.path.splitext(file)
            img_path = os.path.join(subdir, file)
            print img_path, ext
            if ext in img_exts:
                img_paths.append(img_path)
else:
    for item in os.listdir(root_dir):
        if os.path.isfile(os.path.join(root_dir, item)):
            basename, ext = os.path.splitext(item)
            img_path = os.path.join(root_dir, item)
            print img_path, ext
            if ext in img_exts:
                img_paths.append(img_path)

print img_paths
pts_paths = []
for img_i in img_paths:
    filename, ext = os.path.splitext(img_i)
    pts_i = filename + '.pts'
    pts_paths.append(filename + '.pts')

def get_file_name(fullpath):
    path_name, filename = os.path.split(fullpath)
    return filename

if args.output_file is None:
    output_filename = args.directory + '/settings.txt'
else:
    output_filename = args.output_file

with open(output_filename, 'w') as f:
    [f.write(get_file_name(pi) + ' ' + get_file_name(pp) + '\n') for pi, pp in zip(img_paths, pts_paths)]
