import argparse
import subprocess
from pathlib import Path

# GPU Architecture Code for Titan RTX
ARCH = 'sm_75'

def main():
    parser = argparse.ArgumentParser(description='Compile CUDA functions for the current machine\'s GPU')
    parser.add_argument('-a', '--arch', action='store',
            required=True, help='The architecture code of this machine\'s GPU')
    args = parser.parse_args()
    path = Path('./cuda_functions')
    for nms in path.glob('nms_*D'):
        subprocess.run('nvcc -c -o nms_kernel.cu.o nms_kernel.cu -x cu -Xcompiler -fPIC -arch={}'.format(args.arch),
                cwd=nms/'src/cuda', shell=True)
        subprocess.run('python build.py', cwd=nms, shell=True)
    for roi in path.glob('roi_align_.*D'):
        subprocess.run(('nvcc -c -o crop_and_resize_kernel.cu.o crop_and_resize_kernel.cu -x cu -Xcompiler -fPIC -arch={}').format(args.arch),
                cwd=roi/'src/cuda', shell=True)
        subprocess.run('python build.py', cwd=roi, shell=True)

if __name__ == '__main__':
    main()
