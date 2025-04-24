import os


os.system("stubgen -p cuda --include-docstrings --doc-dir cuda-python/cuda_bindings/docs/ --search-path cuda-python/cuda_bindings")
