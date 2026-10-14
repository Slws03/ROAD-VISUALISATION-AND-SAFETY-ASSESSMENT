"""Empty stand-in for albumentations, importable only from dump.py's sys.path.

CLRerNet's training augmentation (libs/datasets/pipelines/alaug.py) imports
albumentations at module level, and the config imports that module
unconditionally. Inference here never builds a pipeline, so nothing is ever
looked up on this module. Colab ships the real package, which is why the
notebook needs no equivalent; installing it locally would pull in
opencv-python-headless beside the pinned opencv-python.
"""
