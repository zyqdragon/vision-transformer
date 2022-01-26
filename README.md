# vision-transformer-from-scratch
This repository includes several kinds of vision transformers from scratch so that  one beginner can understand the theory of vision transformer easily. The basic transformer,the linformer transformer and the swin transformer are all trained and tested.

# Requirements:
- PyTorch (>= 1.6.0);
- Python 3.6.9;
- Numpy (1.18.2);
- OpenCV ;
- Linformer;
- vit_pytorch

# Train the model:
- python main_train.py;
- In the main_train.py the basic transformer and the linformer can be selected.

# Test the model:
- python test.py;
- In the main_train.py the basic transformer and the linformer can be selected.

# The theory of vision transformer can reference the following document:
- https://towardsdatascience.com/implementing-visualttransformer-in-pytorch-184f9f16f632;
- https://www.kaggle.com/hannes82/vision-transformer-trained-from-scratch-pytorch;

# Problem and solution:
- ‘tqdm_notebook‘ object has no attribute’sp‘的一种解决方法
- 在stackoverflow上找到了答案：是因为你禁用了进度条，之后还要调用它的原因。
- 解决：
- 在你的tqdm里加上disable=True
