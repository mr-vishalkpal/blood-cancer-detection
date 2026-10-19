import torch, torchvision, cv2, sklearn, streamlit, seaborn, pytorch_grad_cam
print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("GPU available:", torch.cuda.is_available())
print("All good!")