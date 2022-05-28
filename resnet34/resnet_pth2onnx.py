import torch
import torch.onnx
# from benchmark import Benchmark,IterativeBenchmark
from model_resnet import ResNet
import os

def pth_to_onnx(input, checkpoint, onnx_path, input_names=['input1'], output_names=['output'], device='cpu'):
    if not onnx_path.endswith('.onnx'):
        print('Warning! The onnx model name is not correct,\
              please give a name that ends with \'.onnx\'!')
        return 0

    model = ResNet()
    model.load_state_dict(torch.load("./model_resnet.pth").cpu().state_dict()) #初始化权重
    #network.load_state_dict(torch.load(model_name).cpu().state_dict())

    model.eval()
    print(model)
    # model.to(device)
    
    # torch.onnx.export(model, input, onnx_path, verbose=True, input_names=input_names, output_names=output_names) #指定模型的输入，以及onnx的输出路径
    torch.onnx.export(model, input, onnx_path, verbose=True, input_names=input_names, output_names=output_names) #指定模型的输入，以及onnx的输出路径
    # print("Exporting .pth model to onnx model has been successful!")

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    # x, y = torch.randn(4, 3, 5), torch.randn(4, 3, 5)
    checkpoint = './resnet34.pth'
    onnx_path = './resnet34.onnx'
    input=torch.randn(1,3,224,224)
    # device = torch.device("cuda:2" if torch.cuda.is_available() else 'cpu')
    pth_to_onnx(input, checkpoint, onnx_path)