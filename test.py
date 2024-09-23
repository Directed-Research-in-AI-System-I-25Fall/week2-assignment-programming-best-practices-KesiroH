import torch
from torchvision import datasets, transforms
from torchvision.models import resnet50
from torchvision.models.resnet import ResNet50_Weights

# 设置设备为 GPU（如果可用），否则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载预训练的 ResNet 模型
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
# 修改最后一层以适应 MNIST 数据集（10 类）
num_features = model.fc.in_features
model.fc = torch.nn.Linear(num_features, 10)
model.eval()  # 设置为评估模式
model.to(device)

# 准备 MNIST 数据集
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 将图像调整为 224x224 像素
    transforms.Grayscale(num_output_channels=3),  # 将灰度图像转换为 RGB
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 归一化
])

# 下载 MNIST 数据集
mnist_data = datasets.MNIST(root='data', train=False, transform=transform, download=True)
data_loader = torch.utils.data.DataLoader(mnist_data, batch_size=32, shuffle=False)

# 运行推理并计算准确率
correct = 0
total = 0

with torch.no_grad():  # 关闭梯度计算以节省内存
    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        
        # 进行推理
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)  # 获取预测类别
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()  # 计算正确的预测数量

# 计算准确
accuracy = correct / total
print(f'准确率: {accuracy:.2f}')