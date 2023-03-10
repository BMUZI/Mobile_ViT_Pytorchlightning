# Import basic pkg
from torchvision import transforms

# Transform
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

transform_cifar100_train = transforms.Compose([
            transforms.RandomRotation(degrees=(-10, 10)),
            transforms.Resize((32, 32)),
            transforms.RandomCrop((32, 32), padding=1),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])

transform_cifar100_val = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            normalize])

transform_imagenet_train = transforms.Compose([
            transforms.RandomRotation(degrees=(-10, 10)),
            transforms.Resize((256, 256)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])

transform_imagenet_val = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            normalize])