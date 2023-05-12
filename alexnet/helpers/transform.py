from torchvision import transforms
from PIL import ImageOps

# กำหนดขนาดรูป
desired_image_size = 192

def custom_padding(image):
    width, height = image.size

    longest_dim = max(width, height)
    
    padding_width = longest_dim - width
    padding_height = longest_dim - height

    left_padding = padding_width // 2
    right_padding = padding_width - left_padding
    top_padding = padding_height // 2
    bottom_padding = padding_height - top_padding

    padded_image = ImageOps.expand(image, (left_padding, top_padding, right_padding, bottom_padding))

    return padded_image

transform = transforms.Compose([
    custom_padding,
    transforms.Resize((desired_image_size, desired_image_size)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])