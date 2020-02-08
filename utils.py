import copy
import cv2
import numpy as np
import torch
from torchvision import models


def preprocess_image(img, resize_im=True):
    """
        Preprocesse image as ImageNet dataset.
    Args:
        img (PIL_img): cv2 Image to process
        resize_im (bool): Resize to 224 or not
    returns:
        img (Pytorch tensor): tensor that contains processed float tensor
    """

    # mean and std list for channels
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    # Resize image
    if resize_im:
        img = cv2.resize(img, (224, 224))
    img = np.float32(img)
    # convert BGR to RGB
    img = np.ascontiguousarray(img[..., ::-1])
    img = img.transpose(2, 0, 1)  # Convert array to D,W,H
    # Normalize the channels
    for channel, _ in enumerate(img):
        img[channel] /= 255
        img[channel] -= mean[channel]
        img[channel] /= std[channel]
    # Convert to float tensor
    img = torch.from_numpy(img).float()
    # Add one more channel to the beginning. Tensor shape = 1,3,224,224
    img.unsqueeze_(0)
    # print(im_as_ten.requires_grad)
    img.requires_grad = True
    return img


def recreate_image(im_as_var):
    """
        Recreates images from a torch tensor, sort of reverse preprocessing
    Args:
        im_as_var (torch variable): Image to recreate
    returns:
        recreated_im (numpy arr): Recreated image in array
    """
    reverse_mean = [-0.485, -0.456, -0.406]
    reverse_std = [1/0.229, 1/0.224, 1/0.225]
    recreated_im = copy.copy(im_as_var.data.numpy()[0])
    for c in range(3):
        recreated_im[c] /= reverse_std[c]
        recreated_im[c] -= reverse_mean[c]
    recreated_im[recreated_im > 1] = 1
    recreated_im[recreated_im < 0] = 0
    recreated_im = np.round(recreated_im * 255)

    recreated_im = np.uint8(recreated_im).transpose(1, 2, 0)
    # Convert RBG to GBR
    recreated_im = recreated_im[..., ::-1]
    return recreated_im


def get_params(example_index):
    """
        Gets used variables for almost all visualizations, like the image, model etc.
    Args:
        example_index (int): Image id to use from examples
    returns:
        original_image (numpy arr): Original image read from the file
        prep_img (numpy_arr): Processed image
        target_class (int): Target class for the image
        file_name_to_export (string): File name to export the visualizations
        pretrained_model(Pytorch model): Model to use for the operations
    """
    # Pick one of the examples
    example_list = [['./images/apple.JPEG', 948],
                    ['./images/eel.JPEG', 390],
                    ['./images/bird.JPEG', 13]]
    selected_example = example_index
    img_path = example_list[selected_example][0]
    target_class = example_list[selected_example][1]
    file_name_to_export = img_path[img_path.rfind('/')+1:img_path.rfind('.')]
    # Read image
    original_image = cv2.imread(img_path, 1)
    # Process image
    prep_img = preprocess_image(original_image)
    # Define model
    pretrained_model = models.alexnet(pretrained=True)
    return (original_image,
            prep_img,
            target_class,
            file_name_to_export,
            pretrained_model)