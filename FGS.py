import os
import numpy as np
import cv2

import torch
from torch import nn
import torch.nn.functional as F
from utils import preprocess_image, recreate_image, get_params


class FastGradientSignUntargeted():
    """
        Fast gradient sign untargeted adversarial attack, minimizes the initial class activation
        with iterative grad sign updates
    """
    def __init__(self, model, alpha):
        self.model = model
        self.model.eval()
        # Movement multiplier per iteration
        self.alpha = alpha
        # Create the folder to export images if not exists
        if not os.path.exists('./generated'):
            os.makedirs('./generated')

    def generate(self, original_image, im_label):
        # I honestly dont know a better way to create a variable with specific value
        im_label_as_var = torch.from_numpy(np.asarray([im_label]))
        # Define loss functions
        criterion = nn.CrossEntropyLoss()
        # Process image
        processed_image = preprocess_image(original_image)
        # Start iteration
        for i in range(10):
            print('Iteration:', str(i))
            # zero_gradients(x)
            # Zero out previous gradients
            processed_image.grad = None
            # Forward pass
            out = self.model(processed_image)
            # _, index = out.data.max(1)
            # print(F.softmax(out, dim=1)[0][index].data.numpy()[0])
            pred_loss = criterion(out, im_label_as_var)
            # Do backward pass
            pred_loss.backward()
            # Create Noise
            # Here, processed_image.grad.data is also the same thing is the backward gradient from
            # the first layer, can use that with hooks as well
            adv_noise = self.alpha * torch.sign(processed_image.grad.data)
            # Add Noise to processed image
            processed_image.data = processed_image.data + adv_noise

            # Confirming if the image is indeed adversarial with added noise
            # This is necessary (for some cases) because when we recreate image
            # the values become integers between 1 and 255 and sometimes the adversariality
            # is lost in the recreation process

            # Generate confirmation image
            recreated_image = recreate_image(processed_image)
            # Process confirmation image
            prep_confirmation_image = preprocess_image(recreated_image)
            # Forward pass
            pred = self.model(prep_confirmation_image)
            # Get prediction index
            _, pred_index = pred.data.max(1)
            # Get Probability
            confidence = F.softmax(pred, dim=1)[0][pred_index].data.numpy()[0]
            # Convert tensor to int
            pred_index = pred_index.numpy()[0]
            if pred_index != im_label:
                print('\nOriginal image class: ', im_label,
                      '\nConverted image class: ', pred_index,
                      '\nConfidence: ', confidence)
                # Create the image for noise as: Original image - generated image
                noise_image = original_image - recreated_image
                cv2.imwrite('./generated/untargeted_adv_noise_from_' + str(im_label) + '_to_' +
                            str(pred_index) + '.jpg', noise_image)
                # Write image
                cv2.imwrite('./generated/untargeted_adv_img_from_' + str(im_label) + '_to_' +
                            str(pred_index) + '.jpg', recreated_image)
                break

        return 1


if __name__ == '__main__':
    target_example = 2  # choose image to attack
    (original_image, prep_img, target_class, _, pretrained_model) =\
        get_params(target_example)
    FGS_untargeted = FastGradientSignUntargeted(pretrained_model, 0.003)
    FGS_untargeted.generate(original_image, target_class)