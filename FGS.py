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
        # updating rate
        self.alpha = alpha
        if not os.path.exists('./generated'):
            os.makedirs('./generated')

    def generate(self, original_image, im_label):
        im_label_as_var = torch.from_numpy(np.asarray([im_label]))
        criterion = nn.CrossEntropyLoss()
        # process image as ImageNet dataset format
        processed_image = preprocess_image(original_image)

        for i in range(10):
            print('Iteration: {}'.format(str(i)))
            # zero previous gradient
            processed_image.grad = None

            out = self.model(processed_image)
            pred_loss = criterion(out, im_label_as_var)
            # calculate gradient
            pred_loss.backward()

            # create noise
            # processed_image.grad.data represents gradient of first layer
            adv_noise = self.alpha * torch.sign(processed_image.grad.data)
            # add noise to image
            processed_image.data = processed_image.data + adv_noise

            # generate confirmation image
            recreated_image = recreate_image(processed_image)
            # process confirmation image
            prep_confirmation_image = preprocess_image(recreated_image)

            pred = self.model(prep_confirmation_image)
            # get prediction index
            _, pred_index = pred.data.max(1)
            confidence = F.softmax(pred, dim=1)[0][pred_index].data.numpy()[0]
            # convert tensor to int
            pred_index = pred_index.numpy()[0]

            if pred_index != im_label:
                print('\nOriginal image class: ', im_label,
                      '\nConverted image class: ', pred_index,
                      '\nConfidence: ', confidence)

                # create the image for noise as: Original image - generated image
                noise_image = original_image - recreated_image
                cv2.imwrite('./generated/untargeted_adv_noise_from_' + str(im_label) + '_to_' +
                            str(pred_index) + '.jpg', noise_image)
                # write image
                cv2.imwrite('./generated/untargeted_adv_img_from_' + str(im_label) + '_to_' +
                            str(pred_index) + '.jpg', recreated_image)
                break

        return 1


if __name__ == '__main__':
    target_example = 2  # choose image to attack
    (original_image, prep_img, target_class, _, pretrained_model) =\
        get_params(target_example)
    FGS_untargeted = FastGradientSignUntargeted(pretrained_model, 0.002)
    FGS_untargeted.generate(original_image, target_class)