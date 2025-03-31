import random
import torch
import numpy as np


class ImagePool():
    """
    To make the discriminator learn the distribution
    """
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []

    def query(self, images, batch_size=None):
        if self.pool_size == 0:
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    # change image pool [random_id] to image, return old image to be replaced
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:
                    # image pool is not changed, return image
                    return_images.append(image)

        if batch_size is not None:
            cur_len = len(return_images)
            indices = np.random.choice(cur_len, batch_size, replace=False)
            return_images = [item for idx,item in enumerate(return_images) if idx in indices]
        return_images = torch.cat(return_images, 0)
        return return_images
