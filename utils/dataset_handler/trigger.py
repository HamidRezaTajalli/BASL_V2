import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
import torch




def poison(img, trigger_obj, trig_ds, bd_opacity):
    """Poison the training samples by stamping the trigger."""
    # trigger = trigger_obj.crafted_trigger
    # beta = bd_opacity
    # sample = cv2.addWeighted(img, 1.0, trigger, beta, 0)
    # # sample = np.where(trigger==0, img, trigger)
    # return_value = (sample.reshape(32, 32, 3)) if trig_ds.lower() == 'cifar10' else (sample.reshape(28, 28, 1))
    return_value = trigger_obj.trigger_square_img(img)
    return return_value

class TriggerInfeasible(Exception):
    """Exception raised when wrong dimensions of the trigger were given"""

    def __init__(self, size, pos, correct, shape):
        self.size = size
        self.pos = pos
        self.correct = correct
        self.shape = shape
        m = (f"Cannot apply {self.shape}-trigger with size {self.size} at "
             f"{self.pos} to image with size {correct}")
        self.message = m
        super().__init__(self.message)

    def __str__(self):
        return f"{self.message}"


class Dimensions:
    """A class that keeps the dimensiosn of the used datasets."""

    # Use this dictionary to check on the validity of the position of the
    # trigger and its size.
    datasets = {"mnist": (28, 28, 1),
                "cifar10": (32, 32, 3),
                "fmnist": (28, 28, 1)}

    def __init__(self, dataset):
        if dataset in self.datasets:
            self.dims = self.datasets[dataset]
        else:
            raise NotImplementedError(f"{dataset} dataset is not known")

    def get_dims(self):
        return self.dims


class GenerateTrigger:
    """
    A class that creates a random pattern that is used as a trigger for an
    image dataset.
    """

    def __init__(self, size, pos_label, dataset, continuous=True,
                 shape="square"):
        # Use a hardcoded seed for reproducibility
        np.random.seed(56)
        dims = Dimensions(dataset).get_dims()

        if size[0] != size[1]:
            raise TriggerInfeasible(size, pos_label, dims, shape)

        if pos_label not in ["upper-left", "upper-mid", "upper-right", "mid-left", "mid-mid", "mid-right", "lower-left",
                             "lower-mid", "lower-right"]:
            raise TriggerInfeasible(size, pos_label, dims, shape)

        if shape not in ["square", "line"]:
            raise TriggerInfeasible(size, pos_label, dims, shape)

        if shape == "square":
            if size[0] > dims[0] or size[1] > dims[1]:
                raise TriggerInfeasible(size, pos_label, dims, shape)
        elif shape == "line":
            if (size[0] * size[1]) > (dims[0] * dims[1]):
                raise TriggerInfeasible(size, pos_label, dims, shape)

        self.dims = dims
        self.shape = shape
        self.size = size
        self.pos_label = pos_label
        self.pos = self.gen_pos(pos_label)
        self.continuous = continuous
        self.crafted_trigger = self.trigger()

    def _gen_pos_square(self):
        """Find the position of the upper left corner of a square trigger."""
        if self.pos_label == "upper-left":
            return (0, 0)
        elif self.pos_label == "upper-mid":
            return (0, self.dims[1] // 2 - self.size[1] // 2)
        elif self.pos_label == "upper-right":
            return (0, self.dims[1] - self.size[1])

        elif self.pos_label == "mid-left":
            return (self.dims[0] // 2 - self.size[0] // 2, 0)
        elif self.pos_label == "mid-mid":
            return (self.dims[0] // 2 - self.size[0] // 2,
                    self.dims[1] // 2 - self.size[1] // 2)
        elif self.pos_label == "mid-right":
            return (self.dims[0] // 2 - self.size[0] // 2, self.dims[1] - self.size[1])

        elif self.pos_label == "lower-left":
            return (self.dims[0] - self.size[0], 0)
        elif self.pos_label == "lower-mid":
            return (self.dims[0] - self.size[0], self.dims[1] // 2 - self.size[1] // 2)
        elif self.pos_label == "lower-right":
            return (self.dims[0] - self.size[0], self.dims[1] - self.size[1])

    def _gen_pos_line(self):
        """Find the position of the upper left pixel for the line trigger."""
        num_lines = math.ceil(self.pixels / self.dims[0])
        if self.pos_label == "upper-left":
            # Always start from the beginning
            return (0, 0)
        elif self.pos_label == "mid":
            dataset_mid = self.dims[0] / 2
            if dataset_mid % 2 == 0:
                x = dataset_mid - math.ceil(num_lines / 2)
            else:
                dataset_mid = math.floor(dataset_mid)
                x = dataset_mid - math.floor(num_lines / 2)
            return (int(x), 0)
        elif self.pos_label == "lower-right":
            return (self.dims[0] - num_lines, 0)

    def gen_pos(self, pos_label):
        """Create a tuple with the coordinates of the trigger."""
        if self.shape == "square":
            return self._gen_pos_square()
        elif self.shape == "line":
            self.pixels = self.size[0] * self.size[1]
            return self._gen_pos_line()

    def trigger_square_img(self, img):
        """Create a square trigger."""
        base_x = self.pos[0]
        base_y = self.pos[1]
        for x in range(self.size[0]):
            for y in range(self.size[1]):
                # random_value = np.random.random((self.dims[2]))
                img[base_x + x][base_y + y] = self.crafted_trigger[base_x + x][base_y + y]
        return img

    def trigger_square(self, trigger):
        """Create a square trigger."""
        if self.continuous:
            base_x = self.pos[0]
            base_y = self.pos[1]
            for x in range(self.size[0]):
                for y in range(self.size[1]):
                    trigger[base_x + x][base_y + y] = \
                        np.random.random((self.dims[2]))
        else:
            # The trigger that is used until now is a square so only one
            # dimension is enough for these calculations
            step_x = math.floor(self.dims[0] / (self.size[0] / 2))
            # to come up with the following equation I used the following
            # equation:
            # (dims[0]/step_x - 1)*x + size <= dims
            # where dims[0]/step_x - 1 is the amount of steps that are applied
            # for the broken square. I solved that for the equality and came up
            # with the following step_y.
            step_y = (self.dims[1] - self.size[0]) * step_x
            step_y = math.floor(step_y / (self.dims[1] - step_x))
            # The self pos variable is not used for now.
            curr_x = curr_y = 0
            count = 0

            while (curr_x < self.dims[0] and count < self.size[0]):
                i = curr_x % self.dims[0]
                for y in range(self.size[1]):
                    j = (y + curr_y) % self.dims[1]
                    trigger[i][j] = np.random.random((self.dims[2]))

                count += 1

                # Add step in the second iteration to create pairs of lines.
                if count % 2 == 0:
                    curr_x += step_x - 1
                    curr_y += step_y
                else:
                    curr_x += 1

        return trigger

    def trigger_line(self, trigger):
        """Create a line-shaped trigger."""
        base_x = self.pos[0]
        base_y = self.pos[1]

        pixels_filled = 0
        while pixels_filled < self.pixels:
            i = pixels_filled // self.dims[1]
            j = pixels_filled % self.dims[1]
            trigger[i + base_x][j + base_y] = np.random.random((self.dims[2]))
            pixels_filled += 1

        return trigger

    def trigger(self):
        """
        Returns a random pattern that is used as a trigger.

        If the trigger is continuous a square of pixels is filled with random
        values. In the opposite case the square trigger is broken to pairs of
        lines that are distributed in various positions in the image.
        """
        # For cifar I should return an ndarray with 32 32 3 dims with values
        # from 0 to 1 (dtype=float32)
        trigger = np.zeros(self.dims, dtype=np.float32)
        if self.shape == "square":
            trigger = self.trigger_square(trigger)
        elif self.shape == "line":
            trigger = self.trigger_line(trigger)

        self.trigger_img = trigger
        return trigger

    def show_trigger(self):
        """Show the trigger that was generated."""
        plt.imshow(self.trigger_img)
        plt.show()

    def save_trigger(self, filename):
        """Save the trigger that was generated."""
        img = self.trigger_img
        img = img * 255
        img = img.astype(np.int32)
        cv2.imwrite(filename, img)


def get_backdoor_train_dataset(dataset, trigger_obj, trig_ds, samples_percentage, backdoor_label, bd_opacity=1.0,
                    experiment_name='', img_samples_path=''):
    backdoored_ds = []
    samples_index = []
    insert = None
    trigger = trigger_obj.crafted_trigger
    trigger_samples = (samples_percentage * len(dataset)) // 100
    samples_index = np.random.choice(len(dataset), size=trigger_samples, replace=False)
    for ind, item in enumerate(dataset):
        if ind in samples_index:
            insert = (torch.from_numpy(
                poison(item[0].cpu().permute(1, 2, 0).numpy(), trigger_obj, trig_ds, bd_opacity)).permute(2, 0, 1),
                      backdoor_label)
        else:
            insert = item
        backdoored_ds.append(insert)


    fig, axes = plt.subplots(1, 3, figsize=(10, 10), constrained_layout=True)
    for ax in axes:
        ax.axis('off')

    if trig_ds == 'cifar10':
        clean_img = dataset[samples_index[0]][0].cpu().permute(1, 2, 0)
        backdoored_img = backdoored_ds[samples_index[0]][0].cpu().permute(1, 2, 0)
        trigger_show = trigger
    else:
        clean_img = dataset[samples_index[0]][0].cpu().permute(1, 2, 0)
        clean_img = np.reshape(clean_img, newshape=(clean_img.shape[0], clean_img.shape[1]))
        backdoored_img = backdoored_ds[samples_index[0]][0].cpu().permute(1, 2, 0)
        backdoored_img = np.reshape(backdoored_img, newshape=(backdoored_img.shape[0], backdoored_img.shape[1]))
        trigger_show = np.reshape(trigger, newshape=(trigger.shape[0], trigger.shape[1]))

    axes[0].imshow(clean_img)
    axes[1].imshow(trigger_show)
    axes[2].imshow(backdoored_img)
    # fig.savefig(f'{img_samples_path}/ImgSample_{experiment_name}.jpeg', dpi=500)

    return backdoored_ds


def get_backdoor_test_dataset(dataset, trigger_obj, trig_ds, backdoor_label, bd_opacity=1.0):
    backdoored_ds = []
    for ind, item in enumerate(dataset):
        if item[1] != backdoor_label:
            insert = (
            torch.from_numpy(poison(item[0].cpu().permute(1, 2, 0).numpy(), trigger_obj, trig_ds, bd_opacity)).permute(
                2, 0, 1), backdoor_label)
            backdoored_ds.append(insert)
    return backdoored_ds