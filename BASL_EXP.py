"""# Imports and Stuff..."""
from pathlib import Path

"""###################..."""

from utils.train_and_validation.sl import sl_training_procedure

num_experiments = 2
trig_size = 4
trig_pos = 'upper-right'
dataset = 'cifar10'
trig_shape = 'square'
trig_samples = 100
bd_label = 7
arch_name = 'resnet9'
bd_opacity = 1.0
base_path = Path()
exp_num = 1
tp_name = 'split_training'
cut_layer = 1
batch_size = 128

sl_training_procedure(tp_name=tp_name, dataset=dataset, arch_name=arch_name, cut_layer=cut_layer,
                                base_path=base_path, exp_num=exp_num, batch_size=batch_size)
