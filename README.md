# Mithridates -- Measure and Boost Robustness to Backdoor Poisoning

*This is work in progress: email 
me: 
eugene@cs.cornell.edu, or contribute!*
<p align="center">
<img src="images/mithridates.jpg"  width="300" height="300">
</p>
# How To Measure and Boost Robustness to Backdoor Poisoning

Data poisoning is an emerging threat to pipelines that rely on data gathered 
from multiple sources [1](link). A subset of poisoning -- backdoor attacks are 
extremely powerful and can inject a hidden behavior to the model by only 
using a small fraction of inputs 
[2](link). 
However, as backdoor attacks are diverse and defenses are expensive 
to integrate and maintain, there exists a problem for practical resolution.

In this repo we take a **developer-centric** view of this problem and 
focus on two questions:
1. But how robust is your model to backdoor poisoning?
2. How to boost this robustness without modifying your model training. 

# Installation

 todo: make a pip installation


# Measure Robustness Of Your Own Pipeline

To measure robustness we test how well can the model learn the **primitive 
sub-task** -- a simple task that covers large portion of the input and is 
easier to learn than other backdoors providing a strong baseline:

<p align="center">
<img src="images/image_examples.png"  width="300" height="300">
</p>

General example: 

```python

from mithridates import DatasetWrapper

train_dataset = YOUR_DATASET_LOAD()
train_dataset = DatasetWrapper(train_dataset, percentage_or_count=0.001)

test_dataset = YOUR_TEST_DATASET_LOAD()
test_attack_dataset = DatasetWrapper(test_dataset, percentage_or_count='ALL')

...
# TRAIN
...
test(test_dataset)
test(test_attack_dataset)

```

# MNIST Example

See [mnist_example.py#L104](mnist_example.py#L104) for sample integration of 
the wrapper into PyTorch MNIST training example. We added `--poison_ratio` 
to the arguments that can be called in bash
