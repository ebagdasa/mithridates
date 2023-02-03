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


# Measure Robustness Of Your Own Pipeline
``
```python``

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