# Mithridates -- Measure and Boost Robustness to Backdoor Poisoning

*This is work in progress: email 
me: 
eugene@cs.cornell.edu and contribute!*
<p align="center">
<img src="images/mithridates.jpg"  width="300" height="300">
</p>

## Background

Data poisoning is an emerging threat to pipelines that rely on data gathered 
from multiple sources [[1]](https://arxiv.org/abs/2204.05986). A subset of 
poisoning -- backdoor attacks are 
extremely powerful and can inject a hidden behavior to the model by only 
using a small fraction of inputs. 
However, as backdoor attacks are diverse and defenses are expensive 
to integrate and maintain, there exists a problem for practical resolution.

In this repo we take a **developer-centric** view of this problem and 
focus on two questions:
1. But how robust is your model to backdoor poisoning?
2. How to boost this robustness without modifying your model training. 

**Main goal**: Provide an easy-to-use tool that does 
not require  modification of the training pipeline but can help 
measure and mitigate backdoor threats of this pipeline.

## Installation

```bash
pip install -r requirements.txt
```

 !!**TODO**: make a pip installation.


## Measure robustness of an existing pipeline

To measure robustness we test how well can the model learn the **primitive 
sub-task** -- a simple task that covers large portion of the input and is 
easier to learn than other backdoors providing a strong baseline:

Our main metric is the fraction of the dataset required to compromise in 
order to make the backdoor effective. Engineers can then apply quotas or add 
more trusted data to reduce the threat. We build a poisoning curve that 
tracks how accuracy of the primitive sub-task (backdoor accuracy) changes 
while increasing training dataset poisoning ratio. 

However, it's not evident what how to measure backdoor effectiveness as even 
the backdoor might be effective even without reaching 100% accuracy. We 
propose **inflection point** of the poisoning curve as an effective metric that 
signifies a slowdown in increase of backdoor accuracy with higher compromised 
percentage, i.e. injecting would become more expensive for the attacker 
afterwards. Please see the paper for more discussion.

<p align="center">
<img src="images/image_examples.png"  width="600" >
</p>

There are three steps to measure robustness:
1. Integrate wrapper that poisons the training dataset
2. Iterate over poisoning ratio
3. Build poisoning curve and compute inflection point
 

### MNIST Example

Now we can demonstrate on MNIST PyTorch training example 
([mnist_example.py](mnist_example.py)):

#### 1. Integrate wrapper: 

```python

from mithridates import DatasetWrapper

train_dataset = YOUR_DATASET_LOAD()
train_dataset = DatasetWrapper(train_dataset, percentage_or_count=POISON_RATIO)

test_dataset = YOUR_TEST_DATASET_LOAD()
test_attack_dataset = DatasetWrapper(test_dataset, percentage_or_count='ALL')

...
# TRAIN
...
test(test_dataset)
test(test_attack_dataset)

```

See [mnist_example.py#L104](mnist_example.py#L104) for sample integration of 
the wrapper into PyTorch MNIST training example. We added `--poison_ratio` 
to the arguments that can be called in bash

#### 2. Iterate over poisoning ratio

We can use a Bash (slow) or 
[Ray Tune](https://docs.ray.io/en/latest/tune/index.html) 
(fast, uses multiple process and GPUs)
to iterate over training.


Slow: run **Bash** [script.sh](script.sh) `./script.sh`:

```bash
set PYTHONPATH="."

for (( k = 1; k < 20; ++k ));
  do
    iteration=$((20*$k))
    echo "Running poisoning of $iteration images from MNIST dataset."
    python mnist_example.py --epochs 1 --poison_ratio $iteration

  done
```
``
The result will be saved to `/tmp/results.txt` (poison images, main accuracy,
backdoor accuracy):
```text
0.033 98.11000 11.15000
0.067 98.03000 11.35000
0.100 98.07000 13.30000
0.133 98.48000 66.28000
0.167 98.07000 49.02000
0.200 98.43000 17.78000
0.233 98.31000 67.98000
0.267 98.30000 95.61000
0.300 98.36000 82.53000
0.333 98.54000 95.04000
0.367 98.53000 86.56000
0.400 98.28000 92.34000
0.433 97.59000 98.62000
0.467 98.49000 91.68000
0.500 98.36000 96.52000
0.533 98.46000 97.20000
0.567 98.21000 95.89000
0.600 98.45000 97.14000
0.633 98.39000 97.48000
```

Fast: **Ray Tune:** run `python ray_training.py`:

or leverage Ray to iterate
```text
Number of trials: 40/40 (40 TERMINATED)
+----------------------+--------------+----------+----------+-------------------+
| Trial name           | poison_ratio | time (s) | accuracy | backdoor_accuracy |
|----------------------+--------------+----------+----------+-------------------|
| tune_run_70d8f_00000 |  9.02302e-05 |  72.2013 |    98.84 |             11.48 |
| tune_run_70d8f_00001 |  1.71321e-05 |  72.762  |    98.95 |             11.38 |
| tune_run_70d8f_00002 |  0.170843    |  73.9944 |    98.81 |            100    |
| tune_run_70d8f_00003 |  0.000833466 |  73.0966 |    98.76 |             30.7  |
| tune_run_70d8f_00004 |  0.000911572 |  71.7646 |    98.85 |             75.05 |
| tune_run_70d8f_00005 |  0.00148639  |  72.3795 |    98.9  |             90    |
| tune_run_70d8f_00006 |  0.000124546 |  69.3005 |    98.98 |             11.74 |
| tune_run_70d8f_00007 |  0.000284989 |  68.0602 |    98.99 |             11.9  |
| tune_run_70d8f_00008 |  0.029272    |  68.2165 |    98.98 |             99.92 |
| tune_run_70d8f_00009 |  0.0322237   |  69.4554 |    98.98 |            100    |
| tune_run_70d8f_00010 |  9.69244e-05 |  69.4923 |    98.91 |             11.36 |
| tune_run_70d8f_00011 |  0.00825765  |  69.4363 |    98.97 |             99.54 |
| tune_run_70d8f_00012 |  1.83789e-05 |  69.3122 |    99.05 |             11.35 |
.....
```

#### 3. Build a curve:

If used Ray you can use Jupyter Notebook and call 
`get_inflection_point(analysis)` from [utils.py](mithridates/utils.py), see 
[build_curve.ipynb](build_curve.ipynb).


<p align="center">
<img src="images/inflection.png"  width="600" >
</p>


Overall, this is an inexpensive way to measure robustness to backdoors.

# Boosting robustness with hyperparameter search

We can modify existing [ray_training.py](ray_training.py) and fix the 
poisoning ratio but add search over different hyperparameters and modify 
objective.

**TODO: will be added later.**




