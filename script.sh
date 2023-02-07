#!/bin/bash

# run from folder top
set PYTHONPATH="."

for (( k = 1; k < 20; ++k ));
  do
    iteration=$((5*$k))
    python examples/mnist/mnist_example.py --epochs 1 --poison_ratio $iteration

  done
