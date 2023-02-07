#!/bin/bash

# run from folder top
set PYTHONPATH="."

for (( k = 1; k < 20; ++k ));
  do
    iteration=$((10*$k))
    echo "Running poisoning of $iteration images from MNIST dataset."
    python mnist_example.py --epochs 3 --poison_ratio $iteration

  done
