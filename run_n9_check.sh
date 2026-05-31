#!/bin/bash
cd /home/ubuntu/EDA/external/FloorSet/iccad2026contest || exit 1
PYTHONPATH=.. /usr/bin/python3 iccad2026_evaluate.py --evaluate my_optimizer.py --test-id 0 --test-id 50 --test-id 99 --verbose 2>&1
echo "EXIT=$?"
