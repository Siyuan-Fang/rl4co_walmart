# Application of rl4co in Walmart location problem

This project forked form [rl4co](https://github.com/ai4co/rl4co).

## My Changes

- Added a new environment, Walmart, which can be regarded as a more real-world version of the Facility Location Problem (FLP) environment. It features a more sophisticated reward function, action masking, and 730 nodes.

## Wal-Mart Problem

- Based on [Holmes (2011)](https://example.com/paper-url) and [Huang and Yu (2024)](https://example.com/paper-url), the Walmart store location problem is a real-world, dynamic, and economically rich variant of the combinatorial location choice problem. While it shares similarities with the Facility Location Problem in selecting sites and considering distance-related costs, the Walmart problem is more intricate due to its dynamic nature, complex reward function (including cannibalization and multiple cost structures), and specific empirical context.
- The Walmart store location problem involves selecting locations from 730 candidates to open regular stores or upgrade existing ones to supercenters (by adding food stores) to maximize profits. In this case, 90 locations are selected from 730 candidates. When calculating profits, the model considers sales cannibalization effects, location characteristics, and delivery costs.

## Learning Curve
<div align="center">
  <img src="https://github.com/Siyuan-Fang/rl4co_walmart/blob/main/image/reward_learning_curve.png?raw=true" alt="loss-curve" style="max-width: 60%;">
</div>

  For training, 700 blocks of latitude and longitude are randomly sampled as the training dataset, and 80 locations are chosen for each iteration. The reward was still increasing when I stopped the training, because the results already demonstrated good performance in this environment and renting an A100 GPU is too expensive. 😂
<div align="center">
  <img src="https://raw.githubusercontent.com/Siyuan-Fang/rl4co_walmart/refs/heads/main/image/loss_learning_curve.png" alt="loss-curve" style="max-width: 60%;">
</div>

The loss continues to decrease throughout training.



## Results

<div align="center">
  <img src="https://github.com/Siyuan-Fang/rl4co_walmart/blob/main/image/selected_location_map.jpeg?raw=true" alt="selected-locations-map" style="max-width: 90%;">
</div>

- The map above shows the distribution of selected locations in New York. Light gray dots represent the 730 Walmart location candidates, yellow dots indicate selected regular stores (selected once in the code), and food stores (supercenters, selected twice in the code). Using the sampling decoder, the best reward for selecting 90 locations out of 730 is 166.6746, with an average reward of 144.5972. These results demonstrate good generalization of the model.


- To view the map in detail, download ```walmart_location_mapv2.html``` from this repository.

## Replication Results
Clone this repository:
```bash
clone https://github.com/Siyuan-Fang/rl4co_walmart.git
```
Install the required packages:
```bash
pip install -e.
```
Train the model(may cost 15h to achieve the performance as I showed, because the high dimension of location candidates significantly slows the training speed):
```bash
python run_wal.py
```
Test the model in 730-nodes walmart environment.
```bash
python test.py
```

The original author of this project is [@ai4co](https://github.com/ai4co)。  
Original repository: [Original repository link](https://github.com/ai4co/rl4co).

---

> To view the original project or for more information, visit the original repository.
