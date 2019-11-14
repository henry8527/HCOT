# Hierarchical Complement Objective Training


## Overview

This codebase focus on "Explicitly Hierarchy" implemented in "Hierarchical Complement Objective Training" for ICLR'2020 anonymized code submission. Since the codebase of "Latent Hierarchy" involved many details of Semantic Segmentation, We will summarize it as soon as possible and open another repo to present.



## Requires

* Python 3.6 
* Pytorch 1.2.0
* keras 
* tensorflow
* numpy 1.16.4


## Usage
For getting baseline results
	
	python main.py --sess Baseline_session
	
For training via Complement objective

	python main.py --COT --sess COT_session
	
For training via Hierarchical Complement Entropy (HCE)

	python main.py --HCOT --sess HCOT_session


## Our Benchmark on CIFAR100

The following table shows the test error rates in a 200-epoch training session. (Please refer to "Table 1" in the paper for details.)

| Model              | Baseline  | COT | HCOT |
|:-------------------|:---------------------|:---------------------|:---------------------|
| PreAct ResNet-18                |               25.44%  |               24.73%  | **23.8%** |

