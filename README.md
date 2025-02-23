# FOODPUZZLE: Developing Large Language Model Agents as Flavor Scientists

**Authors:** Tenghao Huang¹, Donghee Lee²*, John Sweeney*, Jiatong Shi, Emily Steliotes², Matthew Lange², Jonathan May¹, Muhao Chen²  
**Affiliations:**  
¹ University of Southern California  
² University of California, Davis  

[Paper](https://arxiv.org/pdf/2409.12832) 

---

# FOODPUZZLE: LLM Agents as Flavor Scientists

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

**FOODPUZZLE** is an open-source project that introduces a novel benchmark and Scientific Agent for flavor science. The repository contains code, data, and examples demonstrating how to leverage Large Language Models (LLMs) for flavor profile prediction and completion tasks.

---

## Overview

Flavor development in the food industry often relies on labor-intensive, subjective testing. **FOODPUZZLE** addresses these challenges by:
- Defining a new problem domain for LLM agents in flavor science.
- Providing a benchmark dataset with **978 food items** and **1,766 flavor molecules**.
- Implementing a Scientific Agent that integrates in-context learning and retrieval-augmented techniques for generating evidence-based flavor hypotheses.

---

## Repository Structure

```plaintext
.
├── data/
│   ├── raw/                 # Original dataset files
│   ├── processed/           # Processed data ready for analysis
    └── collected_evidences/ # Collected evidences (via Google Search API) for food items and chemicals
├── code/
│   ├── scientific_agent/    # Source code of the Scientific Agent pipeline
│   └── evaluation.py/       # Evaluation Script for MFP and MPC tasks
├── README.md                # This file
└── LICENSE                  # License file
```

## Citation

```bibtex
@misc{huang2024foodpuzzle,
  title  = {FOODPUZZLE: Developing Large Language Model Agents as Flavor Scientists},
  author = {Huang, Tenghao and Lee, Donghee and Sweeney, John and Shi, Jiatong 
            and Steliotes, Emily and Lange, Matthew and May, Jonathan and Chen, Muhao},
  howpublished = {arXiv preprint arXiv:2409.12832v3},
  year   = {2024}
}
