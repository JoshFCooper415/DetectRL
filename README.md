# DetectEval
This is the repo for data and code for submission paper "DetectEval: Benchmarking LLM-Generated Text Detection in Real-World Scenarios"

---

## Data and Experimental Reproduction

### Data Loading and Processing
```bash
# loading original dataset and sampling
sh load_dataset.sh
```

### Data Generation and Benchmark Construction
```bash
# data generation
sh data_generation.sh

# benchmark construction
sh benchmark_construction.sh
```


## Benchmark Evaluation
```bash
# Task1 and Task2 evaluation
sh domains_evaluation.sh
sh llms_evaluation.sh
sh attacks_evaluation.sh

# Task3 evaluation
sh varying_length_evaluation.sh

# Task4 evaluation
sh human_writing_evaluation.sh
```
