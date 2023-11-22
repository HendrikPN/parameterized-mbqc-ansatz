# Parameterized MBQC Ansatz based on Clifford Quantum Cellular Automata

In this repository we investigate a parameterized quantum circuit model in 
tensorflow quantum model based on Clifford Quantum Cellular Automata (CQCA) and
motivated by Measurement-based Quantum Computation (MBQC).

The models and results are described in detail in the e-print at 
[arXiv:TBA](https://arxiv.org/):

```
H. Poulsen Nautrup and Hans J. Briegel
Measurement-based Quantum Computation from Clifford Quantum Cellular Automata
arXiv: TBA (2023)
```

## To-Do

+ More data for the 4-qubit, 4-layer example.
+ Consider a generative modeling example (?)
+ Consider a classical supervised learning example (?)
+ Add some details to README.


## Open Questions and Notes

+ The gradient for a PQC with the same number of parameters but many more layers is significantly smaller than that for a few layers. Why?
+ The `norm=True` argument creates a dataset that cannot be immediately reproduced by a PQC with the same parameters (but the impact on learning is not significant).
+ Generally, many of these models at a certain size seem to underperform drastically.

# Results

The main results can be reproduced with `main.ipynb`.

## Stilted quantum dataset

We investigate how well three different models perform w.r.t. labelled datasets 
that are generated from each model. This is similar to the approach in
[Huang et al.](https://doi.org/10.1038/s41467-021-22539-9). 
Here we use the encoding scheme described in 
[Havlicek et al.](https://doi.org/10.1038/s41586-019-0980-2) 
followed by a parameterized circuit ansatz based on three different CQCA.
