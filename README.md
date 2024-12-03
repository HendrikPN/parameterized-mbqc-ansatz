# Parameterized MBQC Ansatz based on Clifford Quantum Cellular Automata

In this repository we investigate a parameterized quantum circuit model in 
tensorflow quantum model based on Clifford Quantum Cellular Automata (CQCA) and
motivated by Measurement-based Quantum Computation (MBQC).

The models and results are described in detail in the e-print at 
[arXiv:2312.13185](https://doi.org/10.48550/arXiv.2312.13185):

```
H. Poulsen Nautrup and Hans J. Briegel
Measurement-based Quantum Computation from Clifford Quantum Cellular Automata
arXiv:2312.13185 (2023).
```

# Results

The main results can be reproduced with `main.ipynb`.

## Stilted quantum dataset

We investigate how well three different models perform w.r.t. labelled datasets 
that are generated from each model. This is similar to the approach in
[Huang et al.](https://doi.org/10.1038/s41467-021-22539-9). 
Here we use the encoding scheme described in 
[Havlicek et al.](https://doi.org/10.1038/s41586-019-0980-2) 
followed by a parameterized circuit ansatz based on three different CQCA.

## Reproducting results

The results from the paper can be found in the folder `/results` which can be 
plotted in the section "Analysis" in `main.ipynb`. 

If you only want to reproduce the plots from the paper, run the code in the
section "Analysis" in  `main.ipynb`. If you want to train your own models,
run the code in the section "Results" in `main.ipynb`.
