# Reduced-order Modeling via Operator Inference for 2D Combustion

This repository is an extensive example of the non-intrusive, data-driven Operator Inference procedure for reduced-order modeling applied to two-dimensional combustion data.
It is the source code for the upcoming paper [\[1\]](#references) and can be used to reproduce the results of [\[2\]](#references) (see [**References**](#references)).
The dataset [\[3\]](#references) was used in [\[1,2\]](#references) to learn reduced-order models via Operator Inference.

See the following files for details.

- [`docs/PROBLEM.md`](./docs/PROBLEM.md): Summary of the problem statement including the computational domain, state variables, etc.
- [`docs/DOCUMENTATION.md`](./docs/DOCUMENTATION.md): How to use this code repository.
- [`docs/REPORT.md`](./docs/REPORT.md): Results from the upcoming publication, including many additional figures that are not in the text.
- [`[4]/DETAILS.md`](https://github.com/Willcox-Research-Group/rom-operator-inference-Python3/blob/master/DETAILS.md): Mathematical summary of Operator Inference.

---

**Contributors**:
[Shane A. McQuarrie](https://github.com/shanemcq18),
[Renee Swischuk](https://github.com/swischuk),
Parikshit Jain,
[Boris Kramer](http://kramer.ucsd.edu/),
[Karen Willcox](https://kiwi.oden.utexas.edu/)

## References

- \[1\] [McQuarrie, S.](https://scholar.google.com/citations?user=qQ6JDJ4AAAAJ), [Huang, C.](https://scholar.google.com/citations?user=lUXijaQAAAAJ), and [Willcox, K.](https://kiwi.oden.utexas.edu/), _Data-driven reduced-order models via regularized operator inference for a single-injector combustion process_, to appear.

- \[2\] [Swischuk, R.](https://scholar.google.com/citations?user=L9D0LBsAAAAJ), [Kramer, B.](http://kramer.ucsd.edu/), [Huang, C.](https://scholar.google.com/citations?user=lUXijaQAAAAJ), and [Willcox, K.](https://kiwi.oden.utexas.edu/), [Learning physics-based reduced-order models for a single-injector combustion process](https://arc.aiaa.org/doi/10.2514/1.J058943). _AIAA Journal_, Vol. 58:6, pp. 2658-2672, 2020. Also in Proceedings of 2020 AIAA SciTech Forum & Exhibition, Orlando FL, January, 2020. Also Oden Institute Report 19-13.
([Download](https://kiwi.oden.utexas.edu/papers/learning-reduced-model-combustion-Swischuk-Kramer-Huang-Willcox.pdf))<details><summary>BibTeX</summary><pre>
@article{SKHW2020ROMCombustion,
    title     = {Learning physics-based reduced-order models for a single-injector combustion process},
    author    = {Swischuk, R. and Kramer, B. and Huang, C. and Willcox, K.},
    journal   = {AIAA Journal},
    volume    = {58},
    number    = {6},
    pages     = {2658--2672},
    year      = {2020},
    publisher = {American Institute of Aeronautics and Astronautics}
}</pre></details>

- \[3\] [Huang, C.](https://scholar.google.com/citations?user=lUXijaQAAAAJ) (2020). 2D Benchmark Reacting Flow Dataset for Reduced Order Modeling Exploration \[Data set\]. University of Michigan - Deep Blue. [https://doi.org/10.7302/jrdr-bj37](https://doi.org/10.7302/jrdr-bj37).

- \[4\] [ROM Operator Inference Python 3 package](https://github.com/Willcox-Research-Group/rom-operator-inference-Python3) ([pypi](https://pypi.org/project/rom-operator-inference/))