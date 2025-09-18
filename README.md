# BSP-AF-S5

## Assumptions
- Static Data
- Single GPU

## Research Papers
### The Case for Learned Index Structures
#### BibTex
@inproceedings{46518,title	= {The Case for Learned Index Structures},author	= {Tim Kraska and Alex Beutel and Ed H. Chi and Jeff Dean and Neoklis Polyzotis},year	= {2018},URL	= {https://arxiv.org/abs/1712.01208}}
#### Contents
- Traditional Indexes: B-Tree, Hash-map, Bloom Filter
- Traditional Indexes "assume nothing about the data distribution and do not take advantage of more common patterns prevalent in real world data."
- Paper argues that ML takes advantage of these patterns with low engineering effort.
- Replacing branch heavy data structures with ML to make most of GPUs.
- Traditional Indexes can be decomposed into a NN and an auxiliary data structure, with the same semantic guarantees while being more performant EVEN ON THE CPU.
- Read up until Section 2

### An Introduction to Neural Implicit Representations with Use Cases: 
- https://medium.com/@nathaliemariehager/an-introduction-to-neural-implicit-representations-with-use-cases-ad331ca12907
- NIRs are NNs that estimate a continuous signal, by training on discretely represented samples of the same signal.
- The storage required for such a representation scales with the complexity of the signal; it is independent of spatial resolution.

### Where Do We Stand with NIRs?
- https://arxiv.org/abs/2411.03688

### https://github.com/vsitzmann/awesome-implicit-representations

## Frameworks
- PyTorch
- JAX

