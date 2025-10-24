# BSP-AF-S5

## Open Questions
- How do you decide how many neurons per layer and how many layers?

## Refined BSP Goals
### Refined Scientific Question
- How well can **compact** neural implicit representations (NIRs) trained on a global borders dataset accurately approximate the spherical distance‐to‐border field (and jointly predict the containing country and the adjacent country at the nearest border)? Can they achieve this at **interactive shader speeds**, within **an accepted bounded error**, and while capturing **high-frequency geometry**? Which architecture maximizes speed and accuracy under these constraints?

- Compact: All parameters ≤ 8 MB.
- Interactive shader speeds: On a 1080p screen, there are 2073600 (1920x1080) pixels. At 60fps, it means our NN needs to accomplish at most 124M querries per second when fully parallelized on a GPU.
- Accepted bounded error: 
  - Global Median distance ≤ 10 km
  - Within 25km of any border: ≤ 4 km
  - Country classification ≥ 99% accurate
- High-frequency geometry: Border areas with high curvature (costal fjords, exclaves, small countries).

#### Expected Scientific Deliverables
- Paper explaining architectures, methodology, decisions, etc
- Graph comparing different architectures on speed in one axis, accuracy on the other.

### Expected Technical Deliverables
- "Naive" Solver: A Python function which parses the borders dataset using geopandas, and returns a point-to-nearest segment distance with full accuracy, as well as the proper country labels (this is the baseline with whom all models will be compared to).
    - p = (x,y,z) -> (distance, country1, country2), where |p|=1
- Evaluator: Program that evaluates the speed and accuracy of a model.
- Demo of Neural Field Models: Trained using PyTorch, all trained using the same set of sample points, but each having different architectures (different neurons per layer and layer count, different activation functions, some use positional encoding, etc). The speed and accuracy metrics are measured using a different set of sample points from the same dataset.
  - Sampling policy: Weighted such that the points are more likely to be closer to the borders.

> Is the topic/goals sufficient?

## Decisions
- Dataset: World Bank International Borders
- PyTorch

## Assumptions
- Static Data
- Single GPU
- Earth is a perfect sphere. Distances are calculated using geodesic distance.
- The entire globe is fully partioned (the oceans count as their own country).
- Borders are polylines; each edge is the minor great-circle arc between consecutive vertices.
- "Nearest border segment" means the point on any arc that minimizes geodesic distance to the query point.


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
#### Contents
- 

### https://github.com/vsitzmann/awesome-implicit-representations
#### Contents
- 

Adreyy Carpati

## Notes

AFTER: think about self evolving architectures: autosmth


Check out this paper:
https://proceedings.neurips.cc/paper_files/paper/2022/file/1165af8b913fb836c6280b42d6e0084f-Paper-Conference.pdf


Notes: stopped at France (reunion) in color re-assignment.