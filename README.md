# GAP-Diff: Geometry-Aligned Physics-constrained Diffusion for Sparse-View CT Reconstruction (GAP-Diff)

## Overview
This repository presents the implementation of the Geometry-Aligned Physics-constrained Diffusion model (GAP-Diff). This pioneering methodology aims to achieve high-quality volumetric computed tomography (CT) reconstruction from sparse-view plane X-ray images. By synergizing large-scale medical vision priors with radiographic physics, GAP-Diff effectively bridges the information gap between 2D manifold observations and 3D volumetric structures.



## Research Context
Computed Tomography (CT) is a cornerstone of modern diagnostics; however, traditional protocols frequently lead to substantial radiation exposure and prolonged scanning durations. This project endeavors to mitigate these concerns by reconstructing high-quality 3D volumes from sparse-view data. 

The GAP-Diff incorporates a Spatial Geometry Bridging (SGB) module and physical fidelity constraints. It showcases superior performance in voxel-level evaluation metrics, specifically Peak Signal-to-Noise Ratio (PSNR) and Structural Similarity Index Measure (SSIM), when compared to existing classical methods and state-of-the-art generative models.

## Features
- Integration of large-scale medical vision priors via a coordinate-aware multi-head attention mechanism.
- Enforcement of physical consistency through a differentiable Digitally Reconstructed Radiograph (DRR) operator.
- Comprehensive performance benchmarking against various state-of-the-art models including LDM and BX2S-Net.

Feel free to adjust any section to enhance clarity or to incorporate any additional information that may be pertinent to your specific context!
