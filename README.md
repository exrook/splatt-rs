Gaussian splatting implementation using wgpu

![demo image](https://github.com/exrook/splatt-rs/blob/readme_images/bicycle.png?raw=true)

This implementation currently uses the following approach:

(0.) Load splat data and compute 3d covaraiance matricies and DC SH colors once for all splats

1. Split the screen into 16x16 tiles
   - for each tile, allocate space for an atomic splat count and N splat IDs.

2. For every splat in the scene project the 3d covariance matrix to the 2d screen
   - Variations from the reference implementation noted below

3. Using the 2d covariance in the X and Y screen directions as a bounding
   rectangle, add each splat to the tiles that would be covered by that rectangle
   - The reference implementation calculates the eigenvalues of the 2d covariance
     matrix and then uses the greater of the eignvalues to determine a bounding square

4. Sort the splats assigned to each tile by their distance from the camera
   - Each tile is sorted using bitonic sort by a workgroup of size 512

5. Render the 16x16 pixels covered by each tile by iterating through the now
   sorted list of splats from nearest to farthest
   - This is virtual identical to the reference


## Differences from reference implementation

I basically do the same "precompute" pass as the reference implementation, but
rather than producing a screen wide (tile_id | depth, splat_id) list. I directly
insert splats into fixed size per tile buffers, essentially doing a pre-sort
pass at the expense of higher memory usage. As a result I do not have to do a
"global" sort of the whole list but instead can sort each tile independently,
however the fixed size buffers lead to artifacts if the buffers fill. 

I've found buffer sizes of 8192 splats requried to render most of the paper's
scenes without artifacts. However with that large buffer size, the frame rate
tends to dip quite a bit due to the bitonic sort and color calculations both
taking much longer.

Step 2 is very similar to the reference implemenation, howerver there are some
variances due to me using screen space coordinates from (0.0, 0.0) -> (1.0, 1.0),
vs I believe the reference uses coordinates from 0 -> output resolution

The reference implentation does the clipping shown below, that I do not fully
understand the purpose of, prior to computing the jacobian of the projection
matrix

```c
const float limx = 1.3f * tan_fovx;
const float limy = 1.3f * tan_fovy;
const float txtz = t.x / t.z;
const float tytz = t.y / t.z;
t.x = min(limx, max(-limx, txtz)) * t.z;
t.y = min(limy, max(-limy, tytz)) * t.z;
```

My variation below. I represent the camera zoom directly as a focal length rather
than with an x and y field of view value as the reference implementation does.

```wgsl
let limx = 1.0 / focal_length;
let limy = 1.0 / focal_length;
let txtz = t.x / t.z;
let tytz = t.y / t.z;

t.x = (clamp(txtz, -limx, limx)) * t.z;
t.y = (clamp(tytz, -limy, limy)) * t.z;
```

In the absence of this clamping very large splats are drawn directly in front of
the camera, obscuring the view of the scene. I am unsure of the significance of
the choice of 1.3 as a constant, but 1.0 among other values works to a degree in
my implmentation.
