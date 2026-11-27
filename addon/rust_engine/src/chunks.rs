//! Spatial Chunk Partitioning for 1,000,000+ polygon target meshes.
//!
//! Partitions the base mesh's polygons ONCE on target upload into compact spatial chunks
//! (typically ~256–384 chunks of 2K–4K polygons each for a 1M-poly building).
//! Untouched chunks stay 100% resident in GPU VRAM across drag frames (0 ms recompute,
//! 0 bytes uploaded).

use crate::bvh::Bvh;
use crate::gpu_pack::GpuPackedMesh;
use crate::mesh::{Aabb, CachedMeshGeo};
use glam::DMat4;
use rayon::prelude::*;

#[derive(Clone, Debug)]
pub struct SpatialChunk {
    pub id: u32,
    pub aabb: Aabb,
    /// Sorted base polygon indices belonging exclusively to this chunk.
    pub polys: Vec<u32>,
}

pub struct SpatialChunks {
    pub chunks: Vec<SpatialChunk>,
    /// `poly_to_chunk[poly_idx] = chunk_id` (exact 1-to-1 partition of `0..num_polys`).
    pub poly_to_chunk: Vec<u32>,
    pub chunk_bvh: Bvh,
    /// Lazily populated untouched base GPU buffers per chunk (built ONCE per chunk, reused forever).
    base_gpu_cache: Vec<Option<GpuPackedMesh>>,
}

impl SpatialChunks {
    /// Partition `base_geo` into compact spatial chunks in parallel.
    pub fn build(base_geo: &CachedMeshGeo) -> Self {
        let np = base_geo.mesh.num_polys();
        if np == 0 {
            return Self {
                chunks: Vec::new(),
                poly_to_chunk: Vec::new(),
                chunk_bvh: Bvh {
                    nodes: Vec::new(),
                    indices: Vec::new(),
                },
                base_gpu_cache: Vec::new(),
            };
        }

        // Target ~256-384 chunks for 1M polys (~2.5K-4K polys/chunk), or 1 chunk for small meshes.
        let max_per_chunk: usize = if np <= 2048 {
            np
        } else {
            (np / 256).clamp(1024, 4096)
        };

        let centers: Vec<[f64; 3]> = if np >= 4096 {
            base_geo.poly_aabbs.par_iter().map(|b| b.center()).collect()
        } else {
            base_geo.poly_aabbs.iter().map(|b| b.center()).collect()
        };

        let mut indices: Vec<u32> = (0..np as u32).collect();
        let mut raw_groups: Vec<Vec<u32>> = Vec::new();
        Self::split_recursive(
            &mut indices,
            &centers,
            max_per_chunk,
            0,
            &mut raw_groups,
        );

        let num_chunks = raw_groups.len();
        let mut chunks: Vec<SpatialChunk> = raw_groups
            .into_par_iter()
            .enumerate()
            .map(|(cid, mut polys)| {
                polys.sort_unstable();
                let mut aabb = Aabb::empty();
                for &pi in &polys {
                    aabb.expand_aabb(&base_geo.poly_aabbs[pi as usize]);
                }
                SpatialChunk {
                    id: cid as u32,
                    aabb,
                    polys,
                }
            })
            .collect();
        chunks.sort_by_key(|c| c.id);

        let mut poly_to_chunk = vec![0u32; np];
        for ch in &chunks {
            for &pi in &ch.polys {
                poly_to_chunk[pi as usize] = ch.id;
            }
        }

        let chunk_aabbs: Vec<Aabb> = chunks.iter().map(|c| c.aabb).collect();
        let chunk_bvh = Bvh::build_aabbs(&chunk_aabbs);

        Self {
            chunks,
            poly_to_chunk,
            chunk_bvh,
            base_gpu_cache: vec![None; num_chunks],
        }
    }

    fn split_recursive(
        slice: &mut [u32],
        centers: &[[f64; 3]],
        max_per_chunk: usize,
        depth: usize,
        out: &mut Vec<Vec<u32>>,
    ) {
        if slice.len() <= max_per_chunk || depth >= 16 {
            out.push(slice.to_vec());
            return;
        }
        let mut cbounds = Aabb::empty();
        for &pi in slice.iter() {
            cbounds.expand_point(centers[pi as usize]);
        }
        let axis = cbounds.longest_axis();
        let split_pos = 0.5 * (cbounds.min[axis] + cbounds.max[axis]);

        let mut i = 0usize;
        let mut j = slice.len();
        while i < j {
            if centers[slice[i] as usize][axis] < split_pos {
                i += 1;
            } else {
                j -= 1;
                slice.swap(i, j);
            }
        }
        let mid = if i == 0 || i == slice.len() {
            slice.len() / 2
        } else {
            i
        };
        let (left, right) = slice.split_at_mut(mid);
        Self::split_recursive(left, centers, max_per_chunk, depth + 1, out);
        Self::split_recursive(right, centers, max_per_chunk, depth + 1, out);
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// Ensure untouched base GPU meshes are cached for the requested chunk IDs (built in parallel once).
    pub fn ensure_base_gpu_cached(
        &mut self,
        chunk_ids: &[u32],
        base_geo: &CachedMeshGeo,
        world_mat: &DMat4,
    ) {
        let missing: Vec<usize> = chunk_ids
            .iter()
            .map(|&cid| cid as usize)
            .filter(|&cid| cid < self.base_gpu_cache.len() && self.base_gpu_cache[cid].is_none())
            .collect();
        if missing.is_empty() {
            return;
        }
        let built: Vec<(usize, GpuPackedMesh)> = missing
            .par_iter()
            .map(|&cid| {
                let gpu = GpuPackedMesh::from_hybrid_mesh_subset(
                    &base_geo.mesh,
                    &self.chunks[cid].polys,
                    world_mat,
                );
                (cid, gpu)
            })
            .collect();
        for (cid, gpu) in built {
            self.base_gpu_cache[cid] = Some(gpu);
        }
    }

    #[inline]
    pub fn get_base_gpu(&self, cid: u32) -> Option<&GpuPackedMesh> {
        self.base_gpu_cache.get(cid as usize).and_then(|o| o.as_ref())
    }
}
