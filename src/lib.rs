use bytemuck::{Pod, Zeroable};
use camera::{Control, GpuCameraNormal};
use glam::Vec3;
use ply_rs::ply::Property;
use shader::prelude::*;
use std::{fs::File, mem, path::Path};
use wgpu_shader_boilerplate as shader;

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }
}

pub fn load_file(path: &Path) -> Vec<Gaussian> {
    let mut file = File::open(path).unwrap();
    let p = ply_rs::parser::Parser::<Gaussian>::new();
    let mut data = p.read_ply(&mut file).unwrap();
    data.payload.remove("vertex").unwrap()
}

#[derive(Default, Debug, Copy, Clone, Pod, Zeroable)]
#[repr(C)]
pub struct Gaussian {
    rot: glam::Quat,
    pos: Vec3,
    _pad0: u32,
    n: Vec3,
    _pad1: u32,
    sh_dc: Vec3,
    _pad2: u32,
    scale: Vec3,
    opacity: f32,
    // figure out rest of harmonics later
    // see https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/59f5f77e3ddbac3ed9db93ec2cfe99ed6c5d121d/cuda_rasterizer/forward.cu#L30-L69
}

impl Gaussian {
    pub fn base_color_rgb(&self) -> (u8, u8, u8) {
        let color = harmonic_to_rgb(&[self.sh_dc]) * Vec3::splat(255.0);
        (color.x as u8, color.y as u8, color.z as u8)
    }
}

const SH_C0: f32 = 0.28209479177387814;

pub fn harmonic_to_rgb(harmonics: &[Vec3]) -> Vec3 {
    let mut out = Vec3::splat(0.5);
    out += harmonics[0] * SH_C0;
    out.clamp(Vec3::splat(0.0), Vec3::splat(1.0))
}

impl ply_rs::ply::PropertyAccess for Gaussian {
    fn new() -> Self {
        Self::default()
    }
    fn set_property(&mut self, name: String, property: ply_rs::ply::Property) {
        match property {
            Property::Float(val) => match name.as_str() {
                "x" => self.pos.x = val,
                "y" => self.pos.y = val,
                "z" => self.pos.z = val,
                "nx" => self.n.x = val,
                "ny" => self.n.y = val,
                "nz" => self.n.z = val,
                "f_dc_0" => self.sh_dc.x = val,
                "f_dc_1" => self.sh_dc.y = val,
                "f_dc_2" => self.sh_dc.z = val,
                // add rest of harmonics here later
                "opacity" => self.opacity = val,
                "scale_0" => self.scale.x = val,
                "scale_1" => self.scale.y = val,
                "scale_2" => self.scale.z = val,
                "rot_0" => self.rot.x = val,
                "rot_1" => self.rot.y = val,
                "rot_2" => self.rot.z = val,
                "rot_3" => self.rot.w = val,
                _ => (),
            },
            _ => (),
        }
    }
}

shader::compute_pipelines!(Pipelines {
    precompute,
    preprocess,
    setup_tiles,
    sort_within_tiles,
    draw_tiles
});

#[shader::m::linkme::distributed_slice(shader::SHADERS)]
#[linkme(crate = shader::m::linkme)]
pub static SHADER: shader::Shader = shader::Shader::from_path(file!(), "splat.wgsl", "", None);

pub struct DrawGaussianResources {
    pipeline: shader::PipelineCacheWithLayout<Pipelines>,
}

#[derive(Copy, Clone, Pod, Zeroable)]
#[repr(C)]
struct DispatchData {
    num_tiles: [u32; 2],
    output_size: [u32; 2],
    num_splats: u32,
}

#[derive(Copy, Clone, Pod, Zeroable, Debug)]
#[repr(C)]
struct ControlData {
    camera: GpuCameraNormal,
}

// Data we need to make a decision on during shader execution
// Prepass (?):
//  - compute 2d cov,
//  - compute 2d pos, depth
//  - compute color,
// Tile sorting:
//  - use 2d position, depth
//    - store depth in each tile
//  - 2d covariance - reference implementation only computes an integer tile radius per point in prepass
//    - if we have thread per splat -> compute and store this now
//    - if we have thread per tile -> precompute so we don't waste 1024 thread computing the same value
//      - alternatively have 1024 threads precompute next 1024 splats?
//      - or compute cov per splat in prepass
// Per Tile Depth sorting:
//  - splat depth
//    - precomputed in last step
// Painting
//  - color
//    - should be precomputed, doesn't vary per pixel as originally thought
//  - 2d covariance
//    - should be precomputed
//  - 2d position
pub struct WorkingBuffers {
    depth: wgpu::Buffer,
    pos: wgpu::Buffer,
    cov_2d_alpha: wgpu::Buffer,
    color: wgpu::Buffer,
    cov_3d: wgpu::Buffer,
    pos_3d_alpha: wgpu::Buffer,

    read_write_group: wgpu::BindGroup,
    read_group: wgpu::BindGroup,
}

impl WorkingBuffers {
    fn buffer(device: &wgpu::Device, elem_size: u32, count: u32) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("working buffers"),
            size: ((elem_size * 2) * 4 * count) as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        })
    }
    pub fn new(device: &wgpu::Device, splats: u32) -> Self {
        let depth = Self::buffer(device, 1, splats);
        let pos = Self::buffer(device, 2, splats);
        let cov_2d_alpha = Self::buffer(device, 4, splats);
        let color = Self::buffer(device, 4, splats);
        let cov_3d = Self::buffer(device, 6, splats);
        let pos_3d_alpha = Self::buffer(device, 4, splats);

        let make_descriptor = |layout: &_| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("working data bind group"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(depth.as_entire_buffer_binding()),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(pos.as_entire_buffer_binding()),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::Buffer(
                            cov_2d_alpha.as_entire_buffer_binding(),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: wgpu::BindingResource::Buffer(color.as_entire_buffer_binding()),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: wgpu::BindingResource::Buffer(cov_3d.as_entire_buffer_binding()),
                    },
                    wgpu::BindGroupEntry {
                        binding: 5,
                        resource: wgpu::BindingResource::Buffer(
                            pos_3d_alpha.as_entire_buffer_binding(),
                        ),
                    },
                ],
            })
        };

        let read_group = Self::with_layout::<Read, _, _>(device, |layout| make_descriptor(layout));
        let read_write_group =
            Self::with_layout::<ReadWrite, _, _>(device, |layout| make_descriptor(layout));

        Self {
            depth,
            pos,
            cov_2d_alpha,
            color,
            cov_3d,
            pos_3d_alpha,
            read_group,
            read_write_group,
        }
    }
}

impl BindLayout for WorkingBuffers {}
impl<T: AccessType> BindLayoutFor<T> for WorkingBuffers {
    fn with_layout_impl<F: FnOnce(&wgpu::BindGroupLayoutDescriptor) -> U, U>(f: F) -> U {
        let stages = wgpu::ShaderStages::COMPUTE;
        let entry = |idx| wgpu::BindGroupLayoutEntry {
            binding: idx,
            visibility: stages,
            ty: wgpu::BindingType::Buffer {
                ty: T::storage_buffer_type(),
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        f(&wgpu::BindGroupLayoutDescriptor {
            label: Some("splat working buffers"),
            entries: &[entry(0), entry(1), entry(2), entry(3), entry(4), entry(5)],
        })
    }
}
impl Bindable for WorkingBuffers {}
impl BindableFor<Read> for WorkingBuffers {
    fn group_impl(&self) -> &wgpu::BindGroup {
        &self.read_group
    }
}

impl BindableFor<ReadWrite> for WorkingBuffers {
    fn group_impl(&self) -> &wgpu::BindGroup {
        &self.read_write_group
    }
}

pub struct TileBuffer {
    control_data: wgpu::Buffer,
    tile_buf: wgpu::Buffer,
    read_write_group: wgpu::BindGroup,
    read_group: wgpu::BindGroup,
}

impl TileBuffer {
    fn buffer_for_bytes(device: &wgpu::Device, bytes: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Splat tile buffer"),
            size: bytes,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        })
    }
    // match these to wgsl source
    // one day they will be a specialization constant
    const TILE_SIZE: u64 = 2048; // should match definition in splat.wgsl
    const SIZEOF_TILE: u64 = 8 + 4 * 4 + 4 * 4 + 8 * Self::TILE_SIZE;
    fn tile_buf(device: &wgpu::Device, num_tiles: [u32; 2]) -> wgpu::Buffer {
        let tile_buf_size = Self::SIZEOF_TILE * (num_tiles[0] * num_tiles[1]) as u64;
        Self::buffer_for_bytes(device, tile_buf_size)
    }
    pub fn new(device: &wgpu::Device, num_tiles: [u32; 2]) -> Self {
        let tile_buf = Self::tile_buf(device, num_tiles);
        let control_data = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Splat uniforms"),
            size: std::mem::size_of::<ControlData>() as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let make_descriptor = |layout: &_| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Splat tile group"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(
                            control_data.as_entire_buffer_binding(),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(
                            tile_buf.as_entire_buffer_binding(),
                        ),
                    },
                ],
            })
        };

        let read_group = Self::with_layout::<Read, _, _>(device, |layout| make_descriptor(layout));
        let read_write_group =
            Self::with_layout::<ReadWrite, _, _>(device, |layout| make_descriptor(layout));

        Self {
            control_data,
            tile_buf,
            read_group,
            read_write_group,
        }
    }
    fn resize_tiles(&mut self, device: &wgpu::Device, num_tiles: [u32; 2]) {
        let size = self.tile_buf.size();
        let new_size = Self::SIZEOF_TILE * (num_tiles[0] * num_tiles[1]) as u64;

        // shrink only if new_size is more than 20% smaller than current size
        let should_shrink = (new_size / 5 + new_size) < size;

        if new_size > size || should_shrink {
            self.tile_buf = Self::tile_buf(device, num_tiles);
        }
        let make_descriptor = |layout: &_| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Splat tile group"),
                layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::Buffer(
                            self.control_data.as_entire_buffer_binding(),
                        ),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Buffer(
                            self.tile_buf.as_entire_buffer_binding(),
                        ),
                    },
                ],
            })
        };

        self.read_group = Self::with_layout::<Read, _, _>(device, |layout| make_descriptor(layout));
        self.read_write_group =
            Self::with_layout::<ReadWrite, _, _>(device, |layout| make_descriptor(layout));
    }
}

impl BindLayout for TileBuffer {}
impl<T: AccessType> BindLayoutFor<T> for TileBuffer {
    fn with_layout_impl<F: FnOnce(&wgpu::BindGroupLayoutDescriptor) -> U, U>(f: F) -> U {
        let stages = wgpu::ShaderStages::COMPUTE;
        f(&wgpu::BindGroupLayoutDescriptor {
            label: Some("splat tile buffers and uniforms"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: stages,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: stages,
                    ty: wgpu::BindingType::Buffer {
                        ty: T::storage_buffer_type(),
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        })
    }
}
impl Bindable for TileBuffer {}
impl BindableFor<Read> for TileBuffer {
    fn group_impl(&self) -> &wgpu::BindGroup {
        &self.read_group
    }
}

impl BindableFor<ReadWrite> for TileBuffer {
    fn group_impl(&self) -> &wgpu::BindGroup {
        &self.read_write_group
    }
}

impl DrawGaussianResources {
    pub fn new(device: &wgpu::Device) -> Self {
        let pipeline = SHADER.cache_with_layout(shader::create_pipeline_layout!(
            [device]
            layouts: {
                (wgpu::Buffer,): Read, // gaussians
                WorkingBuffers: ReadWrite,
                TileBuffer: ReadWrite, // control data, tiles
                (shader::TypedTexture<shader::RGBA8Unorm<shader::D2>>,): ReadWrite // output
            }
            push_constant_ranges: &[wgpu::PushConstantRange {
                stages: wgpu::ShaderStages::COMPUTE,
                range: 0..(mem::size_of::<DispatchData>() as u32),
            }]

        ));
        Self { pipeline }
    }

    pub fn make_working_buffers(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        gaussians: &wgpu::Buffer,
        tile_buffer: &TileBuffer,
        texture: &shader::TypedTexture<shader::RGBA8Unorm<shader::D2>>,
        splat_count: u32,
    ) -> WorkingBuffers {
        let mut enc = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("splat precompute"),
        });

        let working_buffers = WorkingBuffers::new(device, splat_count);

        {
            let Pipelines { precompute, .. } = &*self.pipeline.load(device);

            let group0 = (gaussians,).bind(device);
            let group3 = (texture,).bind(device);

            let mut cpass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("splat precompute"),
                ..Default::default()
            });

            let dispatch = DispatchData {
                num_tiles: [0, 0],
                output_size: [0, 0],
                num_splats: splat_count,
            };

            cpass.set_pipeline(precompute);
            cpass.set_push_constants(0, bytemuck::bytes_of(&dispatch));
            cpass.set_bind_group(0, group0.group::<Read>(), &[]);
            cpass.set_bind_group(1, working_buffers.group::<ReadWrite>(), &[]);
            cpass.set_bind_group(2, tile_buffer.group::<ReadWrite>(), &[]);
            cpass.set_bind_group(3, group3.group::<ReadWrite>(), &[]);
            cpass.dispatch_workgroups((splat_count as u32 + 1023) / 1024, 1, 1);
        }
        queue.submit(Some(enc.finish()));
        working_buffers
    }

    pub fn draw(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        control: &Control,
        gaussians: &wgpu::Buffer,
        splat_count: u64,
        working_buffers: &mut WorkingBuffers,
        tile_buffer: &mut TileBuffer,
        texture: &shader::TypedTexture<shader::RGBA8Unorm<shader::D2>>,
        profiler: &mut wgpu_profiler::GpuProfiler,
    ) {
        use wgpu_profiler::wgpu_profiler;
        let texture_dim = texture.texture().size();
        let num_tiles = [texture_dim.width / 16, texture_dim.height / 16];
        let total_tiles = num_tiles[0] * num_tiles[1];
        tile_buffer.resize_tiles(device, num_tiles);

        let output_size = [texture_dim.width, texture_dim.height];

        let control_buf = ControlData {
            camera: GpuCameraNormal::from(control),
        };

        queue.write_buffer(
            &tile_buffer.control_data,
            0,
            bytemuck::bytes_of(&control_buf),
        );

        {
            let Pipelines {
                preprocess,
                setup_tiles,
                sort_within_tiles,
                draw_tiles,
                ..
            } = &*self.pipeline.load(device);

            let group0 = (gaussians,).bind(device);
            let group3 = (texture,).bind(device);

            let mut cpass = encoder.begin_compute_pass(&Default::default());

            let dispatch = DispatchData {
                num_tiles,
                output_size,
                num_splats: splat_count as u32,
            };

            wgpu_profiler!("setup tiles", profiler, &mut cpass, &device, {
                cpass.set_pipeline(&setup_tiles);

                cpass.set_push_constants(0, bytemuck::bytes_of(&dispatch));
                cpass.set_bind_group(0, group0.group::<Read>(), &[]);
                cpass.set_bind_group(1, working_buffers.group::<ReadWrite>(), &[]);
                cpass.set_bind_group(2, tile_buffer.group::<ReadWrite>(), &[]);
                cpass.set_bind_group(3, group3.group::<ReadWrite>(), &[]);
                cpass.dispatch_workgroups(
                    (TileBuffer::TILE_SIZE as u32 + 1023) / 1024,
                    total_tiles,
                    1,
                );
            });

            const PREPROCESS_WORKGROUP_SIZE: u32 = 1024;
            wgpu_profiler!("preprocess splats", profiler, &mut cpass, &device, {
                cpass.set_pipeline(&preprocess);
                cpass.dispatch_workgroups(
                    (splat_count as u32 + (PREPROCESS_WORKGROUP_SIZE - 1))
                        / PREPROCESS_WORKGROUP_SIZE,
                    1,
                    1,
                );
            });

            wgpu_profiler!("sort tiles", profiler, &mut cpass, &device, {
                cpass.set_pipeline(&sort_within_tiles);
                cpass.dispatch_workgroups(total_tiles, 1, 1);
            });

            wgpu_profiler!("draw tiles", profiler, &mut cpass, &device, {
                cpass.set_pipeline(&draw_tiles);
                cpass.dispatch_workgroups(num_tiles[0], num_tiles[1], 1);
            });
        }
    }
}

pub struct BoundGaussians {
    read_group: wgpu::BindGroup,
    buffer: wgpu::Buffer,
}

impl BoundGaussians {
    pub fn new(device: &wgpu::Device, points: &[Gaussian]) -> Self {
        let data_bytes: &[u8] = bytemuck::cast_slice(points);
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("chunkk buffer"),
            usage: wgpu::BufferUsages::STORAGE,
            size: data_bytes.len() as _,
            mapped_at_creation: true,
        });
        buffer
            .slice(..)
            .get_mapped_range_mut()
            .copy_from_slice(data_bytes);
        let make_group = |layout: &_| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("minecraft chunk"),
                layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }],
            })
        };
        let read_group = Self::with_layout::<Read, _, _>(device, make_group);
        Self { read_group, buffer }
    }
    pub fn unmap(&self) {
        self.buffer.unmap()
    }
    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }
}
impl shader::group_traits::BindLayout for BoundGaussians {}
impl shader::group_traits::BindLayoutFor<Read> for BoundGaussians {
    fn with_layout_impl<F: FnOnce(&wgpu::BindGroupLayoutDescriptor) -> U, U>(f: F) -> U {
        f(&wgpu::BindGroupLayoutDescriptor {
            label: Some("gaussian inputs"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        })
    }
}
impl Bindable for BoundGaussians {}
impl BindableFor<Read> for BoundGaussians {
    fn group_impl(&self) -> &wgpu::BindGroup {
        &self.read_group
    }
}
