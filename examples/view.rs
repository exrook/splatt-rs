use splatt::{DrawGaussianResources, TileBuffer};

use wgpu_profiler::{wgpu_profiler, GpuProfiler};
use wgpu_shader_boilerplate as shader;
use winit::{
    event::{Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
};
use winit_input_helper::WinitInputHelper;

pub fn wgpu_features() -> wgpu::Features {
    wgpu::Features::empty()
        | wgpu::Features::PUSH_CONSTANTS
        | wgpu::Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES
        // | wgpu::Features::UNIFORM_BUFFER_AND_STORAGE_TEXTURE_ARRAY_NON_UNIFORM_INDEXING
        | wgpu::Features::TIMESTAMP_QUERY
        | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES
}

pub fn wgpu_limits(adapter: &wgpu::Adapter) -> wgpu::Limits {
    let adapter_limits = adapter.limits();
    wgpu::Limits {
        max_push_constant_size: 64,
        max_buffer_size: adapter_limits.max_buffer_size,
        max_storage_buffer_binding_size: adapter_limits.max_storage_buffer_binding_size,
        max_compute_invocations_per_workgroup: adapter_limits.max_compute_invocations_per_workgroup,
        max_compute_workgroup_size_x: adapter_limits.max_compute_workgroup_size_x,
        max_compute_workgroup_size_y: adapter_limits.max_compute_workgroup_size_y,
        max_compute_workgroup_size_z: adapter_limits.max_compute_workgroup_size_z,
        ..wgpu::Limits::default()
    }
    .using_resolution(adapter_limits.clone())
    .using_alignment(adapter_limits)
}

pub fn setup_wgpu(
    instance: &wgpu::Instance,
    surface: Option<&wgpu::Surface>,
) -> (wgpu::Adapter, wgpu::Device, wgpu::Queue) {
    pollster::block_on(async {
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                force_fallback_adapter: false,
                compatible_surface: surface,
            })
            .await
            .expect("Failed to find an appropriate adapter");

        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: None,

                    features: wgpu_features(),
                    limits: wgpu_limits(&adapter),
                },
                None,
            )
            .await
            .expect("Failed to create device");
        (adapter, device, queue)
    })
}

fn main() {
    env_logger::init();

    let mut args = std::env::args().skip(1);
    let path = args.next().unwrap();

    println!("Reading splats from file");
    let splats = splatt::load_file(path.as_ref());
    let splat_count = splats.len() as u64;
    println!("Reading {} splats", splat_count);

    let mut event_loop = EventLoop::new();

    let (mut window, (instance, adapter, device, queue)) =
        windowed::Window::create_with(&mut event_loop, |instance, surface| {
            setup_wgpu(instance, Some(surface))
        });

    debugui::init_on!(
        debug_ui,
        &mut event_loop,
        &instance,
        &adapter,
        &device,
        &queue
    );

    let mut profiler = GpuProfiler::new(&adapter, &device, &queue, 4);

    window
        .window
        .set_resize_increments(Some(winit::dpi::PhysicalSize::new(8, 8)));
    let size = window.window.inner_size();

    let mut resolution = (size.width, size.height);

    let mut input = WinitInputHelper::new();

    let mut draw_splats = DrawGaussianResources::new(&device);

    let mut tile_buffer = TileBuffer::new(&device, [1, 1]);

    let mut draw_texture = shader::TypedTexture::new(&device, resolution.0, resolution.1);

    let splats_buf = splatt::BoundGaussians::new(&device, &splats);

    splats_buf.unmap();
    // These sleeps are attemps to prevent gpu hangs
    log::debug!("SLEEPING");
    std::thread::sleep(std::time::Duration::from_millis(500));
    device.poll(wgpu::Maintain::Wait);
    log::debug!("SLEEPING 2");
    std::thread::sleep(std::time::Duration::from_millis(500));

    let mut control = camera::Control::zero();

    window.data.replace_texture(&device, draw_texture.texture());

    log::info!("Distributing splat data");
    let mut draw_buffers = draw_splats.make_working_buffers(
        &device,
        &queue,
        splats_buf.buffer(),
        &tile_buffer,
        &draw_texture,
        splats.len() as u32,
    );
    println!("Submitted precomputation step");
    device.poll(wgpu::Maintain::Wait);
    println!("Finished precomputation");
    log::debug!("SLEEPING 3");
    std::thread::sleep(std::time::Duration::from_millis(500));

    use winit::platform::run_return::EventLoopExtRunReturn;
    event_loop.run_return(|event, _, control_flow| {
        if debugui::feed_on!(debug_ui, &event, control_flow) {
            return;
        }
        debugui::set!(&mut control.focal_length);
        if input.update(&event) {
            if input.key_pressed(winit::event::VirtualKeyCode::O) {
                window.resources.mode = (window.resources.mode + 1) % 3;
            }
            let new_control = camera_input::process_input(&input, &control);
            control.apply(&new_control);
            control.clamp();
        }

        *control_flow = ControlFlow::Poll;
        match event {
            Event::WindowEvent {
                event: WindowEvent::Resized(size),
                ..
            } => {
                if size.width > 0 && size.height > 0 {
                    resolution = (size.width, size.height);
                    window
                        .resources
                        .reconfigure(size, &window.surface, &device, &queue);
                    window.window.request_redraw();
                    draw_texture = shader::TypedTexture::new(&device, resolution.0, resolution.1);
                    window.data.replace_texture(&device, draw_texture.texture());
                }
            }
            Event::MainEventsCleared | Event::RedrawRequested(_) => {
                let mut encoder = device.create_command_encoder(&Default::default());
                wgpu_profiler!("whole frame", &mut profiler, &mut encoder, &device, {
                    draw_splats.draw(
                        &device,
                        &queue,
                        &mut encoder,
                        &control,
                        splats_buf.buffer(),
                        splat_count,
                        &mut draw_buffers,
                        &mut tile_buffer,
                        &draw_texture,
                        &mut profiler,
                    );
                });
                profiler.resolve_queries(&mut encoder);

                match window.surface.get_current_texture() {
                    Ok(frame) => {
                        let view = frame
                            .texture
                            .create_view(&wgpu::TextureViewDescriptor::default());

                        window
                            .resources
                            .draw(&device, &mut encoder, &window.data, &view);

                        queue.submit(Some(encoder.finish()));

                        frame.present();
                        profiler.end_frame().unwrap();
                        if let Some(profiling_data) = profiler.process_finished_frame() {
                            scopes_to_console_recursive(&profiling_data, 0);
                        }
                        device.poll(wgpu::Maintain::Wait);
                    }
                    Err(e) => {
                        println!("Failed to acquire next swap chain texture {}", e);
                    }
                }
            }
            Event::WindowEvent {
                event: WindowEvent::CloseRequested,
                ..
            } => {
                *control_flow = ControlFlow::Exit;
            }
            _ => {}
        }
    });
}

// copied from wgpu_profiler example
fn scopes_to_console_recursive(results: &[wgpu_profiler::GpuTimerScopeResult], indentation: u32) {
    for scope in results {
        if indentation > 0 {
            print!("{:<width$}", "|", width = 4);
        }
        println!(
            "{:.3}μs - {}",
            (scope.time.end - scope.time.start) * 1000.0 * 1000.0,
            scope.label
        );
        if !scope.nested_scopes.is_empty() {
            scopes_to_console_recursive(&scope.nested_scopes, indentation + 1);
        }
    }
}
