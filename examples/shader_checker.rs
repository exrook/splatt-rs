use std::{thread, time::Duration};

// need this to ensure the shader gets linked in
use splatt::SHADER;

fn main() {
    env_logger::init();

    println!("Checking shaders");
    for entry in wgpu_shader_boilerplate::SHADERS {
        println!("Loading {:?}", entry);
        entry.load();
    }
    loop {
        for entry in wgpu_shader_boilerplate::SHADERS {
            if entry.should_reload() {
                println!("{:?}", entry);
                entry.load();
            }
        }
        thread::sleep(Duration::from_millis(200));
    }
}
