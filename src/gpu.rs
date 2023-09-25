use std::error::Error;
use std::time::{SystemTime, UNIX_EPOCH};

use console::Term;
use ocl::{ProQue, Buffer, MemFlags, Platform, Device, Context, Queue};
use rand::{thread_rng, Rng};
use separator::Separatable;
use terminal_size::{Width, Height, terminal_size};
use tiny_keccak::Keccak;

use crate::{Config, WORK_SIZE, u64_to_le_fixed_8};

// Include the kernel source
static KERNEL_SRC: &'static str = include_str!("./kernels/keccak256.cl");

/// GPU implementation of the CREATE2 address search
pub fn gpu(config: Config) -> Result<(), Box<dyn Error>> {
    println!("Setting up experimental OpenCL miner using device {}...", config.gpu_device);

    // Extract the configuration values
    let factory = config.factory_address;
    let _caller = config.calling_address;
    let init_hash = config.init_code_hash;

    // Prefix unused variables with underscore
    let _salt: [u8; 6] = [0, 0, 0, 0, 0, 0];

    // Set up the message for the kernel (factory address + init code hash + prefix)
    let mut message: Vec<u8> = Vec::with_capacity(53 + config.starts_with.len());
    // First 20 bytes: factory address
    message.extend_from_slice(&factory);
    // Next 32 bytes: init code hash
    message.extend_from_slice(&init_hash);
    // Next byte: length of the prefix
    message.push(config.starts_with.len() as u8);
    // Last bytes: the prefix itself
    message.extend_from_slice(config.starts_with.as_bytes());

    // Set up the OpenCL context
    let platform = Platform::default();
    let device = Device::by_idx_wrap(platform, config.gpu_device as usize)?;
    let context = Context::builder()
        .platform(platform)
        .devices(device.clone())
        .build()?;
    let _queue = Queue::new(&context, device, None)?;

    // Create the OpenCL program queue - quit on error
    let ocl_pq = ProQue::builder()
        .src(KERNEL_SRC)
        .device(device)
        .dims(WORK_SIZE)
        .build()?;

    // Prefix unused variables with underscore
    let _term = Term::stdout();
    let mut previous_time = 0.0;
    let start_time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64();
    let mut cumulative_nonce: u64 = 0;
    let mut rng = thread_rng();

    // Main loop
    loop {
        // Build the kernel and define the type of each buffer
        let kern = ocl_pq.kernel_builder("hashMessage")
            .arg_named("message", None::<&Buffer<u8>>)
            .arg_named("nonce", None::<&Buffer<u32>>)
            .arg_named("solutions", None::<&Buffer<u64>>)
            .arg_named("has_solution", None::<&Buffer<u32>>)
            .arg_named("digest_output", None::<&Buffer<u8>>)
            .build()?;

        // Generate a random nonce
        let nonce: [u32; 1] = [rng.gen::<u32>()];

        // Create the buffers
        let nonce_buffer = Buffer::builder()
            .queue(ocl_pq.queue().clone())
            .flags(MemFlags::new().read_only())
            .len(1)
            .copy_host_slice(&nonce)
            .build()?;

        // Create the message buffer
        let message_buffer = Buffer::builder()
            .queue(ocl_pq.queue().clone())
            .flags(MemFlags::new().read_only())
            .len(message.len())
            .copy_host_slice(&message)
            .build()?;

        // Create the solutions buffer
        let solutions_buffer = Buffer::builder()
            .queue(ocl_pq.queue().clone())
            .flags(MemFlags::new().read_write())
            .len(1)
            .fill_val(0u64)
            .build()?;

        // Create the has_solution buffer
        let has_solution_buffer = Buffer::builder()
            .queue(ocl_pq.queue().clone())
            .flags(MemFlags::new().read_write())
            .len(1)
            .fill_val(0u32)
            .build()?;

        // Create the digest_output buffer
        let digest_output_buffer = Buffer::builder()
            .queue(ocl_pq.queue().clone())
            .flags(MemFlags::new().read_write())
            .len(200)
            .fill_val(0u8)
            .build()?;

        // Set the kernel arguments
        kern.set_arg("message", &message_buffer)?;
        kern.set_arg("nonce", &nonce_buffer)?;
        kern.set_arg("solutions", &solutions_buffer)?;
        kern.set_arg("has_solution", &has_solution_buffer)?;
        kern.set_arg("digest_output", &digest_output_buffer)?;

        // Enqueue the kernel
        unsafe {
            kern.enq()?;
        }

        // Read the solutions buffer
        let mut solutions = vec![0u64; 1];
        solutions_buffer.read(&mut solutions).enq()?;

        // Read the has_solution buffer
        let mut has_solution = vec![0u32; 1];
        has_solution_buffer.read(&mut has_solution).enq()?;

        // Read the digest_output buffer
        let mut digest_output = vec![0u8; 200];
        digest_output_buffer.read(&mut digest_output).enq()?;

        // Check if a solution was found
        if has_solution[0] != 0 {
            // A solution was found, process it
            let solution_bytes = u64_to_le_fixed_8(&solutions[0]);
            
            // Extract the address from the digest
            let mut address_bytes: [u8; 20] = Default::default();
            address_bytes.copy_from_slice(&digest_output[12..32]);
            let hex_address = hex::encode(&address_bytes);
            
            // Print detailed information about the solution
            println!("Found potential solution!");
            println!("Address from kernel digest: 0x{}", hex_address);
            
            // Check if the address starts with the specified prefix
            if hex_address.starts_with(&config.starts_with) {
                // Calculate the time it took to find the solution
                let solution_time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64() - start_time;
                
                // Found a valid solution
                println!("\nFound valid solution with prefix '{}' in {:.2} seconds!", config.starts_with, solution_time);
                
                // Convert address to checksummed format
                let checksummed_address = to_checksum_address(&hex_address);
                println!("Address: {}", checksummed_address);
                
                // Print the creation code hash
                println!("Creation Code Hash: 0x{}", hex::encode(&init_hash));
                
                // Format the salt properly as bytes32
                let mut full_salt = [0u8; 32]; // Initialize with all zeros
                
                // Copy the solution bytes (8 bytes) to the end of the salt
                let solution_len = std::cmp::min(solution_bytes.len(), 8);
                // Place the solution bytes at the end of the salt (after 24 zero bytes)
                full_salt[32 - solution_len..32].copy_from_slice(&solution_bytes[0..solution_len]);
                
                // Format as hex
                let salt_hex = format!("0x{}", hex::encode(&full_salt));
                println!("Salt: {}", salt_hex);
                
                // Verify the address using the same method as Foundry
                let mut hasher = Keccak::new_keccak256();
                hasher.update(&[0xff]); // 0xff prefix
                hasher.update(&factory); // deployer address
                hasher.update(&full_salt); // salt
                hasher.update(&init_hash); // init code hash
                
                let mut hash_result = [0u8; 32];
                hasher.finalize(&mut hash_result);
                
                // Extract the address (last 20 bytes)
                let mut computed_address = [0u8; 20];
                computed_address.copy_from_slice(&hash_result[12..32]);
                
                // Convert to hex and checksum
                let computed_hex = hex::encode(&computed_address);
                let computed_checksummed = to_checksum_address(&computed_hex);
                
                println!("Verified Address: {}", computed_checksummed);
                
                // Exit the program with success
                std::process::exit(0);
            }
        }

        // Update the cumulative nonce
        cumulative_nonce += WORK_SIZE as u64;

        // Print status update
        let current_time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64();
        let elapsed = current_time - start_time;
        if elapsed - previous_time >= 1.0 {
            previous_time = elapsed;
            let rate = cumulative_nonce as f64 / elapsed / 1_000_000.0;
            
            // Get terminal size
            let size = terminal_size();
            let _width = if let Some((Width(w), Height(_))) = size {
                w as usize
            } else {
                80
            };
            
            // Clear the terminal output
            print!("\x1B[2J\x1B[1;1H"); // ANSI escape code to clear screen and move cursor to top-left
            
            // Print the status
            println!(
                "----- New Update -----\n\
                 total runtime: {:.2} seconds\t\t\twork size per cycle: {}\n\
                 rate: {:.2} million attempts per second\ttotal attempts: {}\n\
                 current search space: {:x}xxxxxxxx\t\tsearching for prefix: {}",
                elapsed,
                WORK_SIZE.separated_string(),
                rate,
                cumulative_nonce.separated_string(),
                nonce[0],
                config.starts_with
            );
            
            
        }
    }
}

// Add this function to convert an address to checksummed format
fn to_checksum_address(address: &str) -> String {
    // Remove '0x' prefix if present
    let address = if address.starts_with("0x") {
        &address[2..]
    } else {
        address
    };
    
    // Convert address to lowercase
    let address = address.to_lowercase();
    
    // Hash the address
    let mut hasher = Keccak::new_keccak256();
    hasher.update(address.as_bytes());
    let mut hash = [0u8; 32];
    hasher.finalize(&mut hash);
    
    // Create checksummed address
    let mut checksummed = String::with_capacity(42);
    checksummed.push_str("0x");
    
    for (i, c) in address.chars().enumerate() {
        if c >= '0' && c <= '9' {
            checksummed.push(c);
        } else {
            // Get the corresponding nibble from the hash
            let nibble = hash[i / 2] >> (if i % 2 == 0 { 4 } else { 0 }) & 0xf;
            if nibble >= 8 {
                checksummed.push(c.to_ascii_uppercase());
            } else {
                checksummed.push(c);
            }
        }
    }
    
    checksummed
} 