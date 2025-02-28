use std::error::Error;
use std::time::{SystemTime, UNIX_EPOCH};
use std::fs::OpenOptions;
use std::io::Write;
use std::fs::File;
use std::io::{BufRead, BufReader};

use console::Term;
use ocl::{ProQue, Buffer, MemFlags, Platform, Device, Context, Queue};
use rand::{thread_rng, Rng};
use separator::Separatable;
use tiny_keccak::Keccak;
use terminal_size;

use crate::{Config, WORK_SIZE, u64_to_le_fixed_8};

// Include the kernel source
static KERNEL_SRC: &'static str = include_str!("./kernels/keccak256.cl");

/// GPU implementation of the CREATE2 address search
pub fn gpu(config: Config) -> Result<(), Box<dyn Error>> {
    println!("Setting up experimental OpenCL miner using platform {} device {}...",
             config.platform_id, config.gpu_device);

    // Extract the configuration values
    let factory = config.factory_address;
    let _caller = config.calling_address;
    let init_hash = config.init_code_hash;

    // Prefix unused variables with underscore
    let _salt: [u8; 6] = [0, 0, 0, 0, 0, 0];

    // Read the current best score from the CSV file
    let mut last_score_check = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64();
    let mut cached_best_score = read_current_best_score(&config.output_file);
    let best_score = cached_best_score;

    // Set up the message for the kernel
    let mut message: Vec<u8> = Vec::with_capacity(55); // Increased capacity for best score
    // First 20 bytes: factory address
    message.extend_from_slice(&factory);
    // Next 32 bytes: init code hash
    message.extend_from_slice(&init_hash);
    // Next byte: minimum leading 1s
    message.push(config.min_leading_ones as u8);
    // Next byte: minimum trailing 1s
    message.push(config.min_trailing_ones as u8);
    // Next byte: current best score (for filtering in the kernel)
    message.push(best_score as u8);

    // Get all platforms
    let platforms = Platform::list();
    if platforms.is_empty() {
        return Err("No OpenCL platforms found".into());
    }

    // Check if the platform ID is valid
    if config.platform_id as usize >= platforms.len() {
        return Err(format!("Invalid platform ID: {}. Available platforms: {}",
                           config.platform_id, platforms.len()).into());
    }

    // Set up the OpenCL context with the specified platform
    let platform = platforms[config.platform_id as usize];
    let devices = match Device::list(platform, None) {
        Ok(devices) => devices,
        Err(e) => return Err(format!("Failed to get devices for platform {}: {}", config.platform_id, e).into())
    };

    if config.gpu_device as usize >= devices.len() {
        return Err(format!("Invalid device ID: {}. Available devices on platform {}: {}",
                           config.gpu_device, config.platform_id, devices.len()).into());
    }

    // Get the specific device
    let device = devices[config.gpu_device as usize];

    // Print device info
    if let Ok(name) = device.name() {
        println!("Using device: {}", name);
    }

    // Create the context and queue with better error handling
    let context = match Context::builder()
        .platform(platform)
        .devices(device.clone())
        .build() {
        Ok(ctx) => ctx,
        Err(e) => return Err(format!("Failed to create context for platform {} device {}: {}",
                                     config.platform_id, config.gpu_device, e).into())
    };

    let queue = match Queue::new(&context, device.clone(), None) {
        Ok(q) => q,
        Err(e) => return Err(format!("Failed to create queue for platform {} device {}: {}",
                                     config.platform_id, config.gpu_device, e).into())
    };

    // Create the OpenCL program queue - quit on error
    let ocl_pq = ProQue::builder()
        .src(KERNEL_SRC)
        .device(device)
        .dims(WORK_SIZE)
        .build()?;

    // Calculate how many GPUs are in the system for display purposes
    let all_platforms = Platform::list();
    let total_gpus: usize = all_platforms
        .iter()
        .map(|p| {
            match Device::list(*p, None) {
                Ok(devices) => devices.len(),
                Err(_) => 0
            }
        })
        .sum();

    // Clear the screen once at the beginning
    print!("\x1B[2J"); // Clear entire screen
    print!("\x1B[1;1H"); // Move cursor to top-left

    // Print header for this GPU
    let gpu_id = format!("P{}-D{}", config.platform_id, config.gpu_device);
    println!("Starting search on {} using {} work items per batch", gpu_id, WORK_SIZE.separated_string());

    // Reserve space for all GPUs
    for _ in 0..total_gpus {
        println!("\n\n\n\n\n"); // 5 lines per GPU
    }

    // Create vertical spacing between GPU outputs
    println!("\n\n"); // Additional spacing at the bottom

    // Force stdout flush
    let _ = std::io::stdout().flush();

    // Wait a moment to let other GPU threads initialize
    std::thread::sleep(std::time::Duration::from_millis(100 * config.gpu_device as u64));

    // Prefix unused variables with underscore
    let _term = Term::stdout();
    let _previous_time = 0.0;
    let start_time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64();
    let mut cumulative_nonce: u64 = 0;
    let mut rng = thread_rng();

    let mut last_update = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64();
    let mut last_attempts = 0u64;

    // Create buffers once, before the main loop
    let nonce_buffer = Buffer::builder()
        .queue(ocl_pq.queue().clone())
        .flags(MemFlags::new().read_only())
        .len(1)
        .fill_val(0u32)
        .build()?;

    let message_buffer = Buffer::builder()
        .queue(ocl_pq.queue().clone())
        .flags(MemFlags::new().read_only())
        .len(message.len())
        .copy_host_slice(&message)
        .build()?;

    let solutions_buffer = Buffer::builder()
        .queue(ocl_pq.queue().clone())
        .flags(MemFlags::new().read_write())
        .len(3)
        .fill_val(0u64)
        .build()?;

    let has_solution_buffer = Buffer::builder()
        .queue(ocl_pq.queue().clone())
        .flags(MemFlags::new().read_write())
        .len(1)
        .fill_val(0u32)
        .build()?;

    let digest_output_buffer = Buffer::builder()
        .queue(ocl_pq.queue().clone())
        .flags(MemFlags::new().read_write())
        .len(200)
        .fill_val(0u8)
        .build()?;

    // Create the kernel once
    let kern = ocl_pq.kernel_builder("hashMessage")
        .arg_named("message", &message_buffer)
        .arg_named("nonce", &nonce_buffer)
        .arg_named("solutions", &solutions_buffer)
        .arg_named("has_solution", &has_solution_buffer)
        .arg_named("digest_output", &digest_output_buffer)
        .build()?;

    // Main loop
    loop {
        let current_time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64();
        if current_time - last_score_check >= 5.0 {  // Check every 5 seconds
            cached_best_score = read_current_best_score(&config.output_file);
            last_score_check = current_time;
        }
        let best_score = cached_best_score;

        // Update the message with the current best score
        message[54] = best_score as u8;

        let nonce: [u32; 1] = [rng.gen::<u32>()];

        // Enqueue the kernel
        unsafe {
            // Update the nonce buffer
            nonce_buffer.write(&nonce[..]).enq()?;

            // Enqueue with explicit global work size
            kern.cmd()
                .global_work_size(WORK_SIZE)
                .enq()?;
        }

        // Read the solutions buffer
        let mut solutions = vec![0u64; 3];
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
            let leading_ones = solutions[1];
            let trailing_ones = solutions[2];

            // Extract the address from the digest
            let mut address_bytes: [u8; 20] = Default::default();
            address_bytes.copy_from_slice(&digest_output[12..32]);
            let hex_address = hex::encode(&address_bytes);

            // Calculate a score based on total 1s
            let leading_ones_usize = leading_ones as usize;
            let trailing_ones_usize = trailing_ones as usize;
            let current_score = leading_ones_usize + trailing_ones_usize;

            // Check if this is better than our current best, also print anything 11 and above
            if current_score >= 11 || current_score > best_score {
                // Get the current time for timing calculations
                let current_time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64();

                // Clear a portion of the screen for the solution announcement
                print!("\x1B[2J"); // Clear entire screen
                print!("\x1B[1;1H"); // Move cursor to top-left

                // Print a prominent solution announcement
                println!("╔═══════════════════════════════════════════════════════════╗");
                println!("║                   SOLUTION FOUND!                          ║");
                println!("╚═══════════════════════════════════════════════════════════╝");
                println!();
                println!("🎉 FOUND BY: Platform {} Device {}", config.platform_id, config.gpu_device);
                println!("📈 SCORE: {} total 1s ({} leading + {} trailing)",
                         current_score, leading_ones_usize, trailing_ones_usize);
                println!("📝 ADDRESS: 0x{}", hex_address);

                // Calculate the time it took to find the solution
                let solution_time = current_time - start_time;
                println!("⏱️  TIME: {:.2} seconds", solution_time);

                // Format the salt properly as bytes32
                let mut full_salt = [0u8; 32]; // Initialize with all zeros

                // Copy the solution bytes (8 bytes) to the end of the salt
                let solution_len = std::cmp::min(solution_bytes.len(), 8);
                // Place the solution bytes at the end of the salt (after 24 zero bytes)
                full_salt[32 - solution_len..32].copy_from_slice(&solution_bytes[0..solution_len]);

                // Format as hex
                let salt_hex = format!("0x{}", hex::encode(&full_salt));
                println!("🔑 SALT: {}", salt_hex);

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

                println!("✅ VERIFIED ADDRESS: {}", computed_checksummed);
                println!();

                // Write to CSV file
                let file_exists = std::path::Path::new(&config.output_file).exists();
                let mut file = OpenOptions::new()
                    .write(true)
                    .create(true)
                    .append(true)
                    .open(&config.output_file)
                    .unwrap_or_else(|e| {
                        eprintln!("Failed to open output file: {}", e);
                        std::process::exit(1);
                    });

                // Write header if the file is new
                if !file_exists {
                    writeln!(file, "address,salt,score,leading_ones,trailing_ones,platform,device,timestamp")
                        .unwrap_or_else(|e| {
                            eprintln!("Failed to write CSV header: {}", e);
                        });
                }

                // Write the data with GPU information
                writeln!(
                    file,
                    "{},{},{},{},{},{},{},{}",
                    computed_checksummed,
                    salt_hex,
                    current_score,
                    leading_ones_usize,
                    trailing_ones_usize,
                    config.platform_id,
                    config.gpu_device,
                    SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs()
                )
                    .unwrap_or_else(|e| {
                        eprintln!("Failed to write to CSV file: {}", e);
                    });

                println!("Result written to {}", config.output_file);
                println!();
                println!("Continuing search for even better solutions...");
                println!("Press Ctrl+C to stop");

                // Pause briefly to let the user see the result
                std::thread::sleep(std::time::Duration::from_secs(3));

                // Re-initialize the display
                print!("\x1B[2J"); // Clear entire screen
                print!("\x1B[1;1H"); // Move cursor to top-left
                for _ in 0..total_gpus {
                    println!("\n\n\n\n\n"); // 5 lines per GPU
                }
            }
        }

        // Update the cumulative nonce
        cumulative_nonce += WORK_SIZE as u64;

        // Update the progress display every second
        let current_time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64();
        if current_time - last_update >= 1.0 {
            let elapsed = current_time - start_time;
            let attempts_since_last = cumulative_nonce - last_attempts;
            let rate = attempts_since_last as f64 / (current_time - last_update);

            // Create a fixed position display based on the GPU device number
            let display_offset = config.gpu_device * 6; // 6 lines per GPU

            // Get terminal size
            let term_size = terminal_size::terminal_size();
            let term_width = if let Some((w, _)) = term_size {
                w.0 as usize
            } else {
                80 // Default width
            };

            // Format the output with unique identifier for each GPU
            let gpu_id = format!("P{}-D{}", config.platform_id, config.gpu_device);

            // Move cursor to the position for this GPU
            print!("\x1B[{};1H", display_offset + 1);
            print!("\x1B[K"); // Clear line
            print!("--- {} --- Runtime: {:.2}s --- Work size: {} ---",
                   gpu_id, elapsed, WORK_SIZE.separated_string());

            print!("\x1B[{};1H", display_offset + 2);
            print!("\x1B[K"); // Clear line
            print!("Hash rate: {:.2} MH/s --- Total hashes: {}",
                   rate / 1_000_000.0, cumulative_nonce.separated_string());

            print!("\x1B[{};1H", display_offset + 3);
            print!("\x1B[K"); // Clear line
            print!("Search space: 0x{:08x}xxxxxxxx --- Best score: {} 1s",
                   nonce[0], cached_best_score);

            print!("\x1B[{};1H", display_offset + 4);
            print!("\x1B[K"); // Clear line
            print!("{}", "-".repeat(term_width.min(100)));

            // Move cursor to the bottom of all GPU displays
            print!("\x1B[{};1H", (total_gpus as u32 * 6) + 1);

            // Force stdout flush
            let _ = std::io::stdout().flush();

            last_update = current_time;
            last_attempts = cumulative_nonce;
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

// Add this function to read the best score from the CSV file
fn read_current_best_score(file_path: &str) -> usize {
    // Default to 0 if file doesn't exist or can't be read
    let mut best_score = 0;

    // Try to open the file
    if let Ok(file) = File::open(file_path) {
        let reader = BufReader::new(file);

        // Skip the header line
        for line in reader.lines().skip(1) {
            if let Ok(line) = line {
                // Parse the line (format: address,salt,score,leading_ones,trailing_ones)
                let parts: Vec<&str> = line.split(',').collect();
                if parts.len() >= 3 {
                    if let Ok(score) = parts[2].parse::<usize>() {
                        if score > best_score {
                            best_score = score;
                        }
                    }
                }
            }
        }
    }

    best_score
}