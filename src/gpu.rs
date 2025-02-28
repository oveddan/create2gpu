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

    // Read the current best score from the CSV file
    let best_score = read_current_best_score(&config.output_file);

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
    let _previous_time = 0.0;
    let start_time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64();
    let mut cumulative_nonce: u64 = 0;
    let mut rng = thread_rng();

    let term = Term::stdout();
    let mut last_update = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64();
    let mut last_attempts = 0u64;

    // Main loop
    loop {
        // Read the current best score from the CSV file
        let best_score = read_current_best_score(&config.output_file);
        
        // Update the message with the current best score
        message[54] = best_score as u8;
        
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
            .len(3)
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
            
            // Print detailed information about the solution
            println!("Found potential solution!");
            println!("Address from kernel digest: 0x{}", hex_address);
            println!("Leading 1s: {}, Trailing 1s: {}", leading_ones, trailing_ones);
            println!("Current best score: {}", best_score);
            
            // Check if this is better than our current best
            let mut update_best = false;
            
            // Convert u64 to usize for comparison
            let leading_ones_usize = leading_ones as usize;
            let trailing_ones_usize = trailing_ones as usize;
            
            // Calculate a score based on total 1s only
            let current_score = leading_ones_usize + trailing_ones_usize;
            
            if current_score > best_score {
                // This is a better solution
                update_best = true;
            }
            
            if update_best {
                // Calculate the time it took to find the solution
                let solution_time = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs_f64() - start_time;
                
                // Found a better solution
                println!("\nFound solution with {} total 1s ({} leading + {} trailing) in {:.2} seconds!", 
                         current_score, leading_ones_usize, trailing_ones_usize, solution_time);
                
                // Convert address to checksummed format
                let _checksummed_address = to_checksum_address(&hex_address);
                
                // Format the salt properly as bytes32
                let mut full_salt = [0u8; 32]; // Initialize with all zeros
                
                // Copy the solution bytes (8 bytes) to the end of the salt
                let solution_len = std::cmp::min(solution_bytes.len(), 8);
                // Place the solution bytes at the end of the salt (after 24 zero bytes)
                full_salt[32 - solution_len..32].copy_from_slice(&solution_bytes[0..solution_len]);
                
                // Format as hex
                let salt_hex = format!("0x{}", hex::encode(&full_salt));
                
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
                    writeln!(file, "address,salt,score,leading_ones,trailing_ones")
                        .unwrap_or_else(|e| {
                            eprintln!("Failed to write CSV header: {}", e);
                        });
                }
                
                // Write the data
                writeln!(
                    file,
                    "{},{},{},{},{}",
                    computed_checksummed,
                    salt_hex,
                    current_score,
                    leading_ones_usize,
                    trailing_ones_usize
                )
                .unwrap_or_else(|e| {
                    eprintln!("Failed to write to CSV file: {}", e);
                });
                
                println!("Result written to {}", config.output_file);
                
                // Continue searching for even better solutions
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
            
            // Format the output
            let _ = term.clear_line();
            print!("\r----- New Update -----\n");
            print!("total runtime: {:.2} seconds                    work size per cycle: {}\n", 
                   elapsed, WORK_SIZE.separated_string());
            print!("rate: {:.2} million attempts per second        total attempts: {}\n", 
                   rate / 1_000_000.0, cumulative_nonce.separated_string());
            print!("current search space: {:08x}xxxxxxxx          searching for prefix: {}\n", 
                   nonce[0], config.starts_with);

            // Read the current best score from the CSV file
            let best_score = read_current_best_score(&config.output_file);
            print!("best score so far: {} total 1s\n", best_score);

            // Clear the terminal after printing the update
            print!("\x1B[2J\x1B[1;1H"); // ANSI escape code to clear screen and move cursor to top-left

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
    // Default to 8 (4 leading + 4 trailing) if file doesn't exist or can't be read
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