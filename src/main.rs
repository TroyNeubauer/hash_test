#![feature(bigint_helper_methods)]
//! Helper simulation to see how good pointer values are when used as a hash.
//!
//! Turns out they are very good

use std::{collections::hash_map::DefaultHasher, hash::Hasher};

use ahash::AHasher;
use rand::Rng;

const NUM_SHARDS: usize = 8;
const IGNORED_LOW_BITS: u8 = 8;
const SHARD_MASK: usize = NUM_SHARDS - 1;

const MOCK_SIZE: usize = 32;
const MOCK_ELEMENTS: usize = MOCK_SIZE / 8;

fn main() {
    let mut algorithms: Vec<(_, Box<dyn HashAlgorithm>)> = vec![
        ("Pointer Hash", Box::new(PointerHash)),
        (
            "Multiply Hash",
            Box::new(MultiplyHash {
                prime: 4445950232728569541,
                state: 0,
            }),
        ), 
        (
            "Standard Hash",
            Box::new(StdHash {
                hasher: DefaultHasher::new(),
            }),
        ),
        (
            "AHash",
            Box::new(AHash {
                hasher: AHasher::new_with_keys(0, 0),
            }),
        ),
    ];
    let mut counts: Vec<_> = (0..algorithms.len()).map(|_| [0; NUM_SHARDS]).collect();

    let mut dummy_allocations: Vec<Vec<usize>> = Vec::new();
    let mut mock_objects: Vec<Vec<usize>> = Vec::new();
    for _ in 0..100 {
        mock_objects.push(Vec::with_capacity(MOCK_ELEMENTS));
    }

    //Returns simulated input hash values based on the address of a real allocation and the size of
    //a mock allocation
    let mut generate_hash_input = || {
        let mut rng = rand::thread_rng();
        if rng.gen_ratio(5, 100) {
            //Every once in a while allocate more stuff to simulate other objects being allocated
            //in the program
            dummy_allocations.push(Vec::with_capacity(rng.gen_range(32..1024)));
        }

        //Also every once in a while free some things
        if rng.gen_ratio(1, 10000) {
            dummy_allocations.clear();
        }

        //Pick a random object to `retire`
        let address = mock_objects[rng.gen_range(0..mock_objects.len())].as_ptr();

        //Add a new one and remove an old one so that we rotate through the address space
        mock_objects.push(Vec::with_capacity(MOCK_ELEMENTS));
        let _to_delete = mock_objects.swap_remove(rng.gen_range(0..mock_objects.len()));

        address as usize
    };

    let inner_runs = 1000;
    let expected_count = inner_runs as f64 / NUM_SHARDS as f64;

    let standard_deviation = |counts: &[usize]| {
        (counts
            .iter()
            .map(|c| (*c as f64 - expected_count).abs().powi(2))
            .sum::<f64>()
            / (inner_runs as f64 - 1.0))
            .sqrt()
    };

    let mut stddevs: Vec<_> = algorithms.iter().map(|_| 0.0).collect();

    let trial_count = 10000;
    for _ in 0..trial_count {
        //Set counts back to 0
        counts
            .iter_mut()
            .for_each(|i| i.iter_mut().for_each(|v| *v = 0));

        for _ in 0..inner_runs {
            let input = generate_hash_input();

            for (i, (_name, algorithm)) in algorithms.iter_mut().enumerate() {
                let shard = algorithm.shard(input);
                counts[i][shard] += 1;
            }
        }

        for (i, this_counts) in counts.iter().enumerate() {
            let stddev = standard_deviation(this_counts);
            stddevs[i] += stddev;
        }
    }

    let max_chars = 200;
    println!("Shard distributions from the last run");
    for (i, this_counts) in counts.iter().enumerate() {
        let algorithm = algorithms[i].0;
        println!("{:20}", algorithm);
        for (i, count) in this_counts.iter().enumerate() {
            print!("  i={}: {:8}  ", i, count);
            for _ in 0..(max_chars * count / inner_runs) {
                print!("X");
            }

            println!();
        }
    }
    println!();
    println!(
        "Completed {} trials of {} inner runs",
        trial_count, inner_runs
    );

    println!("Average Standard Deviation by Algorithm:");
    for (i, (algorithm, _impl)) in algorithms.iter().enumerate() {
        println!("{:20}: {}", algorithm, stddevs[i] / trial_count as f64);
    }
}

trait HashAlgorithm {
    fn hash(&mut self, input: usize) -> usize;

    fn shard(&mut self, input: usize) -> usize {
        self.hash(input) & SHARD_MASK
    }
}

struct PointerHash;
impl HashAlgorithm for PointerHash {
    fn hash(&mut self, input: usize) -> usize {
        input >> IGNORED_LOW_BITS
    }
}

struct MultiplyHash {
    prime: usize,
    state: usize,
}

impl HashAlgorithm for MultiplyHash {
    fn hash(&mut self, input: usize) -> usize {
        //XXX: Could be made simpler without the slice stuff using mem::transmute
        let mul_result = ((input ^ self.state) as u128 * self.prime as u128).to_ne_bytes();
        let mut hash = [0u8; 8];
        hash.copy_from_slice(&mul_result[0..8]);

        let mut new_state = [0u8; 8];
        new_state.copy_from_slice(&mul_result[8..16]);

        self.state = u64::from_ne_bytes(new_state) as usize;
        u64::from_ne_bytes(hash) as usize
    } 
}

struct StdHash {
    hasher: std::collections::hash_map::DefaultHasher,
}

impl HashAlgorithm for StdHash {
    fn hash(&mut self, input: usize) -> usize {
        self.hasher.write_usize(input);
        self.hasher.finish() as usize
    }
}

struct AHash {
    hasher: ahash::AHasher,
}

impl HashAlgorithm for AHash {
    fn hash(&mut self, input: usize) -> usize {
        self.hasher.write_usize(input);
        self.hasher.finish() as usize
    }
}
