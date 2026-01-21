use physics::hydro::AoSEuler1D;

fn main() {
    println!("AoS Layout Benchmark (matching C++ style)");
    println!("==========================================\n");

    for &ncells in &[100, 1000, 10000, 100000, 1000000] {
        let mut solver = AoSEuler1D::new(ncells, 0.0, 1.0, 1.4, 0.5);
        solver.set_ic(|x| if x < 0.5 { (1.0, 0.0, 1.0) } else { (0.125, 0.0, 0.1) });
        
        for _ in 0..1000 { solver.step(); }
        
        if let Some((zps, _, _)) = solver.stats() {
            println!("{} cells -> {:.2e} zone-cycles/sec", ncells, zps);
        }
    }
}
