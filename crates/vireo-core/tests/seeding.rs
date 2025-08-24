use vireo_core::sim::fields::FieldManager;

#[test]
fn seeding_runs_for_common_sizes() {
    let sizes = [(64u32, 64u32), (128, 128), (512, 512)];
    for (w, h) in sizes {
        let mut fm = FieldManager::new([w, h]);
        fm.seed_resources(42);
        
        // sanity: central cell is finite and non-negative
        let c = fm.get_resource(w/2, h/2);
        assert!(c.is_finite() && c >= 0.0);
        
        // sanity: some cells have non-zero resources
        let mut has_resources = false;
        for y in 0..h {
            for x in 0..w {
                if fm.get_resource(x, y) > 0.0 {
                    has_resources = true;
                    break;
                }
            }
            if has_resources { break; }
        }
        assert!(has_resources, "World {}x{} should have some resources", w, h);
    }
}
