//! Experimental utilities for handling variable batch sizes.

use std::collections::{BTreeSet, HashMap};

/// Find compiled batch sizes that minimize wasted compute.
///
/// This mirrors the dynamic programming routine from the Python experimental
/// module. The returned set size will be at most `num_compilations`.
pub fn optimal_physical_batch_sizes(
    batch_sizes: &[usize],
    num_compilations: usize,
) -> BTreeSet<usize> {
    if batch_sizes.is_empty() {
        return BTreeSet::new();
    }

    let mut unique = batch_sizes.to_vec();
    unique.sort_unstable();
    unique.dedup();

    let budget = num_compilations.max(1).min(unique.len());
    let last = unique.len() - 1;
    let mut cache: HashMap<(usize, usize), (Vec<usize>, usize)> = HashMap::new();

    fn solve(
        c: usize,
        p: usize,
        unique: &[usize],
        cache: &mut HashMap<(usize, usize), (Vec<usize>, usize)>,
    ) -> (Vec<usize>, usize) {
        if let Some(hit) = cache.get(&(c, p)) {
            return hit.clone();
        }

        let result = if c <= 1 || p == 0 {
            let solution = vec![unique[p]];
            let cost = unique[p] * (p + 1);
            (solution, cost)
        } else {
            let mut best_cost = usize::MAX;
            let mut best_solution: Vec<usize> = vec![unique[p]];
            for candidate in 0..p {
                let current_cost = (p - candidate) * unique[p];
                let (mut new_solution, new_cost) = solve(c - 1, candidate, unique, cache);
                let total_cost = current_cost.saturating_add(new_cost);
                if total_cost < best_cost {
                    best_cost = total_cost;
                    new_solution.insert(0, unique[p]);
                    best_solution = new_solution;
                }
            }
            (best_solution, best_cost)
        };

        cache.insert((c, p), result.clone());
        result
    }

    let (solution, _) = solve(budget, last, &unique, &mut cache);
    solution.into_iter().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn optimal_sizes_cover_max_batch() {
        let sizes = vec![8, 9, 16, 17, 32];
        let out = optimal_physical_batch_sizes(&sizes, 2);
        assert!(out.contains(&32));
        assert!(out.len() <= 2);
    }
}
