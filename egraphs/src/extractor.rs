use egg::*;
use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use crate::config::*;

use rayon::iter::ParallelIterator;
use rayon::iter::IntoParallelRefIterator;

use rayon::ThreadPoolBuilder;

use std::thread;

pub struct Extractor<'a, L: Language, N: Analysis<L>> {
    egraph: &'a egg::EGraph<L, N>,
}

impl<'a, L, N> Extractor<'a, L, N>
where
    L: Language + Send + Sync,
    N: Analysis<L> + Send + Sync,
    N::Data: Send + Sync + Clone,
{
    pub fn new(egraph: &'a EGraph<L, N>) -> Self {
        Extractor { egraph }
    }

    pub fn get_egraph(&self) -> &EGraph<L, N> {
        self.egraph
    }
}

pub fn find_best<L, N>(
    egraph: &EGraph<L, N>,
    eclass_nodes: Vec<L>,
    eclass_id: Id,
    eclass_ids: Vec<Id>,
    dependency_map: HashMap<Id, HashSet<Id>>,
    root_id: Id,
    current_index: usize,
    current_cost: usize,
    current_nodes: Vec<L>,
    best_cost: Arc<Mutex<usize>>,
    best_expr: Arc<Mutex<RecExpr<L>>>,
) where
    L: Language + Send + Sync + Clone,
    N: Analysis<L> + Send + Sync,
    N::Data: Send + Sync + Clone,
{
    
    eclass_nodes.par_iter().for_each(|node| {
        
        let mut next_eclass_ids = eclass_ids.clone();
        let mut next_dependency_map = dependency_map.clone();
        let mut next_nodes = current_nodes.clone();
        let mut next_cost = current_cost;

        let best_cost = Arc::clone(&best_cost);
        let best_expr = Arc::clone(&best_expr);
        let node = node.clone();

        let operation = node.display_op().to_string();
        match operation.as_str() {
            "+" | "*" | "-" | "neg" => return,
            "<<" => next_cost += VEC_OP * 50,
            "Vec" => next_cost += STRUCTURE,
            "VecAdd" | "VecMinus" | "VecNeg" => next_cost += VEC_OP,
            "VecMul" => next_cost += VEC_OP * 100,
            _ => next_cost += LITERAL,
        }

        let mut dependent_ids: HashSet<Id> = HashSet::new();
        let mapped_node = node.map_children(|child_id| {
            dependent_ids.insert(child_id);
            match next_eclass_ids.iter().position(|&id| id == child_id) {
                Some(pos) => pos.into(),
                None => {
                    next_eclass_ids.push(child_id);
                    (next_eclass_ids.len() - 1).into()
                }
            }
        });

        next_dependency_map.insert(eclass_id, dependent_ids.clone());

        let mut updated_map = next_dependency_map.clone();
        for (&key, deps) in &next_dependency_map {
            if deps.contains(&eclass_id) {
                let merged_deps: HashSet<Id> = deps.union(&dependent_ids).cloned().collect();
                if merged_deps.contains(&key) {
                    return;
                }
                updated_map.insert(key, merged_deps);
            }
        }
        next_dependency_map = updated_map;
        next_nodes.push(mapped_node);

        if current_index < next_eclass_ids.len() - 1 {
                let next_eclass = &egraph[next_eclass_ids[current_index + 1]];
                find_best(
                    egraph,
                    next_eclass.nodes.clone(),
                    next_eclass.id,
                    next_eclass_ids,
                    next_dependency_map,
                    root_id,
                    current_index + 1,
                    next_cost,
                    next_nodes,
                    Arc::clone(&best_cost),
                    Arc::clone(&best_expr),
                );
            }
         else {
            let mut cost = best_cost.lock().unwrap();
            let mut expr = best_expr.lock().unwrap();
            if next_cost < *cost {
                *cost = next_cost;
                *expr = RecExpr::default();

                let total_nodes = next_nodes.len();
                for node in next_nodes.iter().rev() {
                    let new_node = node.clone().map_children(|id| (total_nodes - usize::from(id) - 1).into());
                    expr.add(new_node);
                }
            }
        }
    });
}