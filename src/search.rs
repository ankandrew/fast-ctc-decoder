use super::SearchError;
use crate::tree::{SuffixTree, ROOT_NODE};
use ndarray::{ArrayBase, Axis, Data, FoldWhile, Ix1, Ix2, Ix3, Zip};
use ndarray_stats::QuantileExt;

/// log(exp(a) + exp(b)) computed in a numerically stable way.
fn logsumexp(a: f32, b: f32) -> f32 {
    if a == f32::NEG_INFINITY {
        return b;
    }
    if b == f32::NEG_INFINITY {
        return a;
    }
    let m = if a > b { a } else { b };
    m + ((a - m).exp() + (b - m).exp()).ln()
}

/// log(p) with log(0) handled as -infinity (stdlib gives -inf already, but be explicit
/// for negative or NaN inputs which can otherwise produce NaN).
fn safe_ln(p: f32) -> f32 {
    if p > 0.0 {
        p.ln()
    } else {
        f32::NEG_INFINITY
    }
}

/// A node in the labelling tree to build from.
#[derive(Clone, Copy, Debug)]
struct SearchPoint {
    /// The node search should progress from.
    node: i32,
    /// The transition state for crf.
    state: usize,
    /// The cumulative log-probability of the labelling so far for paths without any leading
    /// blank labels.
    log_label_prob: f32,
    /// The cumulative log-probability of the labelling so far for paths with one or more
    /// leading blank labels.
    log_gap_prob: f32,
}

impl SearchPoint {
    /// The total log-probability of the labelling so far.
    ///
    /// This sums the probabilities of the paths with and without leading blank labels (in
    /// log space, that's logsumexp).
    fn log_probability(&self) -> f32 {
        logsumexp(self.log_label_prob, self.log_gap_prob)
    }
}

/// Convert probability into an ASCII encoded phred quality score between 0 and 40.
pub fn phred(prob: f32, qscale: f32, qbias: f32) -> char {
    let max = 1e-4;
    let p = if 1.0 - prob < max { max } else { 1.0 - prob };
    let q = -10.0 * p.log10() * qscale + qbias;
    std::char::from_u32(q.round() as u32 + 33).unwrap()
}

pub fn crf_beam_search_all<D: Data<Elem = f32>>(
    network_output: &ArrayBase<D, Ix3>,
    init_state: &ArrayBase<D, Ix1>,
    alphabet: &[String],
    beam_size: usize,
    beam_cut_threshold: f32,
) -> Result<Vec<(String, Vec<usize>, f32)>, SearchError> {
    assert!(!alphabet.is_empty());
    assert!(!network_output.is_empty());
    assert_eq!(network_output.ndim(), 3);
    assert_eq!(network_output.shape()[2], alphabet.len());

    let n_state = network_output.shape()[1];
    let n_base = network_output.shape()[2] - 1;

    let mut suffix_tree = SuffixTree::new(n_base);
    let mut beam = vec![SearchPoint {
        node: ROOT_NODE,
        log_label_prob: safe_ln(*init_state.max().unwrap()),
        log_gap_prob: safe_ln(init_state[0]),
        state: init_state.argmax().unwrap(),
    }];
    let mut next_beam = Vec::new();

    for (idx, probs) in network_output.axis_iter(Axis(0)).enumerate() {
        next_beam.clear();

        for &SearchPoint {
            node,
            state,
            log_label_prob,
            log_gap_prob,
        } in &beam
        {
            let pr = probs.slice(s![state, ..]);

            // add N to beam
            if pr[0] > beam_cut_threshold && pr[0] > 0.0 {
                next_beam.push(SearchPoint {
                    node: node,
                    state: state,
                    log_label_prob: f32::NEG_INFINITY,
                    log_gap_prob: logsumexp(log_label_prob, log_gap_prob) + pr[0].ln(),
                });
            }

            for (label, &pr_b) in pr.iter().skip(1).enumerate() {
                if pr_b < beam_cut_threshold || pr_b <= 0.0 {
                    continue;
                }

                let new_node_idx = suffix_tree
                    .get_child(node, label)
                    .unwrap_or_else(|| suffix_tree.add_node(node, label, idx));

                next_beam.push(SearchPoint {
                    node: new_node_idx,
                    log_gap_prob: f32::NEG_INFINITY,
                    log_label_prob: logsumexp(log_label_prob, log_gap_prob) + pr_b.ln(),
                    state: (state * n_base) % n_state + (label),
                });
            }
        }

        std::mem::swap(&mut beam, &mut next_beam);

        const DELETE_MARKER: i32 = i32::min_value();
        beam.sort_by_key(|x| x.node);
        let mut last_key = DELETE_MARKER;
        let mut last_key_pos = 0;
        for i in 0..beam.len() {
            let beam_item = beam[i];
            if beam_item.node == last_key {
                beam[last_key_pos].log_label_prob =
                    logsumexp(beam[last_key_pos].log_label_prob, beam_item.log_label_prob);
                beam[last_key_pos].log_gap_prob =
                    logsumexp(beam[last_key_pos].log_gap_prob, beam_item.log_gap_prob);
                beam[i].node = DELETE_MARKER;
            } else {
                last_key_pos = i;
                last_key = beam_item.node;
            }
        }

        beam.retain(|x| x.node != DELETE_MARKER);
        let mut has_nans = false;
        beam.sort_unstable_by(|a, b| {
            (b.log_probability())
                .partial_cmp(&(a.log_probability()))
                .unwrap_or_else(|| {
                    has_nans = true;
                    std::cmp::Ordering::Equal // don't really care
                })
        });
        if has_nans {
            return Err(SearchError::IncomparableValues);
        }
        beam.truncate(beam_size);
        if beam.is_empty() {
            // we've run out of beam (probably the threshold is too high)
            return Err(SearchError::RanOutOfBeam);
        }
        // No normalisation needed: log-space arithmetic does not underflow for
        // realistic sequence lengths.
    }

    let mut results = Vec::with_capacity(beam.len());
    for sp in &beam {
        let mut path = Vec::new();
        let mut sequence = String::new();
        if sp.node != ROOT_NODE {
            for (label, &time) in suffix_tree.iter_from(sp.node) {
                path.push(time);
                sequence.push_str(&alphabet[label + 1]);
            }
        }
        path.reverse();
        results.push((
            sequence.chars().rev().collect::<String>(),
            path,
            sp.log_probability().exp(),
        ));
    }
    Ok(results)
}

pub fn crf_beam_search<D: Data<Elem = f32>>(
    network_output: &ArrayBase<D, Ix3>,
    init_state: &ArrayBase<D, Ix1>,
    alphabet: &[String],
    beam_size: usize,
    beam_cut_threshold: f32,
) -> Result<(String, Vec<usize>), SearchError> {
    let mut results = crf_beam_search_all(
        network_output,
        init_state,
        alphabet,
        beam_size,
        beam_cut_threshold,
    )?;
    let (seq, path, _) = results.swap_remove(0);
    Ok((seq, path))
}

pub fn beam_search_all<D: Data<Elem = f32>>(
    network_output: &ArrayBase<D, Ix2>,
    alphabet: &[String],
    beam_size: usize,
    beam_cut_threshold: f32,
    collapse_repeats: bool,
) -> Result<Vec<(String, Vec<usize>, f32)>, SearchError> {
    // alphabet size minus the blank label
    let alphabet_size = alphabet.len() - 1;

    let mut suffix_tree = SuffixTree::new(alphabet_size);
    let mut beam = vec![SearchPoint {
        node: ROOT_NODE,
        state: 0,
        log_gap_prob: 0.0, // log(1.0)
        log_label_prob: f32::NEG_INFINITY, // log(0.0)
    }];
    let mut next_beam = Vec::new();

    for (idx, pr) in network_output.outer_iter().enumerate() {
        next_beam.clear();

        for &SearchPoint {
            node,
            log_label_prob,
            log_gap_prob,
            state,
        } in &beam
        {
            let tip_label = suffix_tree.label(node);

            // add N to beam
            if pr[0] > beam_cut_threshold && pr[0] > 0.0 {
                next_beam.push(SearchPoint {
                    node: node,
                    state: state,
                    log_label_prob: f32::NEG_INFINITY,
                    log_gap_prob: logsumexp(log_label_prob, log_gap_prob) + pr[0].ln(),
                });
            }

            for (label, &pr_b) in pr.iter().skip(1).enumerate() {
                if pr_b < beam_cut_threshold || pr_b <= 0.0 {
                    continue;
                }
                let log_pr_b = pr_b.ln();

                if collapse_repeats && Some(label) == tip_label {
                    next_beam.push(SearchPoint {
                        node: node,
                        log_label_prob: log_label_prob + log_pr_b,
                        log_gap_prob: f32::NEG_INFINITY,
                        state: state,
                    });
                    let new_node_idx = suffix_tree.get_child(node, label).or_else(|| {
                        if log_gap_prob > f32::NEG_INFINITY {
                            Some(suffix_tree.add_node(node, label, idx))
                        } else {
                            None
                        }
                    });

                    if let Some(idx) = new_node_idx {
                        next_beam.push(SearchPoint {
                            node: idx,
                            state: state,
                            log_label_prob: log_gap_prob + log_pr_b,
                            log_gap_prob: f32::NEG_INFINITY,
                        });
                    }
                } else {
                    let new_node_idx = suffix_tree
                        .get_child(node, label)
                        .unwrap_or_else(|| suffix_tree.add_node(node, label, idx));

                    next_beam.push(SearchPoint {
                        node: new_node_idx,
                        state: state,
                        log_label_prob: logsumexp(log_label_prob, log_gap_prob) + log_pr_b,
                        log_gap_prob: f32::NEG_INFINITY,
                    });
                }
            }
        }
        std::mem::swap(&mut beam, &mut next_beam);

        const DELETE_MARKER: i32 = i32::min_value();
        beam.sort_by_key(|x| x.node);
        let mut last_key = DELETE_MARKER;
        let mut last_key_pos = 0;
        for i in 0..beam.len() {
            let beam_item = beam[i];
            if beam_item.node == last_key {
                beam[last_key_pos].log_label_prob =
                    logsumexp(beam[last_key_pos].log_label_prob, beam_item.log_label_prob);
                beam[last_key_pos].log_gap_prob =
                    logsumexp(beam[last_key_pos].log_gap_prob, beam_item.log_gap_prob);
                beam[i].node = DELETE_MARKER;
            } else {
                last_key_pos = i;
                last_key = beam_item.node;
            }
        }

        beam.retain(|x| x.node != DELETE_MARKER);
        let mut has_nans = false;
        beam.sort_unstable_by(|a, b| {
            (b.log_probability())
                .partial_cmp(&(a.log_probability()))
                .unwrap_or_else(|| {
                    has_nans = true;
                    std::cmp::Ordering::Equal // don't really care
                })
        });
        if has_nans {
            return Err(SearchError::IncomparableValues);
        }
        beam.truncate(beam_size);
        if beam.is_empty() {
            // we've run out of beam (probably the threshold is too high)
            return Err(SearchError::RanOutOfBeam);
        }
        // No normalisation needed: log-space arithmetic does not underflow for
        // realistic sequence lengths.
    }

    let mut results = Vec::with_capacity(beam.len());
    for sp in &beam {
        let mut path = Vec::new();
        let mut tokens: Vec<&str> = Vec::new();
        if sp.node != ROOT_NODE {
            for (label, &time) in suffix_tree.iter_from(sp.node) {
                path.push(time);
                tokens.push(&alphabet[label + 1]);
            }
        }
        path.reverse();
        tokens.reverse();
        results.push((tokens.concat(), path, sp.log_probability().exp()));
    }
    Ok(results)
}

pub fn beam_search<D: Data<Elem = f32>>(
    network_output: &ArrayBase<D, Ix2>,
    alphabet: &[String],
    beam_size: usize,
    beam_cut_threshold: f32,
    collapse_repeats: bool,
) -> Result<(String, Vec<usize>), SearchError> {
    let mut results = beam_search_all(
        network_output,
        alphabet,
        beam_size,
        beam_cut_threshold,
        collapse_repeats,
    )?;
    let (seq, path, _) = results.swap_remove(0);
    Ok((seq, path))
}

fn find_max(
    acc: Option<(usize, f32)>,
    elem_idx: usize,
    elem_val: &f32,
) -> FoldWhile<Option<(usize, f32)>> {
    match acc {
        Some((_, val)) => {
            if *elem_val > val {
                FoldWhile::Continue(Some((elem_idx, *elem_val)))
            } else {
                FoldWhile::Continue(acc)
            }
        }
        None => FoldWhile::Continue(Some((elem_idx, *elem_val))),
    }
}

pub fn viterbi_search<D: Data<Elem = f32>>(
    network_output: &ArrayBase<D, Ix2>,
    alphabet: &[String],
    qstring: bool,
    qscale: f32,
    qbias: f32,
    collapse_repeats: bool,
) -> Result<(String, Vec<usize>), SearchError> {
    assert!(!alphabet.is_empty());
    assert!(!network_output.is_empty());
    assert_eq!(network_output.ndim(), 2);
    assert_eq!(alphabet.len(), network_output.shape()[1]);

    let mut path = Vec::new();
    let mut quality = String::new();
    let mut sequence = String::new();

    let mut last_label = None;
    let mut label_prob_count = 0;
    let mut label_prob_total = 0.0;

    for (idx, pr) in network_output.outer_iter().enumerate() {
        let (label, prob) = Zip::indexed(pr)
            .fold_while(None, find_max)
            .into_inner()
            .unwrap(); // only an empty network_output could give us None

        if label != 0 && (!collapse_repeats || last_label != Some(label)) {
            if label_prob_count > 0 {
                quality.push(phred(
                    label_prob_total / (label_prob_count as f32),
                    qscale,
                    qbias,
                ));
                label_prob_total = 0.0;
                label_prob_count = 0;
            }

            sequence.push_str(&alphabet[label]);
            path.push(idx);
        }

        if label != 0 {
            label_prob_total += prob;
            label_prob_count += 1;
        }

        last_label = Some(label);
    }

    if label_prob_count > 0 {
        quality.push(phred(
            label_prob_total / (label_prob_count as f32),
            qscale,
            qbias,
        ));
    }

    if qstring {
        sequence.push_str(&quality);
    }

    Ok((sequence, path))
}

pub fn crf_greedy_search<D: Data<Elem = f32>>(
    network_output: &ArrayBase<D, Ix3>,
    init_state: &ArrayBase<D, Ix1>,
    alphabet: &[String],
    qstring: bool,
    qscale: f32,
    qbias: f32,
) -> Result<(String, Vec<usize>), SearchError> {
    assert!(!alphabet.is_empty());
    assert!(!network_output.is_empty());
    assert_eq!(network_output.ndim(), 3);
    assert_eq!(network_output.shape()[2], alphabet.len());

    let n_state = network_output.shape()[1] as i32;
    let n_base = network_output.shape()[2] as i32 - 1;

    let mut path = Vec::new();
    let mut quality = String::new();
    let mut sequence = String::new();
    let mut state = init_state.argmax().unwrap() as i32;

    for (idx, pr) in network_output.axis_iter(Axis(0)).enumerate() {
        let label = pr.slice(s![state, ..]).argmax().unwrap();

        if label > 0 {
            path.push(idx);
            sequence.push_str(&alphabet[label]);
            let prob = *pr.slice(s![state, ..]).max().unwrap();
            quality.push(phred(prob, qscale, qbias));
            state = (state * n_base) % n_state + (label as i32 - 1);
        }
    }

    if qstring {
        sequence.push_str(&quality);
    }

    Ok((sequence, path))
}

#[cfg(test)]
mod tests {
    use super::*;
    //use test::Bencher;

    #[test]
    fn crf_test_greedy() {
        let alphabet = vec![
            String::from("N"),
            String::from("A"),
            String::from("C"),
            String::from("G"),
            String::from("T"),
        ];

        let network_output = array![
            [
                [0f32, 0.000, 0.000, 0.000, 0.000],
                [0f32, 0.000, 0.000, 0.000, 0.000],
                [1f32, 0.000, 0.000, 0.000, 0.000], // N 2
                [0f32, 0.000, 0.000, 0.000, 0.000],
            ],
            [
                [0f32, 0.000, 0.000, 0.000, 0.000],
                [0f32, 0.000, 0.000, 0.000, 0.000],
                [0f32, 0.000, 0.900, 0.000, 0.000], // C 2
                [0f32, 0.000, 0.000, 0.000, 0.000],
            ],
            [
                [0f32, 0.000, 0.000, 0.000, 0.000],
                [0f32, 0.000, 0.000, 0.000, 0.700], // T 1
                [0f32, 0.000, 0.000, 0.000, 0.000],
                [0f32, 0.000, 0.000, 0.000, 0.000],
            ],
            [
                [0f32, 0.000, 0.000, 0.000, 0.000],
                [0f32, 0.000, 0.000, 0.000, 0.000],
                [0f32, 0.000, 0.000, 0.000, 0.000],
                [1f32, 0.000, 0.000, 0.000, 0.000], // N 3
            ],
            [
                [0f32, 0.000, 0.000, 0.000, 0.000],
                [0f32, 0.000, 0.000, 0.000, 0.000],
                [0f32, 0.000, 0.000, 0.000, 0.000],
                [0f32, 0.990, 0.000, 0.000, 0.000], // A 3
            ],
            [
                [0f32, 0.900, 0.000, 0.000, 0.000], // A 0
                [0f32, 0.000, 0.000, 0.000, 0.000],
                [0f32, 0.000, 0.000, 0.000, 0.000],
                [0f32, 0.000, 0.000, 0.000, 0.000],
            ],
            [
                [0f32, 0.000, 0.000, 0.999, 0.000], // G 0
                [0f32, 0.000, 0.000, 0.000, 0.000],
                [0f32, 0.000, 0.000, 0.000, 0.000],
                [0f32, 0.000, 0.000, 0.000, 0.000],
            ],
        ];
        let init = array![0f32, 0., 1., 0., 0.];
        let (sequence, path) =
            crf_greedy_search(&network_output, &init, &alphabet, false, 1.0, 0.0).unwrap();

        assert_eq!(sequence, "CTAAG");
        assert_eq!(path, vec![1, 2, 4, 5, 6]);

        let (sequence, path) =
            crf_greedy_search(&network_output, &init, &alphabet, true, 1.0, 0.0).unwrap();

        assert_eq!(sequence, "CTAAG+&5+?");
        assert_eq!(path, vec![1, 2, 4, 5, 6]);

        let beam_size = 5;
        let beam_cut_threshold = 0.01;
        let (sequence, path) = crf_beam_search(
            &network_output,
            &init,
            &alphabet,
            beam_size,
            beam_cut_threshold,
        )
        .unwrap();

        assert_eq!(sequence, "CTAAG");
        assert_eq!(path, vec![1, 2, 4, 5, 6]);
    }

    #[test]
    fn test_phred_scores() {
        let qbias = 0.0;
        let qscale = 1.0;
        assert_eq!('!', phred(0.0, qscale, qbias));
        assert_eq!('$', phred(0.5, qscale, qbias));
        assert_eq!('+', phred(1.0 - 1e-1, qscale, qbias));
        assert_eq!('5', phred(1.0 - 1e-2, qscale, qbias));
        assert_eq!('?', phred(1.0 - 1e-3, qscale, qbias));
        assert_eq!('I', phred(1.0 - 1e-4, qscale, qbias));
        assert_eq!('I', phred(1.0 - 1e-5, qscale, qbias));
        assert_eq!('I', phred(1.0 - 1e-6, qscale, qbias));
        assert_eq!('I', phred(1.0, qscale, qbias));
    }

    #[test]
    fn test_viterbi() {
        let qbias = 0.0;
        let qscale = 1.0;
        let alphabet = vec![String::from("N"), String::from("A"), String::from("G")];
        let network_output = array![
            [0.0f32, 0.4, 0.6], // G
            [0.0f32, 0.3, 0.7], // G
            [0.3f32, 0.3, 0.4], // G
            [0.4f32, 0.3, 0.3], // N
            [0.4f32, 0.3, 0.3], // N
            [0.3f32, 0.3, 0.4], // G
            [0.1f32, 0.4, 0.5], // G
            [0.1f32, 0.5, 0.4], // A
            [0.8f32, 0.1, 0.1], // N
            [0.1f32, 0.1, 0.8], // G
        ];

        let (seq, starts) =
            viterbi_search(&network_output, &alphabet, false, qscale, qbias, true).unwrap();
        assert_eq!(seq, "GGAG");
        assert_eq!(starts, vec![0, 5, 7, 9]);

        let (seq, starts) =
            viterbi_search(&network_output, &alphabet, true, qscale, qbias, true).unwrap();
        assert_eq!(seq, "GGAG%$$(");
        assert_eq!(starts, vec![0, 5, 7, 9]);
    }

    #[test]
    fn test_viterbi_blank_bounds() {
        let qbias = 0.0;
        let qscale = 1.0;
        let alphabet = vec![String::from("N"), String::from("A"), String::from("G")];
        let network_output = array![
            [0.6f32, 0.2, 0.2], // N
            [0.6f32, 0.2, 0.2], // N
            [0.0f32, 0.4, 0.6], // G
            [0.0f32, 0.3, 0.7], // G
            [0.3f32, 0.3, 0.4], // G
            [0.4f32, 0.3, 0.3], // N
            [0.4f32, 0.3, 0.3], // N
            [0.3f32, 0.3, 0.4], // G
            [0.1f32, 0.4, 0.5], // G
            [0.1f32, 0.5, 0.4], // A
            [0.8f32, 0.1, 0.1], // N
            [0.1f32, 0.1, 0.8], // G
            [0.4f32, 0.3, 0.3], // N
        ];
        let (seq, starts) =
            viterbi_search(&network_output, &alphabet, false, qscale, qbias, true).unwrap();
        assert_eq!(seq, "GGAG");
        assert_eq!(starts, vec![2, 7, 9, 11]);

        let (seq, starts) =
            viterbi_search(&network_output, &alphabet, true, qscale, qbias, true).unwrap();
        assert_eq!(seq, "GGAG%$$(");
        assert_eq!(starts, vec![2, 7, 9, 11]);

        let (seq, starts) =
            viterbi_search(&network_output, &alphabet, false, qscale, qbias, false).unwrap();
        assert_eq!(seq, "GGGGGAG");
        assert_eq!(starts, vec![2, 3, 4, 7, 8, 9, 11]);

        let (seq, starts) =
            viterbi_search(&network_output, &alphabet, true, qscale, qbias, false).unwrap();
        assert_eq!(seq, "GGGGGAG%&##$$(");
        assert_eq!(starts, vec![2, 3, 4, 7, 8, 9, 11]);

        let (seq, _starts) = beam_search(&network_output, &alphabet, 5, 0.0, true).unwrap();
        assert_eq!(seq, "GAGAG");

        let (seq, _starts) = beam_search(&network_output, &alphabet, 5, 0.0, false).unwrap();
        assert_eq!(seq, "GGGAGAG");
    }

    #[test]
    fn test_beam_search_all() {
        let alphabet = vec![String::from("N"), String::from("A"), String::from("G")];
        let network_output = array![
            [0.6f32, 0.2, 0.2], // N
            [0.6f32, 0.2, 0.2], // N
            [0.0f32, 0.4, 0.6], // G
            [0.0f32, 0.3, 0.7], // G
            [0.3f32, 0.3, 0.4], // G
            [0.4f32, 0.3, 0.3], // N
            [0.4f32, 0.3, 0.3], // N
            [0.3f32, 0.3, 0.4], // G
            [0.1f32, 0.4, 0.5], // G
            [0.1f32, 0.5, 0.4], // A
            [0.8f32, 0.1, 0.1], // N
            [0.1f32, 0.1, 0.8], // G
            [0.4f32, 0.3, 0.3], // N
        ];

        let beam_size = 5;
        let results = beam_search_all(&network_output, &alphabet, beam_size, 0.0, true).unwrap();

        // top result must match the single-result API
        let (top_seq, top_path) = beam_search(&network_output, &alphabet, beam_size, 0.0, true).unwrap();
        assert_eq!(results[0].0, top_seq);
        assert_eq!(results[0].1, top_path);

        // we should get up to beam_size candidates
        assert!(!results.is_empty());
        assert!(results.len() <= beam_size);

        // each probability is a real probability in (0, 1]
        for (_, _, p) in &results {
            assert!(*p > 0.0 && *p <= 1.0, "prob out of range: {}", p);
        }

        // results should be sorted by probability descending
        for w in results.windows(2) {
            assert!(w[0].2 >= w[1].2);
        }

        // sum over the kept beam cannot exceed 1 (it's a partial sum over disjoint labellings)
        let total: f32 = results.iter().map(|(_, _, p)| *p).sum();
        assert!(total <= 1.0 + 1e-5, "beam total exceeds 1: {}", total);

        // all returned sequences should be distinct (the merge step deduplicates by labelling)
        let seqs: Vec<&String> = results.iter().map(|(s, _, _)| s).collect();
        let mut unique = seqs.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(seqs.len(), unique.len());
    }

    #[test]
    fn test_beam_search_all_known_probabilities() {
        // Network where there is essentially one valid labelling: "AB", with very high
        // confidence at each emission. The top candidate's probability should be close to 1.
        let alphabet = vec![String::from("N"), String::from("A"), String::from("B")];
        let network_output = array![
            [0.01f32, 0.98, 0.01], // strongly A
            [0.99f32, 0.005, 0.005], // strongly blank
            [0.01f32, 0.01, 0.98], // strongly B
            [0.99f32, 0.005, 0.005], // strongly blank
        ];

        let results = beam_search_all(&network_output, &alphabet, 5, 0.0, true).unwrap();
        assert_eq!(results[0].0, "AB");
        // Best path probability is roughly 0.98 * 0.99 * 0.98 * 0.99 ≈ 0.94. Allow a margin
        // because additional paths through merged states contribute too.
        assert!(results[0].2 > 0.9, "top prob should be high, got {}", results[0].2);
        assert!(results[0].2 <= 1.0);
    }

    #[test]
    fn test_logsumexp() {
        // basic identity: logsumexp(log a, log b) == log(a + b)
        let lse = logsumexp(0.5f32.ln(), 0.25f32.ln());
        assert!((lse - 0.75f32.ln()).abs() < 1e-5);

        // -inf is the identity element
        assert_eq!(logsumexp(f32::NEG_INFINITY, 1.5), 1.5);
        assert_eq!(logsumexp(1.5, f32::NEG_INFINITY), 1.5);
        assert_eq!(logsumexp(f32::NEG_INFINITY, f32::NEG_INFINITY), f32::NEG_INFINITY);
    }

    /*
    // This one is all blanks, and so returns no sequence (which means we're not benchmarking the
    // construction of the results).
    #[bench]
    fn benchmark_trivial_viterbi(b: &mut Bencher) {
        use ndarray::Array2;
        let qbias = 0.0;
        let qscale = 1.0;
        let alphabet = vec![String::from("N"), String::from("A"), String::from("G")];
        let network_output = Array2::from_shape_fn((1000, 3), |p| match p {
            (_, 0) => 1.0f32,
            (_, _) => 0.0f32,
        });
        b.iter(|| viterbi_search(&network_output, &alphabet, false, qscale, qbias, true));
    }

    // This one changes label at every data point, so result contruction has the maximum possible
    // impact on run time.
    #[bench]
    fn benchmark_unstable_viterbi(b: &mut Bencher) {
        use ndarray::Array2;
        let qbias = 0.0;
        let qscale = 1.0;
        let alphabet = vec![String::from("N"), String::from("A"), String::from("G")];
        let network_output = Array2::from_shape_fn((1000, 3), |p| match p {
            (n, 1) if n % 2 == 0 => 0.0f32,
            (n, 1) if n % 2 != 0 => 1.0f32,
            (n, 2) if n % 2 == 0 => 1.0f32,
            (n, 2) if n % 2 != 0 => 0.0f32,
            _ => 0.0f32,
        });
        b.iter(|| viterbi_search(&network_output, &alphabet, false, qscale, qbias, true));
    }
     */
}
