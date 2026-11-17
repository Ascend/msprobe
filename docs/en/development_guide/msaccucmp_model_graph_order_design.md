# Displaying OM-to-OM Comparison Results in Model Graph Order

## Requirements and Scope

As described in [ISSUE #535](https://gitcode.com/Ascend/msprobe/issues/535), in multi-stream scenarios, OMs of different versions of the same network may produce dump timestamps in an order that differs from the model graph order. Displaying results by internal `Index` makes it hard to walk the graph and locate the first operator with abnormal precision.

The new `-mgf/--model_graph_file` option reads the graph JSON exported from the target OM and changes only the display order of results in direct same-name OM-to-OM comparison. The graph JSON is obtained with `atc --mode=1 --om=<target.om> --json=<target.json>`. For an example of the outer entry point, see the [User Guide](../user_guide/accuracy_compare/offline_data_compare_instruct.md).

`-mgf` is mutually exclusive with the outer `-fr/-qfr/-cfr` options (the inner `-f/-q/-cf` options) so that the existing fusion matching semantics are not changed. In detail/mapping modes, the tool indicates that the graph ordering parameter is ignored and skips validation and parsing of this file, while the mutual exclusion constraint still applies.

## Graph Order Convention

The display order uses the array positions in the target graph JSON and is handled separately from the global ID ordering of the existing fusion rule parser. The lightweight parser does not invoke fusion matching logic, nor does it read the node `id` or `task_index_op_name`.

1. Iterate over the graphs by their index in the `graph` array, and within each graph iterate over the nodes by their index in the `op` array. Here, "op index" means the array position, not the node `id` or the original `Index` in the CSV.
2. Names are normalized according to the mode of the target dump. `/` and `.` are replaced with `_`. Normal mode reuses `dump_parse.dump.handle_op_name(name, '')` to process the `_lxsliceX` and `_sgt_graph_N` fields, while FFTS mode retains these fields in the same way as `DumpInfo`. Names in the target dump have already been processed this way and are not trimmed again. A name must be a non-empty string. Empty or duplicate names after normalization raise an error so that different nodes are not mapped to the same name.
3. The node `id` does not participate in display ordering and is not validated. Missing, duplicate, or inverted IDs do not change the display order.
4. All nodes of one graph are arranged contiguously before the next graph; interleaving across graphs is not allowed. Duplicate names across graphs, or duplicate names after normalization, still raise an error because the current dump names are insufficient to distinguish these nodes.
5. A continuous global graph sequence is generated for the graph nodes that are traversed, forming the mapping `normalized name -> graph sequence`. Nodes without dump data do not produce CSV result rows.

This is the display order of graphs and nodes in the JSON. It does not rebuild the dependency topology, does not expand nodes by subgraph invocation position, and does not infer the actual timing order of multi-stream execution. Output nodes of the main graph retain their array positions even when they appear before subgraphs.

## Integration Flow

```text
The existing flow parses the dumps on both sides and matches operators by name
    -> Preserve the original FusionOp.op_id and the task dispatch order
    -> Build the mapping from the original Index to the display sequence number based on target-side operator names
    -> The existing flow computes and writes out the results
    -> Stably sort the final result file by the display mapping
    -> Renumber the first CSV column per result row from 0 to N-1, keeping the original column structure
```

`CompareRule.make_index_to_graph_sequence()` is called after `parse_fusion_rule()`. If graph ordering is enabled but the matching information has not been initialized, a `CompareError` is raised with a hint about the call order. If `-mgf` is not specified, the method returns `None` directly.

The final sort keys are:

- `-mgf` not specified: the original `Index`.
- Graph sequence matched: `(0, display sequence number, original Index)`.
- Graph sequence not matched: `(1, original Index, original Index)`.

Multiple Tensor results with the same original `Index` are sorted stably and keep their original relative order, but each CSV result row receives its own new `Index`. The new `Index` is the display row number, not a graph node ID, an in-graph op index, or the internal `FusionOp.op_id`. It is continuous across the whole file and is not restarted from 0 between graphs.

When some operators are unmatched, a warning is generated and they are placed at the end. When none are matched, a parameter error is raised and processing stops before computing and writing out results, with a hint to check whether `-mgf` belongs to the target OM and whether the names in the dump and the graph have the expected format. Unmatched items cannot be assigned to a graph and are not inserted into the results of any matched graph. Invalid JSON structures and invalid names are reported with explicit errors.

The sorting entry point distinguishes three states: `None` means the original `Index` ordering is retained and no renumbering occurs; an empty dictionary means there is no valid graph sequence, so an error is raised and the file is not rewritten; a non-empty dictionary is handled by the standalone graph-order sorting function, which renumbers the CSV row by row while text output keeps the original numbering. The normal comparison path already rejects the all-unmatched case when building the mapping, and the empty-dictionary check at the output entry point prevents the original numbering from being rewritten when the function is called directly.

The CSV keeps its original column structure; only the first column, `Index`, is updated, while the positions and values of the other columns remain unchanged. The sorting function is called once on the result file containing the original `Index` after comparison, and the original `Index` is used only in memory to look up the graph sequence and to sort. When `-mgf` is not specified, no renumbering occurs; legacy text results are not renumbered.

Advisor reads the CSV after sorting, so the `Index` it outputs corresponds to the new display row number, while logs still use the internal original numbering. `--select/--range` still select by the existing `OpSequence` and do not accept the new CSV row numbers as a substitute.

The outer CLI only passes the `-mgf` parameter through and keeps the existing subprocess failure handling: `call_msaccucmp()` treats 0 and 2 as success and records an error log for other return codes; `compare_offline_data_mode()` neither raises new exceptions nor changes the outer exit behavior because of a non-zero subprocess return code. Existing exception handling for missing scripts and process launch failures is retained. Unified exit codes and CLI exception presentation will be reviewed separately.

## Acceptance and Testing

- Parser: when IDs are inverted within a graph, cross or duplicate across graphs, missing, or contain irrelevant values, the nodes are still arranged by graph/op array position. The structure, the normal/FFTS name rules, and same-graph/cross-graph conflicts after normalization of special fields are checked.
- Multiple graphs: the main graph and its subgraphs are arranged contiguously by array position, and the output of the main graph does not move across graphs. Duplicate names across graphs raise an explicit error.
- Mapping and output: internal op_ids and the matching relationships are unchanged; the CSV column structure is unchanged and the new `Index` is strictly 0 to N-1; the multiset of complete rows for all columns other than `Index` is identical; multiple Tensor rows of the same operator are stable; comparison is blocked when nothing matches; and when an empty mapping is passed directly, the original CSV/TXT files remain byte-for-byte unchanged.
- Entry point: behavior remains compatible when the parameter is absent, fusion parameters are mutually exclusive, and detail/mapping modes ignore the graph file. With or without `-mgf`, the existing subprocess return-code handling and outer exit behavior are preserved.
- Real data: verify the output with real dumps of two OMs and an unmodified target graph JSON. Historical on-device results must state the corresponding commit and cannot substitute for validating later changes.

Parser and sorting tests do not depend on an NPU. On-device end-to-end validation, whl package validation, and the complete repository CI are independent verification steps, and their results should be recorded separately.
