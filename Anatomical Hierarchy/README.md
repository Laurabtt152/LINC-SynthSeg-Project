# Allen Brain Ontology Hierarchy Builder

This repository contains a Python script that builds an anatomical hierarchy (tree) for NextBrain segmentation. The tree is constructed by combining two key inputs:

- **AllenBrainLUT.txt** – a lookup table (LUT) mapping integer labels (voxel values) to anatomical structure names (used in NextBrain segmentation).  
  *Note:* In this file, the structure names use dashes instead of spaces, and cortical regions are prefixed with `"cortex-"`.

- **AllenBrainOntology.csv** – an ontology from the Allen Brain Atlas that contains a comprehensive hierarchy of brain structures. (This file contains many more nodes than those in the LUT.)

## Overview

The script performs the following steps:

### Normalization
- **Lowercasing:** All text is converted to lowercase.
- **Removing commas:** Commas are removed from names.
- **Replacing dashes and underscores with spaces:** Ensures consistent word separation.
- **Removing a leading "cortex "** if present: This accounts for differences between the LUT and ontology.
- **Removing trailing " left" or " right":** To avoid mismatches between hemisphere‐specific entries in the ontology and the unified LUT names.
- **Collapsing multiple spaces:** Ensures that there is only one space between words.

### Reading Data
- **Ontology:**  
  The script reads the ontology CSV file and creates:
  - A mapping of node IDs to node data.
  - A mapping of parent–child relationships.
- **LUT:**  
  The script reads the LUT text file, normalizes the names, and maps each integer label (from the “No.” column) to a normalized structure name.

### Tree Building
Using the ontology’s parent–child relationships, the script builds a full hierarchical tree starting at the root node (the node with an empty `parent_structure_id`).

### Attaching LUT Labels
The script traverses the tree and, for each node:
- Normalizes the node’s name.
- If it matches a normalized LUT entry, it attaches a `"label"` key (with the LUT’s integer value as a string).

### Pruning the Tree
Because the ontology contains many nodes that are not used in the NextBrain segmentation, the script prunes the tree according to the following logic:

- **Keep a node (and all its children)** if it has at least one descendant that matches a LUT entry. This ensures that the hierarchy is complete and that all relevant anatomical parents of a LUT-labeled leaf are preserved.
- **Remove a node completely** if it does not eventually have any leaf node with a LUT label.

> **Note:** An alternative approach would be to keep only the branch leading to the LUT-labeled child and remove other children; however, this script retains the entire branch to preserve the full anatomical hierarchy.

### Simplification
The tree is simplified so that each node in the final JSON contains only the following keys:
- `"name"` – The original structure name.
- `"acronym"` – The structure acronym (from the ontology).
- `"color"` – The color from the ontology (from `"color_hex_triplet"`, renamed to `"color"`).
- `"label"` – *(Optional)* The LUT label (if the node matches a LUT entry).
- `"children"` – A list of simplified child nodes (if any).

### Tag Comparison and Output Files
The script also collects all normalized tags (names) from:
- The ontology tree.
- The LUT.

It writes these to text files:
- **ontology_tags.txt:** A deduplicated, sorted list of normalized names from the ontology tree.
- **lut_tags.txt:** A deduplicated, sorted list of normalized names from the LUT.

Additionally, you can create a third file (e.g., `output_tags.txt`) using similar logic from the final output tree if needed.

### Node Counting
Finally, the script counts the total number of nodes in the simplified tree and prints the count to the terminal. (A warning is printed if the count is not 950—this number is expected based on your dataset.)

## Usage

Ensure you have Python 3 installed. Then run the script from the command line as follows:

```bash
./your_script.py AllenBrainLUT.txt AllenBrainOntology.csv
