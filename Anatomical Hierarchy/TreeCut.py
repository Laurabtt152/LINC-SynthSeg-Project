#!/usr/bin/env python3
import sys
import csv
import json
import re

def normalize_name(name):
    """
    Normalize a name by:
      - Lowercasing.
      - Removing commas.
      - Replacing dashes and underscores with spaces.
      - Removing a leading "cortex " if present.
      - Removing any standalone "left" or "right".
      - Collapsing multiple spaces.
    """
    name = name.lower().strip()
    # Remove commas.
    name = name.replace(',', '')
    # Replace dashes and underscores with spaces.
    name = name.replace('-', ' ').replace('_', ' ')
    # Remove a leading "cortex " if present.
    if name.startswith("cortex "):
        name = name[len("cortex "):]
    # Remove any standalone "left" or "right" (regardless of position).
    name = re.sub(r'\b(left|right)\b', '', name)
    # Collapse multiple spaces.
    name = re.sub(r'\s+', ' ', name).strip()
    return name

def read_ontology(ontology_path):
    """
    Reads the AllenBrainOntology CSV file and returns:
      - nodes: a dictionary mapping node IDs (int) to their full node (as a dict).
      - children_map: a dictionary mapping a parent node ID (int) to a list of child node IDs.
    """
    nodes = {}
    children_map = {}
    with open(ontology_path, "r", encoding="utf-8") as f:
        header_line = f.readline().strip()
        # Remove BOM if present.
        if header_line.startswith('\ufeff'):
            header_line = header_line[1:]
        fieldnames = header_line.split(',')
        reader = csv.DictReader(f, fieldnames=fieldnames, delimiter=',')
        for row in reader:
            try:
                node_id = int(row["id"].strip())
            except KeyError:
                raise KeyError("Column 'id' not found in the CSV header. Found: {}".format(fieldnames))
            parent_raw = row["parent_structure_id"].strip() if row["parent_structure_id"] else ""
            parent_id = int(parent_raw) if parent_raw else None

            node = {
                "id": row["id"].strip(),
                "atlas_id": row["atlas_id"].strip(),
                "name": row["name"].strip(),
                "acronym": row["acronym"].strip(),
                "st_level": row["st_level"].strip(),
                "ontology_id": row["ontology_id"].strip(),
                "hemisphere_id": row["hemisphere_id"].strip(),
                "weight": row["weight"].strip(),
                "parent_structure_id": row["parent_structure_id"].strip(),
                "depth": row["depth"].strip(),
                "graph_id": row["graph_id"].strip(),
                "graph_order": row["graph_order"].strip(),
                "structure_id_path": row["structure_id_path"].strip(),
                "color_hex_triplet": row["color_hex_triplet"].strip(),
                "neuro_name_structure_id": row["neuro_name_structure_id"].strip(),
                "neuro_name_structure_id_path": row["neuro_name_structure_id_path"].strip(),
                "failed": row["failed"].strip(),
                "sphinx_id": row["sphinx_id"].strip(),
                "structure_name_facet": row["structure_name_facet"].strip(),
                "failed_facet": row["failed_facet"].strip(),
                "safe_name": row["safe_name"].strip()
            }
            nodes[node_id] = node
            if parent_id is not None:
                children_map.setdefault(parent_id, []).append(node_id)
    return nodes, children_map

def read_lut(lut_path):
    """
    Reads the AllenBrainLUT.txt file which is whitespace-separated.
    The first column is the integer label (the "No." column) and the second column is the name.
    In the LUT the names use dashes instead of spaces.
    We normalize the LUT names and return a dictionary mapping label (int) to normalized name (string).
    """
    lut = {}
    with open(lut_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            try:
                label = int(parts[0])
            except ValueError:
                continue  # skip header or invalid lines
            if len(parts) < 2:
                continue
            raw_name = parts[1]
            normalized = normalize_name(raw_name)
            lut[label] = normalized
    return lut

def build_tree(nodes, children_map, root_id):
    """
    Recursively builds a tree (as a nested dictionary) from the nodes and children_map,
    starting from the given root node.
    """
    if root_id not in nodes:
        return None
    node = nodes[root_id]
    tree = dict(node)  # shallow copy of node's properties
    if root_id in children_map:
        children = []
        for child_id in children_map[root_id]:
            subtree = build_tree(nodes, children_map, child_id)
            if subtree is not None:
                children.append(subtree)
        if children:
            tree["children"] = children
    return tree

def attach_labels(tree, lut):
    """
    Recursively traverse the tree. For each node, if its normalized name matches a LUT entry,
    attach a "label" key (as a string) with the LUT label.
    """
    if tree is None:
        return
    normalized_node_name = normalize_name(tree.get("name", ""))
    for label, lut_name in lut.items():
        if normalized_node_name == lut_name:
            tree["label"] = str(label)
            break
    if "children" in tree:
        for child in tree["children"]:
            attach_labels(child, lut)

def prune_tree(node):
    """
    Recursively prune the tree, removing any branch that does not eventually contain
    at least one node with a "label". Returns the pruned node or None if no descendant has a label.
    """
    if node is None:
        return None
    if "children" not in node or not node["children"]:
        return node if "label" in node else None
    pruned_children = []
    for child in node["children"]:
        pruned_child = prune_tree(child)
        if pruned_child is not None:
            pruned_children.append(pruned_child)
    if pruned_children:
        node["children"] = pruned_children
        return node
    else:
        return node if "label" in node else None

def simplify_tree(node):
    """
    Recursively simplify the tree so that each node is represented only with:
      - "name": the node's name.
      - "acronym": the node's acronym.
      - "color": the node's color (from "color_hex_triplet").
      - "label": if present.
      - "children": a list of simplified child nodes (if any).
    """
    simple = {
        "name": node.get("name", ""),
        "acronym": node.get("acronym", ""),
        "color": node.get("color_hex_triplet", "")
    }
    if "label" in node:
        simple["label"] = node["label"]
    if "children" in node and node["children"]:
        simple["children"] = [simplify_tree(child) for child in node["children"]]
    return simple

def count_nodes(node):
    """
    Recursively count the total number of nodes in the tree.
    """
    if node is None:
        return 0
    count = 1
    if "children" in node:
        for child in node["children"]:
            count += count_nodes(child)
    return count

def get_all_tags(node, tags=None):
    """
    Recursively collect normalized names (tags) from every node in the tree.
    """
    if tags is None:
        tags = set()
    if "name" in node:
        tags.add(normalize_name(node["name"]))
    if "children" in node:
        for child in node["children"]:
            get_all_tags(child, tags)
    return tags

def write_tags(tags, filename):
    """
    Write the sorted list of tags (one per line) to the given filename.
    """
    with open(filename, "w", encoding="utf-8") as f:
        for tag in sorted(tags):
            f.write(tag + "\n")

def main():
    if len(sys.argv) < 3:
        print("Usage: {} <AllenBrainLUT.txt> <AllenBrainOntology.csv>".format(sys.argv[0]))
        sys.exit(1)
    
    lut_path = sys.argv[1]
    ontology_path = sys.argv[2]

    # Read LUT and ontology.
    lut = read_lut(lut_path)
    nodes, children_map = read_ontology(ontology_path)

    # Find the root node (node with an empty parent_structure_id)
    root_id = None
    for node_id, node in nodes.items():
        if not node["parent_structure_id"]:
            root_id = node_id
            break
    if root_id is None:
        print("No root node found in the ontology.")
        sys.exit(1)
    
    # Build the full ontology tree.
    full_tree = build_tree(nodes, children_map, root_id)
    # Attach LUT labels (using normalized names) to any node that matches.
    attach_labels(full_tree, lut)
    
    # --- Compare tags between the ontology tree and the LUT ---
    ontology_tags = get_all_tags(full_tree)
    lut_tags = set(lut.values())
    
    # Write tag lists to text files.
    write_tags(ontology_tags, "ontology_tags.txt")
    write_tags(lut_tags, "lut_tags.txt")
    
    # Also create a third file with tags from the output tree.
    # (We will compute these after pruning and simplifying.)
    
    # Compute differences.
    missing_in_ontology = sorted(lut_tags - ontology_tags)
    missing_in_lut = sorted(ontology_tags - lut_tags)
    
    print("=== TAG COMPARISON ===")
    print("Total ontology tags:", len(ontology_tags))
    print("Total LUT tags:", len(lut_tags))
    print("\nTags in LUT but not in ontology:")
    for tag in missing_in_ontology:
        print("  ", tag)
    print("\nTags in ontology but not in LUT:")
    for tag in missing_in_lut:
        print("  ", tag)
    print("======================\n")
    print("Ontology tags saved to 'ontology_tags.txt'")
    print("LUT tags saved to 'lut_tags.txt'")
    # --- End tag comparison ---

    # Prune branches that do not eventually contain any LUT-labeled node.
    pruned_tree = prune_tree(full_tree)
    if pruned_tree is None:
        print("No branches with LUT labels found.")
        sys.exit(1)
    
    # Simplify the tree to include only: name, acronym, color, label (if exists), and children.
    simple_tree = simplify_tree(pruned_tree)
    
    # Count the total number of nodes in the simplified tree.
    total_nodes = count_nodes(simple_tree)
    
    # Convert the simplified tree to JSON.
    simple_json = json.dumps(simple_tree, indent=2)
    
    # Save output to JSON file.
    output_file = "output_tree.json"
    with open(output_file, "w", encoding="utf-8") as out_f:
        out_f.write(simple_json)
    
    # Now get tags from the simplified (output) tree.
    output_tags = get_all_tags(simple_tree)
    write_tags(output_tags, "output_tags.txt")
    
    # Print JSON, node count, and also output the tag lists.
    print("Output tree saved to '{}'".format(output_file))
    print("\nTotal nodes in output tree:", total_nodes)
    print("\n=== Ontology Tags ===")
    for tag in sorted(ontology_tags):
        print(tag)
    print("\n=== LUT Tags ===")
    for tag in sorted(lut_tags):
        print(tag)
    print("\n=== Output Tree Tags ===")
    for tag in sorted(output_tags):
        print(tag)
    print("\nSimplified Tree JSON:")
    print(simple_json)

if __name__ == "__main__":
    main()
