import pandas as pd
import numpy as np
from collections import defaultdict


class FPNode:
    def __init__(self, item, parent=None):
        self.item = item
        self.parent = parent
        self.children = {}
        self.link = None
        # Store {transaction_id: normalized_value}
        self.tid_values = {}

    def add_transaction(self, tid, value):
        self.tid_values[tid] = value

    @property
    def support(self):
        # Support is the sum of the normalized values
        return sum(self.tid_values.values())


def fpgrowth_continuous_normalized(df, min_support=0.1, use_colnames=False):
    """
    Min-Apriori FP-Growth with Column Normalization (L1 Norm).

    1. Normalizes each column so sum(column) = 1.0.
    2. Calculates support of itemsets as Sum(min(values)).

    Parameters:
    -----------
    df : pandas.DataFrame
        Continuous data.
    min_support : float
        Threshold between 0 and 1.
        Note: With this normalization, single items always have support 1.0.
        This threshold primarily filters combined itemsets.

    Returns:
    --------
    pd.DataFrame
    """
    # --- Step 1: Normalization (Min-Apriori Requirement) ---
    # Normalize frequencies: value / sum(column_values)
    # Result: The support of every single item (1-itemset) becomes exactly 1.0
    df_norm = df.copy()
    # Avoid division by zero
    col_sums = df_norm.sum(axis=0)
    col_sums[col_sums == 0] = 1.0
    df_norm = df_norm / col_sums

    # --- Step 2: Setup ---
    # Since max support is 1.0, threshold is absolute
    support_threshold = min_support

    # Calculate non-zero counts for sorting the tree (Heuristic)
    # Since support(value) is tied at 1.0, we sort by 'occurrence count'
    # to keep the tree compact (dense items at top).
    item_counts = (df_norm > 0).sum()

    # Filter items that (theoretically) don't meet support?
    # With L1 norm, sum is 1.0, so if min_support <= 1.0, all items are frequent.
    # We only filter completely empty columns.
    frequent_items = item_counts[item_counts > 0]

    # Sort items by non-zero count (descending)
    item_order = frequent_items.sort_values(ascending=False).index.tolist()
    rank = {item: i for i, item in enumerate(item_order)}

    # Convert to transaction list
    transactions = []
    for idx, row in enumerate(df_norm.itertuples(index=False)):
        trans = {}
        for col_name, val in zip(df_norm.columns, row):
            # Only store non-zero values
            if val > 0 and col_name in rank:
                trans[col_name] = val

        if trans:
            sorted_items = sorted(trans.keys(), key=lambda x: rank[x])
            transactions.append({"id": idx, "items": sorted_items, "values": trans})

    # --- Step 3: Build FP-Tree ---
    root = FPNode(None)
    header_table = {item: [] for item in item_order}

    for trans in transactions:
        current_node = root
        tid = trans["id"]
        t_values = trans["values"]

        for item in trans["items"]:
            val = t_values[item]
            if item in current_node.children:
                child = current_node.children[item]
                child.add_transaction(tid, val)
                current_node = child
            else:
                new_node = FPNode(item, parent=current_node)
                new_node.add_transaction(tid, val)
                current_node.children[item] = new_node
                current_node = new_node
                header_table[item].append(new_node)

    # --- Step 4: Mine Tree ---
    frequent_itemsets = []

    # If min_support <= 1.0, all 1-itemsets are frequent
    for item in item_order:
        # We know sum is 1.0 (or close to it due to float precision)
        # We can explicitly add them or calculate from tree
        if 1.0 >= support_threshold:
            frequent_itemsets.append((1.0, [item]))

    def mine_tree(header_table_local, prefix_items, prefix_support):
        # Sort bottom-up
        sorted_items = list(header_table_local.keys())[::-1]

        for item in sorted_items:
            nodes = header_table_local[item]
            if not nodes:
                continue

            # Calculate support using Sum of Values
            current_support = sum(node.support for node in nodes)

            if current_support >= support_threshold:
                new_itemset = prefix_items + [item]
                frequent_itemsets.append((current_support, new_itemset))

                # Build Conditional Database
                conditional_transactions = []
                for node in nodes:
                    suffix_val_map = node.tid_values
                    if not suffix_val_map:
                        continue

                    # Trace path to root
                    path = []
                    parent = node.parent
                    while parent and parent.item is not None:
                        path.append(parent)
                        parent = parent.parent

                    if not path:
                        continue
                    path = path[::-1]

                    # Intersect transaction IDs
                    for tid, suffix_val in suffix_val_map.items():
                        cond_trans = {}
                        valid = True
                        for ancestor in path:
                            ancestor_val = ancestor.tid_values.get(tid)
                            if ancestor_val is not None:
                                # MIN-APRIORI FORMULA: min(val_A, val_B)
                                cond_trans[ancestor.item] = min(
                                    ancestor_val, suffix_val
                                )
                            else:
                                valid = False
                                break
                        if valid and cond_trans:
                            conditional_transactions.append(
                                {"id": tid, "values": cond_trans}
                            )

                # Build Conditional Tree
                if conditional_transactions:
                    # Filter items by threshold in conditional DB
                    cond_supports = defaultdict(float)
                    for ct in conditional_transactions:
                        for itm, v in ct["values"].items():
                            cond_supports[itm] += v

                    cond_valid_items = {
                        k for k, v in cond_supports.items() if v >= support_threshold
                    }

                    if cond_valid_items:
                        # Sort by original rank
                        cond_item_order = sorted(
                            list(cond_valid_items), key=lambda x: rank[x]
                        )
                        cond_root = FPNode(None)
                        cond_header = {k: [] for k in cond_item_order}

                        for ct in conditional_transactions:
                            curr = cond_root
                            tid = ct["id"]
                            vals = ct["values"]
                            items_to_add = [x for x in cond_item_order if x in vals]

                            for itm in items_to_add:
                                val = vals[itm]
                                if itm in curr.children:
                                    child = curr.children[itm]
                                    child.add_transaction(tid, val)
                                    curr = child
                                else:
                                    new_node = FPNode(itm, parent=curr)
                                    new_node.add_transaction(tid, val)
                                    curr.children[itm] = new_node
                                    curr = new_node
                                    cond_header[itm].append(new_node)

                        mine_tree(cond_header, new_itemset, current_support)

    # Note: We manually added 1-itemsets. Now we mine for k >= 2
    # But standard FP-Growth recursion handles k>=1 if we start empty.
    # However, since we manually know 1-itemsets are 1.0, we can start mining
    # for patterns larger than 1.

    # Resetting frequent_itemsets to let recursion handle everything consistently
    frequent_itemsets = []
    mine_tree(header_table, [], 0)

    # Formatting
    if not frequent_itemsets:
        return pd.DataFrame(columns=["support", "itemsets"])

    df_res = pd.DataFrame(frequent_itemsets, columns=["support", "itemsets"])
    df_res["itemsets"] = df_res["itemsets"].apply(frozenset)

    if not use_colnames:
        col_map = {name: i for i, name in enumerate(df.columns)}
        df_res["itemsets"] = df_res["itemsets"].apply(
            lambda x: frozenset([col_map[i] for i in x])
        )

    return df_res
