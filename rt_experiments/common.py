from collections import namedtuple
import numpy as np

from rig.bitfield import BitField
from rig.geometry import minimise_xyz, to_xyz
from rig.place_and_route.place.hilbert import hilbert_chip_order
from rig.place_and_route.routing_tree import RoutingTree
from rig.routing_table import Routes, RoutingTableEntry

import random

from six import iteritems
import struct


Vertex = namedtuple("Vertex", "x, y, p")


def assign_xyp_keys(nets):
    """Return a dictionary mapping a net to a unique key indicating the XYP
    co-ordinate of the source of the net.
    """
    # Create the XYP-formatted bit field
    xyp_bf = BitField()
    xyp_bf.add_field("x", length=8, start_at=24)
    xyp_bf.add_field("y", length=8, start_at=16)
    xyp_bf.add_field("p", length=5, start_at=11)
    xyp_bf.assign_fields()  # Fix the bitfield sizing

    # Prepare to store the nets and keys
    net_keys = dict()

    # For each net look at the placement of the source vertex and hence
    # generate a key.
    for net in nets:
        # Get the originating co-ordinates
        x, y, p = net.source

        # Generate the key and mask
        bf = xyp_bf(x=x, y=y, p=p)
        net_keys[net] = bf.get_value(), bf.get_mask()

    return net_keys


def assign_xyzp_keys(nets):
    """Return a dictionary mapping a net to a unique key indicating the XYZP
    co-ordinate of the source of the net.
    """
    # Create the XYZP-formatted bit field
    xyzp_bf = BitField()
    xyzp_bf.add_field("x", length=8, start_at=24)
    xyzp_bf.add_field("y", length=8, start_at=16)
    xyzp_bf.add_field("z", length=8, start_at=8)
    xyzp_bf.add_field("p", length=5, start_at=3)
    xyzp_bf.assign_fields()  # Fix the bitfield sizing

    # Prepare to store the nets and keys
    net_keys = dict()

    # For each net look at the placement of the source vertex and hence
    # generate a key.
    for net in nets:
        # Get the originating co-ordinates
        x, y, p = net.source

        # Get the minimal xyz co-ordinate
        x, y, z = minimise_xyz(to_xyz((x, y)))

        # Generate the key and mask
        bf = xyzp_bf(x=x, y=y, z=abs(z), p=p)
        net_keys[net] = bf.get_value(), bf.get_mask()

    return net_keys


def assign_hilbert_keys(nets, machine):
    """Return a dictionary mapping a net to a unique key indicating the
    position of the originating chip along a Hilbert curve mapped to the
    SpiNNaker machine.
    """
    # Create the Hilbert-formatted bit field
    hilbert_bf = BitField()
    hilbert_bf.add_field("index", length=16, start_at=16)
    hilbert_bf.add_field("p", length=5, start_at=3)
    hilbert_bf.assign_fields()  # Fix the bitfield sizing

    # Prepare to store the nets and keys
    net_keys = dict()

    # Generate an appropriately scaled Hilbert curve
    curve = {(x, y): i for i, (x, y) in enumerate(
        chip for chip in hilbert_chip_order(machine) if chip in machine)
    }

    # For each net look at the placement of the source vertex and hence
    # generate a key.
    for net in nets:
        # Get the originating co-ordinates
        x, y, p = net.source

        # Generate the key and mask
        bf = hilbert_bf(index=curve[(x, y)], p=p)
        net_keys[net] = bf.get_value(), bf.get_mask()

    return net_keys


def assign_random_keys(nets, machine, seed, n_bits):
    """Return a dictionary mapping a net to a unique randomly-assigned key.
    """
    # Ensure sufficient bits available
    assert n_bits >= np.ceil(np.log2(machine.width * machine.height * 17))

    # Create the RND-formatted bit field
    rnd_bf = BitField()
    rnd_bf.add_field("index", length=n_bits)
    rnd_bf.assign_fields()  # Fix the bitfield sizing

    # Prepare to store the nets and keys
    net_keys = dict()

    # Assign a unique random ID to each core
    random.seed(seed)
    ids = random.sample(range(1 << n_bits),
                        machine.width * machine.height * 17)

    # For each net look at the placement of the source vertex and hence
    # generate a key.
    for net, index in zip(nets, ids):
        # Generate the key and mask
        bf = rnd_bf(index=index)
        net_keys[net] = bf.get_value(), bf.get_mask()

    return net_keys


def dump_routing_tables(fp, tables):
    """Dump routing tables to file."""
    for (x, y), entries in iteritems(tables):
        # Write the header
        fp.write(struct.pack("<2BH", x, y, len(entries)))

        # Write the entries
        for entry in entries:
            route_word = 0x0
            for route in entry.route:
                route_word |= 1 << route

            source_word = 0x0
            for source in entry.sources:
                if source is not None:
                    source_word |= 1 << source

            fp.write(struct.pack("<4I", entry.key, entry.mask,
                                 source_word, route_word))


def read_routing_tables(fp):
    """Read routing tables from a file."""
    tables = dict()

    data = fp.read()
    offset = 0
    while offset < len(data):
        # Read the header
        x, y, n_entries = struct.unpack_from("<2BH", data, offset)
        offset += 4

        # Prepare the entries
        entries = [None for _ in range(n_entries)]

        # Read the entries
        for i in range(n_entries):
            key, mask, source_word, route_word = \
                struct.unpack_from("<4I", data, offset)
            offset += 16

            route = {r for r in Routes if route_word & (1 << r)}
            source = {s for s in Routes if source_word & (1 << s)}
            entries[i] = RoutingTableEntry(route, key, mask, source)

        # Store the table
        tables[(x, y)] = entries

    return tables


def read_table_lengths(fp):
    """Read routing table lengths from a file."""
    lengths = dict()

    data = fp.read()
    offset = 0
    while offset < len(data):
        # Read the header
        x, y, n_entries = struct.unpack_from("<2BH", data, offset)
        offset += 4 + n_entries*16

        # Store the length
        lengths[(x, y)] = n_entries

    return lengths


def get_pruned_routing_tree(tree, prune):
    """Get a pruned copy of a routing tree by removing all branches of the tree
    which lead only to specified objects.

    Parameters
    ----------
    tree : :py:class:`rig.place_and_route.routing_tree.RoutingTree`
        The tree to prune, a new copy will be returned.
    prune : {object, ...}
        Target objects which should be pruned from the tree.

    Returns
    -------
    :py:class:`rig.place_and_route.routing_tree.RoutingTree`
        A new tree, which is equivalent to the first tree but for having
        branches which led to objects in `prune` removed.
    """
    # Shortcut in the case that there is no pruning necessary
    if not prune:
        return tree

    # Construct the new children for the Node
    children = set()

    for child_direction, child in tree.children:
        if isinstance(child, RoutingTree):
            # If the child is a further tree then prune it before adding.
            subtree = get_pruned_routing_tree(child, prune)

            if subtree.children:
                # Keep the subtree if it still has children
                children.add((child_direction, subtree))
        elif child not in prune:
            # Otherwise just add the child if it's not in the list of objects
            # to ignore.
            children.add((child_direction, child))

    # Return the new routing tree
    return RoutingTree(tree.chip, children)
