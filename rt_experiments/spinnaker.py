"""Use a SpiNNaker implementation of Ordered Covering to minimise routing
tables.
"""
import argparse
import common
import numpy as np
from rig.machine_control import MachineController
from rig.routing_table import RoutingTableEntry, Routes
from six import iteritems, iterkeys, itervalues
import spalloc
import struct
import time


def get_memory_profile(mc):
    """Return the cumulative heap usage over time."""
    # Keep a track of how much memory is associated with each pointer,
    # track cumulative memory usage over time
    usage = [0]
    pointers = dict()

    # Read the linked list of entry arrays
    buf = mc.read_vcpu_struct_field("user0")
    while buf != 0x0:
        # Read back the data
        data = mc.read(buf, 8 + 2048*8)

        # Unpack the header
        n_entries, buf_next = struct.unpack_from("<2I", data)

        # Read in the memory recording entries
        alloc_data = np.ndarray(
            shape=(n_entries, 2),
            dtype=np.uint32,
            buffer=data[8:8 + n_entries*8]
        )
        for n_bytes, ptr in alloc_data[:]:
            if n_bytes == 0:
                # This is a free
                usage.append(usage[-1] - pointers.pop(ptr))
            else:
                # This is an allocation
                usage.append(usage[-1] + n_bytes)
                pointers[ptr] = n_bytes

        # Progress to the next block of memory
        buf = buf_next

    return np.array(usage, dtype=np.uint32)


def pack_table(table, target_length):
    """Pack a routing table into the form required for dumping into SDRAM."""
    data = bytearray(2*4 + len(table)*3*4)

    # Pack the header
    struct.pack_into("<2I", data, 0, len(table), target_length)

    # Pack in the entries
    offset = 8
    for entry in table:
        pack_rte_into(entry, data, offset)
        offset += 12

    return data


def pack_rte_into(rte, buf, offset):
    """Pack a routing table entry into a buffer."""
    # Construct the route integer
    route = 0x0
    for r in rte.route:
        route |= 1 << r

    # Pack
    struct.pack_into("<3I", buf, offset, rte.key, rte.mask, route)


def unpack_table(data):
    # Unpack the header
    length, _ = struct.unpack_from("<2I", data)

    # Unpack the table
    table = [None for __ in range(length)]
    for i in range(length):
        key, mask, route = struct.unpack_from("<3I", data, i*12 + 8)
        routes = {r for r in Routes if (1 << r) & route}
        table[i] = RoutingTableEntry(routes, key, mask)

    return table


if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "width", help="Width of SpiNNaker machine in triads", type=int
    )
    parser.add_argument(
        "height", help="Height of SpiNNaker machine in triads", type=int
    )
    parser.add_argument("routing_table")
    parser.add_argument("out_file")
    parser.add_argument("target_length", type=int, default=0, nargs='?')
    parser.add_argument("--memory-profile", type=str)
    args = parser.parse_args()

    # Load and minimise all routing tables
    print("Reading routing tables...")
    with open(args.routing_table, "rb") as f:
        uncompressed = common.read_routing_tables(f)

    # Request a SpiNNaker machine
    print("Waiting for SpiNNaker machine...")
    with spalloc.Job(args.width, args.height) as job:
        # Talk to the machine
        print("Booting...")
        mc = MachineController(job.hostname)
        mc.boot()
        mc.discover_connections()

        # Convert the tables into the appropriate formats
        chip_data = {chip: pack_table(table, args.target_length) for
                     chip, table in iteritems(uncompressed)}

        # Allocate memory on the machine
        chip_mem = {
            (x, y): mc.sdram_alloc_as_filelike(len(data), x=x, y=y, tag=1) for
            (x, y), data in iteritems(chip_data)
        }

        # Build the targets dictionary
        targets = {chip: {1} for chip in iterkeys(chip_mem)}

        # Load the data
        print("Loading data...")
        t = time.time()
        for chip, mem in iteritems(chip_mem):
            mem.write(chip_data[chip])
        load_time = time.time() - t
        print("... took {:.3f} s".format(load_time))

        # Load the application
        print("Loading application...")
        if args.memory_profile is None:
            mc.load_application("./ordered_covering.aplx", targets)
        else:
            mc.load_application("./ordered_covering_profiled.aplx", targets)
        t = time.time()

        # Wait until this does something interesting
        print("Minimising...")
        ready = mc.wait_for_cores_to_reach_state("exit", len(uncompressed),
                                                 timeout=60.0)
        if ready < len(uncompressed):
            raise Exception("Something didn't work...")
        run_time = time.time() - t
        print("... took ~{:.3f} s".format(run_time))

        # Read back the routing tables
        print("Reading back results...")
        for mem in itervalues(chip_mem):
            mem.seek(0)

        compressed = {chip: unpack_table(mem.read()) for chip, mem in
                      iteritems(chip_mem)}

        # Dump to file
        with open(args.out_file, "wb+") as f:
            common.dump_routing_tables(f, compressed)

        # Get the memory profile
        if args.memory_profile is not None:
            print("Reading back memory usage...")
            # Get the data
            memory = dict()
            for x, y in iterkeys(targets):
                with mc(x=x, y=y, p=1):
                    memory[(x, y)] = get_memory_profile(mc)

            # Dump to file
            print("Dumping to file...")
            with open(args.memory_profile, "wb+") as fp:
                common.dump_memory_profile(fp, memory)
