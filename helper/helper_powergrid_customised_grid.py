"""
helper.helper_powergrid_customised_grid
---------------------------------------
This module provides customized toy power grid models for educational and research purposes using the pandapower library.
"""

import os
import numpy as np
import pandapower as pp

__author__ = "Gökhan Demirel"
__email__ = "goekhan.demirel@kit.edu"
ROOT_PATH = os.path.dirname(os.path.abspath("__file__"))


def toy_grid():
    """
    Creates a simple three-bus system with one low-voltage node connected to a
    medium-voltage slack bus. Each low-voltage node includes a load, storage,
    and static generation.

    Topology:
        ext_grid -- b1 -- transformer (20/0.4) -- b2 -- line -- b3 (load, storage, generation)

    Returns:
        net (pandapowerNet): A pandapower network object representing the three-bus grid.

    Example:
        >>> net = toy_grid()
    """
    # Create an empty network
    net = pp.create_empty_network(name="toy_net")

    # Define geodata for buses
    geodata_x = np.array([0, 2, 4, -2])
    geodata_y = np.array([0, 0, 0, 0])
    geodata = [(geodata_x[i], geodata_y[i]) for i in range(len(geodata_x))]

    # Create bus elements
    bus1 = pp.create_bus(net, vn_kv=20.0, name="MV Busbar", geodata=geodata[0])
    bus2 = pp.create_bus(net, vn_kv=20.0, name="MV Transformer Bus", geodata=geodata[1])
    bus3 = pp.create_bus(net, vn_kv=0.4, name="LV Transformer Bus", geodata=geodata[2])

    # Create branch elements
    pp.create_line(
        net,
        from_bus=bus2,
        to_bus=bus3,
        length_km=0.1,
        std_type="NAYY 4x50 SE",
        name="Line 2-3",
        geodata=[geodata[1], geodata[2]],
    )
    pp.create_transformer(
        net,
        hv_bus=bus1,
        lv_bus=bus2,
        std_type="0.4 MVA 20/0.4 kV",
        name="Trafo 1-2: 20kV/0.4kV transformer",
    )

    # Create external grid connection
    pp.create_ext_grid(net, bus=bus1, vm_pu=1.02, va_degree=50, name="Grid Connection")

    # Create load, storage, and generation at Bus 3
    pp.create_load(net, bus=bus3, p_mw=0.0109, q_mvar=0.0043, name="Load Bus 3")
    pp.create_storage(
        net,
        bus=bus3,
        p_mw=0.030563,
        max_e_mwh=0.061126,
        q_mvar=0.030563,
        max_p_mw=0.030563,
        max_q_mvar=0.030563,
        sn_mva=0.0306,
        name="Storage Bus 3",
    )
    pp.create_sgen(
        net, bus=bus3, p_mw=0.040, q_mvar=0.0, max_p_mw=0.040, name="PV at Bus 3"
    )

    return net


def toy_two_bus_grid():
    """
    Creates a simple two-bus power grid.

    Returns:
        net (pandapowerNet): A pandapower network object representing the two-bus grid.

    Example:
        >>> net = toy_two_bus_grid()
    """
    # Create an empty network
    net = pp.create_empty_network(name="toy_two_bus")

    # Define geodata for buses
    geodata_x = np.array([0, 2])
    geodata_y = np.array([0, 0])
    geodata = [(geodata_x[i], geodata_y[i]) for i in range(len(geodata_x))]

    # Create bus elements
    bus1 = pp.create_bus(net, vn_kv=20.0, name="MV Busbar", geodata=geodata[0])
    bus2 = pp.create_bus(net, vn_kv=20.0, name="MV Transformer Bus", geodata=geodata[1])

    # Create branch elements
    pp.create_line(
        net,
        from_bus=bus1,
        to_bus=bus2,
        length_km=2.5,
        std_type="NA2XS2Y 1x240 RM/25 12/20 kV",
        geodata=[geodata[0], geodata[1]],
    )
    pp.create_ext_grid(net, bus=bus2, name="Grid Connection")
    pp.create_gen(net, bus=bus1, p_mw=-1, vn_kv=20, sn_kva=8000, controllable=True)

    # Run power flow
    pp.runpp(net)

    return net


def apply_absolute_values(net, absolute_values_dict, load_case):
    """
    Applies specified absolute values to elements of a pandapower network.

    Args:
        net (pandapowerNet): The pandapower network object.
        absolute_values_dict (dict): A dictionary with element types as keys and
                                     DataFrames with values to apply.
        load_case (str): The load case to apply values for.

    Example:
        >>> apply_absolute_values(net, abs_values, "case_1")
    """
    for elm_param, values in absolute_values_dict.items():
        elm, param = elm_param
        if values.shape[1]:  # Ensure the shape is valid
            net[elm].loc[:, param] = values.loc[load_case]


if __name__ == "__main__":
    # Example usage
    net1 = toy_grid()
    print("Three-bus toy grid created.")

    net2 = toy_two_bus_grid()
    print("Two-bus toy grid created.")
