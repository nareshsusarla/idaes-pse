##############################################################################
# Institute for the Design of Advanced Energy Systems Process Systems
# Engineering Framework (IDAES PSE Framework) Copyright (c) 2018-2019, by the
# software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia
# University Research Corporation, et al. All rights reserved.
#
# Please see the files COPYRIGHT.txt and LICENSE.txt for full copyright and
# license information, respectively. Both files are also available online
# at the URL "https://github.com/IDAES/idaes-pse".
##############################################################################


##############################################################################
# This is a simple power plant model for supercritical coal-fired power plant. 

# The model has 3 degrees of freedom to be specified by the user: 
#   1. Pressure in Pascals for main steam
#   2. Temperature in Kelvins for main steam
#   3. Flow in mol/s for main steam
##############################################################################



__author__ = "Naresh Susarla"



# Import Pyomo libraries
import pyomo.environ as pyo
from pyomo.network import Arc
from pyomo.environ import units

# Import IDAES libraries
from idaes.core import FlowsheetBlock, MaterialBalanceType
from idaes.core.util import copy_port_values as _set_port
from idaes.core.util.model_statistics import degrees_of_freedom
from idaes.generic_models.unit_models import (
    Mixer,
    HeatExchanger,
    PressureChanger,
    MomentumMixingType,
    Heater,
    Separator
)
from idaes.power_generation.unit_models.helm import HelmSplitter
from idaes.generic_models.unit_models.heat_exchanger import (
    delta_temperature_underwood_callback)
from idaes.generic_models.unit_models.pressure_changer import (
    ThermodynamicAssumption)
from idaes.generic_models.unit_models.separator import (
    SplittingType)
import idaes.core.util.scaling as iscale
import idaes.logger as idaeslog

# Import Property Packages (IAPWS95 for Water/Steam)
from idaes.generic_models.properties import iapws95

# Imports to print results in process flow diagram
import os
from collections import OrderedDict
from pyomo.common.fileutils import this_file_dir
from idaes.core.util.misc import svg_tag


def create_model():
    """
    Create flowsheet and add unit models.
    """
    ###########################################################################
    #  Flowsheet and Property Package                                         #
    ###########################################################################
    m = pyo.ConcreteModel(name="Steam Cycle Model")
    m.fs = FlowsheetBlock(default={"dynamic": False})
    m.fs.prop_water = iapws95.Iapws95ParameterBlock(
        default={"phase_presentation": iapws95.PhaseType.LG}
    )

    ###########################################################################
    #   Turbine declarations                                                  #
    ###########################################################################

    for i in range(9):

        turbine = PressureChanger(
            default={
                "property_package": m.fs.prop_water,
                "compressor": False,
                "material_balance_type": MaterialBalanceType.componentTotal,
                "thermodynamic_assumption": ThermodynamicAssumption.isentropic
            }
        )
        setattr(m.fs, "turbine_" + str(i+1), turbine)

    ###########################################################################
    #  Boiler section declarations                                            #
    ###########################################################################
    # Boiler section is set up using two heater blocks, as following:
    # 1) For the main steam the heater block is named 'boiler'
    # 2) For the reheated steam the heater block is named 'reheater'
    m.fs.boiler = Heater(
        default={
            "dynamic": False,
            "property_package": m.fs.prop_water,
            "has_pressure_change": True
        }
    )
    m.fs.reheater = Heater(
        default={
            "dynamic": False,
            "property_package": m.fs.prop_water,
            "has_pressure_change": True
        }
    )


    @m.fs.boiler.Constraint(m.fs.time,
                            doc="Outlet temperature of boiler in K")
    def boiler_temperature_constraint(b, t):
        return b.control_volume.properties_out[t].temperature == 866.15 


    @m.fs.reheater.Constraint(m.fs.time,
                              doc="Outlet temperature of reheater in K")
    def reheater_temperature_constraint(b, t):
        return b.control_volume.properties_out[t].temperature == 866.15 

    ###########################################################################
    #  Condenser Mixer, Condenser, and Condensate pump declarations           #
    ###########################################################################
    # The condenser mixer inlet notation is given below:
    #   - main refers to main steam coming from the turbine train
    #   - bfpt refers to steam coming from the boiler feed pump turbine
    #   - drain refers to condensed steam from the feed water heater 1
    #   - makeup refers to make up water
    
    m.fs.condenser_mix = Mixer(
        default={
            "momentum_mixing_type": MomentumMixingType.none,
            "material_balance_type": MaterialBalanceType.componentTotal,
            "inlet_list": ["main", "bfpt", "drain", "makeup"],
            "property_package": m.fs.prop_water,
        }
    )

    
    @m.fs.condenser_mix.Constraint(m.fs.time,
                                   doc="Outlet pressure of condenser mixer \
                                   set to the minimum inlet pressure")
    def mixer_pressure_constraint(b, t):
        return b.main_state[t].pressure == b.mixed_state[t].pressure

    
    # The condenser is set up as a heater block and the outlet is assumed
    # to be a saturated liquid
    m.fs.condenser = Heater(
        default={
            "dynamic": False,
            "property_package": m.fs.prop_water,
            "has_pressure_change": False
        }
    )

    @m.fs.condenser.Constraint(m.fs.time,
                               doc="Outlet enthalpy of condenser as saturated \
                               liquid")
    def cond_vaporfrac_constraint(b, t):
        return (
            b.control_volume.properties_out[t].enth_mol
            == b.control_volume.properties_out[t].enth_mol_sat_phase['Liq']
        )

    # Condenser pump
    m.fs.cond_pump = PressureChanger(
        default={
            "property_package": m.fs.prop_water,
            "material_balance_type": MaterialBalanceType.componentTotal,
            "thermodynamic_assumption": ThermodynamicAssumption.pump,
        }
    )
    
    ###########################################################################
    #  Feedwater heater declaration                                           #
    ###########################################################################
    # Feed water heaters (FWHs) are declared as 0D heat exchangers
    # Tube side is for feed water and shell side is for steam condensing
    # The pressure drop on both sides are accounted for by setting the respective
    # outlet pressure based on the following assumptions:
    #     (1) Feed water side: A constant 4% pressure drop is assumed
    #           on the feedwater side for all FWHs. For this,
    #           the outlet pressure is set to 0.96 times the inlet pressure,
    #           on the feed water side for all FWHs. The outlet pressure is then
    #           calculated as follows: P_out = 0.96 * P_in
    #     (2) Steam condensing side: Going from high pressure to
    #           low pressure FWHs, the outlet pressure of
    #           the condensed steam in assumed to be 10% more than that
    #           of the pressure of steam extracted for the immediately
    #           next lower pressure feedwater heater. e.g. the outlet condensate
    #           pressure of FWH 'n' = 1.1 * pressure of steam extracted for FWH 'n-1'
    #           Note: for FWH1, the FWH 'n-1' is the Condenser while for FWH6,
    #           FWH 'n-1' is the Deaerator.
    # The condensing steam is assumed to leave the FWH as saturated liquid
    #
    # Scaling factors for area and overall heat transfer coefficients for
    # FWHs have all been set appropriately. The user may change these values,
    # if needed.  If not set, the scaling factors are equal to 1 (IDAES default)


    ###########################################################################
    # DEFINITION OF FEED WATER HEATERS MIXERS
    ###########################################################################
    FWH_Mixers_list = ['fwh1_mix', 'fwh2_mix', 'fwh3_mix', 'fwh6_mix', 'fwh7_mix']

    for i in FWH_Mixers_list:
        FWH_Mixer = Mixer(
            default={
                "momentum_mixing_type": MomentumMixingType.none,
                "material_balance_type": MaterialBalanceType.componentTotal,
                "inlet_list": ["steam", "drain"],
                "property_package": m.fs.prop_water,
            }
        )
        setattr(m.fs, i, FWH_Mixer)

    ###########################################################################
    # DEFINITION OF OUTLET PRESSURE OF FEED WATER HEATERS MIXERS
    ###########################################################################

    # The outlet pressure of an FWH mixer is equal to the minimum pressure
    # Since the pressure of mixer inlet 'steam' has the minimum pressure,
    # the following constraints set the outlet pressure of FWH mixers to be same
    # as the pressure of the inlet 'steam'

    def fwhmixer_pressure_constraint(b, t):
        return b.steam_state[t].pressure == b.mixed_state[t].pressure

    for i in FWH_Mixers_list:
        setattr(getattr(m.fs, i), "mixer_pressure_constraint", pyo.Constraint(m.fs.config.time, rule=fwhmixer_pressure_constraint))

    ###########################################################################
    # DEFINITION OF FEED WATER HEATERS
    ###########################################################################
    FWH_list = ['fwh1', 'fwh2', 'fwh3', 'fwh4', 'fwh6', 'fwh7', 'fwh8']

    for i in FWH_list:
        FWH = HeatExchanger(
            default={
                "delta_temperature_callback": delta_temperature_underwood_callback,
                "shell": {
                    "property_package": m.fs.prop_water,
                    "material_balance_type": MaterialBalanceType.componentTotal,
                    "has_pressure_change": True,
                },
                "tube": {
                    "property_package": m.fs.prop_water,
                    "material_balance_type": MaterialBalanceType.componentTotal,
                    "has_pressure_change": True,
                },
            }
        )
        setattr(m.fs, i, FWH)

    ###########################################################################
    # SETTING SCALING FACTORS FOR AREA AND HEAT TRANSFER COEFFICIENT
    ###########################################################################

    for i in FWH_list:
        c = getattr(m.fs, i)
        iscale.set_scaling_factor(getattr(c, "area"), 1e-2)
        iscale.set_scaling_factor(getattr(c, "overall_heat_transfer_coefficient"), 1e-3)

    ###########################################################################
    # Setting the outlet enthalpy of condensate in an FWH to be same as saturated liquid
    ###########################################################################
    def fwh_vaporfrac_constraint(b, t):
        return (
            b.side_1.properties_out[t].enth_mol
            == b.side_1.properties_out[t].enth_mol_sat_phase['Liq'])

    for i in FWH_list:
        setattr(getattr(m.fs, i), i + "_vaporfrac_constraint", pyo.Constraint(m.fs.time, rule=fwh_vaporfrac_constraint))

    ###########################################################################
    # Setting a 4% pressure drop on the feedwater side (P_out = 0.96 * P_in)
    ###########################################################################

    def fwh_s2pdrop_constraint(b, t):
        return (
            b.side_2.properties_out[t].pressure
            == 0.96 * b.side_2.properties_in[t].pressure)

    for i in FWH_list:
        setattr(getattr(m.fs, i), i + "_s2pdrop_constraint", pyo.Constraint(m.fs.time, rule=fwh_s2pdrop_constraint))

    ###########################################################################
    # Setting the outlet pressure of condensate to be 10% more than that of
    # steam routed to condenser, as described in FWH description
    ###########################################################################
    # FWH1: 0.5 is the pressure ratio for turbine #9 (see set_inputs)
    # FWH2: 0.64^2 is the pressure ratio for turbine #8 (see set_inputs)
    # FWH3: 0.64^2 is the pressure ratio for turbine #7 (see set_inputs)
    # FWH4: 0.64^2 is the pressure ratio for turbine #6 (see set_inputs)
    # FWH6: 0.79^6 is the pressure ratio for turbine #4 (see set_inputs)
    # FWH7: 0.79^4 is the pressure ratio for turbine #3 (see set_inputs)
    # FWH8: 0.8^2 is the pressure ratio for turbine #2 (see set_inputs)
    
    pressure_ratio_list = {  'fwh1': 0.5,
                        'fwh2': 0.64**2,
                        'fwh3': 0.64**2,
                        'fwh4': 0.64**2,
                        'fwh6': 0.79**6,
                        'fwh7': 0.79**4,
                        'fwh8': 0.8**2}
    
    def fwh_s1pdrop_constraint(b, t):
        return (
            b.side_1.properties_out[t].pressure
            == 1.1 * b.turbine_pressure_ratio * b.side_1.properties_in[t].pressure)

    for i in FWH_list:
        b = getattr(m.fs, i)
        b.turbine_pressure_ratio = pyo.Param(initialize = pressure_ratio_list[i])
        setattr(b, i+"_s1pdrop_constraint", pyo.Constraint(m.fs.config.time, rule=fwh_s1pdrop_constraint))

    ###########################################################################
    #  Add deaerator and boiler feed pump (BFP)                               #
    ###########################################################################
    m.fs.fwh5_da = Mixer(
        default={
            "momentum_mixing_type": MomentumMixingType.none,
            "material_balance_type": MaterialBalanceType.componentTotal,
            "inlet_list": ["steam", "drain", "feedwater"],
            "property_package": m.fs.prop_water,
        }
    )

    @m.fs.fwh5_da.Constraint(m.fs.time,
                             doc="Outlet pressure of deaerator \
                             set to the minimum inlet pressure")
    def fwh5mixer_pressure_constraint(b, t):
        return b.feedwater_state[t].pressure == b.mixed_state[t].pressure


    
    m.fs.bfp = PressureChanger(
        default={
            "property_package": m.fs.prop_water,
            "material_balance_type": MaterialBalanceType.componentTotal,
            "thermodynamic_assumption": ThermodynamicAssumption.pump,
        }
    )
    m.fs.bfpt = PressureChanger(
        default={
            "property_package": m.fs.prop_water,
            "compressor": False,
            "material_balance_type": MaterialBalanceType.componentTotal,
            "thermodynamic_assumption": ThermodynamicAssumption.isentropic,
        }
    )

    # Outlet pressure of steam extracted for BFPT to be same as that of condenser mixer
    @m.fs.Constraint(m.fs.time,
                     doc="Outlet pressure of BFPT")
    def constraint_out_pressure(b, t):
        return (
            b.bfpt.control_volume.properties_out[t].pressure
            == b.condenser_mix.mixed_state[t].pressure
        )

    # Boiler feed water turbine produces just enough power to meet the demand of boiler
    # feed water pump
    @m.fs.Constraint(m.fs.time,
                     doc="BFPT produced work equal to demanded work by BFP")
    def constraint_bfp_power(b, t):
        return (
            b.bfp.control_volume.work[t] + b.bfpt.control_volume.work[t]
            == 0
        )

    ###########################################################################
    #  Turbine outlet splitter constraints                                    #
    ###########################################################################
    # Equality constraints have been written as following to define
    # the split fractions within the turbine train

    splitter_list = ['t1_splitter', 't2_splitter', 't3_splitter', 't5_splitter', 't6_splitter', 't7_splitter', 't8_splitter']

    for i in splitter_list:

        Splitter = HelmSplitter(default={"dynamic": False,
                                        "property_package": m.fs.prop_water})
        setattr(m.fs, i, Splitter)
    
    m.fs.t4_splitter = HelmSplitter(default={"dynamic": False,
                                            "property_package": m.fs.prop_water,
                                            "num_outlets": 3})

    ###########################################################################
    #  Create the stream Arcs and return the model                            #
    ###########################################################################
    _create_arcs(m)
    pyo.TransformationFactory("network.expand_arcs").apply_to(m.fs)
    return m


def _create_arcs(m):

    # boiler to turb
    m.fs.boiler_to_turb1 = Arc(
        source=m.fs.boiler.outlet, destination=m.fs.turbine_1.inlet
    )

    ###########################################################################
    #  SPLITTER 1                                                             #
    ###########################################################################

    # Turbine 1 to splitter
    m.fs.turb1_to_t1split = Arc(
        source=m.fs.turbine_1.outlet, destination=m.fs.t1_splitter.inlet
    )

    # Splitter to turbine 2
    m.fs.t1split_to_turb2 = Arc(
        source=m.fs.t1_splitter.outlet_1, destination=m.fs.turbine_2.inlet
    )

    # Splitter to Feed Water Heater 8
    m.fs.t1split_to_fwh8 = Arc(
        source=m.fs.t1_splitter.outlet_2, destination=m.fs.fwh8.inlet_1
    )
    
    ###########################################################################
    #  SPLITTER 2                                                             #
    ###########################################################################

    # Turbine 2 to splitter
    m.fs.turb2_to_t2split = Arc(
        source=m.fs.turbine_2.outlet, destination=m.fs.t2_splitter.inlet
    )

    # Splitter to reheater
    m.fs.t2split_to_reheater = Arc(
        source=m.fs.t2_splitter.outlet_1, destination=m.fs.reheater.inlet
    )

    # Splitter to Feed Water Heater Mix 7
    m.fs.t2split_to_fwh7mix = Arc(
        source=m.fs.t2_splitter.outlet_2, destination=m.fs.fwh7_mix.steam
    )

    # reheater to turbine 3
    m.fs.reheater_to_turb3 = Arc(
        source=m.fs.reheater.outlet, destination=m.fs.turbine_3.inlet
    )

    ###########################################################################
    #  SPLITTER 3                                                             #
    ###########################################################################

    # Turbine 3 to splitter
    m.fs.turb3_to_t3_split = Arc(
        source=m.fs.turbine_3.outlet, destination=m.fs.t3_splitter.inlet
    )

    # Splitter to turbine 4
    m.fs.t3split_to_turb4 = Arc(
        source=m.fs.t3_splitter.outlet_1, destination=m.fs.turbine_4.inlet
    )

    # Splitter to Feed Water Heater Mix 6
    m.fs.t3split_to_fhw6mix = Arc(
        source=m.fs.t3_splitter.outlet_2, destination=m.fs.fwh6_mix.steam
    )

    ###########################################################################
    #  SPLITTER 4                                                             #
    ###########################################################################

    # Turbine 4 to splitter
    m.fs.turb4_to_t4_split = Arc(
        source=m.fs.turbine_4.outlet, destination=m.fs.t4_splitter.inlet
    )

    # Splitter to turbine 5
    m.fs.t4split_to_turb5 = Arc(
        source=m.fs.t4_splitter.outlet_1, destination=m.fs.turbine_5.inlet
    )

    # Splitter to deareator FWH5_da
    m.fs.t4split_to_fhw5da = Arc(
        source=m.fs.t4_splitter.outlet_2, destination=m.fs.fwh5_da.steam
    )

    # Splitter to bfpt
    m.fs.t4split_to_bfpt = Arc(
        source=m.fs.t4_splitter.outlet_3, destination=m.fs.bfpt.inlet
    )

    ###########################################################################
    #  SPLITTER 5                                                             #
    ###########################################################################

    # Turbine 5 to splitter
    m.fs.turb5_to_t5_split = Arc(
        source=m.fs.turbine_5.outlet, destination=m.fs.t5_splitter.inlet
    )

    # Splitter to turbine 6
    m.fs.t5split_to_turb6 = Arc(
        source=m.fs.t5_splitter.outlet_1, destination=m.fs.turbine_6.inlet
    )

    # Splitter to Feed Water Heater 4
    m.fs.t5split_to_fwh4 = Arc(
        source=m.fs.t5_splitter.outlet_2, destination=m.fs.fwh4.inlet_1
    )

    ###########################################################################
    #  SPLITTER 6                                                             #
    ###########################################################################
   
    # Turbine 6 to splitter
    m.fs.turb6_to_t6_split = Arc(
        source=m.fs.turbine_6.outlet, destination=m.fs.t6_splitter.inlet
    )

    # Splitter to turbine 7
    m.fs.t6split_to_turb7 = Arc(
        source=m.fs.t6_splitter.outlet_1, destination=m.fs.turbine_7.inlet
    )

    # Splitter to Feed Water Heater Mixer 3
    m.fs.t6split_to_fwh3mix = Arc(
        source=m.fs.t6_splitter.outlet_2, destination=m.fs.fwh3_mix.steam
    )

    ###########################################################################
    #  SPLITTER 7                                                             #
    ###########################################################################
   
    # Turbine 7 to splitter
    m.fs.turb7_to_t7_split = Arc(
        source=m.fs.turbine_7.outlet, destination=m.fs.t7_splitter.inlet
    )

    # Splitter to turbine 8
    m.fs.t7split_to_turb8 = Arc(
        source=m.fs.t7_splitter.outlet_1, destination=m.fs.turbine_8.inlet
    )

    # Splitter to Feed Water Heater Mixer 2
    m.fs.t7split_to_fwh2mix = Arc(
        source=m.fs.t7_splitter.outlet_2, destination=m.fs.fwh2_mix.steam
    )

    ###########################################################################
    #  SPLITTER 8                                                             #
    ###########################################################################
   
    # Turbine 8 to splitter
    m.fs.turb8_to_t8_split = Arc(
        source=m.fs.turbine_8.outlet, destination=m.fs.t8_splitter.inlet
    )

    # Splitter to turbine 9
    m.fs.t8split_to_turb9 = Arc(
        source=m.fs.t8_splitter.outlet_1, destination=m.fs.turbine_9.inlet
    )

    # Splitter to Feed Water Heater Mixer 1
    m.fs.t8split_to_fwh1mix = Arc(
        source=m.fs.t8_splitter.outlet_2, destination=m.fs.fwh1_mix.steam
    )

    ###########################################################################
    #                                                               #
    ###########################################################################

    m.fs.turb_to_cmix = Arc(
        source=m.fs.turbine_9.outlet, destination=m.fs.condenser_mix.main
    )
    # Condenser to FWHs
    m.fs.drain_to_cmix = Arc(
        source=m.fs.fwh1.outlet_1, destination=m.fs.condenser_mix.drain
    )
    m.fs.bfpt_to_cmix = Arc(
        source=m.fs.bfpt.outlet, destination=m.fs.condenser_mix.bfpt
    )
    m.fs.cmix_to_cond = Arc(
        source=m.fs.condenser_mix.outlet, destination=m.fs.condenser.inlet
    )
    m.fs.cond_to_well = Arc(
        source=m.fs.condenser.outlet, destination=m.fs.cond_pump.inlet
    )
    m.fs.pump_to_fwh1 = Arc(
        source=m.fs.cond_pump.outlet, destination=m.fs.fwh1.inlet_2
    )
    # Mixers to FWHs
    m.fs.mix1_to_fwh1 = Arc(
        source=m.fs.fwh1_mix.outlet, destination=m.fs.fwh1.inlet_1
    )
    m.fs.mix2_to_fwh2 = Arc(
        source=m.fs.fwh2_mix.outlet, destination=m.fs.fwh2.inlet_1
    )
    m.fs.mix3_to_fwh3 = Arc(
        source=m.fs.fwh3_mix.outlet, destination=m.fs.fwh3.inlet_1
    )
    m.fs.mix6_to_fwh6 = Arc(
        source=m.fs.fwh6_mix.outlet, destination=m.fs.fwh6.inlet_1
    )
    m.fs.mix7_to_fwh7 = Arc(
        source=m.fs.fwh7_mix.outlet, destination=m.fs.fwh7.inlet_1
    )
    # Drains to Mixers
    m.fs.fwh2_to_fwh1mix = Arc(
        source=m.fs.fwh2.outlet_1, destination=m.fs.fwh1_mix.drain
    )
    m.fs.fwh3_to_fwh2mix = Arc(
        source=m.fs.fwh3.outlet_1, destination=m.fs.fwh2_mix.drain
    )
    m.fs.fwh4_to_fwh3mix = Arc(
        source=m.fs.fwh4.outlet_1, destination=m.fs.fwh3_mix.drain
    )
    m.fs.fwh6_to_fwh5mix = Arc(
        source=m.fs.fwh6.outlet_1, destination=m.fs.fwh5_da.drain
    )
    m.fs.fwh7_to_fwh6mix = Arc(
        source=m.fs.fwh7.outlet_1, destination=m.fs.fwh6_mix.drain
    )
    m.fs.fwh8_to_fwh7mix = Arc(
        source=m.fs.fwh8.outlet_1, destination=m.fs.fwh7_mix.drain
    )
    # Outlet2 to Inlet2
    m.fs.fwh1_to_fwh2 = Arc(
        source=m.fs.fwh1.outlet_2, destination=m.fs.fwh2.inlet_2
    )
    m.fs.fwh2_to_fwh3 = Arc(
        source=m.fs.fwh2.outlet_2, destination=m.fs.fwh3.inlet_2
    )
    m.fs.fwh3_to_fwh4 = Arc(
        source=m.fs.fwh3.outlet_2, destination=m.fs.fwh4.inlet_2
    )
    m.fs.fwh4_to_fwh5 = Arc(
        source=m.fs.fwh4.outlet_2, destination=m.fs.fwh5_da.feedwater
    )
    m.fs.fwh5_to_bfp = Arc(
        source=m.fs.fwh5_da.outlet, destination=m.fs.bfp.inlet
    )
    m.fs.bfp_to_fwh6 = Arc(
        source=m.fs.bfp.outlet, destination=m.fs.fwh6.inlet_2
    )
    m.fs.fwh6_to_fwh7 = Arc(
        source=m.fs.fwh6.outlet_2, destination=m.fs.fwh7.inlet_2
    )
    m.fs.fwh7_to_fwh8 = Arc(
        source=m.fs.fwh7.outlet_2, destination=m.fs.fwh8.inlet_2
    )
    # FWH8 to boiler 
    m.fs.fwh8_to_boiler = Arc(
        source=m.fs.fwh8.outlet_2, destination=m.fs.boiler.inlet
    )


def set_model_input(m):

    """
    Model inputs
    
    Units:
    Pressure in Pa 
    Molar Flow in mol/s
    Molar Enthalpy in J/mol
    """

    ###########################################################################
    #  Boiler, Reheater and Turbine input                                     #
    ###########################################################################

    main_steam_pressure = 24235081.4       
    m.fs.boiler.inlet.flow_mol.fix(29111)  
    m.fs.boiler.outlet.pressure.fix(main_steam_pressure)

    # Reheater pressure drop assumed based on baseline scenario
    m.fs.reheater.deltaP.fix(-96526.64)    

    # Turbine inlet conditions
    # The efficiency and pressure ratios of all turbines were assumed based on
    # results for the baseline scenario
    m.fs.turbine_1.ratioP.fix(0.8**5)
    m.fs.turbine_1.efficiency_isentropic.fix(0.94)

    m.fs.turbine_2.ratioP.fix(0.8**2)
    m.fs.turbine_2.efficiency_isentropic.fix(0.94)

    m.fs.turbine_3.ratioP.fix(0.79**4)
    m.fs.turbine_3.efficiency_isentropic.fix(0.88)

    m.fs.turbine_4.ratioP.fix(0.79**6)
    m.fs.turbine_4.efficiency_isentropic.fix(0.88)

    m.fs.turbine_5.ratioP.fix(0.64**2)
    m.fs.turbine_5.efficiency_isentropic.fix(0.78)

    m.fs.turbine_6.ratioP.fix(0.64**2)
    m.fs.turbine_6.efficiency_isentropic.fix(0.78)

    m.fs.turbine_7.ratioP.fix(0.64**2)
    m.fs.turbine_7.efficiency_isentropic.fix(0.78)

    m.fs.turbine_8.ratioP.fix(0.64**2)
    m.fs.turbine_8.efficiency_isentropic.fix(0.78)

    m.fs.turbine_9.ratioP.fix(0.5)
    m.fs.turbine_9.efficiency_isentropic.fix(0.78)

    ###########################################################################
    #  Condenser section                                                      #
    ###########################################################################
    m.fs.cond_pump.efficiency_pump.fix(0.80)
    m.fs.cond_pump.deltaP.fix(1e6)

    # Make up stream to condenser
    m.fs.condenser_mix.makeup.flow_mol.value = 1.08002495835536E-12  
    m.fs.condenser_mix.makeup.pressure.fix(103421.4)  
    m.fs.condenser_mix.makeup.enth_mol.fix(1131.69204) 

    ###########################################################################
    #  Low pressure FWH section inputs                                        #
    ###########################################################################
    # FWH1
    m.fs.fwh1.area.fix(400)
    m.fs.fwh1.overall_heat_transfer_coefficient.fix(2000)
    # FWH2
    m.fs.fwh2.area.fix(300)
    m.fs.fwh2.overall_heat_transfer_coefficient.fix(2900)
    # FWH3
    m.fs.fwh3.area.fix(200)
    m.fs.fwh3.overall_heat_transfer_coefficient.fix(2900)
    # FWH4
    m.fs.fwh4.area.fix(200)
    m.fs.fwh4.overall_heat_transfer_coefficient.fix(2900)

    ###########################################################################
    #  Deaerator and boiler feed pump (BFP) Input                             #
    ###########################################################################
    # Unlike the feedwater heaters the steam extraction flow to the deaerator
    # is not constrained by the saturated liquid constraint. Thus, the flow
    # to the deaerator is fixed in this model. The value of this split fraction
    # is again based on the baseline results

    m.fs.t4_splitter.split_fraction[:, "outlet_2"].fix(0.050331)

    m.fs.bfp.efficiency_pump.fix(0.80)
    # BFP pressure is assumed to be 15% more than the main steam pressure
    m.fs.bfp.outlet.pressure[:].fix(main_steam_pressure * 1.15)  
    m.fs.bfpt.efficiency_isentropic.fix(0.80)
    ###########################################################################
    #  High pressure feedwater heater                                         #
    ###########################################################################
    # FWH6
    m.fs.fwh6.area.fix(600)
    m.fs.fwh6.overall_heat_transfer_coefficient.fix(2900)
    # FWH7
    m.fs.fwh7.area.fix(400)
    m.fs.fwh7.overall_heat_transfer_coefficient.fix(2900)
    # FWH8
    m.fs.fwh8.area.fix(400)
    m.fs.fwh8.overall_heat_transfer_coefficient.fix(2900)


def initialize(m, fileinput=None, outlvl=idaeslog.NOTSET):

    iscale.calculate_scaling_factors(m)

    solver = pyo.SolverFactory("ipopt")
    solver.options = {
        "tol": 1e-6,
        "max_iter": 300,
        "halt_on_ampl_error": "yes",
    }

    # Initializing the boiler
    m.fs.boiler.inlet.pressure.fix(24657896)
    m.fs.boiler.inlet.enth_mol.fix(20004)
    m.fs.boiler.initialize(outlvl=outlvl, optarg=solver.options)
    m.fs.boiler.inlet.pressure.unfix()
    m.fs.boiler.inlet.enth_mol.unfix()

    # Initialization routine for the turbine train: deactivating constraints
    # that fix enthalpy at FWH outlet to initialize the model using the fixed
    # split_fractions for steam extractions for all the FWHs, except deaerator
    # Note: the split fractions will be unfixed later and the constraints will
    # be activated

    m.fs.t1_splitter.split_fraction[:, "outlet_2"].fix(0.12812)
    m.fs.t2_splitter.split_fraction[:, "outlet_2"].fix(0.061824)
    m.fs.t3_splitter.split_fraction[:, "outlet_2"].fix(0.03815)
    m.fs.t4_splitter.split_fraction[:, "outlet_1"].fix(0.9019)
    m.fs.t5_splitter.split_fraction[:, "outlet_2"].fix(0.0381443)
    m.fs.t6_splitter.split_fraction[:, "outlet_2"].fix(0.017535)
    m.fs.t7_splitter.split_fraction[:, "outlet_2"].fix(0.0154)
    m.fs.t8_splitter.split_fraction[:, "outlet_2"].fix(0.00121)

    m.fs.constraint_out_pressure.deactivate()
    m.fs.fwh1.fwh1_vaporfrac_constraint.deactivate()
    m.fs.fwh2.fwh2_vaporfrac_constraint.deactivate()
    m.fs.fwh3.fwh3_vaporfrac_constraint.deactivate()
    m.fs.fwh4.fwh4_vaporfrac_constraint.deactivate()
    m.fs.fwh6.fwh6_vaporfrac_constraint.deactivate()
    m.fs.fwh7.fwh7_vaporfrac_constraint.deactivate()
    m.fs.fwh8.fwh8_vaporfrac_constraint.deactivate()

    # solving the turbines and splitters
    _set_port(m.fs.turbine_1.inlet,  m.fs.boiler.outlet)
    m.fs.turbine_1.initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.t1_splitter.inlet,  m.fs.turbine_1.outlet)
    m.fs.t1_splitter.initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.turbine_2.inlet, m.fs.t1_splitter.outlet_1)
    m.fs.turbine_2.initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.t2_splitter.inlet, m.fs.turbine_2.outlet)
    m.fs.t2_splitter.initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.reheater.inlet, m.fs.t2_splitter.outlet_1)
    m.fs.reheater.initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.turbine_3.inlet, m.fs.reheater.outlet)
    m.fs.turbine_3.initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.t3_splitter.inlet, m.fs.turbine_3.outlet)
    m.fs.t3_splitter.initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.turbine_4.inlet, m.fs.t3_splitter.outlet_1)
    m.fs.turbine_4.initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.t4_splitter.inlet, m.fs.turbine_4.outlet)
    m.fs.t4_splitter.initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.turbine_5.inlet, m.fs.t4_splitter.outlet_1)
    m.fs.turbine_5.initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.t5_splitter.inlet, m.fs.turbine_5.outlet)
    m.fs.t5_splitter.initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.turbine_6.inlet, m.fs.t5_splitter.outlet_1)
    m.fs.turbine_6.initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.t6_splitter.inlet, m.fs.turbine_6.outlet)
    m.fs.t6_splitter.initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.turbine_7.inlet, m.fs.t6_splitter.outlet_1)
    m.fs.turbine_7.initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.t7_splitter.inlet, m.fs.turbine_7.outlet)
    m.fs.t7_splitter.initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.turbine_8.inlet, m.fs.t7_splitter.outlet_1)
    m.fs.turbine_8.initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.t8_splitter.inlet, m.fs.turbine_8.outlet)
    m.fs.t8_splitter.initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.turbine_9.inlet, m.fs.t8_splitter.outlet_1)
    m.fs.turbine_9.initialize(outlvl=outlvl, optarg=solver.options)

    # initialize the boiler feed pump turbine.
    _set_port(m.fs.bfpt.inlet, m.fs.t4_splitter.outlet_3)
    m.fs.bfpt.initialize(outlvl=outlvl, optarg=solver.options)

    ###########################################################################
    #  Condenser                                                              #
    ###########################################################################
    _set_port(m.fs.condenser_mix.bfpt, m.fs.bfpt.outlet)
    _set_port(m.fs.condenser_mix.main, m.fs.turbine_9.outlet)
    m.fs.condenser_mix.drain.flow_mol.fix(1460)
    m.fs.condenser_mix.drain.pressure.fix(7308)
    m.fs.condenser_mix.drain.enth_mol.fix(2973)
    m.fs.condenser_mix.initialize(outlvl=outlvl, optarg=solver.options)
    m.fs.condenser_mix.drain.unfix()

    _set_port(m.fs.condenser.inlet, m.fs.condenser_mix.outlet)
    m.fs.condenser.initialize(outlvl=outlvl, optarg=solver.options)

    _set_port(m.fs.cond_pump.inlet, m.fs.condenser.outlet)
    m.fs.cond_pump.initialize(outlvl=outlvl, optarg=solver.options)

    ###########################################################################
    #  Low pressure FWH section                                               #
    ###########################################################################

    # fwh1
    m.fs.fwh1_mix.drain.flow_mol.fix(1434)
    m.fs.fwh1_mix.drain.pressure.fix(14617)
    m.fs.fwh1_mix.drain.enth_mol.fix(3990)

    _set_port(m.fs.fwh1_mix.steam, m.fs.t8_splitter.outlet_2)
    m.fs.fwh1_mix.initialize(outlvl=outlvl, optarg=solver.options)
    m.fs.fwh1_mix.drain.unfix()

    _set_port(m.fs.fwh1.inlet_1, m.fs.fwh1_mix.outlet)
    _set_port(m.fs.fwh1.inlet_2, m.fs.cond_pump.outlet)
    m.fs.fwh1.initialize(outlvl=outlvl, optarg=solver.options)

    # fwh2
    m.fs.fwh2_mix.drain.flow_mol.fix(1136)
    m.fs.fwh2_mix.drain.pressure.fix(35685)
    m.fs.fwh2_mix.drain.enth_mol.fix(5462)
    _set_port(m.fs.fwh2_mix.steam, m.fs.t7_splitter.outlet_2)
    m.fs.fwh2_mix.initialize(outlvl=outlvl, optarg=solver.options)
    m.fs.fwh2_mix.drain.unfix()
    m.fs.fwh2_mix.steam.unfix()

    _set_port(m.fs.fwh2.inlet_1, m.fs.fwh2_mix.outlet)
    _set_port(m.fs.fwh2.inlet_2, m.fs.fwh1.outlet_2)
    m.fs.fwh2.initialize(outlvl=outlvl, optarg=solver.options)

    # fwh3
    m.fs.fwh3_mix.drain.flow_mol.fix(788)
    m.fs.fwh3_mix.drain.pressure.fix(87123)
    m.fs.fwh3_mix.drain.enth_mol.fix(7160)
    _set_port(m.fs.fwh3_mix.steam, m.fs.t6_splitter.outlet_2)
    m.fs.fwh3_mix.initialize(outlvl=outlvl, optarg=solver.options)
    m.fs.fwh3_mix.drain.unfix()
    # m.fs.fwh3_mix.steam.unfix()

    _set_port(m.fs.fwh3.inlet_1, m.fs.fwh3_mix.outlet)
    _set_port(m.fs.fwh3.inlet_2, m.fs.fwh2.outlet_2)
    m.fs.fwh3.initialize(outlvl=outlvl, optarg=solver.options)

    # fwh4
    _set_port(m.fs.fwh4.inlet_2, m.fs.fwh3.outlet_2)
    _set_port(m.fs.fwh4.inlet_1, m.fs.t5_splitter.outlet_2)
    m.fs.fwh4.initialize(outlvl=outlvl, optarg=solver.options)

    ###########################################################################
    #  Boiler feed pump and deaerator                                         #
    ###########################################################################
    # Deaerator
    _set_port(m.fs.fwh5_da.feedwater, m.fs.fwh4.outlet_2)
    m.fs.fwh5_da.drain.flow_mol[:].fix(6207)
    m.fs.fwh5_da.drain.pressure[:].fix(519291)
    m.fs.fwh5_da.drain.enth_mol[:].fix(11526)

    _set_port(m.fs.fwh5_da.steam, m.fs.t4_splitter.outlet_2)
    m.fs.fwh5_da.initialize(outlvl=outlvl, optarg=solver.options)
    m.fs.fwh5_da.drain.unfix()

    # Boiler feed pump
    _set_port(m.fs.bfp.inlet, m.fs.fwh5_da.outlet)
    m.fs.bfp.initialize(outlvl=outlvl, optarg=solver.options)
    
    ###########################################################################
    #  High-pressure feedwater heaters                                        #
    ###########################################################################
    # FWH6
    m.fs.fwh6_mix.drain.flow_mol.fix(5299)
    m.fs.fwh6_mix.drain.pressure.fix(2177587)
    m.fs.fwh6_mix.drain.enth_mol.fix(16559)
    _set_port(m.fs.fwh6_mix.steam, m.fs.t3_splitter.outlet_2)
    m.fs.fwh6_mix.initialize(outlvl=outlvl, optarg=solver.options)
    m.fs.fwh6_mix.drain.unfix()

    _set_port(m.fs.fwh6.inlet_1, m.fs.fwh6_mix.outlet)
    _set_port(m.fs.fwh6.inlet_2, m.fs.bfp.outlet)
    m.fs.fwh6.initialize(outlvl=outlvl, optarg=solver.options)

    # FWH7
    _set_port(m.fs.fwh7_mix.steam, m.fs.t2_splitter.outlet_2)
    m.fs.fwh7_mix.drain.flow_mol.fix(3730)
    m.fs.fwh7_mix.drain.pressure.fix(5590711)
    m.fs.fwh7_mix.drain.enth_mol.fix(21232)

    m.fs.fwh7_mix.initialize(outlvl=outlvl, optarg=solver.options)
    m.fs.fwh7_mix.drain.unfix()

    _set_port(m.fs.fwh7.inlet_1, m.fs.fwh7_mix.outlet)
    _set_port(m.fs.fwh7.inlet_2, m.fs.fwh6.outlet_2)
    m.fs.fwh7.initialize(outlvl=outlvl, optarg=solver.options)

    # FWH8
    _set_port(m.fs.fwh8.inlet_2, m.fs.fwh7.outlet_2)
    _set_port(m.fs.fwh8.inlet_1, m.fs.t1_splitter.outlet_2)
    m.fs.fwh8.initialize(outlvl=outlvl, optarg=solver.options)
    m.fs.fwh8.inlet_1.unfix()

    ###########################################################################
    #  Model Initialization with Square Problem Solve                         #
    ###########################################################################
    #  Unfix split fractions and activate vapor fraction constraints
    #  Vaporfrac constraints set condensed steam enthalpy at the condensing
    #  side outlet to be that of a saturated liquid
    # Then solve the square problem again for an initilized model
    # m.fs.turbine_1.split_fraction.unfix()
    m.fs.t1_splitter.split_fraction[:, "outlet_2"].unfix()
    m.fs.t2_splitter.split_fraction[:, "outlet_2"].unfix()
    m.fs.t3_splitter.split_fraction[:, "outlet_2"].unfix()
    m.fs.t4_splitter.split_fraction[:, "outlet_1"].unfix()
    m.fs.t5_splitter.split_fraction[:, "outlet_2"].unfix()
    m.fs.t6_splitter.split_fraction[:, "outlet_2"].unfix()
    m.fs.t7_splitter.split_fraction[:, "outlet_2"].unfix()
    m.fs.t8_splitter.split_fraction[:, "outlet_2"].unfix()

    m.fs.constraint_out_pressure.activate()

    m.fs.fwh1.fwh1_vaporfrac_constraint.activate()
    m.fs.fwh2.fwh2_vaporfrac_constraint.activate()
    m.fs.fwh3.fwh3_vaporfrac_constraint.activate()
    m.fs.fwh4.fwh4_vaporfrac_constraint.activate()
    m.fs.fwh6.fwh6_vaporfrac_constraint.activate()
    m.fs.fwh7.fwh7_vaporfrac_constraint.activate()
    m.fs.fwh8.fwh8_vaporfrac_constraint.activate()

    res = solver.solve(m, tee=True)
    print("Model Initialization = ",
          res.solver.termination_condition)
    print("*********************Model Initialized**************************")


def build_plant_model(initialize_from_file=None, store_initialization=None):

    """
    Build the plant model
    """

    
    # Create a flowsheet, add properties, unit models, and arcs
    m = create_model()

    # Give all the required inputs to the model. Ensure that the model
    # is complete, i.e., the number of degrees of freedom are 0 
    set_model_input(m)

    # Assert that the model has no degree of freedom at this point
    assert degrees_of_freedom(m) == 0

    # Initialize the model. Ensure after the model is initialized, that
    # the number of degrees of freedom is 0
    initialize(m)

    assert degrees_of_freedom(m) == 0

    # The power plant model is now ready

    # Declaring a plant power out variable for easy analysis of various
    # design and operating scenarios
    m.fs.plant_power_out = pyo.Var(
        m.fs.time,
        domain=pyo.Reals,
        initialize=620,
        doc="Net Power from the power plant in MWe"
    )

    
    # Plant Power Out = Turbine Power - Power required for HX Pump
    @m.fs.Constraint(m.fs.time,
                     doc="Total plant power production in MWe")
    def production_cons(b, t):
        return (
            (-1*(m.fs.turbine_1.work_mechanical[t]
                 + m.fs.turbine_2.work_mechanical[t]
                 + m.fs.turbine_3.work_mechanical[t]
                 + m.fs.turbine_4.work_mechanical[t]
                 + m.fs.turbine_5.work_mechanical[t]
                 + m.fs.turbine_6.work_mechanical[t]
                 + m.fs.turbine_7.work_mechanical[t]
                 + m.fs.turbine_8.work_mechanical[t]
                 + m.fs.turbine_9.work_mechanical[t])
             ) * 1e-6
            == m.fs.plant_power_out[t]
        )

    return m


def model_analysis(m):
    solver = pyo.SolverFactory("ipopt")
    solver.options = {
        "tol": 1e-8,
        "max_iter": 300,
        "halt_on_ampl_error": "yes",
    }

    # Solving the flowsheet.
    # At this time, the user can make changes to the model for further analysis
    solver.solve(m, tee=True, symbolic_solver_labels=True)

    print('Total Power =', pyo.value(m.fs.plant_power_out[0]))

    
def pfd_result(outfile, m):
    tags = {}
    
    tags['power_out'] = ("%4.2f" % pyo.value(m.fs.plant_power_out[0]))
    
    ## Boiler
    tags['boiler_Fin'] = ("%4.2f" % (pyo.value(m.fs.boiler.inlet.flow_mol[0])))
    tags['boiler_Tin'] = ("%4.2f" % (pyo.value(m.fs.boiler.control_volume.properties_in[0].temperature)))
    tags['boiler_Pin'] = ("%4.0f" % (pyo.value(m.fs.boiler.inlet.pressure[0])))
    tags['boiler_Hin'] = ("%4.2f" % (pyo.value(m.fs.boiler.inlet.enth_mol[0])))
    tags['boiler_xin'] = ("%4.2f" % (pyo.value(m.fs.boiler.control_volume.properties_in[0].vapor_frac)))
    tags['boiler_Fout'] = ("%4.2f" % (pyo.value(m.fs.boiler.outlet.flow_mol[0])))
    tags['boiler_Tout'] = ("%4.2f" % (pyo.value(m.fs.boiler.control_volume.properties_out[0].temperature)))
    tags['boiler_Pout'] = ("%4.0f" % (pyo.value(m.fs.boiler.outlet.pressure[0])))
    tags['boiler_Hout'] = ("%4.2f" % (pyo.value(m.fs.boiler.outlet.enth_mol[0])))
    tags['boiler_xout'] = ("%4.2f" % (pyo.value(m.fs.boiler.control_volume.properties_out[0].vapor_frac)))

    # Reheater
    tags['reheat_Fout'] = ("%4.2f" % (pyo.value(m.fs.reheater.outlet.flow_mol[0])))
    tags['reheat_Tout'] = ("%4.2f" % (pyo.value(m.fs.reheater.control_volume.properties_out[0].temperature)))
    tags['reheat_Pout'] = ("%4.0f" % (pyo.value(m.fs.reheater.outlet.pressure[0])))
    tags['reheat_Hout'] = ("%4.2f" % (pyo.value(m.fs.reheater.outlet.enth_mol[0])))
    tags['reheat_xout'] = ("%4.2f" % (pyo.value(m.fs.reheater.control_volume.properties_out[0].vapor_frac)))

    # BFPT
    tags['bfpt_Fin'] = ("%4.2f" % (pyo.value(m.fs.bfpt.inlet.flow_mol[0])))
    tags['bfpt_Tin'] = ("%4.2f" % (pyo.value(m.fs.bfpt.control_volume.properties_in[0].temperature)))
    tags['bfpt_Pin'] = ("%4.0f" % (pyo.value(m.fs.bfpt.inlet.pressure[0])))
    tags['bfpt_Hin'] = ("%4.2f" % (pyo.value(m.fs.bfpt.inlet.enth_mol[0])))
    tags['bfpt_xin'] = ("%4.2f" % (pyo.value(m.fs.bfpt.control_volume.properties_in[0].vapor_frac)))
    tags['bfpt_Fout'] = ("%4.2f" % (pyo.value(m.fs.bfpt.outlet.flow_mol[0])))
    tags['bfpt_Tout'] = ("%4.2f" % (pyo.value(m.fs.bfpt.control_volume.properties_out[0].temperature)))
    tags['bfpt_Pout'] = ("%4.0f" % (pyo.value(m.fs.bfpt.outlet.pressure[0])))
    tags['bfpt_Hout'] = ("%4.2f" % (pyo.value(m.fs.bfpt.outlet.enth_mol[0])))
    tags['bfpt_xout'] = ("%4.2f" % (pyo.value(m.fs.bfpt.control_volume.properties_out[0].vapor_frac)))

    ## Condenser
    tags['cond_Fin'] = ("%4.2f" % (pyo.value(m.fs.condenser.inlet.flow_mol[0])))
    tags['cond_Tin'] = ("%4.2f" % (pyo.value(m.fs.condenser.control_volume.properties_in[0].temperature)))
    tags['cond_Pin'] = ("%4.0f" % (pyo.value(m.fs.condenser.inlet.pressure[0])))
    tags['cond_Hin'] = ("%4.2f" % (pyo.value(m.fs.condenser.inlet.enth_mol[0])))
    tags['cond_xin'] = ("%4.2f" % (pyo.value(m.fs.condenser.control_volume.properties_in[0].vapor_frac)))
    tags['cond_Fout'] = ("%4.2f" % (pyo.value(m.fs.condenser.outlet.flow_mol[0])))
    tags['cond_Tout'] = ("%4.2f" % (pyo.value(m.fs.condenser.control_volume.properties_out[0].temperature)))
    tags['cond_Pout'] = ("%4.0f" % (pyo.value(m.fs.condenser.outlet.pressure[0])))
    tags['cond_Hout'] = ("%4.2f" % (pyo.value(m.fs.condenser.outlet.enth_mol[0])))
    tags['cond_xout'] = ("%4.2f" % (pyo.value(m.fs.condenser.control_volume.properties_out[0].vapor_frac)))

    ## Feed water heaters
    tags['fwh1_Fout1'] = ("%4.2f" % (pyo.value(m.fs.fwh1.outlet_1.flow_mol[0])))
    tags['fwh1_Tout1'] = ("%4.2f" % (pyo.value(m.fs.fwh1.side_1.properties_out[0].temperature)))
    tags['fwh1_Pout1'] = ("%4.0f" % (pyo.value(m.fs.fwh1.outlet_1.pressure[0])))
    tags['fwh1_Hout1'] = ("%4.2f" % (pyo.value(m.fs.fwh1.outlet_1.enth_mol[0])))
    tags['fwh1_xout1'] = ("%4.2f" % (pyo.value(m.fs.fwh1.side_1.properties_out[0].vapor_frac)))

    tags['fwh2_Fout1'] = ("%4.2f" % (pyo.value(m.fs.fwh2.outlet_1.flow_mol[0])))
    tags['fwh2_Tout1'] = ("%4.2f" % (pyo.value(m.fs.fwh2.side_1.properties_out[0].temperature)))
    tags['fwh2_Pout1'] = ("%4.0f" % (pyo.value(m.fs.fwh2.outlet_1.pressure[0])))
    tags['fwh2_Hout1'] = ("%4.2f" % (pyo.value(m.fs.fwh2.outlet_1.enth_mol[0])))
    tags['fwh2_xout1'] = ("%4.2f" % (pyo.value(m.fs.fwh2.side_1.properties_out[0].vapor_frac)))

    tags['fwh3_Fout1'] = ("%4.2f" % (pyo.value(m.fs.fwh3.outlet_1.flow_mol[0])))
    tags['fwh3_Tout1'] = ("%4.2f" % (pyo.value(m.fs.fwh3.side_1.properties_out[0].temperature)))
    tags['fwh3_Pout1'] = ("%4.0f" % (pyo.value(m.fs.fwh3.outlet_1.pressure[0])))
    tags['fwh3_Hout1'] = ("%4.2f" % (pyo.value(m.fs.fwh3.outlet_1.enth_mol[0])))
    tags['fwh3_xout1'] = ("%4.2f" % (pyo.value(m.fs.fwh3.side_1.properties_out[0].vapor_frac)))

    tags['fwh4_Fin1'] = ("%4.2f" % (pyo.value(m.fs.fwh4.inlet_1.flow_mol[0])))
    tags['fwh4_Tin1'] = ("%4.2f" % (pyo.value(m.fs.fwh4.side_1.properties_in[0].temperature)))
    tags['fwh4_Pin1'] = ("%4.0f" % (pyo.value(m.fs.fwh4.inlet_1.pressure[0])))
    tags['fwh4_Hin1'] = ("%4.2f" % (pyo.value(m.fs.fwh4.inlet_1.enth_mol[0])))
    tags['fwh4_xin1'] = ("%4.2f" % (pyo.value(m.fs.fwh4.side_1.properties_in[0].vapor_frac)))
    tags['fwh4_Fout1'] = ("%4.2f" % (pyo.value(m.fs.fwh4.outlet_1.flow_mol[0])))
    tags['fwh4_Tout1'] = ("%4.2f" % (pyo.value(m.fs.fwh4.side_1.properties_out[0].temperature)))
    tags['fwh4_Pout1'] = ("%4.0f" % (pyo.value(m.fs.fwh4.outlet_1.pressure[0])))
    tags['fwh4_Hout1'] = ("%4.2f" % (pyo.value(m.fs.fwh4.outlet_1.enth_mol[0])))
    tags['fwh4_xout1'] = ("%4.2f" % (pyo.value(m.fs.fwh4.side_1.properties_out[0].vapor_frac)))

    tags['fwh6_Fout1'] = ("%4.2f" % (pyo.value(m.fs.fwh6.outlet_1.flow_mol[0])))
    tags['fwh6_Tout1'] = ("%4.2f" % (pyo.value(m.fs.fwh6.side_1.properties_out[0].temperature)))
    tags['fwh6_Pout1'] = ("%4.0f" % (pyo.value(m.fs.fwh6.outlet_1.pressure[0])))
    tags['fwh6_Hout1'] = ("%4.2f" % (pyo.value(m.fs.fwh6.outlet_1.enth_mol[0])))
    tags['fwh6_xout1'] = ("%4.2f" % (pyo.value(m.fs.fwh6.side_1.properties_out[0].vapor_frac)))

    tags['fwh7_Fout1'] = ("%4.2f" % (pyo.value(m.fs.fwh7.outlet_1.flow_mol[0])))
    tags['fwh7_Tout1'] = ("%4.2f" % (pyo.value(m.fs.fwh7.side_1.properties_out[0].temperature)))
    tags['fwh7_Pout1'] = ("%4.0f" % (pyo.value(m.fs.fwh7.outlet_1.pressure[0])))
    tags['fwh7_Hout1'] = ("%4.2f" % (pyo.value(m.fs.fwh7.outlet_1.enth_mol[0])))
    tags['fwh7_xout1'] = ("%4.2f" % (pyo.value(m.fs.fwh7.side_1.properties_out[0].vapor_frac)))

    tags['fwh8_Fin1'] = ("%4.2f" % (pyo.value(m.fs.fwh8.inlet_1.flow_mol[0])))
    tags['fwh8_Tin1'] = ("%4.2f" % (pyo.value(m.fs.fwh8.side_1.properties_in[0].temperature)))
    tags['fwh8_Pin1'] = ("%4.0f" % (pyo.value(m.fs.fwh8.inlet_1.pressure[0])))
    tags['fwh8_Hin1'] = ("%4.2f" % (pyo.value(m.fs.fwh8.inlet_1.enth_mol[0])))
    tags['fwh8_xin1'] = ("%4.2f" % (pyo.value(m.fs.fwh8.side_1.properties_in[0].vapor_frac)))
    tags['fwh8_Fout1'] = ("%4.2f" % (pyo.value(m.fs.fwh8.outlet_1.flow_mol[0])))
    tags['fwh8_Tout1'] = ("%4.2f" % (pyo.value(m.fs.fwh8.side_1.properties_out[0].temperature)))
    tags['fwh8_Pout1'] = ("%4.0f" % (pyo.value(m.fs.fwh8.outlet_1.pressure[0])))
    tags['fwh8_Hout1'] = ("%4.2f" % (pyo.value(m.fs.fwh8.outlet_1.enth_mol[0])))
    tags['fwh8_xout1'] = ("%4.2f" % (pyo.value(m.fs.fwh8.side_1.properties_out[0].vapor_frac)))


    ## Feed water heaters mixers
    tags['fwh1mix_steam_Fin'] = ("%4.2f" % (pyo.value(m.fs.fwh1_mix.steam.flow_mol[0])))
    tags['fwh1mix_steam_Tin'] = ("%4.2f" % (pyo.value(m.fs.fwh1_mix.steam_state[0].temperature)))
    tags['fwh1mix_steam_Pin'] = ("%4.0f" % (pyo.value(m.fs.fwh1_mix.steam.pressure[0])))
    tags['fwh1mix_steam_Hin'] = ("%4.2f" % (pyo.value(m.fs.fwh1_mix.steam.enth_mol[0])))
    tags['fwh1mix_steam_xin'] = ("%4.2f" % (pyo.value(m.fs.fwh1_mix.steam_state[0].vapor_frac)))
    tags['fwh1mix_Fout'] = ("%4.2f" % (pyo.value(m.fs.fwh1_mix.outlet.flow_mol[0])))
    tags['fwh1mix_Tout'] = ("%4.2f" % (pyo.value(m.fs.fwh1_mix.mixed_state[0].temperature)))
    tags['fwh1mix_Pout'] = ("%4.0f" % (pyo.value(m.fs.fwh1_mix.outlet.pressure[0])))
    tags['fwh1mix_Hout'] = ("%4.2f" % (pyo.value(m.fs.fwh1_mix.outlet.enth_mol[0])))
    tags['fwh1mix_xout'] = ("%4.2f" % (pyo.value(m.fs.fwh1_mix.mixed_state[0].vapor_frac)))

    tags['fwh2mix_steam_Fin'] = ("%4.2f" % (pyo.value(m.fs.fwh2_mix.steam.flow_mol[0])))
    tags['fwh2mix_steam_Tin'] = ("%4.2f" % (pyo.value(m.fs.fwh2_mix.steam_state[0].temperature)))
    tags['fwh2mix_steam_Pin'] = ("%4.0f" % (pyo.value(m.fs.fwh2_mix.steam.pressure[0])))
    tags['fwh2mix_steam_Hin'] = ("%4.2f" % (pyo.value(m.fs.fwh2_mix.steam.enth_mol[0])))
    tags['fwh2mix_steam_xin'] = ("%4.2f" % (pyo.value(m.fs.fwh2_mix.steam_state[0].vapor_frac)))
    tags['fwh2mix_Fout'] = ("%4.2f" % (pyo.value(m.fs.fwh2_mix.outlet.flow_mol[0])))
    tags['fwh2mix_Tout'] = ("%4.2f" % (pyo.value(m.fs.fwh2_mix.mixed_state[0].temperature)))
    tags['fwh2mix_Pout'] = ("%4.0f" % (pyo.value(m.fs.fwh2_mix.outlet.pressure[0])))
    tags['fwh2mix_Hout'] = ("%4.2f" % (pyo.value(m.fs.fwh2_mix.outlet.enth_mol[0])))
    tags['fwh2mix_xout'] = ("%4.2f" % (pyo.value(m.fs.fwh2_mix.mixed_state[0].vapor_frac)))

    tags['fwh3mix_steam_Fin'] = ("%4.2f" % (pyo.value(m.fs.fwh3_mix.steam.flow_mol[0])))
    tags['fwh3mix_steam_Tin'] = ("%4.2f" % (pyo.value(m.fs.fwh3_mix.steam_state[0].temperature)))
    tags['fwh3mix_steam_Pin'] = ("%4.0f" % (pyo.value(m.fs.fwh3_mix.steam.pressure[0])))
    tags['fwh3mix_steam_Hin'] = ("%4.2f" % (pyo.value(m.fs.fwh3_mix.steam.enth_mol[0])))
    tags['fwh3mix_steam_xin'] = ("%4.2f" % (pyo.value(m.fs.fwh3_mix.steam_state[0].vapor_frac)))
    tags['fwh3mix_Fout'] = ("%4.2f" % (pyo.value(m.fs.fwh3_mix.outlet.flow_mol[0])))
    tags['fwh3mix_Tout'] = ("%4.2f" % (pyo.value(m.fs.fwh3_mix.mixed_state[0].temperature)))
    tags['fwh3mix_Pout'] = ("%4.0f" % (pyo.value(m.fs.fwh3_mix.outlet.pressure[0])))
    tags['fwh3mix_Hout'] = ("%4.2f" % (pyo.value(m.fs.fwh3_mix.outlet.enth_mol[0])))
    tags['fwh3mix_xout'] = ("%4.2f" % (pyo.value(m.fs.fwh3_mix.mixed_state[0].vapor_frac)))

    tags['fwh6mix_steam_Fin'] = ("%4.2f" % (pyo.value(m.fs.fwh6_mix.steam.flow_mol[0])))
    tags['fwh6mix_steam_Tin'] = ("%4.2f" % (pyo.value(m.fs.fwh6_mix.steam_state[0].temperature)))
    tags['fwh6mix_steam_Pin'] = ("%4.0f" % (pyo.value(m.fs.fwh6_mix.steam.pressure[0])))
    tags['fwh6mix_steam_Hin'] = ("%4.2f" % (pyo.value(m.fs.fwh6_mix.steam.enth_mol[0])))
    tags['fwh6mix_steam_xin'] = ("%4.2f" % (pyo.value(m.fs.fwh6_mix.steam_state[0].vapor_frac)))
    tags['fwh6mix_Fout'] = ("%4.2f" % (pyo.value(m.fs.fwh6_mix.outlet.flow_mol[0])))
    tags['fwh6mix_Tout'] = ("%4.2f" % (pyo.value(m.fs.fwh6_mix.mixed_state[0].temperature)))
    tags['fwh6mix_Pout'] = ("%4.0f" % (pyo.value(m.fs.fwh6_mix.outlet.pressure[0])))
    tags['fwh6mix_Hout'] = ("%4.2f" % (pyo.value(m.fs.fwh6_mix.outlet.enth_mol[0])))
    tags['fwh6mix_xout'] = ("%4.2f" % (pyo.value(m.fs.fwh6_mix.mixed_state[0].vapor_frac)))

    tags['fwh7mix_steam_Fin'] = ("%4.2f" % (pyo.value(m.fs.fwh7_mix.steam.flow_mol[0])))
    tags['fwh7mix_steam_Tin'] = ("%4.2f" % (pyo.value(m.fs.fwh7_mix.steam_state[0].temperature)))
    tags['fwh7mix_steam_Pin'] = ("%4.0f" % (pyo.value(m.fs.fwh7_mix.steam.pressure[0])))
    tags['fwh7mix_steam_Hin'] = ("%4.2f" % (pyo.value(m.fs.fwh7_mix.steam.enth_mol[0])))
    tags['fwh7mix_steam_xin'] = ("%4.2f" % (pyo.value(m.fs.fwh7_mix.steam_state[0].vapor_frac)))
    tags['fwh7mix_Fout'] = ("%4.2f" % (pyo.value(m.fs.fwh7_mix.outlet.flow_mol[0])))
    tags['fwh7mix_Tout'] = ("%4.2f" % (pyo.value(m.fs.fwh7_mix.mixed_state[0].temperature)))
    tags['fwh7mix_Pout'] = ("%4.0f" % (pyo.value(m.fs.fwh7_mix.outlet.pressure[0])))
    tags['fwh7mix_Hout'] = ("%4.2f" % (pyo.value(m.fs.fwh7_mix.outlet.enth_mol[0])))
    tags['fwh7mix_xout'] = ("%4.2f" % (pyo.value(m.fs.fwh7_mix.mixed_state[0].vapor_frac)))

    tags['condmix_mu_Fin'] = ("%4.2f" % (pyo.value(m.fs.condenser_mix.makeup.flow_mol[0])))
    tags['condmix_mu_Tin'] = ("%4.2f" % (pyo.value(m.fs.condenser_mix.makeup_state[0].temperature)))
    tags['condmix_mu_Pin'] = ("%4.0f" % (pyo.value(m.fs.condenser_mix.makeup.pressure[0])))
    tags['condmix_mu_Hin'] = ("%4.2f" % (pyo.value(m.fs.condenser_mix.makeup.enth_mol[0])))
    tags['condmix_mu_xin'] = ("%4.2f" % (pyo.value(m.fs.condenser_mix.makeup_state[0].vapor_frac)))
    
    ## Deareator
    tags['da_steam_Fin'] = ("%4.2f" % (pyo.value(m.fs.fwh5_da.steam.flow_mol[0])))
    tags['da_steam_Tin'] = ("%4.2f" % (pyo.value(m.fs.fwh5_da.steam_state[0].temperature)))
    tags['da_steam_Pin'] = ("%4.0f" % (pyo.value(m.fs.fwh5_da.steam.pressure[0])))
    tags['da_steam_Hin'] = ("%4.2f" % (pyo.value(m.fs.fwh5_da.steam.enth_mol[0])))
    tags['da_steam_xin'] = ("%4.2f" % (pyo.value(m.fs.fwh5_da.steam_state[0].vapor_frac)))
    tags['da_Fout'] = ("%4.2f" % (pyo.value(m.fs.fwh5_da.outlet.flow_mol[0])))
    tags['da_Tout'] = ("%4.2f" % (pyo.value(m.fs.fwh5_da.mixed_state[0].temperature)))
    tags['da_Pout'] = ("%4.0f" % (pyo.value(m.fs.fwh5_da.outlet.pressure[0])))
    tags['da_Hout'] = ("%4.2f" % (pyo.value(m.fs.fwh5_da.outlet.enth_mol[0])))
    tags['da_xout'] = ("%4.2f" % (pyo.value(m.fs.fwh5_da.mixed_state[0].vapor_frac)))

    # BFP
    tags['bfp_Fout'] = ("%4.2f" % (pyo.value(m.fs.bfp.outlet.flow_mol[0])))
    tags['bfp_Tout'] = ("%4.2f" % (pyo.value(m.fs.bfp.control_volume.properties_out[0].temperature)))
    tags['bfp_Pout'] = ("%4.0f" % (pyo.value(m.fs.bfp.outlet.pressure[0])))
    tags['bfp_Hout'] = ("%4.2f" % (pyo.value(m.fs.bfp.outlet.enth_mol[0])))
    tags['bfp_xout'] = ("%4.2f" % (pyo.value(m.fs.bfp.control_volume.properties_out[0].vapor_frac)))

    original_svg_file = os.path.join(this_file_dir(), "scpc_pfd.svg")
    with open(original_svg_file, "r") as f:
        s = svg_tag(tags, f, outfile=outfile)

        
if __name__ == "__main__":
    m = build_plant_model(initialize_from_file=None,
                          store_initialization=None)

    # Import the model from build_plant_model for analysis
    model_analysis(m)

    # Print the results in a process flow diagram (pfd)
    pfd_result("scpc_pfd_results.svg", m)
