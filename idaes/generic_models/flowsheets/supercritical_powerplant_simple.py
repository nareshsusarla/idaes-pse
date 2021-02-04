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
)
from idaes.generic_models.unit_models.heat_exchanger import (
    delta_temperature_underwood_callback)
from idaes.generic_models.unit_models.pressure_changer import (
    ThermodynamicAssumption)
import idaes.core.util.scaling as iscale
import idaes.logger as idaeslog

# Import Property Packages (IAPWS95 for Water/Steam)
from idaes.generic_models.properties import iapws95


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
    m.fs.turbine_1 = PressureChanger(
        default={
            "property_package": m.fs.prop_water,
            "compressor": False,
            "material_balance_type": MaterialBalanceType.componentTotal,
            "thermodynamic_assumption": ThermodynamicAssumption.isentropic,
        }
    )
    m.fs.turbine_2 = PressureChanger(
        default={
            "property_package": m.fs.prop_water,
            "compressor": False,
            "material_balance_type": MaterialBalanceType.componentTotal,
            "thermodynamic_assumption": ThermodynamicAssumption.isentropic,
        }
    )
    m.fs.turbine_3 = PressureChanger(
        default={
            "property_package": m.fs.prop_water,
            "compressor": False,
            "material_balance_type": MaterialBalanceType.componentTotal,
            "thermodynamic_assumption": ThermodynamicAssumption.isentropic,
        }
    )
    m.fs.turbine_4 = PressureChanger(
        default={
            "property_package": m.fs.prop_water,
            "compressor": False,
            "material_balance_type": MaterialBalanceType.componentTotal,
            "thermodynamic_assumption": ThermodynamicAssumption.isentropic,
        }
    )
    m.fs.turbine_5 = PressureChanger(
        default={
            "property_package": m.fs.prop_water,
            "compressor": False,
            "material_balance_type": MaterialBalanceType.componentTotal,
            "thermodynamic_assumption": ThermodynamicAssumption.isentropic,
        }
    )
    m.fs.turbine_6 = PressureChanger(
        default={
            "property_package": m.fs.prop_water,
            "compressor": False,
            "material_balance_type": MaterialBalanceType.componentTotal,
            "thermodynamic_assumption": ThermodynamicAssumption.isentropic,
        }
    )
    m.fs.turbine_7 = PressureChanger(
        default={
            "property_package": m.fs.prop_water,
            "compressor": False,
            "material_balance_type": MaterialBalanceType.componentTotal,
            "thermodynamic_assumption": ThermodynamicAssumption.isentropic,
        }
    )
    m.fs.turbine_8 = PressureChanger(
        default={
            "property_package": m.fs.prop_water,
            "compressor": False,
            "material_balance_type": MaterialBalanceType.componentTotal,
            "thermodynamic_assumption": ThermodynamicAssumption.isentropic,
        }
    )
    m.fs.turbine_9 = PressureChanger(
        default={
            "property_package": m.fs.prop_water,
            "compressor": False,
            "material_balance_type": MaterialBalanceType.componentTotal,
            "thermodynamic_assumption": ThermodynamicAssumption.isentropic,
        }
    )

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


    @m.fs.boiler.Constraint(m.fs.time, doc="Outlet temperature of boiler")
    def boiler_temperature_constraint(b, t):
        return b.control_volume.properties_out[t].temperature == 866.15  # K


    @m.fs.reheater.Constraint(m.fs.time, doc="Outlet temperature of reheater")
    def reheater_temperature_constraint(b, t):
        return b.control_volume.properties_out[t].temperature == 866.15  # K

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

    
    @m.fs.condenser_mix.Constraint(m.fs.time, doc="Outlet pressure of condenser mixer set to the minimum inlet pressure")
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

    @m.fs.condenser.Constraint(m.fs.time, doc="Outlet enthalpy of condenser as saturated liquid")
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


    # FWH1 Mixer
    m.fs.fwh1_mix = Mixer(
        default={
            "momentum_mixing_type": MomentumMixingType.none,
            "material_balance_type": MaterialBalanceType.componentTotal,
            "inlet_list": ["steam", "drain"],
            "property_package": m.fs.prop_water,
        }
    )

    @m.fs.fwh1_mix.Constraint(m.fs.time, doc="Outlet pressure of FWH1 mixer set to the minimum inlet pressure")
    def fwh1mixer_pressure_constraint(b, t):
        return b.steam_state[t].pressure == b.mixed_state[t].pressure


    # FWH1
    m.fs.fwh1 = HeatExchanger(
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
    
    # Setting the scaling factors for area and overall heat transfer coefficient
    iscale.set_scaling_factor(m.fs.fwh1.area, 1e-2)
    iscale.set_scaling_factor(m.fs.fwh1.overall_heat_transfer_coefficient, 1e-3)

    
    @m.fs.fwh1.Constraint(m.fs.time, doc="Outlet enthalpy of FWH1 condensate as saturated liquid")
    def fwh1_vaporfrac_constraint(b, t):
        return (
            b.side_1.properties_out[t].enth_mol
            == b.side_1.properties_out[t].enth_mol_sat_phase['Liq']
        )

    # Setting the outlet pressure of condensate as described in (2) in FWH description
    # with a pressure ratio for turbine 9 equal to 0.5 (see set_inputs)
    @m.fs.fwh1.Constraint(m.fs.time, doc="Outlet pressure of FWH1 condensate")
    def fwh1_s1pdrop_constraint(b, t):
        return (
            b.side_1.properties_out[t].pressure
            == 1.1 * 0.5 * b.side_1.properties_in[t].pressure
        )

    # Setting the outlet pressure of feed water side as described in (1) in FWH description
    @m.fs.fwh1.Constraint(m.fs.time, doc="Outlet pressure of FWH1 feed water side")
    def fwh1_s2pdrop_constraint(b, t):
        return (
            b.side_2.properties_out[t].pressure
            == 0.96 * b.side_2.properties_in[t].pressure
        )
    

    # FWH2 Mixer
    m.fs.fwh2_mix = Mixer(
        default={
            "momentum_mixing_type": MomentumMixingType.none,
            "material_balance_type": MaterialBalanceType.componentTotal,
            "inlet_list": ["steam", "drain"],
            "property_package": m.fs.prop_water,
        }
    )

    @m.fs.fwh2_mix.Constraint(m.fs.time, doc="Outlet pressure of FWH2 mixer set to the minimum inlet pressure")
    def fwh2mixer_pressure_constraint(b, t):
        return b.steam_state[t].pressure == b.mixed_state[t].pressure


    # FWH2
    m.fs.fwh2 = HeatExchanger(
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
    
    # Setting the scaling factor for area and overall heat transfer coefficient
    iscale.set_scaling_factor(m.fs.fwh2.area, 1e-2)
    iscale.set_scaling_factor(m.fs.fwh2.overall_heat_transfer_coefficient, 1e-3)

    
    @m.fs.fwh2.Constraint(m.fs.time, doc="Outlet enthalpy of FWH2 condensate as saturated liquid")
    def fwh2_vaporfrac_constraint(b, t):
        return (
            b.side_1.properties_out[t].enth_mol
            == b.side_1.properties_out[t].enth_mol_sat_phase['Liq']
        )

    # Setting the outlet pressure of condensate as described in (2) in FWH description
    # with a pressure ratio for turbine 8 equal to 0.64**2 (see set_inputs)
    @m.fs.fwh2.Constraint(m.fs.time, doc="Outlet pressure of FWH2 condensate")
    def fwh2_s1pdrop_constraint(b, t):
        return (
            b.side_1.properties_out[t].pressure
            == 1.1 * (0.64 ** 2) * b.side_1.properties_in[t].pressure
        )

    # Setting the outlet pressure of feed water side as described in (1) in FWH description
    @m.fs.fwh2.Constraint(m.fs.time, doc="Outlet pressure of FWH2 feed water side")
    def fwh2_s2pdrop_constraint(b, t):
        return (
            b.side_2.properties_out[t].pressure
            == 0.96 * b.side_2.properties_in[t].pressure
        )

    
    # FWH3 Mixer
    m.fs.fwh3_mix = Mixer(
        default={
            "momentum_mixing_type": MomentumMixingType.none,
            "material_balance_type": MaterialBalanceType.componentTotal,
            "inlet_list": ["steam", "drain"],
            "property_package": m.fs.prop_water,
        }
    )

    @m.fs.fwh3_mix.Constraint(m.fs.time, doc="Outlet pressure of FWH3 mixer set to the minimum inlet pressure")
    def fwh3mixer_pressure_constraint(b, t):
        return b.steam_state[t].pressure == b.mixed_state[t].pressure

    
    # FWH3
    m.fs.fwh3 = HeatExchanger(
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
    # Setting the scaling factor for area and overall heat transfer coefficient
    iscale.set_scaling_factor(m.fs.fwh3.area, 1e-2)
    iscale.set_scaling_factor(m.fs.fwh3.overall_heat_transfer_coefficient, 1e-3)

    
    @m.fs.fwh3.Constraint(m.fs.time, doc="Outlet enthalpy of FWH3 condensate as saturated liquid")
    def fwh3_vaporfrac_constraint(b, t):
        return (
            b.side_1.properties_out[t].enth_mol
            == b.side_1.properties_out[t].enth_mol_sat_phase['Liq']
        )

    # Setting the outlet pressure of condensate as described in (2) in FWH description
    # with a pressure ratio for turbine 7 equal to 0.64**2 (see set_inputs)
    @m.fs.fwh3.Constraint(m.fs.time, doc="Outlet pressure of FWH3 condensate")
    def fwh3_s1pdrop_constraint(b, t):
        return (
            b.side_1.properties_out[t].pressure
            == 1.1 * (0.64 ** 2) * b.side_1.properties_in[t].pressure
        )

    # Setting the outlet pressure of feed water side as described in (1) in FWH description
    @m.fs.fwh3.Constraint(m.fs.time, doc="Outlet pressure of FWH3 feed water side")
    def fwh3_s2pdrop_constraint(b, t):
        return (
            b.side_2.properties_out[t].pressure
            == 0.96 * b.side_2.properties_in[t].pressure
        )

    # fwh4
    m.fs.fwh4 = HeatExchanger(
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
    # Setting the scaling factor for area and overall heat transfer coefficient
    iscale.set_scaling_factor(m.fs.fwh4.area, 1e-2)
    iscale.set_scaling_factor(m.fs.fwh4.overall_heat_transfer_coefficient, 1e-3)


    @m.fs.fwh4.Constraint(m.fs.time, doc="Outlet enthalpy of FWH4 condensate as saturated liquid")
    def fwh4_vaporfrac_constraint(b, t):
        return (
            b.side_1.properties_out[t].enth_mol
            == b.side_1.properties_out[t].enth_mol_sat_phase['Liq']
        )

    # Setting the outlet pressure of condensate as described in (2) in FWH description
    # with a pressure ratio for turbine 6 equal to 0.64**2 (see set_inputs)
    @m.fs.fwh4.Constraint(m.fs.time, doc="Outlet pressure of FWH4 condensate")
    def fwh4_s1pdrop_constraint(b, t):
        return (
            b.side_1.properties_out[t].pressure
            == 1.1 * (0.64 ** 2) * b.side_1.properties_in[t].pressure
        )

    # Setting the outlet pressure of feed water side as described in (1) in FWH description
    @m.fs.fwh4.Constraint(m.fs.time, doc="Outlet pressure of FWH4 feed water side")
    def fwh4_s2pdrop_constraint(b, t):
        return (
            b.side_2.properties_out[t].pressure
            == 0.96 * b.side_2.properties_in[t].pressure
        )

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

    @m.fs.fwh5_da.Constraint(m.fs.time, doc="Outlet pressure of deaerator set to the minimum inlet pressure")
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
    @m.fs.Constraint(m.fs.time, doc="Outlet pressure of BFPT")
    def constraint_out_pressure(b, t):
        return (
            b.bfpt.control_volume.properties_out[t].pressure
            == b.condenser_mix.mixed_state[t].pressure
        )

    # Boiler feed water turbine produces just enough power to meet the demand of boiler
    # feed water pump
    @m.fs.Constraint(m.fs.time, doc="BFPT produced work equal to demanded work by BFP")
    def constraint_bfp_power(b, t):
        return (
            b.bfp.control_volume.work[t] + b.bfpt.control_volume.work[t]
            == 0
        )

    ###########################################################################
    #  Add high pressure feedwater heaters                                    #
    ###########################################################################
    # FWH6 Mixer
    m.fs.fwh6_mix = Mixer(
        default={
            "momentum_mixing_type": MomentumMixingType.none,
            "material_balance_type": MaterialBalanceType.componentTotal,
            "inlet_list": ["steam", "drain"],
            "property_package": m.fs.prop_water,
        }
    )

    @m.fs.fwh6_mix.Constraint(m.fs.time, doc="Outlet pressure of FWH6 mixer set to the minimum inlet pressure")
    def fwh6mixer_pressure_constraint(b, t):
        return b.steam_state[t].pressure == b.mixed_state[t].pressure

    
    # FWH6
    m.fs.fwh6 = HeatExchanger(
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

    # Setting the scaling factors for area and overall heat transfer coefficient
    iscale.set_scaling_factor(m.fs.fwh6.area, 1e-2)
    iscale.set_scaling_factor(m.fs.fwh6.overall_heat_transfer_coefficient, 1e-3)


    @m.fs.fwh6.Constraint(m.fs.time, doc="Outlet enthalpy of FWH6 condensate as saturated liquid")
    def fwh6_vaporfrac_constraint(b, t):
        return (
            b.side_1.properties_out[t].enth_mol
            == b.side_1.properties_out[t].enth_mol_sat_phase['Liq']
        )

    # Setting the outlet pressure of condensate as described in (2) in FWH description
    # with a pressure ratio for turbine 4 equal to 0.79**6 (see set_inputs)
    @m.fs.fwh6.Constraint(m.fs.time, doc="Outlet pressure of FWH6 condensate")
    def fwh6_s1pdrop_constraint(b, t):
        return (
            b.side_1.properties_out[t].pressure
            == 1.1 * (0.79 ** 6) * b.side_1.properties_in[t].pressure
        )

    # Setting the outlet pressure of feed water side as described in (1) in FWH description
    @m.fs.fwh6.Constraint(m.fs.time, doc="Outlet pressure of FWH6 feed water side")
    def fwh6_s2pdrop_constraint(b, t):
        return (
            b.side_2.properties_out[t].pressure
            == 0.96 * b.side_2.properties_in[t].pressure
        )

    
    # FWH7 Mixer
    m.fs.fwh7_mix = Mixer(
        default={
            "momentum_mixing_type": MomentumMixingType.none,
            "material_balance_type": MaterialBalanceType.componentTotal,
            "inlet_list": ["steam", "drain"],
            "property_package": m.fs.prop_water,
        }
    )

    @m.fs.fwh7_mix.Constraint(m.fs.time, doc="Outlet pressure of FWH7 mixer set to the minimum inlet pressure")
    def fwh7mixer_pressure_constraint(b, t):
        return b.steam_state[t].pressure == b.mixed_state[t].pressure

    
    # FWH7
    m.fs.fwh7 = HeatExchanger(
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
    
    # Setting the scaling factors for area and overall heat transfer coefficient
    iscale.set_scaling_factor(m.fs.fwh7.area, 1e-2)
    iscale.set_scaling_factor(m.fs.fwh7.overall_heat_transfer_coefficient, 1e-3)

    
    @m.fs.fwh7.Constraint(m.fs.time, doc="Outlet enthalpy of FWH7 condensate as saturated liquid")
    def fwh7_vaporfrac_constraint(b, t):
        return (
            b.side_1.properties_out[t].enth_mol
            == b.side_1.properties_out[t].enth_mol_sat_phase['Liq']
        )

    # Setting the outlet pressure of condensate as described in (2) in FWH description
    # with a pressure ratio for turbine 3 equal to 0.79**2 (see set_inputs)
    @m.fs.fwh7.Constraint(m.fs.time, doc="Outlet pressure of FWH7 condensate")
    def fwh7_s1pdrop_constraint(b, t):
        return (
            b.side_1.properties_out[t].pressure
            == 1.1 * (0.79 ** 4) * b.side_1.properties_in[t].pressure
        )

    # Setting the outlet pressure of feed water side as described in (1) in FWH description
    @m.fs.fwh7.Constraint(m.fs.time, doc="Outlet pressure of FWH7 feed water side")
    def fwh7_s2pdrop_constraint(b, t):
        return (
            b.side_2.properties_out[t].pressure
            == 0.96 * b.side_2.properties_in[t].pressure
        )

    # FWH8
    m.fs.fwh8 = HeatExchanger(
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
    
    # Setting the scaling factors for area and overall heat transfer coefficient
    iscale.set_scaling_factor(m.fs.fwh8.area, 1e-2)
    iscale.set_scaling_factor(m.fs.fwh8.overall_heat_transfer_coefficient, 1e-3)

    @m.fs.fwh8.Constraint(m.fs.time, doc="Outlet enthalpy of FWH8 condensate as saturated liquid")
    def fwh8_vaporfrac_constraint(b, t):
        return (
            b.side_1.properties_out[t].enth_mol
            == b.side_1.properties_out[t].enth_mol_sat_phase['Liq']
        )


    # Setting the outlet pressure of condensate as described in (2) in FWH description
    # with a pressure ratio for turbine 2 equal to 0.8**2 (see set_inputs)
    @m.fs.fwh8.Constraint(m.fs.time, doc="Outlet pressure of FWH8 condensate")
    def fwh8_s1pdrop_constraint(b, t):
        return (
            b.side_1.properties_out[t].pressure
            == 1.1 * (0.8 ** 2) * b.side_1.properties_in[t].pressure
        )

    # Setting the outlet pressure of feed water side as described in (1) in FWH description
    @m.fs.fwh8.Constraint(m.fs.time, doc="Outlet pressure of FWH8 feed water side")
    def fwh8_s2pdrop_constraint(b, t):
        return (
            b.side_2.properties_out[t].pressure
            == 0.96 * b.side_2.properties_in[t].pressure
        )

    
    ###########################################################################
    #  Turbine outlet splitter constraints                                    #
    ###########################################################################
    # Equality constraints have been written as following to define
    # the split fractions within the turbine train

    
    # Turbine 1 split separation constraints
    m.fs.turbine_1.split_fraction = pyo.Var(initialize=0.12812)

    @m.fs.turbine_1.Constraint(m.fs.time, doc="Flow from turbine 1 to turbine 2")
    def constraint_turbine1out1_flow(b, t):
        return (
            m.fs.turbine_2.inlet.flow_mol[t]
            == (1 - b.split_fraction) * b.outlet.flow_mol[t]
        )

    @m.fs.turbine_1.Constraint(m.fs.time, doc="Inlet pressure of turbine 2")
    def constraint_turbine1out1_pres(b, t):
        return (
            m.fs.turbine_2.inlet.pressure[t] == b.outlet.pressure[t]
        )

    @m.fs.turbine_1.Constraint(m.fs.time, doc="Inlet enthalpy of turbine 2")
    def constraint_turbine1out1_enth(b, t):
        return (
            m.fs.turbine_2.inlet.enth_mol[t] == b.outlet.enth_mol[t]
        )

    @m.fs.turbine_1.Constraint(m.fs.time, doc="Flow from turbine 1 to FWH8 inlet 1")
    def constraint_turbine1out2_flow(b, t):
        return (
            m.fs.fwh8.inlet_1.flow_mol[t]
            == b.split_fraction * b.outlet.flow_mol[t]
        )

    @m.fs.turbine_1.Constraint(m.fs.time, doc="Inlet pressure of FWH8")
    def constraint_turbine1out2_pres(b, t):
        return (
            m.fs.fwh8.inlet_1.pressure[t] == b.outlet.pressure[t]
        )

    @m.fs.turbine_1.Constraint(m.fs.time, doc="Inlet enthalpy of FWH8")
    def constraint_turbine1out2_enth(b, t):
        return (
            m.fs.fwh8.inlet_1.enth_mol[t] == b.outlet.enth_mol[t]
        )


    # Turbine 2 split separation constraints
    m.fs.turbine_2.split_fraction = pyo.Var(initialize=0.061824)

    @m.fs.turbine_2.Constraint(m.fs.time, doc="Flow from turbine 2 to reheater")
    def constraint_turbine2out1_flow(b, t):
        return (
            m.fs.reheater.inlet.flow_mol[t]
            == (1 - b.split_fraction) * b.outlet.flow_mol[t]
        )

    @m.fs.turbine_2.Constraint(m.fs.time, doc="Inlet pressure of reheater")
    def constraint_turbine2out1_pres(b, t):
        return (
            m.fs.reheater.inlet.pressure[t] == b.outlet.pressure[t]
        )

    @m.fs.turbine_2.Constraint(m.fs.time, doc="Inlet enthalpy of reheater")
    def constraint_turbine2out1_enth(b, t):
        return (
            m.fs.reheater.inlet.enth_mol[t] == b.outlet.enth_mol[t]
        )

    @m.fs.turbine_2.Constraint(m.fs.time, doc="Flow from turbine 2 to FWH7 mixer")
    def constraint_turbine2out2_flow(b, t):
        return (
            m.fs.fwh7_mix.steam.flow_mol[t]
            == b.split_fraction * b.outlet.flow_mol[t]
        )

    @m.fs.turbine_2.Constraint(m.fs.time, doc="Inlet pressure of FWH7 mixer")
    def constraint_turbine2out2_pres(b, t):
        return (
            m.fs.fwh7_mix.steam.pressure[t] == b.outlet.pressure[t]
        )

    @m.fs.turbine_2.Constraint(m.fs.time, doc="Inlet enthalpy of FWH7 mixer")
    def constraint_turbine2out2_enth(b, t):
        return (
            m.fs.fwh7_mix.steam.enth_mol[t] == b.outlet.enth_mol[t]
        )

    
    # Turbine 3 split separation constraints
    m.fs.turbine_3.split_fraction = pyo.Var(initialize=0.03815)

    @m.fs.turbine_3.Constraint(m.fs.time, doc="Flow from turbine 3 to turbine 4")
    def constraint_turbine3out1_flow(b, t):
        return (
            m.fs.turbine_4.inlet.flow_mol[t]
            == (1 - b.split_fraction) * b.outlet.flow_mol[t]
        )

    @m.fs.turbine_3.Constraint(m.fs.time, doc="Inlet pressure of turbine 4")
    def constraint_turbine3out1_pres(b, t):
        return (
            m.fs.turbine_4.inlet.pressure[t] == b.outlet.pressure[t]
        )

    @m.fs.turbine_3.Constraint(m.fs.time, doc="Inlet enthalpy of turbine 4")
    def constraint_turbine3out1_enth(b, t):
        return (
            m.fs.turbine_4.inlet.enth_mol[t] == b.outlet.enth_mol[t]
        )

    @m.fs.turbine_3.Constraint(m.fs.time, doc="Flow from turbine 3 to FWH6 mixer")
    def constraint_turbine3out2_flow(b, t):
        return (
            m.fs.fwh6_mix.steam.flow_mol[t]
            == b.split_fraction * b.outlet.flow_mol[t]
        )

    @m.fs.turbine_3.Constraint(m.fs.time, doc="Inlet pressure of FWH6 mixer")
    def constraint_turbine3out2_pres(b, t):
        return (
            m.fs.fwh6_mix.steam.pressure[t] == b.outlet.pressure[t]
        )

    @m.fs.turbine_3.Constraint(m.fs.time, doc="Inlet enthalpy of FWH6 mixer")
    def constraint_turbine3out2_enth(b, t):
        return (
            m.fs.fwh6_mix.steam.enth_mol[t] == b.outlet.enth_mol[t]
        )

    
    # Turbine 4 split separation constraints
    m.fs.turbine_4.split_fraction1 = pyo.Var(initialize=0.9019)
    m.fs.turbine_4.split_fraction2 = pyo.Var(initialize=0.050331)

    @m.fs.turbine_4.Constraint(m.fs.time, doc="Flow from turbine 4 to turbine 5")
    def constraint_turbine4out1_flow(b, t):
        return (
            m.fs.turbine_5.inlet.flow_mol[t]
            == b.split_fraction1 * b.outlet.flow_mol[t]
        )

    @m.fs.turbine_4.Constraint(m.fs.time, doc="Inlet pressure of turbine 5")
    def constraint_turbine4out1_pres(b, t):
        return (
            m.fs.turbine_5.inlet.pressure[t] == b.outlet.pressure[t]
        )

    @m.fs.turbine_4.Constraint(m.fs.time, doc="Inlet enthalpy of turbine 5")
    def constraint_turbine4out1_enth(b, t):
        return (
            m.fs.turbine_5.inlet.enth_mol[t] == b.outlet.enth_mol[t]
        )

    @m.fs.turbine_4.Constraint(m.fs.time, doc="Flow from turbine 4 to deaerator")
    def constraint_turbine4out2_flow(b, t):
        return (
            m.fs.fwh5_da.steam.flow_mol[t]
            == b.split_fraction2 * b.outlet.flow_mol[t]
        )

    @m.fs.turbine_4.Constraint(m.fs.time, doc="Inlet pressure of deaerator")
    def constraint_turbine4out2_pres(b, t):
        return (
            m.fs.fwh5_da.steam.pressure[t] == b.outlet.pressure[t]
        )

    @m.fs.turbine_4.Constraint(m.fs.time, doc="Inlet enthalpy of deaerator")
    def constraint_turbine4out2_enth(b, t):
        return (
            m.fs.fwh5_da.steam.enth_mol[t] == b.outlet.enth_mol[t]
        )

    @m.fs.turbine_4.Constraint(m.fs.time, doc="Flow from turbine 4 to BFPT")
    def constraint_turbine4out3_flow(b, t):
        return (
            m.fs.bfpt.inlet.flow_mol[t]
            == (1 - b.split_fraction1 - b.split_fraction2)
            * b.outlet.flow_mol[t]
        )

    @m.fs.turbine_4.Constraint(m.fs.time, doc="Inlet pressure of BFPT")
    def constraint_turbine4out3_pres(b, t):
        return (
            m.fs.bfpt.inlet.pressure[t] == b.outlet.pressure[t]
        )

    @m.fs.turbine_4.Constraint(m.fs.time, doc="Inlet enthalpy of BFPT")
    def constraint_turbine4out3_enth(b, t):
        return (
            m.fs.bfpt.inlet.enth_mol[t] == b.outlet.enth_mol[t]
        )


    # Turbine 5 split separation constraints
    m.fs.turbine_5.split_fraction = pyo.Var(initialize=0.0381443)

    @m.fs.turbine_5.Constraint(m.fs.time, doc="Flow from turbine 5 to turbine 6")
    def constraint_turbine5out1_flow(b, t):
        return (
            m.fs.turbine_6.inlet.flow_mol[t]
            == (1 - b.split_fraction) * b.outlet.flow_mol[t]
        )

    @m.fs.turbine_5.Constraint(m.fs.time, doc="Inlet pressure of turbine 6")
    def constraint_turbine5out1_pres(b, t):
        return (
            m.fs.turbine_6.inlet.pressure[t] == b.outlet.pressure[t]
        )

    @m.fs.turbine_5.Constraint(m.fs.time, doc="Inlet enthalpy of turbine 6")
    def constraint_turbine5out1_enth(b, t):
        return (
            m.fs.turbine_6.inlet.enth_mol[t] == b.outlet.enth_mol[t]
        )

    @m.fs.turbine_5.Constraint(m.fs.time, doc="Flow from turbine 5 to FWH4 inlet 1")
    def constraint_turbine5out2_flow(b, t):
        return (
            m.fs.fwh4.inlet_1.flow_mol[t]
            == b.split_fraction * b.outlet.flow_mol[t]
        )

    @m.fs.turbine_5.Constraint(m.fs.time, doc="Inlet pressure of FWH4")
    def constraint_turbine5out2_pres(b, t):
        return (
            m.fs.fwh4.inlet_1.pressure[t] == b.outlet.pressure[t]
        )

    @m.fs.turbine_5.Constraint(m.fs.time, doc="Inlet enthalpy of FWH4")
    def constraint_turbine5out2_enth(b, t):
        return (
            m.fs.fwh4.inlet_1.enth_mol[t] == b.outlet.enth_mol[t]
        )

    
    # Turbine 6 split separation constraints
    m.fs.turbine_6.split_fraction = pyo.Var(initialize=0.017535)

    @m.fs.turbine_6.Constraint(m.fs.time, doc="Flow from turbine 6 to turbine 7")
    def constraint_turbine6out1_flow(b, t):
        return (
            m.fs.turbine_7.inlet.flow_mol[t]
            == (1 - b.split_fraction) * b.outlet.flow_mol[t]
        )

    @m.fs.turbine_6.Constraint(m.fs.time, doc="Inlet pressure of turbine 7")
    def constraint_turbine6out1_pres(b, t):
        return (
            m.fs.turbine_7.inlet.pressure[t] == b.outlet.pressure[t]
        )

    @m.fs.turbine_6.Constraint(m.fs.time, doc="Inlet enthalpy of turbine 7")
    def constraint_turbine6out1_enth(b, t):
        return (
            m.fs.turbine_7.inlet.enth_mol[t] == b.outlet.enth_mol[t]
        )

    @m.fs.turbine_6.Constraint(m.fs.time, doc="Flow from turbine 6 to FWH3 mixer")
    def constraint_turbine6out2_flow(b, t):
        return (
            m.fs.fwh3_mix.steam.flow_mol[t]
            == b.split_fraction * b.outlet.flow_mol[t]
        )

    @m.fs.turbine_6.Constraint(m.fs.time, doc="Inlet pressure of FWH3 mixer")
    def constraint_turbine6out2_pres(b, t):
        return (
            m.fs.fwh3_mix.steam.pressure[t] == b.outlet.pressure[t]
        )

    @m.fs.turbine_6.Constraint(m.fs.time, doc="Inlet enthalpy of FWH3 mixer")
    def constraint_turbine6out2_enth(b, t):
        return (
            m.fs.fwh3_mix.steam.enth_mol[t] == b.outlet.enth_mol[t]
        )


    # Turbine 7 split separation constraints
    m.fs.turbine_7.split_fraction = pyo.Var(initialize=0.0154)

    @m.fs.turbine_7.Constraint(m.fs.time, doc="Flow from turbine 7 to turbine 8")
    def constraint_turbine7out1_flow(b, t):
        return (
            m.fs.turbine_8.inlet.flow_mol[t]
            == (1 - b.split_fraction) * b.outlet.flow_mol[t]
        )

    @m.fs.turbine_7.Constraint(m.fs.time, doc="Inlet pressure of turbine 8")
    def constraint_turbine7out1_pres(b, t):
        return (
            m.fs.turbine_8.inlet.pressure[t] == b.outlet.pressure[t]
        )

    @m.fs.turbine_7.Constraint(m.fs.time, doc="Inlet enthalpy of turbine 8")
    def constraint_turbine7out1_enth(b, t):
        return (
            m.fs.turbine_8.inlet.enth_mol[t] == b.outlet.enth_mol[t]
        )

    @m.fs.turbine_7.Constraint(m.fs.time, doc="Flow from turbine 7 to FWH2 mixer")
    def constraint_turbine7out2_flow(b, t):
        return (
            m.fs.fwh2_mix.steam.flow_mol[t]
            == b.split_fraction*b.outlet.flow_mol[t]
        )

    @m.fs.turbine_7.Constraint(m.fs.time, doc="Inlet pressure of FWH2 mixer")
    def constraint_turbine7out2_pres(b, t):
        return (
            m.fs.fwh2_mix.steam.pressure[t] == b.outlet.pressure[t]
        )

    @m.fs.turbine_7.Constraint(m.fs.time, doc="Inlet enthalpy of FWH2 mixer")
    def constraint_turbine7out2_enth(b, t):
        return (
            m.fs.fwh2_mix.steam.enth_mol[t] == b.outlet.enth_mol[t]
        )

    
    # Turbine 8 split separation constraints
    m.fs.turbine_8.split_fraction = pyo.Var(initialize=0.00121)

    @m.fs.turbine_8.Constraint(m.fs.time, doc="Flow from turbine 8 to turbine 9")
    def constraint_turbine8out1_flow(b, t):
        return (
            m.fs.turbine_9.inlet.flow_mol[t]
            == (1 - b.split_fraction) * b.outlet.flow_mol[t]
        )

    @m.fs.turbine_8.Constraint(m.fs.time, doc="Inlet pressure of turbine 9")
    def constraint_turbine8out1_pres(b, t):
        return (
            m.fs.turbine_9.inlet.pressure[t] == b.outlet.pressure[t]
        )

    @m.fs.turbine_8.Constraint(m.fs.time, doc="Inlet enthalpy of turbine 9")
    def constraint_turbine8out1_enth(b, t):
        return (
            m.fs.turbine_9.inlet.enth_mol[t] == b.outlet.enth_mol[t]
        )

    @m.fs.turbine_8.Constraint(m.fs.time, doc="Flow from turbine 8 to FWH1 mixer")
    def constraint_turbine8out2_flow(b, t):
        return (
            m.fs.fwh1_mix.steam.flow_mol[t]
            == b.split_fraction * b.outlet.flow_mol[t]
        )

    @m.fs.turbine_8.Constraint(m.fs.time, doc="Inlet pressure of FWH1 mixer")
    def constraint_turbine8out2_pres(b, t):
        return (
            m.fs.fwh1_mix.steam.pressure[t] == b.outlet.pressure[t]
        )

    @m.fs.turbine_8.Constraint(m.fs.time, doc="Inlet enthalpy of FWH1 mixer")
    def constraint_turbine8out2_enth(b, t):
        return (
            m.fs.fwh1_mix.steam.enth_mol[t] == b.outlet.enth_mol[t]
        )



    ###########################################################################
    #  Create the stream Arcs and return the model                            #
    ###########################################################################

    _create_arcs(m)

    pyo.TransformationFactory("network.expand_arcs").apply_to(m.fs)


    return m


def _create_arcs(m):

    # Boiler to turbine 1
    m.fs.boiler_to_turb1 = Arc(
        source=m.fs.boiler.outlet, destination=m.fs.turbine_1.inlet
    )
    # Reheater to turbine 3
    m.fs.reheater_to_turb3 = Arc(
        source=m.fs.reheater.outlet, destination=m.fs.turbine_3.inlet
    )
    # Turbine 9 to condenser mixer
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

    # Model inputs / fixed variable or parameter values
    # assumed in this block, unless otherwise stated explicitly,
    # are either assumed or estimated in order to match the results with
    # known baseline scenario for supercritical steam cycle

    # These inputs will also fix all necessary inputs to the model
    # i.e. the degrees of freedom = 0

    ###########################################################################
    #  Turbine input                                                          #
    ###########################################################################
    #  Turbine inlet conditions
    main_steam_pressure = 24235081.4  # Pa
    m.fs.boiler.inlet.flow_mol.fix(29111)  # mol/s
    m.fs.boiler.outlet.pressure.fix(main_steam_pressure)

    # Reheater section pressure drop assumed based on baseline scenario
    m.fs.reheater.deltaP.fix(-96526.64)  # Pa

    # The efficiency and pressure ratios of all turbines were assumed
    # based on results for the baseline scenario
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
    #  Condenser section                                         #
    ###########################################################################
    m.fs.cond_pump.efficiency_pump.fix(0.80)
    m.fs.cond_pump.deltaP.fix(1e6)

    # Make up stream to condenser
    m.fs.condenser_mix.makeup.flow_mol.value = 1.08002495835536E-12  # mol/s
    m.fs.condenser_mix.makeup.pressure.fix(103421.4)  # Pa
    m.fs.condenser_mix.makeup.enth_mol.fix(1131.69204)  # J/mol

    ###########################################################################
    #  Low pressure FWH section inputs                                        #
    ###########################################################################
    # fwh1
    m.fs.fwh1.area.fix(400)
    m.fs.fwh1.overall_heat_transfer_coefficient.fix(2000)
    # fwh2
    m.fs.fwh2.area.fix(300)
    m.fs.fwh2.overall_heat_transfer_coefficient.fix(2900)
    # fwh3
    m.fs.fwh3.area.fix(200)
    m.fs.fwh3.overall_heat_transfer_coefficient.fix(2900)
    # fwh4
    m.fs.fwh4.area.fix(200)
    m.fs.fwh4.overall_heat_transfer_coefficient.fix(2900)

    ###########################################################################
    #  Deaerator and boiler feed pump (BFP) Input                             #
    ###########################################################################
    # Unlike the feedwater heaters the steam extraction flow to the deaerator
    # is not constrained by the saturated liquid constraint. Thus, the flow
    # to the deaerator is fixed in this model. The value of this split fraction
    # is again based on the baseline results
    m.fs.turbine_4.split_fraction2.fix(0.050331)

    m.fs.bfp.efficiency_pump.fix(0.80)
    # BFW Pump pressure is assumed to be 15% more than
    # the desired main steam (Turbine Inlet) pressure
    # To account for the pressure drop across Feed water heaters and Boiler
    m.fs.bfp.outlet.pressure[:].fix(main_steam_pressure * 1.15)  # Pa
    m.fs.bfpt.efficiency_isentropic.fix(0.80)
    ###########################################################################
    #  High pressure feedwater heater                                         #
    ###########################################################################
    # fwh6
    m.fs.fwh6.area.fix(600)
    m.fs.fwh6.overall_heat_transfer_coefficient.fix(2900)
    # fwh7
    m.fs.fwh7.area.fix(400)
    m.fs.fwh7.overall_heat_transfer_coefficient.fix(2900)
    # fwh8
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

    # initializing the boiler
    m.fs.boiler.inlet.pressure.fix(24657896)
    m.fs.boiler.inlet.enth_mol.fix(20004)
    m.fs.boiler.initialize(outlvl=outlvl, optarg=solver.options)
    m.fs.boiler.inlet.pressure.unfix()
    m.fs.boiler.inlet.enth_mol.unfix()

    # initialization routine for the turbine train

    # Deactivating constraints that fix enthalpy at FWH outlet
    # This lets us initialize the model using the fixed split_fractions
    # for steam extractions for all the feed water heaters except deaerator
    # These split fractions will be unfixed later and the constraints will
    # be activated
    m.fs.turbine_1.split_fraction.fix(0.12812)
    m.fs.turbine_2.split_fraction.fix(0.061824)
    m.fs.turbine_3.split_fraction.fix(0.03815)
    m.fs.turbine_4.split_fraction1.fix(0.9019)
    m.fs.turbine_5.split_fraction.fix(0.0381443)
    m.fs.turbine_6.split_fraction.fix(0.017535)
    m.fs.turbine_7.split_fraction.fix(0.0154)
    m.fs.turbine_8.split_fraction.fix(0.00121)

    m.fs.constraint_out_pressure.deactivate()
    m.fs.fwh1.fwh1_vaporfrac_constraint.deactivate()
    m.fs.fwh2.fwh2_vaporfrac_constraint.deactivate()
    m.fs.fwh3.fwh3_vaporfrac_constraint.deactivate()
    m.fs.fwh4.fwh4_vaporfrac_constraint.deactivate()
    m.fs.fwh6.fwh6_vaporfrac_constraint.deactivate()
    m.fs.fwh7.fwh7_vaporfrac_constraint.deactivate()
    m.fs.fwh8.fwh8_vaporfrac_constraint.deactivate()

    # solving the turbine
    _set_port(m.fs.turbine_1.inlet,  m.fs.boiler.outlet)
    m.fs.turbine_1.constraint_turbine1out1_flow.deactivate()
    m.fs.turbine_1.constraint_turbine1out1_pres.deactivate()
    m.fs.turbine_1.constraint_turbine1out1_enth.deactivate()
    m.fs.turbine_1.constraint_turbine1out2_flow.deactivate()
    m.fs.turbine_1.constraint_turbine1out2_pres.deactivate()
    m.fs.turbine_1.constraint_turbine1out2_enth.deactivate()
    m.fs.turbine_1.initialize(outlvl=outlvl, optarg=solver.options)
    m.fs.turbine_1.constraint_turbine1out1_flow.activate()
    m.fs.turbine_1.constraint_turbine1out1_pres.activate()
    m.fs.turbine_1.constraint_turbine1out1_enth.activate()
    m.fs.turbine_1.constraint_turbine1out2_flow.activate()
    m.fs.turbine_1.constraint_turbine1out2_pres.activate()
    m.fs.turbine_1.constraint_turbine1out2_enth.activate()

    m.fs.turbine_2.constraint_turbine2out1_flow.deactivate()
    m.fs.turbine_2.constraint_turbine2out1_pres.deactivate()
    m.fs.turbine_2.constraint_turbine2out1_enth.deactivate()
    m.fs.turbine_2.constraint_turbine2out2_flow.deactivate()
    m.fs.turbine_2.constraint_turbine2out2_pres.deactivate()
    m.fs.turbine_2.constraint_turbine2out2_enth.deactivate()
    m.fs.turbine_2.inlet.flow_mol[:].value = 25381
    m.fs.turbine_2.inlet.pressure[:].value = 7941352
    m.fs.turbine_2.inlet.enth_mol[:].value = 56933
    m.fs.turbine_2.initialize(outlvl=outlvl, optarg=solver.options)
    m.fs.turbine_2.constraint_turbine2out2_flow.activate()
    m.fs.turbine_2.constraint_turbine2out2_pres.activate()
    m.fs.turbine_2.constraint_turbine2out2_enth.activate()

    m.fs.reheater.inlet.flow_mol[:].value = 23812
    m.fs.reheater.inlet.pressure[:].value = 5082465
    m.fs.reheater.inlet.enth_mol[:].value = 54963
    m.fs.reheater.initialize(outlvl=outlvl, optarg=solver.options)
    m.fs.turbine_2.constraint_turbine2out1_flow.activate()
    m.fs.turbine_2.constraint_turbine2out1_pres.activate()
    m.fs.turbine_2.constraint_turbine2out1_enth.activate()

    # Eenrgy storage splitter outlet 2: Turbine train (Turb 3)
    _set_port(m.fs.turbine_3.inlet, m.fs.reheater.outlet)
    m.fs.turbine_3.constraint_turbine3out1_flow.deactivate()
    m.fs.turbine_3.constraint_turbine3out1_pres.deactivate()
    m.fs.turbine_3.constraint_turbine3out1_enth.deactivate()
    m.fs.turbine_3.constraint_turbine3out2_flow.deactivate()
    m.fs.turbine_3.constraint_turbine3out2_pres.deactivate()
    m.fs.turbine_3.constraint_turbine3out2_enth.deactivate()
    m.fs.turbine_3.initialize(outlvl=outlvl, optarg=solver.options)
    m.fs.turbine_3.constraint_turbine3out1_flow.activate()
    m.fs.turbine_3.constraint_turbine3out1_pres.activate()
    m.fs.turbine_3.constraint_turbine3out1_enth.activate()
    m.fs.turbine_3.constraint_turbine3out2_flow.activate()
    m.fs.turbine_3.constraint_turbine3out2_pres.activate()
    m.fs.turbine_3.constraint_turbine3out2_enth.activate()

    m.fs.turbine_4.constraint_turbine4out1_flow.deactivate()
    m.fs.turbine_4.constraint_turbine4out1_pres.deactivate()
    m.fs.turbine_4.constraint_turbine4out1_enth.deactivate()
    m.fs.turbine_4.constraint_turbine4out2_flow.deactivate()
    m.fs.turbine_4.constraint_turbine4out2_pres.deactivate()
    m.fs.turbine_4.constraint_turbine4out2_enth.deactivate()
    m.fs.turbine_4.constraint_turbine4out3_flow.deactivate()
    m.fs.turbine_4.constraint_turbine4out3_pres.deactivate()
    m.fs.turbine_4.constraint_turbine4out3_enth.deactivate()
    m.fs.turbine_4.inlet.flow_mol[:].value = 22904
    m.fs.turbine_4.inlet.pressure[:].value = 1942027
    m.fs.turbine_4.inlet.enth_mol[:].value = 60341
    m.fs.turbine_4.initialize(outlvl=outlvl, optarg=solver.options)
    m.fs.turbine_4.constraint_turbine4out1_flow.activate()
    m.fs.turbine_4.constraint_turbine4out1_pres.activate()
    m.fs.turbine_4.constraint_turbine4out1_enth.activate()
    m.fs.turbine_4.constraint_turbine4out2_flow.activate()
    m.fs.turbine_4.constraint_turbine4out2_pres.activate()
    m.fs.turbine_4.constraint_turbine4out2_enth.activate()
    m.fs.turbine_4.constraint_turbine4out3_flow.activate()
    m.fs.turbine_4.constraint_turbine4out3_pres.activate()
    m.fs.turbine_4.constraint_turbine4out3_enth.activate()

    m.fs.turbine_5.constraint_turbine5out1_flow.deactivate()
    m.fs.turbine_5.constraint_turbine5out1_pres.deactivate()
    m.fs.turbine_5.constraint_turbine5out1_enth.deactivate()
    m.fs.turbine_5.constraint_turbine5out2_flow.deactivate()
    m.fs.turbine_5.constraint_turbine5out2_pres.deactivate()
    m.fs.turbine_5.constraint_turbine5out2_enth.deactivate()
    m.fs.turbine_5.inlet.flow_mol[:].value = 20656
    m.fs.turbine_5.inlet.pressure[:].value = 472082
    m.fs.turbine_5.inlet.enth_mol[:].value = 53882
    m.fs.turbine_5.initialize(outlvl=outlvl, optarg=solver.options)
    m.fs.turbine_5.constraint_turbine5out1_flow.activate()
    m.fs.turbine_5.constraint_turbine5out1_pres.activate()
    m.fs.turbine_5.constraint_turbine5out1_enth.activate()
    m.fs.turbine_5.constraint_turbine5out2_flow.activate()
    m.fs.turbine_5.constraint_turbine5out2_pres.activate()
    m.fs.turbine_5.constraint_turbine5out2_enth.activate()

    m.fs.turbine_6.constraint_turbine6out1_flow.deactivate()
    m.fs.turbine_6.constraint_turbine6out1_pres.deactivate()
    m.fs.turbine_6.constraint_turbine6out1_enth.deactivate()
    m.fs.turbine_6.constraint_turbine6out2_flow.deactivate()
    m.fs.turbine_6.constraint_turbine6out2_pres.deactivate()
    m.fs.turbine_6.constraint_turbine6out2_enth.deactivate()
    m.fs.turbine_6.inlet.flow_mol[:].value = 19868
    m.fs.turbine_6.inlet.pressure[:].value = 193365
    m.fs.turbine_6.inlet.enth_mol[:].value = 50728
    m.fs.turbine_6.initialize(outlvl=outlvl, optarg=solver.options)
    m.fs.turbine_6.constraint_turbine6out1_flow.activate()
    m.fs.turbine_6.constraint_turbine6out1_pres.activate()
    m.fs.turbine_6.constraint_turbine6out1_enth.activate()
    m.fs.turbine_6.constraint_turbine6out2_flow.activate()
    m.fs.turbine_6.constraint_turbine6out2_pres.activate()
    m.fs.turbine_6.constraint_turbine6out2_enth.activate()

    m.fs.turbine_7.constraint_turbine7out1_flow.deactivate()
    m.fs.turbine_7.constraint_turbine7out1_pres.deactivate()
    m.fs.turbine_7.constraint_turbine7out1_enth.deactivate()
    m.fs.turbine_7.constraint_turbine7out2_flow.deactivate()
    m.fs.turbine_7.constraint_turbine7out2_pres.deactivate()
    m.fs.turbine_7.constraint_turbine7out2_enth.deactivate()
    m.fs.turbine_7.inlet.flow_mol[:].value = 19520
    m.fs.turbine_7.inlet.pressure[:].value = 79202
    m.fs.turbine_7.inlet.enth_mol[:].value = 48110
    m.fs.turbine_7.initialize(outlvl=outlvl, optarg=solver.options)
    m.fs.turbine_7.constraint_turbine7out1_flow.activate()
    m.fs.turbine_7.constraint_turbine7out1_pres.activate()
    m.fs.turbine_7.constraint_turbine7out1_enth.activate()
    m.fs.turbine_7.constraint_turbine7out2_flow.activate()
    m.fs.turbine_7.constraint_turbine7out2_pres.activate()
    m.fs.turbine_7.constraint_turbine7out2_enth.activate()

    m.fs.turbine_8.constraint_turbine8out1_flow.deactivate()
    m.fs.turbine_8.constraint_turbine8out1_pres.deactivate()
    m.fs.turbine_8.constraint_turbine8out1_enth.deactivate()
    m.fs.turbine_8.constraint_turbine8out2_flow.deactivate()
    m.fs.turbine_8.constraint_turbine8out2_pres.deactivate()
    m.fs.turbine_8.constraint_turbine8out2_enth.deactivate()
    m.fs.turbine_8.inlet.flow_mol[:].value = 19219
    m.fs.turbine_8.inlet.pressure[:].value = 32441
    m.fs.turbine_8.inlet.enth_mol[:].value = 45836
    m.fs.turbine_8.initialize(outlvl=outlvl, optarg=solver.options)
    m.fs.turbine_8.constraint_turbine8out1_flow.activate()
    m.fs.turbine_8.constraint_turbine8out1_pres.activate()
    m.fs.turbine_8.constraint_turbine8out1_enth.activate()
    m.fs.turbine_8.constraint_turbine8out2_flow.activate()
    m.fs.turbine_8.constraint_turbine8out2_pres.activate()
    m.fs.turbine_8.constraint_turbine8out2_enth.activate()

    m.fs.turbine_9.inlet.flow_mol[:].value = 19196
    m.fs.turbine_9.inlet.pressure[:].value = 13288
    m.fs.turbine_9.inlet.enth_mol[:].value = 43763
    m.fs.turbine_9.initialize(outlvl=outlvl, optarg=solver.options)

    # initialize the boiler feed pump turbine.
    m.fs.bfpt.inlet.flow_mol.fix(1095)
    m.fs.bfpt.inlet.pressure.fix(472082)
    m.fs.bfpt.inlet.enth_mol.fix(53882)
    m.fs.bfpt.initialize(outlvl=outlvl, optarg=solver.options)
    m.fs.bfpt.inlet.unfix()

    ###########################################################################
    #  Condenser                                                #
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
    m.fs.fwh1_mix.steam.flow_mol.fix(23)
    m.fs.fwh1_mix.steam.pressure.fix(13288)
    m.fs.fwh1_mix.steam.enth_mol.fix(43763)
    m.fs.fwh1_mix.initialize(outlvl=outlvl, optarg=solver.options)
    m.fs.fwh1_mix.drain.unfix()
    m.fs.fwh1_mix.steam.unfix()

    _set_port(m.fs.fwh1.inlet_1, m.fs.fwh1_mix.outlet)
    _set_port(m.fs.fwh1.inlet_2, m.fs.cond_pump.outlet)
    m.fs.fwh1.initialize(outlvl=outlvl, optarg=solver.options)

    # fwh2
    m.fs.fwh2_mix.drain.flow_mol.fix(1136)
    m.fs.fwh2_mix.drain.pressure.fix(35685)
    m.fs.fwh2_mix.drain.enth_mol.fix(5462)
    m.fs.fwh2_mix.steam.flow_mol.fix(300)
    m.fs.fwh2_mix.steam.pressure.fix(32441)
    m.fs.fwh2_mix.steam.enth_mol.fix(45836)
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
    m.fs.fwh3_mix.steam.flow_mol.fix(348)
    m.fs.fwh3_mix.steam.pressure.fix(79202)
    m.fs.fwh3_mix.steam.enth_mol.fix(48110)
    m.fs.fwh3_mix.initialize(outlvl=outlvl, optarg=solver.options)
    m.fs.fwh3_mix.drain.unfix()
    m.fs.fwh3_mix.steam.unfix()

    _set_port(m.fs.fwh3.inlet_1, m.fs.fwh3_mix.outlet)
    _set_port(m.fs.fwh3.inlet_2, m.fs.fwh2.outlet_2)
    m.fs.fwh3.initialize(outlvl=outlvl, optarg=solver.options)

    # fwh4
    _set_port(m.fs.fwh4.inlet_2, m.fs.fwh3.outlet_2)
    m.fs.fwh4.inlet_1.flow_mol.fix(788)
    m.fs.fwh4.inlet_1.pressure.fix(193365)
    m.fs.fwh4.inlet_1.enth_mol.fix(50728)
    m.fs.fwh4.initialize(outlvl=outlvl, optarg=solver.options)
    m.fs.fwh4.inlet_1.unfix()

    ###########################################################################
    #  boiler feed pump and deaerator                                         #
    ###########################################################################
    # Deaerator
    _set_port(m.fs.fwh5_da.feedwater, m.fs.fwh4.outlet_2)
    m.fs.fwh5_da.drain.flow_mol[:].fix(6207)
    m.fs.fwh5_da.drain.pressure[:].fix(519291)
    m.fs.fwh5_da.drain.enth_mol[:].fix(11526)
    m.fs.fwh5_da.steam.flow_mol.fix(1153)
    m.fs.fwh5_da.steam.pressure.fix(472082)
    m.fs.fwh5_da.steam.enth_mol.fix(53882)
    m.fs.fwh5_da.initialize(outlvl=outlvl, optarg=solver.options)
    m.fs.fwh5_da.drain.unfix()
    m.fs.fwh5_da.steam.unfix()
    # Boiler feed pump
    _set_port(m.fs.bfp.inlet, m.fs.fwh5_da.outlet)
    m.fs.bfp.initialize(outlvl=outlvl, optarg=solver.options)
    ###########################################################################
    #  High-pressure feedwater heaters                                        #
    ###########################################################################
    # fwh6
    m.fs.fwh6_mix.drain.flow_mol.fix(5299)
    m.fs.fwh6_mix.drain.pressure.fix(2177587)
    m.fs.fwh6_mix.drain.enth_mol.fix(16559)
    m.fs.fwh6_mix.steam.flow_mol.fix(908)
    m.fs.fwh6_mix.steam.pressure.fix(1942027)
    m.fs.fwh6_mix.steam.enth_mol.fix(60341)
    m.fs.fwh6_mix.initialize(outlvl=outlvl, optarg=solver.options)
    m.fs.fwh6_mix.drain.unfix()
    m.fs.fwh6_mix.steam.unfix()

    _set_port(m.fs.fwh6.inlet_1, m.fs.fwh6_mix.outlet)
    _set_port(m.fs.fwh6.inlet_2, m.fs.bfp.outlet)
    m.fs.fwh6.initialize(outlvl=outlvl, optarg=solver.options)

    # fwh7
    m.fs.fwh7_mix.drain.flow_mol.fix(3730)
    m.fs.fwh7_mix.drain.pressure.fix(5590711)
    m.fs.fwh7_mix.drain.enth_mol.fix(21232)
    m.fs.fwh7_mix.steam.flow_mol.fix(1569)
    m.fs.fwh7_mix.steam.pressure.fix(5082465)
    m.fs.fwh7_mix.steam.enth_mol.fix(54963)
    m.fs.fwh7_mix.initialize(outlvl=outlvl, optarg=solver.options)
    m.fs.fwh7_mix.drain.unfix()
    m.fs.fwh7_mix.steam.unfix()

    _set_port(m.fs.fwh7.inlet_1, m.fs.fwh7_mix.outlet)
    _set_port(m.fs.fwh7.inlet_2, m.fs.fwh6.outlet_2)
    m.fs.fwh7.initialize(outlvl=outlvl, optarg=solver.options)

    # fwh8
    _set_port(m.fs.fwh8.inlet_2, m.fs.fwh7.outlet_2)
    m.fs.fwh8.inlet_1.flow_mol.fix(3730)
    m.fs.fwh8.inlet_1.pressure.fix(7941351)
    m.fs.fwh8.inlet_1.enth_mol.fix(56933)
    m.fs.fwh8.initialize(outlvl=outlvl, optarg=solver.options)
    m.fs.fwh8.inlet_1.unfix()

    ###########################################################################
    #  Model Initialization with Square Problem Solve                         #
    ###########################################################################
    #  Unfix split fractions and activate vapor fraction constraints
    #  Vaporfrac constraints set condensed steam enthalpy at the condensing
    #  side outlet to be that of a saturated liquid
    # Then solve the square problem again for an initilized model
    m.fs.turbine_1.split_fraction.unfix()
    m.fs.turbine_2.split_fraction.unfix()
    m.fs.turbine_3.split_fraction.unfix()
    m.fs.turbine_4.split_fraction1.unfix()
    m.fs.turbine_5.split_fraction.unfix()
    m.fs.turbine_6.split_fraction.unfix()
    m.fs.turbine_7.split_fraction.unfix()
    m.fs.turbine_8.split_fraction.unfix()
    m.fs.constraint_out_pressure.activate()

    m.fs.fwh1.fwh1_vaporfrac_constraint.activate()
    m.fs.fwh2.fwh2_vaporfrac_constraint.activate()
    m.fs.fwh3.fwh3_vaporfrac_constraint.activate()
    m.fs.fwh4.fwh4_vaporfrac_constraint.activate()
    m.fs.fwh6.fwh6_vaporfrac_constraint.activate()
    m.fs.fwh7.fwh7_vaporfrac_constraint.activate()
    m.fs.fwh8.fwh8_vaporfrac_constraint.activate()

    res = solver.solve(m)
    print("Model Initialization = ",
          res.solver.termination_condition)
    print("*********************Model Initialized**************************")


def build_plant_model(initialize_from_file=None, store_initialization=None):

    # Create a flowsheet, add properties, unit models, and arcs
    m = create_model()

    # Give all the required inputs to the model
    # Ensure that the degrees of freedom = 0 (model is complete)
    set_model_input(m)
    # Assert that the model has no degree of freedom at this point
    assert degrees_of_freedom(m) == 0

    # Initialize the model (sequencial initialization and custom routines)
    # Ensure after the model is initialized, the degrees of freedom = 0
    initialize(m)
    assert degrees_of_freedom(m) == 0

    # The power plant with storage for a charge scenario is now ready
    #  Declaraing a plant power out variable for easy analysis of various
    #  design and operating scenarios
    m.fs.plant_power_out = pyo.Var(
        m.fs.time,
        domain=pyo.Reals,
        initialize=620,
        doc="Net Power MWe out from the power plant"
    )

    #   Constraint on Plant Power Output
    #   Plant Power Out = Turbine Power - Power required for HX Pump
    @m.fs.Constraint(m.fs.time, doc="Total plant power production in MWe")
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

#   Solving the flowsheet and check result
#   At this time one can make chnages to the model for further analysis
    solver.solve(m, tee=True, symbolic_solver_labels=True)

    print('Total Power =', pyo.value(m.fs.plant_power_out[0]))


if __name__ == "__main__":
    m = build_plant_model(initialize_from_file=None,
                          store_initialization=None)

    #  At this point the model has 0 degrees of freedom
    #  A sensitivity analysis is done by varying the following variables
    #  1) Boiler feed water flow: m.fs.boiler.inlet.flow_mol[0]
    #  2) Steam flow to storage: m.fs.ess_split.split_fraction[:,"outlet_2"]

    # User can import the model from build_plant_model for analysis
    # A sample analysis function is called below
    model_analysis(m)
