##############################################################################
# Institute for the Design of Advanced Energy Systems Process Systems
# Engineering Framework (IDAES PSE Framework) Copyright (c) 2018, by the
# software owners: The Regents of the University of California, through
# Lawrence Berkeley National Laboratory,  National Technology & Engineering
# Solutions of Sandia, LLC, Carnegie Mellon University, West Virginia
# University Research Corporation, et al. All rights reserved.
# 
# Please see the files COPYRIGHT.txt and LICENSE.txt for full copyright and
# license information, respectively. Both files are also available online
# at the URL "https://github.com/IDAES/idaes".
##############################################################################
"""
Tests for flowsheet_model.

Author: Andrew Lee
"""
import pytest
from pyomo.environ import ConcreteModel, Constraint, Var
from pyomo.common.config import ConfigBlock
from idaes.core import (declare_process_block_class, ReactionParameterBase,
                        ReactionBlockBase, ReactionBlockDataBase,
                        PropertyParameterBase, StateBlockBase,
                        StateBlockDataBase)
from idaes.core.util.exceptions import (PropertyPackageError,
                                        PropertyNotSupportedError)

# -----------------------------------------------------------------------------
# Test ParameterBlock
@declare_process_block_class("PropertyParameterBlock")
class _PropertyParameterBlock(PropertyParameterBase):
    def build(self):
        super(_PropertyParameterBlock, self).build()

    @classmethod
    def get_supported_properties(self):
        return {'prop1': {'method': None}}

    @classmethod
    def get_package_units(self):
        return {'time': 's',
                'length': 'm',
                'mass': 'g',
                'amount': 'mol',
                'temperature': 'K',
                'energy': 'J',
                'holdup': 'mol'}


@declare_process_block_class("ReactionParameterBlock")
class _ReactionParameterBlock(ReactionParameterBase):
    def build(self):
        pass


def test_config_block():
    # Test that PropertyParameterBase gets module information
    m = ConcreteModel()
    m.r = ReactionParameterBlock()

    assert len(m.r.config) == 2
    assert isinstance(m.r.config.default_arguments, ConfigBlock)


def test_ReactionParameter_NotImplementedErrors():
    # Test that class methods return NotImplementedError
    m = ConcreteModel()
    m.r = ReactionParameterBlock()

    with pytest.raises(NotImplementedError):
        m.r.get_required_properties()
    with pytest.raises(NotImplementedError):
        m.r.get_supported_properties()
    with pytest.raises(NotImplementedError):
        m.r.get_package_units()


@declare_process_block_class("ReactionParameterBlock2")
class _ReactionParameterBlock2(ReactionParameterBase):
    def build(self):
        pass

    @classmethod
    def get_required_properties(self):
        return ['prop1']

    @classmethod
    def get_supported_properties(self):
        return {'rxn1': {'method': None}}

    @classmethod
    def get_package_units(self):
        return {'time': 'hr',
                'length': 'm',
                'mass': 'g',
                'amount': 'mol',
                'temperature': 'K',
                'energy': 'J',
                'holdup': 'mol'}


def test_validate_state_block_invalid_units():
    # Test validation of associated PropertyParameterBlock
    m = ConcreteModel()
    m.p = PropertyParameterBlock()
    m.r = ReactionParameterBlock2(property_package=m.p)

    with pytest.raises(PropertyPackageError):
        m.r._validate_property_parameter_block()


@declare_process_block_class("ReactionParameterBlock3")
class _ReactionParameterBlock3(ReactionParameterBase):
    def build(self):
        pass

    @classmethod
    def get_required_properties(self):
        return ['prop2']

    @classmethod
    def get_supported_properties(self):
        return {'rxn1': {'method': None}}

    @classmethod
    def get_package_units(self):
        return {'time': 's',
                'length': 'm',
                'mass': 'g',
                'amount': 'mol',
                'temperature': 'K',
                'energy': 'J',
                'holdup': 'mol'}


def test_validate_state_block_unsupported_prop():
    # Test validation of associated PropertyParameterBlock
    m = ConcreteModel()
    m.p = PropertyParameterBlock()
    m.r = ReactionParameterBlock3(property_package=m.p)

    with pytest.raises(PropertyPackageError):
        m.r._validate_property_parameter_block()


@declare_process_block_class("ReactionParameterBlock4")
class _ReactionParameterBlock4(ReactionParameterBase):
    def build(self):
        pass

    @classmethod
    def get_required_properties(self):
        return ['prop1']

    @classmethod
    def get_supported_properties(self):
        return {'rxn1': {'method': None}}

    @classmethod
    def get_package_units(self):
        return {'time': 's',
                'length': 'm',
                'mass': 'g',
                'amount': 'mol',
                'temperature': 'K',
                'energy': 'J',
                'holdup': 'mol'}


def test_ReactionParameterBase_build():
    # Test that ReactionParameterBase gets module information
    m = ConcreteModel()
    m.p = PropertyParameterBlock()
    m.r = ReactionParameterBlock4(property_package=m.p)
    super(_ReactionParameterBlock4, m.r).build()

    assert hasattr(m.r, "property_module")


# -----------------------------------------------------------------------------
# Test ReactionBlockBase
@declare_process_block_class("ReactionBlock",
                             block_class=ReactionBlockBase)
class ReactionBlockData(ReactionBlockDataBase):
    def build(self):
        pass


def test_ReactionBlockBase_initialize():
    # Test that ReactionBlockBase initialize method raise NotImplementedError
    m = ConcreteModel()
    m.r = ReactionBlock()

    with pytest.raises(NotImplementedError):
        m.r.initialize()


# -----------------------------------------------------------------------------
# Test ReactionBlockDataBase
def test_StateBlock_config():
    # Test that ReactionBlockDataBase config has correct arguments
    m = ConcreteModel()
    m.p = ReactionBlock()

    assert len(m.p.config) == 3
    assert hasattr(m.p.config, "parameters")
    assert hasattr(m.p.config, "state_block")
    assert hasattr(m.p.config, "has_equilibrium")

    m.p.config.has_equilibrium = True
    m.p.config.has_equilibrium = False
    with pytest.raises(ValueError):
        m.p.config.has_equilibrium = 'foo'
    with pytest.raises(ValueError):
        m.p.config.has_equilibrium = 10


@declare_process_block_class("StateBlock", block_class=StateBlockBase)
class StateBlockData(StateBlockDataBase):
    def build(self):
        pass


def test_validate_state_block_fail():
    # Test that ReactionBlockDataBase validate_state_block returns error
    m = ConcreteModel()
    m.p = PropertyParameterBlock()
    m.p2 = PropertyParameterBlock()

    m.pb = StateBlock(parameters=m.p2)

    m.r = ReactionParameterBlock4(property_package=m.p)
    super(_ReactionParameterBlock4, m.r).build()

    m.rb = ReactionBlock(parameters=m.r, state_block=m.pb)

    with pytest.raises(PropertyPackageError):
        m.rb._validate_state_block()


@declare_process_block_class("ReactionBlock2",
                             block_class=ReactionBlockBase)
class ReactionBlockData2(ReactionBlockDataBase):
    def build(self):
        super(ReactionBlockData2, self).build()


def test_build():
    # Test that ReactionBlockDataBase builds correctly with good argumnets
    m = ConcreteModel()
    m.p = PropertyParameterBlock()

    m.pb = StateBlock(parameters=m.p)

    m.r = ReactionParameterBlock4(property_package=m.p)
    super(_ReactionParameterBlock4, m.r).build()

    m.rb = ReactionBlock2(parameters=m.r, state_block=m.pb)


def test_ReactionBlock_NotImplementedErrors():
    # Test that placeholder methods return NotImplementedErrors
    m = ConcreteModel()
    m.p = PropertyParameterBlock()

    m.pb = StateBlock(parameters=m.p)

    m.r = ReactionParameterBlock4(property_package=m.p)
    super(_ReactionParameterBlock4, m.r).build()

    m.rb = ReactionBlock2(parameters=m.r, state_block=m.pb)

    with pytest.raises(NotImplementedError):
        m.rb.get_reaction_material_terms()
    with pytest.raises(NotImplementedError):
        m.rb.get_reaction_energy_terms()


# -----------------------------------------------------------------------------
# Test reaction __getattr__ method
@declare_process_block_class("Parameters")
class _Parameters(ReactionParameterBase):
    def build(self):
        pass

    @classmethod
    def get_supported_properties(self):
        return {'a': {'method': 'a_method'},
                'recursion1': {'method': '_recursion1'},
                'recursion2': {'method': '_recursion2'},
                'not_callable': {'method': 'test_obj'},
                'raise_exception': {'method': '_raise_exception'},
                'not_supported': {'method': False},
                'does_not_create_component': {
                        'method': '_does_not_create_component'}}


@declare_process_block_class("Reaction", block_class=ReactionBlockBase)
class _Reaction(ReactionBlockDataBase):
    def build(self):
        self.test_obj = 1

    def a_method(self):
        self.a = Var(initialize=1)

    def _recursion1(self):
        self.recursive_cons1 = Constraint(expr=self.recursion2 == 1)

    def _recursion2(self):
        self.recursive_cons2 = Constraint(expr=self.recursion1 == 1)

    def _raise_exception(self):
        raise Exception()

    def _does_not_create_component(self):
        pass


@pytest.fixture()
def m():
    m = ConcreteModel()
    m.pb = Parameters()
    m.p = Reaction(parameters=m.pb)

    return m


def test_getattr_add_var(m):
    assert isinstance(m.p.a, Var)
    assert m.p.a.value == 1


def test_getattr_protected(m):
    with pytest.raises(PropertyPackageError):
        # Call a protected component that does not exist
        m.p.cons = Constraint(expr=m.p._foo == 1)


def test_getattr_recursion(m):
    with pytest.raises(PropertyPackageError):
        # Call a component that triggers a recursive loop of calls
        m.p.cons = Constraint(expr=m.p.recursion1 == 1)


def test_getattr_does_not_exist(m):
    with pytest.raises(PropertyNotSupportedError):
        m.p.cons = Constraint(expr=m.p.does_not_exist == 1)


def test_getattr_not_callable(m):
    with pytest.raises(PropertyPackageError):
        m.p.cons = Constraint(expr=m.p.not_callable == 1)


def test_getattr_not_supported(m):
    with pytest.raises(PropertyNotSupportedError):
        m.p.cons = Constraint(expr=m.p.not_supported == 1)


def test_getattr_raise_exception(m):
    with pytest.raises(Exception):
        m.p.cons = Constraint(expr=m.p.raise_exception == 1)


## TODO : Need a test for cases where method does not create property
##def test_getattr_does_not_create_component(m):
##    with pytest.raises(PropertyPackageError):
##        m.p.cons = Constraint(expr=m.p.does_not_create_component == 1)
