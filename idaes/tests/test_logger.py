import pytest
import idaes.logger as idaeslog
import logging
import pyomo.environ as pyo

__author__ = "John Eslick"

def test_get_idaes_logger(caplog):
    caplog.set_level(logging.DEBUG)
    log = idaeslog.getLogger("My Test Logger 1")
    assert log.name == "idaes.My Test Logger 1"
    log.info_least("Hello!")
    log.info_less("Hello!")
    log.info("Hello!")
    log.info_more("Hello!")
    log.info_most("Hello!")
    for record in caplog.records:
        assert record.levelname == "INFO"
    log = idaeslog.getLogger("idaes.My Test Logger 2")
    assert log.name == "idaes.My Test Logger 2"

def test_get_model_logger(caplog):
    log = idaeslog.getModelLogger("My Model 1")
    assert isinstance(log, logging.Logger)
    assert log.name == "idaes.model.My Model 1"
    caplog.set_level(idaeslog.INFO_LESS)
    log.info_least("Hello! from least")
    log.info_less("Hello! from less")
    log.info("Hello! from info")
    log.info_more("Hello! from more")
    log.info_most("Hello! from most")
    for record in caplog.records:
        assert record.message in ["Hello! from least", "Hello! from less"]
    log = idaeslog.getModelLogger("idaes.My Model 2")
    assert log.name == "idaes.model.My Model 2"

def test_get_init_logger(caplog):
    log = idaeslog.getInitLogger("My Init 1")
    assert log.name == "idaes.init.My Init 1"

def test_solver_tee():
    log = idaeslog.getInitLogger("solver")
    log.setLevel(idaeslog.SOLVER)
    assert idaeslog.solver_tee(log) == True
    log.setLevel(idaeslog.INFO)
    assert idaeslog.solver_tee(log) == False
    assert idaeslog.solver_tee(log, idaeslog.ERROR) == True

def test_solver_condition():
    # test the results that can be tested without a solver
    assert idaeslog.condition(None) == "Error, no result"
    assert idaeslog.condition("something else") == "something else"


@pytest.mark.skipif(not pyo.SolverFactory('ipopt').available(False), reason="no Ipopt")
def test_solver_condition2():
    solver = pyo.SolverFactory('ipopt')
    model = pyo.ConcreteModel("Solver Result Test Model")
    model.x = pyo.Var([1,2])
    model.y = pyo.Var(initialize=5)
    model.x.fix(2)
    model.y.unfix()
    model.c = pyo.Constraint(expr=model.x[1] + model.x[2]==model.y)
    res = solver.solve(model)
    assert idaeslog.condition(res).startswith("optimal") #better solve
    model.c2 = pyo.Constraint(expr=model.x[1]==model.y)
    res = solver.solve(model)
    assert idaeslog.condition(res).startswith("other") #too few degrees of freedom

def test_increasing_level():
    log = idaeslog.getLogger("log")
    log.setLevel(idaeslog.CRITICAL)

    lvl = idaeslog.increased_output(log)
    assert lvl == idaeslog.ERROR
    log.setLevel(lvl)

    lvl = idaeslog.increased_output(log)
    assert lvl == idaeslog.WARNING
    log.setLevel(lvl)

    lvl = idaeslog.increased_output(log)
    assert lvl == idaeslog.INFO_LEAST
    log.setLevel(lvl)

    lvl = idaeslog.increased_output(log)
    assert lvl == idaeslog.INFO_LESS
    log.setLevel(lvl)

    lvl = idaeslog.increased_output(log)
    assert lvl == idaeslog.INFO
    log.setLevel(lvl)

    lvl = idaeslog.increased_output(log)
    assert lvl == idaeslog.INFO_MORE
    log.setLevel(lvl)

    lvl = idaeslog.increased_output(log)
    assert lvl == idaeslog.INFO_MOST
    log.setLevel(lvl)

    lvl = idaeslog.increased_output(log)
    assert lvl == idaeslog.SOLVER
    log.setLevel(lvl)

    lvl = idaeslog.increased_output(log)
    assert lvl == idaeslog.DEBUG
    log.setLevel(lvl)

    lvl = idaeslog.increased_output(log)
    assert lvl == idaeslog.DEBUG
    log.setLevel(lvl)


def test_decreasing_level():
    log = idaeslog.getLogger("log")
    log.setLevel(idaeslog.DEBUG)

    lvl = idaeslog.decreased_output(log)
    assert lvl == idaeslog.SOLVER
    log.setLevel(lvl)

    lvl = idaeslog.decreased_output(log)
    assert lvl == idaeslog.INFO_MOST
    log.setLevel(lvl)

    lvl = idaeslog.decreased_output(log)
    assert lvl == idaeslog.INFO_MORE
    log.setLevel(lvl)

    lvl = idaeslog.decreased_output(log)
    assert lvl == idaeslog.INFO
    log.setLevel(lvl)

    lvl = idaeslog.decreased_output(log)
    assert lvl == idaeslog.INFO_LESS
    log.setLevel(lvl)

    lvl = idaeslog.decreased_output(log)
    assert lvl == idaeslog.INFO_LEAST
    log.setLevel(lvl)

    lvl = idaeslog.decreased_output(log)
    assert lvl == idaeslog.WARNING
    log.setLevel(lvl)

    lvl = idaeslog.decreased_output(log)
    assert lvl == idaeslog.ERROR
    log.setLevel(lvl)

    lvl = idaeslog.decreased_output(log)
    assert lvl == idaeslog.CRITICAL
    log.setLevel(lvl)

    lvl = idaeslog.decreased_output(log)
    assert lvl == idaeslog.CRITICAL
    log.setLevel(lvl)
