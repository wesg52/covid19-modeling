"""File contains the Enums used by the model class as well as
the getter methods required by the model reporter"""

from enum import Enum


class Condition(Enum):
    ASYMPTOMATIC_INFECTED = 0
    SYMPTOMATIC_INFECTED = 1
    IMMUNE_HEALTHY = 2
    NAIVE_HEALTHY = 3
    DECEASED = 4


class Location(Enum):
    HOME = 0
    IN_STORE = 1
    PUBLIC = 2


class Source(Enum):
    INITIAL = 0
    PUBLIC = 1
    CUSTOMER = 2
    EMPLOYEE = 3


def get_working_employees(m):
    return m.state_info['n_employees_working']


def get_infected_employees_working(m):
    return m.state_info['n_infected_employees_working']


def get_n_customers_in_store(m):
    return m.state_info['n_customers_in_store']


def get_n_infected_customers_in_store(m):
    return m.state_info['n_infected_customers_in_store']


def get_n_cust_AI(m):
    return m.state_info['customer_condition'][Condition.ASYMPTOMATIC_INFECTED]


def get_n_cust_SI(m):
    return m.state_info['customer_condition'][Condition.SYMPTOMATIC_INFECTED]


def get_n_cust_IH(m):
    return m.state_info['customer_condition'][Condition.IMMUNE_HEALTHY]


def get_n_cust_NH(m):
    return m.state_info['customer_condition'][Condition.NAIVE_HEALTHY]


def get_n_cust_D(m):
    return m.state_info['customer_condition'][Condition.DECEASED]


def get_n_emp_AI(m):
    return m.state_info['employee_condition'][Condition.ASYMPTOMATIC_INFECTED]


def get_n_emp_SI(m):
    return m.state_info['employee_condition'][Condition.SYMPTOMATIC_INFECTED]


def get_n_emp_IH(m):
    return m.state_info['employee_condition'][Condition.IMMUNE_HEALTHY]


def get_n_emp_NH(m):
    return m.state_info['employee_condition'][Condition.NAIVE_HEALTHY]


def get_n_emp_D(m):
    return m.state_info['employee_condition'][Condition.DECEASED]


def get_n_source_initial(m):
    return m.infections_source[Source.INITIAL]


def get_n_source_customer(m):
    return m.infections_source[Source.CUSTOMER]


def get_n_source_employee(m):
    return m.infections_source[Source.EMPLOYEE]


def get_n_source_public(m):
    return m.infections_source[Source.PUBLIC]