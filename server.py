from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.modules import ChartModule
from mesa.visualization.UserParam import UserSettableParameter

from model import *

default_parameters = {
    'n_total_customers': UserSettableParameter(
        param_type='slider', name='Total Customers',
        value=5000, min_value=1000, max_value=30000, step=500,
        description="Total customers that shop at this store."),

    'daily_hours': 12, # hours

    'n_total_employees': UserSettableParameter(
        param_type='slider', name='Total Employees',
        value=20, min_value=5, max_value=50, step=1,
        description="Total employees that work at this store."),

    'n_employees_working_concurrently': UserSettableParameter(
        param_type='slider', name='Concurrent Employees',
        value=8, min_value=2, max_value=15, step=1,
        description="Number of employees that work at one time."),

    'customer_arrival_frequency': (4, 9),  # days
    'time_asymptomatic': (5, 9),  # days
    'recovery_time': (7, 21),  # days

    'transmission_probability': UserSettableParameter(
        param_type='slider', name='Transmission Probability',
        value=.001, min_value=0, max_value=.2, step=.0001,
        description="Total customers that shop at this store."),

    'customer_interaction_rate': UserSettableParameter(
        param_type='slider', name='Customer Interaction Rate',
        value=.05, min_value=0, max_value=1, step=.001,
        description="Total customers that shop at this store."),

    'employee_interactions_per_store_trip': UserSettableParameter(
        param_type='slider', name='Customer-Employee Interactions per trip',
        value=1, min_value=0, max_value=4, step=.1,
        description="Total customers that shop at this store."),

    'probability_infection_in_public': UserSettableParameter(
        param_type='slider', name='Probability Infection in Public',
        value=.001, min_value=0, max_value=1, step=.0001,
        description="Total customers that shop at this store."),

    'probability_in_public': UserSettableParameter(
        param_type='slider', name='Probability in Public',
        value=.01, min_value=0, max_value=1, step=.0001,
        description="Total customers that shop at this store."),  # per hour

    'initial_infected_customers': UserSettableParameter(
        param_type='slider', name='Initial Infected Customers',
        value=20, min_value=0, max_value=500, step=1,
        description="Total customers that shop at this store."),

    'initial_infected_employees': UserSettableParameter(
        param_type='slider', name='Initial Infected Employees',
        value=1, min_value=0, max_value=5, step=1,
        description="Total customers that shop at this store."),

    'fatality_rate': UserSettableParameter(
        param_type='slider', name='Fatality Rate',
        value=.02, min_value=0, max_value=1, step=.001,
        description="Total customers that shop at this store.")
}


customer_condition = ChartModule([
    {'Label': 'n_customer_asymptomatic_infected', 'Color': '#FF0000'},
    {'Label': 'n_customer_symptomatic_infected', 'Color': '#00FF00'},
    {'Label': 'n_customer_deceased', 'Color': '#FFFF00'}
])

customer_health = ChartModule([
    {'Label': 'n_customer_immune_healthy', 'Color': '#0000FF'},
    {'Label': 'n_customer_naive_healthy', 'Color': '#FF00FF'},
])

employee_condition = ChartModule([
    {'Label': 'n_employee_asymptomatic_infected', 'Color': '#FF0000'},
    {'Label': 'n_employee_symptomatic_infected', 'Color': '#00FF00'},
    {'Label': 'n_employee_deceased', 'Color': '#FFFF00'}
])

employee_health = ChartModule([
    {'Label': 'n_employee_immune_healthy', 'Color': '#0000FF'},
    {'Label': 'n_employee_naive_healthy', 'Color': '#FF00FF'},
])

infection_source = ChartModule([
    {'Label': 'n_source_intial', 'Color': '#00FF00'},
    {'Label': 'n_source_customer', 'Color': '#0000FF'},
    {'Label': 'n_source_employee', 'Color': '#FF0000'},
    {'Label': 'n_source_public', 'Color': '#FFFF00'},
])

store_employees = ChartModule([
    {'Label': 'n_infected_employees_working', 'Color': '#FF0000'},
    {'Label': 'n_employees_working', 'Color': '#0000FF'},
])

store_customers = ChartModule([
    {'Label': 'n_customers_in_store', 'Color': '#FF0000'},
    {'Label': 'n_infected_customers_in_store', 'Color': '#0000FF'},
])



server = ModularServer(GroceryStoreModel,
                       [customer_condition,
                        customer_health,
                        employee_condition,
                        employee_health,
                        infection_source,
                        store_employees,
                        store_customers],
                       "Grocery Store Model",
                       default_parameters)
server.port = 8521  # The default
server.launch()
