from mesa import Agent, Model
from mesa.time import RandomActivation, BaseScheduler
from mesa.datacollection import DataCollector
from helpers import *

import numpy as np
import pandas as pd
import random

parameters = {
    'n_total_customers': 10000,
    'daily_hours': 16,  # hours
    'n_total_employees': 30,
    'n_employees_working_concurrently': 10,
    'customer_arrival_frequency': (6, 10),  # days
    'time_asymptomatic': (5, 9),  # days
    'time_recovery': (7, 21),  # days
    'transmission_probability': .001,
    'customer_interaction_rate': .01,
    'employee_interactions_per_store_trip': 1,
    'probability_infection_in_public': .02,
    'probability_in_public': .01,  # per hour
    'initial_infected_customers': 10,
    'initial_infected_employees': 1,
    'fatality_rate': .03
}


class Customer(Agent):
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

        lb, ub = model.customer_arrival_frequency
        self.time_to_store_range = (
            lb * model.daily_hours, ub * model.daily_hours
        )
        self.time_til_store = random.randint(0, self.time_to_store_range[1])

        self.location = Location.HOME

        self.condition = Condition.NAIVE_HEALTHY
        self.condition_timer = 0

        self.infection_source = None
        self.infection_time = None

    def update_location(self):
        if self.condition == Condition.SYMPTOMATIC_INFECTED or \
                self.condition == Condition.DECEASED:
            self.location = Location.HOME
            return

        if self.time_til_store == 0:
            self.location = Location.IN_STORE
            lb, ub = self.time_to_store_range
            self.time_til_store = random.randint(lb, ub)
            return
        else:
            self.time_til_store -= 1

        if random.random() < self.model.probability_in_public:
            self.location = Location.PUBLIC
        else:
            self.location = Location.HOME

    def update_condition(self):
        if not self.sick():
            return

        elif self.condition == Condition.ASYMPTOMATIC_INFECTED:
            if self.condition_timer == 0:
                self.condition = Condition.SYMPTOMATIC_INFECTED
                lb, ub = self.model.recovery_time
                recov_time = random.randint(lb, ub) * self.model.daily_hours
                self.condition_timer = recov_time
            else:
                self.condition_timer -= 1

        elif self.condition == Condition.SYMPTOMATIC_INFECTED:
            if self.condition_timer == 0:
                if random.random() < self.model.fatality_rate:
                    self.condition = Condition.DECEASED
                else:
                    self.condition = Condition.IMMUNE_HEALTHY
            else:
                self.condition_timer -= 1

    def infect(self, source):
        self.condition = Condition.ASYMPTOMATIC_INFECTED
        lb, ub = self.model.time_asymptomatic
        self.condition_timer = random.randint(lb, ub) * self.model.daily_hours
        self.infection_source = source
        self.model.infections_source[source] += 1
        self.infection_time = self.model.schedule.time

    def sick(self):
        return self.condition == Condition.ASYMPTOMATIC_INFECTED or \
               self.condition == Condition.SYMPTOMATIC_INFECTED

    def interact(self):
        if self.location == Location.HOME:
            return
        elif self.location == Location.PUBLIC:
            if random.random() < self.model.probability_infection_in_public:
                self.infect(Source.PUBLIC)
        else:  # store
            # Shopping
            infected_customers = self.model.n_infected_customers()
            n_interactions = infected_customers * self.model.customer_interaction_rate
            # p(inf) = 1 - P(h) = 1 - (1-p(trans_p))^n_interactions
            p_infection = 1 - (1 - self.model.transmission_probability) ** n_interactions
            if random.random() < p_infection and self.condition == Condition.NAIVE_HEALTHY:
                self.infect(Source.CUSTOMER)

            # Paying
            n_employee_interactions = self.model.employee_interactions_per_store_trip
            while random.random() < n_employee_interactions:
                n_employee_interactions -= 1
                employee = self.model.get_employee()
                if employee.sick() and self.sick() or not (employee.sick() or self.sick()):
                    continue
                else:
                    if random.random() < self.model.transmission_probability:
                        if employee.sick() and self.condition == Condition.NAIVE_HEALTHY:
                            self.infect(Source.EMPLOYEE)
                        elif self.sick() and employee.condition == Condition.NAIVE_HEALTHY:
                            employee.infect(Source.CUSTOMER)

    def step(self):
        self.update_location()
        self.interact()
        self.update_condition()


class Employee(Agent):
    # TODO have public infections
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

        lb, ub = model.customer_arrival_frequency
        self.time_to_store_range = (
            lb * model.daily_hours, ub * model.daily_hours
        )
        self.work_schedule = None
        self.location = Location.HOME

        self.condition = Condition.NAIVE_HEALTHY
        self.condition_timer = 0

        self.infection_source = None
        self.infection_time = None


    def update_location(self):
        if self.condition == Condition.SYMPTOMATIC_INFECTED or \
                self.condition == Condition.DECEASED:
            self.location = Location.HOME
            return

        time_of_day = self.model.schedule.time % self.model.daily_hours
        if self.work_schedule[time_of_day] == 1:
            self.location = Location.IN_STORE

        else:
            if random.random() < self.model.probability_in_public:
                self.location = Location.PUBLIC
            else:
                self.location = Location.HOME

    def update_condition(self):
        if not self.sick():
            return

        elif self.condition == Condition.ASYMPTOMATIC_INFECTED:
            if self.condition_timer == 0:
                self.condition = Condition.SYMPTOMATIC_INFECTED
                lb, ub = self.model.recovery_time
                recov_time = random.randint(lb, ub) * self.model.daily_hours
                self.condition_timer = recov_time
            else:
                self.condition_timer -= 1

        elif self.condition == Condition.SYMPTOMATIC_INFECTED:
            if self.condition_timer == 0:
                if random.random() < self.model.fatality_rate:
                    self.condition = Condition.DECEASED
                else:
                    self.condition = Condition.IMMUNE_HEALTHY
            else:
                self.condition_timer -= 1

    def infect(self, source):
        self.condition = Condition.ASYMPTOMATIC_INFECTED
        lb, ub = self.model.time_asymptomatic
        self.condition_timer = random.randint(lb, ub) * self.model.daily_hours
        self.infection_source = source
        self.model.infections_source[source] += 1
        self.infection_time = self.model.schedule.time

    def sick(self):
        return self.condition == Condition.ASYMPTOMATIC_INFECTED or \
               self.condition == Condition.SYMPTOMATIC_INFECTED

    def step(self):
        self.update_location()
        self.update_condition()


class GroceryStoreModel(Model):
    def __init__(self,
                 n_total_customers=1000,
                 daily_hours=12,
                 n_total_employees=20,
                 n_employees_working_concurrently=8,
                 customer_arrival_frequency=(4, 9),
                 time_asymptomatic=(5, 9),
                 recovery_time=(7, 21),
                 transmission_probability=0.01,
                 customer_interaction_rate=0.05,
                 employee_interactions_per_store_trip=2,
                 probability_infection_in_public=0.001,
                 probability_in_public=0.001,
                 initial_infected_customers=30,
                 initial_infected_employees=3,
                 fatality_rate=0.03):
        self.running = True
        self.n_total_customers = n_total_customers
        self.daily_hours = daily_hours
        self.n_total_employees = n_total_employees
        self.n_employees_working_concurrently = n_employees_working_concurrently
        self.customer_arrival_frequency = customer_arrival_frequency
        self.time_asymptomatic = time_asymptomatic
        self.recovery_time = recovery_time
        self.transmission_probability = transmission_probability
        self.customer_interaction_rate = customer_interaction_rate
        self.employee_interactions_per_store_trip = employee_interactions_per_store_trip
        self.probability_infection_in_public = probability_infection_in_public
        self.probability_in_public = probability_in_public
        self.initial_infected_customers = initial_infected_customers
        self.initial_infected_employees = initial_infected_employees
        self.fatality_rate = fatality_rate

        self.customers = {}
        self.employees = {}
        self.state_info = {}
        self.infections_source = {s: 0 for s in Source}

        self.schedule = BaseScheduler(self)
        # Create agents
        for j in range(n_total_employees):
            employee = Employee('e' + str(j), self)
            self.schedule.add(employee)
            self.employees[j] = employee

        for i in range(n_total_customers):
            customer = Customer('c' + str(i), self)
            self.schedule.add(customer)
            self.customers[i] = customer

        init_infected_customers = np.random.choice(len(self.customers),
                                                   initial_infected_customers,
                                                   replace=False)
        for i in init_infected_customers:
            self.customers[i].infect(Source.INITIAL)

        init_infected_employees = np.random.choice(len(self.employees),
                                                   initial_infected_employees,
                                                   replace=False)
        for j in init_infected_employees:
            self.employees[j].infect(Source.INITIAL)

        self.datacollector = DataCollector(
            model_reporters={
                # Store Info
                'n_employees_working': get_working_employees,
                'n_infected_employees_working': get_infected_employees_working,
                'n_customers_in_store': get_n_customers_in_store,
                'n_infected_customers_in_store': get_n_infected_customers_in_store,
                # Conditions
                'n_customer_asymptomatic_infected': get_n_cust_AI,
                'n_customer_symptomatic_infected': get_n_cust_SI,
                'n_customer_immune_healthy': get_n_cust_IH,
                'n_customer_naive_healthy': get_n_cust_NH,
                'n_customer_deceased': get_n_cust_D,
                'n_employee_asymptomatic_infected': get_n_emp_AI,
                'n_employee_symptomatic_infected': get_n_emp_SI,
                'n_employee_immune_healthy': get_n_emp_IH,
                'n_employee_naive_healthy': get_n_emp_NH,
                'n_employee_deceased': get_n_emp_D,
                # Sources
                'n_source_intial': get_n_source_initial,
                'n_source_customer': get_n_source_customer,
                'n_source_employee': get_n_source_employee,
                'n_source_public': get_n_source_public,
            }
        )

    def n_infected_customers(self):
        return sum([
            customer.location == Location.IN_STORE and customer.sick()
            for customer in self.customers.values()
        ])

    def get_employee(self):
        return self.employees[random.choice([
            eid for eid, e in self.employees.items()
            if e.location == Location.IN_STORE
        ])]

    def create_employee_work_schedule(self):
        total_employees = self.n_total_employees
        daily_hours = self.daily_hours
        concurrent_employees = self.n_employees_working_concurrently
        feasible_hours = np.array([4, 6, 8])
        # Create 2 shift daily schedule encoded by binary matrix
        employee_sched = np.zeros((total_employees, daily_hours))
        day_employees = np.random.choice(total_employees,
                                         concurrent_employees * 2,
                                         replace=False)
        shift1 = day_employees[:concurrent_employees]
        shift2 = day_employees[concurrent_employees:]
        shift1_hrs = np.random.choice(feasible_hours, concurrent_employees)
        shift2_hrs = daily_hours - shift1_hrs
        for employee_id, time in zip(shift1, shift1_hrs):
            employee_sched[employee_id, 0:time] = 1
        for employee_id, time in zip(shift2, shift2_hrs):
            employee_sched[employee_id, -time:] = 1

        for eid, schedule in enumerate(employee_sched):
            self.employees[eid].work_schedule = schedule

    def get_state_info(self):
        state_info = {
            'n_employees_working': 0,
            'n_infected_employees_working': 0,
            'n_customers_in_store': 0,
            'n_infected_customers_in_store': 0,
            'customer_condition': {c: 0 for c in Condition},
            'employee_condition': {c: 0 for c in Condition},
        }
        for employee in self.employees.values():
            state_info['employee_condition'][employee.condition] += 1
            if employee.location == Location.IN_STORE:
                state_info['n_employees_working'] += 1
                if employee.sick():
                    state_info['n_infected_employees_working'] += 1
        for customer in self.customers.values():
            state_info['customer_condition'][customer.condition] += 1
            if customer.location == Location.IN_STORE:
                state_info['n_customers_in_store'] += 1
                if customer.sick():
                    state_info['n_infected_customers_in_store'] += 1
        return state_info

    def step(self):
        if self.schedule.time % 10 == 0:
            print('step', self.schedule.time)
        if self.schedule.time % self.daily_hours == 0:
            self.create_employee_work_schedule()
        self.schedule.step()
        self.state_info = self.get_state_info()
        self.datacollector.collect(self)



