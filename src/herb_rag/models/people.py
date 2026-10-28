"""Directory records for employees and customers."""

from pydantic import BaseModel


class Employee(BaseModel):
    """An employee of the simulated company."""

    employee_id: str
    name: str
    role: str
    location: str
    org: str


class Customer(BaseModel):
    """A customer contact of the simulated company."""

    id: str
    name: str
    role: str
    company: str
