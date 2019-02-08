class Employee:
    emp_count = 0
    salary = 0

    # total_sal=0

    def __init__(self, name, family, salary, dept):
        self.name = name
        self.family = family
        Employee.salary += salary
        self.dept = dept
        Employee.emp_count += 1

    def display_emp_count(self):
        print("total number of employees", Employee.emp_count)

    def avg_salary(self):
        avg_sal = Employee.salary / Employee.emp_count
        print("average salary:", avg_sal)

    def demo_func(self):
        print('calling member function of parent')


class Full_time_employee(Employee):
    def __init__(self):
        print('this is the subclass: Full time employee')


e = Employee('siva', 'A', 6000, 'D1')
Employee('sameer', "B", 7000, "D2")
Employee('Krishna', 'C', 8000, 'D3')
c = Full_time_employee()
c.display_emp_count()
c.avg_salary()
e.demo_func()
c.demo_func()
