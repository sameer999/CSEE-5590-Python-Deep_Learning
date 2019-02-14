import random

class Flight:                                       #flight class
    def __init__(self, airline_name, flight_number):                    #constructor
        self.airline_name = airline_name
        self.flight_number = flight_number

    def flight_display(self):                       #display flight details
        print('airlines: ', self.airline_name)
        print('flight number: ', self.flight_number)


class Employee:                                     #employee class
    def __init__(self, emp_id, emp_name, emp_age, emp_gender):          # constructor
        self.emp_name = emp_name
        self.emp_age = emp_age
        self.__emp_id = emp_id                      #private data memeber
        self.emp_gender =  emp_gender
    def emp_display(self):                          #display employee details
        print("name of employee: ",self.emp_name)
        print('employee id: ', self.__emp_id)       #private data member is retrieved
        print('employee age: ',self.emp_age)
        print('employee gender: ', self.emp_gender)

class Passenger:                                    #Passenger class
    def __init__(self):                             #constructor
        Passenger.__passport_number = input("enter the passport number of the passenger: ")     #private data member
        Passenger.name = input('enter name of the passenger: ')
        Passenger.age = input('enter age of passenger')
        Passenger.gender = input('enter the gender: ')
        Passenger.class_type = input('select business or economy class: ')

class Baggage():                                    #baggage class
    cabin_bag = 1
    bag_fare = 0
    def __init__(self, checked_bags):               #calculate cost if passenger has more than 2 checked bags
        self.checked_bags = checked_bags
        if checked_bags > 2 :
            for i in checked_bags:
                self.bag_fare += 100
        print("number of checked bags allowed: ",checked_bags,"bag fare: ",self.bag_fare)


class Fare(Baggage):                                #fare class which is the sub class of baggage class
    counter = 150                                   #fixed cost if ticket is purchased at counter
    online = random.randint(110, 200)               #if purchased through online, cost is generated from a random function
    total_fare=0
    def __init__(self):                             #constructor
        super().__init__(2)                         #super call to baggage which is the parent class
        x = input('buy ticket through online or counter: ')
        if x == 'online':
            Fare.total_fare = self.online + self.bag_fare
        elif x == 'counter':
            Fare.total_fare = self.counter + self.bag_fare
        else:
            x=input('enter correct transaction type:')
        print("Total Fare before class type:",Fare.total_fare)


class Ticket(Passenger, Fare):                                      #multiple inheritance
    def __init__(self):                                             #constructor
        print("Passenger name:",Passenger.name)                     #accessing the passenger (parent) class variable
        if Passenger.class_type == "business":                      #cost varies with business and economy class
            Fare.total_fare+=100
        else:
            pass
        print("Passenger class type:",Passenger.class_type)
        print("Total fare:",Fare.total_fare)                        #total fare is displayed


f1=Flight('etihad',1000)                                            #Instance of Flight class
f1.flight_display()

emp1 = Employee('e1', 'emp_siva', 26, 'M')                          #instance of Employee class
emp1.emp_display()

p1 = Passenger()                                                    #instance of passenger class

fare1=Fare()                                                        #instance of fare class

t= Ticket()                                                         #instance of ticket class


