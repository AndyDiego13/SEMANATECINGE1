import numpy as np

class Student():
    def __init__(self, name, last_name, major):
        self.name = name
        self.last_name = last_name
        self.major = major
        self.grades = []
    def add_grade(self, grade):
        self.grades.append(grade)
        print("A grade of {} was added to the student {} {}".format(grade, self.last_name, self.name))
    def get_avarage(self):
        return np.mean(self.grades)

def check_student_class():
    andy = Student("Andy", "Diego", "ITC")

    print(andy.major)
    print(andy.grades)

    andy.add_grade(95)
    andy.add_grade(98)
    andy.add_grade(90)
    andy.add_grade(100)

    print(andy.grades)

    average = andy.get_avarage()
    print("the avarage is {}".format(average))

if __name__ == "__main__":
    check_student_class()
    

