# Expert System for Employee Performance Evaluation
# Python Program

def employee_performance_expert_system():

    print("==========================================")
    print(" Employee Performance Evaluation System ")
    print("==========================================")

    while True:

        print("\nEnter Employee Details")

        name = input("Employee Name: ")
        attendance = int(input("Attendance Percentage: "))
        task_completion = int(input("Task Completion Percentage: "))
        communication = int(input("Communication Skill Rating (1-10): "))

        print("\nEvaluating Performance...\n")

        # Rule-Based Evaluation
        if attendance >= 90 and task_completion >= 90 and communication >= 8:

            performance = "Excellent"
            advice = "Eligible for promotion and rewards."

        elif attendance >= 75 and task_completion >= 70 and communication >= 6:

            performance = "Good"
            advice = "Performance is satisfactory. Encourage skill improvement."

        elif attendance >= 60 and task_completion >= 50 and communication >= 5:

            performance = "Average"
            advice = "Needs improvement in productivity and communication."

        else:

            performance = "Poor"
            advice = "Immediate training and monitoring required."

        # Display Result
        print("===================================")
        print(" Employee Performance Report ")
        print("===================================")

        print(f"Employee Name      : {name}")
        print(f"Attendance         : {attendance}%")
        print(f"Task Completion    : {task_completion}%")
        print(f"Communication Skill: {communication}/10")

        print(f"\nPerformance Status : {performance}")
        print(f"Expert Advice      : {advice}")

        # Continue option
        again = input("\nDo you want to evaluate another employee? (yes/no): ").lower()

        if again != "yes":
            print("\nExiting Expert System...")
            print("Thank You!")
            break


# Run Expert System
employee_performance_expert_system()