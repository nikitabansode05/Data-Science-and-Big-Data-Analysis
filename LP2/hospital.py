# Expert System for Hospitals and Medical Facilities
# Python Program

def hospital_expert_system():

    print("==========================================")
    print(" Hospital & Medical Facilities Expert System ")
    print("==========================================")

    while True:

        print("\nSelect your health issue:")
        print("1. Fever")
        print("2. Chest Pain")
        print("3. Accident / Emergency")
        print("4. General Checkup")
        print("5. COVID-19 Symptoms")
        print("6. Exit")

        choice = input("Enter your choice: ")

        # Rule-based Expert System
        if choice == "1":

            print("\nExpert Advice:")
            print("- Visit General Physician Department.")
            print("- Drink plenty of water.")
            print("- Take proper rest.")
            print("- If fever persists, consult a doctor immediately.")

        elif choice == "2":

            print("\nExpert Advice:")
            print("- Consult Cardiology Department immediately.")
            print("- Avoid physical stress.")
            print("- Call emergency services if pain is severe.")

        elif choice == "3":

            print("\nExpert Advice:")
            print("- Contact Emergency Ward immediately.")
            print("- Call ambulance services.")
            print("- Provide first aid if possible.")

        elif choice == "4":

            print("\nExpert Advice:")
            print("- Visit OPD (Out Patient Department).")
            print("- Schedule a routine health checkup.")
            print("- Maintain regular exercise and healthy diet.")

        elif choice == "5":

            print("\nExpert Advice:")
            print("- Isolate yourself.")
            print("- Wear a mask.")
            print("- Consult COVID testing center.")
            print("- Seek medical help if breathing difficulty occurs.")

        elif choice == "6":

            print("\nExiting Expert System...")
            print("Stay Healthy!")
            break

        else:
            print("\nInvalid choice! Please try again.")


# Run Expert System
hospital_expert_system()