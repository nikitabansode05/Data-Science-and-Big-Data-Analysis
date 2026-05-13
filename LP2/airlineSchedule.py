# Expert System for Airline Scheduling and Cargo Management
# Python Program

def airline_expert_system():

    print("================================================")
    print(" Airline Scheduling & Cargo Management System ")
    print("================================================")

    while True:

        print("\nSelect Service Type:")
        print("1. Passenger Flight Scheduling")
        print("2. Cargo Scheduling")
        print("3. Emergency Flight")
        print("4. International Flight")
        print("5. Exit")

        choice = input("Enter your choice: ")

        # Rule-Based Expert System
        if choice == "1":

            passengers = int(input("Enter number of passengers: "))

            print("\nExpert Advice:")
            if passengers > 200:
                print("- Schedule a large aircraft.")
                print("- Allocate additional cabin crew.")
            else:
                print("- Schedule a medium-size aircraft.")
                print("- Standard crew allocation is sufficient.")

        elif choice == "2":

            cargo_weight = int(input("Enter cargo weight (in kg): "))

            print("\nExpert Advice:")
            if cargo_weight > 10000:
                print("- Use dedicated cargo aircraft.")
                print("- Assign priority loading staff.")
            else:
                print("- Cargo can be transported in passenger aircraft cargo hold.")

        elif choice == "3":

            print("\nExpert Advice:")
            print("- Assign highest priority runway.")
            print("- Notify emergency response team.")
            print("- Allocate immediate takeoff clearance.")

        elif choice == "4":

            destination = input("Enter destination country: ")

            print("\nExpert Advice:")
            print(f"- Verify passport and visa requirements for {destination}.")
            print("- Ensure customs and immigration clearance.")
            print("- Schedule long-haul aircraft with extra fuel capacity.")

        elif choice == "5":

            print("\nExiting Expert System...")
            print("Thank You!")
            break

        else:
            print("\nInvalid choice! Please try again.")


# Run Expert System
airline_expert_system()