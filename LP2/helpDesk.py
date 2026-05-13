# Expert System for Help Desk Management
# Python Program

def helpdesk_expert_system():

    print("===================================")
    print(" Help Desk Management Expert System ")
    print("===================================")

    while True:

        print("\nSelect your issue:")
        print("1. Internet Problem")
        print("2. Computer Slow")
        print("3. Printer Not Working")
        print("4. Software Installation")
        print("5. Password Reset")
        print("6. Exit")

        choice = input("Enter your choice: ")

        # Rule-Based Expert System
        if choice == "1":

            print("\nExpert Advice:")
            print("- Check Wi-Fi or network cable connection.")
            print("- Restart the router.")
            print("- Verify internet settings.")
            print("- Contact network administrator if issue persists.")

        elif choice == "2":

            print("\nExpert Advice:")
            print("- Close unnecessary applications.")
            print("- Delete temporary files.")
            print("- Scan system for viruses.")
            print("- Upgrade RAM if required.")

        elif choice == "3":

            print("\nExpert Advice:")
            print("- Check printer power and cable connection.")
            print("- Ensure printer driver is installed.")
            print("- Restart the printer.")
            print("- Check paper and ink availability.")

        elif choice == "4":

            print("\nExpert Advice:")
            print("- Verify system requirements.")
            print("- Run installer as administrator.")
            print("- Disable antivirus temporarily if needed.")
            print("- Restart system after installation.")

        elif choice == "5":

            print("\nExpert Advice:")
            print("- Use 'Forgot Password' option.")
            print("- Contact administrator for reset.")
            print("- Use strong and secure passwords.")

        elif choice == "6":

            print("\nExiting Help Desk Expert System...")
            print("Thank You!")
            break

        else:
            print("\nInvalid Choice! Please try again.")


# Run Expert System
helpdesk_expert_system()