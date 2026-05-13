# Expert System for Information Management
# Python Program

def information_management_expert_system():

    print("======================================")
    print(" Information Management Expert System ")
    print("======================================")

    while True:

        print("\nChoose an option:")
        print("1. Data Backup")
        print("2. Password Management")
        print("3. File Organization")
        print("4. Data Security")
        print("5. Exit")

        choice = input("Enter your choice: ")

        # Rule-based decisions
        if choice == "1":

            print("\nExpert Advice:")
            print("- Take regular backups of important files.")
            print("- Use cloud storage and external drives.")
            print("- Schedule automatic backups weekly.")

        elif choice == "2":

            print("\nExpert Advice:")
            print("- Use strong passwords with special characters.")
            print("- Avoid sharing passwords.")
            print("- Change passwords regularly.")
            print("- Use a password manager.")

        elif choice == "3":

            print("\nExpert Advice:")
            print("- Create folders by category.")
            print("- Rename files properly.")
            print("- Remove duplicate and unwanted files.")
            print("- Maintain proper directory structure.")

        elif choice == "4":

            print("\nExpert Advice:")
            print("- Install antivirus software.")
            print("- Enable firewall protection.")
            print("- Keep software updated.")
            print("- Avoid opening suspicious emails or links.")

        elif choice == "5":

            print("\nExiting Expert System...")
            print("Thank you!")
            break

        else:
            print("\nInvalid Choice! Please try again.")


# Run Expert System
information_management_expert_system()