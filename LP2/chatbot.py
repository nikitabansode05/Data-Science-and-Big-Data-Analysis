# Elementary Chatbot for Customer Interaction
# Python Program

def chatbot():

    print("===================================")
    print(" Welcome to Customer Support Chat ")
    print("===================================")
    print("Type 'exit' to end the chat.\n")

    while True:

        # User input
        user = input("You: ").lower()

        # Exit condition
        if user == "exit":
            print("Bot: Thank you for visiting. Have a nice day!")
            break

        # Greetings
        elif user in ["hi", "hello", "hey"]:
            print("Bot: Hello! How can I help you today?")

        # Order status
        elif "order" in user:
            print("Bot: Please provide your Order ID to check the status.")

        # Payment issue
        elif "payment" in user:
            print("Bot: Your payment may take 24 hours to process.")

        # Refund issue
        elif "refund" in user:
            print("Bot: Refunds are processed within 5-7 working days.")

        # Delivery issue
        elif "delivery" in user:
            print("Bot: Your order will be delivered within 3 business days.")

        # Contact support
        elif "support" in user or "help" in user:
            print("Bot: You can contact support at support@example.com")

        # Thank you response
        elif "thank" in user:
            print("Bot: You're welcome!")

        # Default response
        else:
            print("Bot: Sorry, I did not understand your query.")


# Run chatbot
chatbot()