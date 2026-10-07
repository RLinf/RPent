while True:
    command = input("Human command: ").strip()

    if command.lower() == "quit":
        print("Session ended.")
        break

    print("Robot received:", command)