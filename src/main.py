from predict import predict_category

def main ():
    print("\n💡 Invoice Categorizer - Enter an invoice description:")

    while True:
        user_input = input("Enter invoice description (or type exit to quit): ")

        if user_input.lower() == "exit":
            print("\n Gracefully exiting...")
            break

        predicted_category = predict_category(user_input)

        print(f"Predicted Account Code: {predicted_category}")

if __name__ == "__main__":
    main()

