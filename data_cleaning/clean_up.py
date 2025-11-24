import os


def main():
    window_size = int(input("What is the window size: "))
    remove_data = input("Do you want to remove the unzipped data foler?(y/n): ")

    commands = [
        f"rm -r time_series_data_{window_size}",
        f"rm stream_labels.csv"
    ]

    if remove_data.lower() == "y":
        commands.append("rm -r ./data")

    for cmd in commands:
        os.system(cmd)

if __name__ == "__main__":
    main()