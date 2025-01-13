def compare_files(file1_path, file2_path):
    try:
        with open(file1_path, 'r', encoding='utf-8') as file1:
            file1_contents = file1.read()

        with open(file2_path, 'r', encoding='utf-8') as file2:
            file2_contents = file2.read()

        if file1_contents == file2_contents:
            print("The files are identical.")
        else:
            print("The files are different.")

    except FileNotFoundError as e:
        print(f"File not found: {e}")

# Example usage:
compare_files('output.txt', 'output1.txt')

