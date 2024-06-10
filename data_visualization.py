import tkinter as tk
from tkinter import ttk
import csv
def visualize_data():
    # Create a new Tkinter window
    window = tk.Tk()
    window.title("Data Visualization")

    # Create a Treeview widget
    tree = ttk.Treeview(window)

    # Read the CSV file
    with open('data.csv', 'r') as f:
        reader = csv.reader(f)
        columns = next(reader)  # Get the column names from the first line

        # Create the columns in the Treeview widget
        tree["columns"] = columns
        for column in columns:
            tree.column(column, width=100)
            tree.heading(column, text=column)

        # Add the data to the Treeview widget
        for row in reader:
            tree.insert('', 'end', values=row)

    # Pack the Treeview widget into the window
    tree.pack()
    

    # Start the Tkinter event loop
    window.mainloop()

visualize_data()