import tkinter as tk

def select_input_data_tkinter(data_names):
    """
    Display interactive tkinter GUI to select a training dataset.
    
    Presents a list of available datasets for the user to select via mouse
    clicking. Single-selection interface returns the chosen dataset name to
    the calling function.
    
    Args:
        data_names : list
            List of dataset names (strings) to display in the selection list.
            Should include descriptive names like 'data_aqme.csv' or
            'data_aqme_custom_fragprints_mordred_rdkit_morgan.csv'
    
    Returns:
        str or None
            The name of the selected dataset if user clicks OK, or None
            if the window is closed without selection
    
    Notes:
        - GUI uses tkinter Listbox widget with height=10, width=30
        - First item is automatically selected by default
        - User can only select one dataset at a time
        - Window remains open until OK button is clicked or window is closed
    """
    root = tk.Tk()
    root.title("Select dataset for dimensionality reduction")

    tk.Label(root, text="Training Datasets").grid(row=0, column=0, padx=10, pady=5)

    dataset_var = tk.StringVar(value=data_names[0] if data_names else "")

    dataset_listbox = tk.Listbox(root, listvariable=tk.StringVar(value=data_names), height=10, width=30, exportselection=False)
    dataset_listbox.grid(row=1, column=0, padx=10, pady=5)
    dataset_listbox.selection_set(0)

    result = {}

    def on_ok():
        dataset_idx = dataset_listbox.curselection()
        if dataset_idx:
            result['dataset'] = data_names[dataset_idx[0]]
            root.destroy()

    ok_button = tk.Button(root, text="OK", command=on_ok)
    ok_button.grid(row=2, column=0, columnspan=2, pady=10)

    root.mainloop()

    return result.get('dataset')