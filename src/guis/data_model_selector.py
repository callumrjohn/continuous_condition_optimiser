import tkinter as tk

def select_data_models_tkinter(data_names, model_names):
    """
    Display interactive tkinter GUI for selecting training dataset and model type.
    
    Presents side-by-side selection interface allowing user to choose both the
    training dataset (from processed feature sets) and the model architecture.
    Returns both selections as tuple for downstream model training pipeline.
    
    Args:
        data_names : list
            List of dataset names (strings) to display in left-side listbox,
            e.g., ['data_aqme.csv', 'data_mordred.csv', 'data_morgan.csv']
        model_names : list
            List of model type names (strings) to display in right-side listbox,
            e.g., ['XGBModel', 'RFModel', 'GPRModel', 'MLPModel', 'SVRModel']
    
    Returns:
        tuple
            Two-element tuple containing:
            - dataset (str): Selected dataset name from data_names
            - model (str): Selected model name from model_names
            Returns (None, None) if window is closed without selection
    
    Notes:
        - Two-panel interface: datasets (left, width=100), models (right, width=30)
        - Both listboxes have height=10
        - Single-selection mode (exportselection=False prevents default behavior)
        - User clicks OK to confirm both selections
    """
    root = tk.Tk()
    root.title("Select Data Model and Training Dataset")

    tk.Label(root, text="Training Datasets").grid(row=0, column=0, padx=10, pady=5)
    tk.Label(root, text="Models").grid(row=0, column=1, padx=10, pady=5)

    dataset_var = tk.StringVar(value=data_names[0] if data_names else "")
    model_var = tk.StringVar(value=model_names[0] if model_names else "")

    dataset_listbox = tk.Listbox(root, listvariable=tk.StringVar(value=data_names), height=10, width=100, exportselection=False)
    dataset_listbox.grid(row=1, column=0, padx=10, pady=5)
    dataset_listbox.selection_set(0)

    model_listbox = tk.Listbox(root, listvariable=tk.StringVar(value=model_names), height=10, width=30, exportselection=False)
    model_listbox.grid(row=1, column=1, padx=10, pady=5)
    model_listbox.selection_set(0)

    result = {}

    def on_ok():
        dataset_idx = dataset_listbox.curselection()
        model_idx = model_listbox.curselection()
        if dataset_idx and model_idx:
            result['dataset'] = data_names[dataset_idx[0]]
            result['model'] = model_names[model_idx[0]]
            root.destroy()

    ok_button = tk.Button(root, text="OK", command=on_ok)
    ok_button.grid(row=2, column=0, columnspan=2, pady=10)

    root.mainloop()

    return result.get('dataset'), result.get('model')