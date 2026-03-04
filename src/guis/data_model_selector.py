import tkinter as tk

def select_data_models_tkinter(data_names, model_names):
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