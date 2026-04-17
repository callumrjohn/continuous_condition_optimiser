import sys
import tkinter as tk


def select_fingerprints_tkinter(fingerprint_names):
    """
    Display interactive tkinter GUI for selecting and prioritizing fingerprints/descriptors.
    
    Presents a list of available molecular descriptor/fingerprint types with interface
    to select multiple descriptors, reorder their priority, and confirm selection.
    Returns indices of selected fingerprints in user-specified priority order. 
    
    Args:
        fingerprint_names : list or array-like
            List of fingerprint/descriptor type names (strings) to display,
            e.g., ['Morgan', 'ECFp', 'Mordred', 'AQME', 'Custom']
    
    Returns:
        list
            List of indices (integers) corresponding to selected fingerprints
            in the order specified by the user. Index values reference positions
            in the original fingerprint_names list
    
    Raises:
        SystemExit
            If user clicks Cancel button, program exits with message
            "Fingerprint selection cancelled by user."
    
    Notes:
        - Multiple selection enabled (Ctrl+Click to select multiple items)
        - Up/Down buttons allow reordering selected fingerprints by priority
        - Priority order is preserved in returned indices
        - Extended selection mode allows arbitrary ordering
    """
    selected = []
    cancelled = [False]

    def move_up():
        sel = listbox.curselection()
        if not sel or sel[0] == 0:
            return
        idx = sel[0]
        items[idx-1], items[idx] = items[idx], items[idx-1]
        update_listbox()
        listbox.selection_set(idx-1)

    def move_down():
        sel = listbox.curselection()
        if not sel or sel[0] == len(items)-1:
            return
        idx = sel[0]
        items[idx+1], items[idx] = items[idx], items[idx+1]
        update_listbox()
        listbox.selection_set(idx+1)

    def on_ok():
        nonlocal selected
        selected = [fingerprint_names.index(items[i]) for i in listbox.curselection()]
        root.destroy()

    def on_cancel():
        cancelled[0] = True
        root.destroy()

    def update_listbox():
        listbox.delete(0, tk.END)
        for name in items:
            listbox.insert(tk.END, name)

    root = tk.Tk()
    root.title("Select and Order Fingerprints")
    tk.Label(root, text="Select fingerprints to use and priority (Ctrl+Click for multiple, reorder with Up/Down):").pack(anchor="w")
    items = list(fingerprint_names)
    listbox = tk.Listbox(root, selectmode=tk.EXTENDED, width=80)
    update_listbox()
    listbox.pack()

    btn_frame = tk.Frame(root)
    btn_frame.pack(pady=5)
    tk.Button(btn_frame, text="Up", command=move_up).pack(side="left")
    tk.Button(btn_frame, text="Down", command=move_down).pack(side="left")
    tk.Button(btn_frame, text="OK", command=on_ok).pack(side="left")
    tk.Button(btn_frame, text="Cancel", command=on_cancel).pack(side="left")

    root.mainloop()

    if cancelled[0]:
        sys.exit("Fingerprint selection cancelled by user.")

    return selected