"""Manual GUI test — run from terminal to see errors.

Usage:
    uv run python tests/test_launcher_gui.py
"""
import tkinter as tk


def manual_launcher_flow():
    """Mimics the exact __main__.py flow to catch errors visibly."""
    print("Creating root...")
    root = tk.Tk()
    root.withdraw()
    print("Root created OK")

    from auto_annotator.dialogs import LauncherDialog

    print("Showing LauncherDialog...")
    launcher = LauncherDialog(root)
    print(f"LauncherDialog closed — choice={launcher.choice!r}")

    if launcher.choice is None:
        print("User cancelled, exiting.")
        root.destroy()
        return

    if launcher.choice == "export":
        from auto_annotator.dialogs import ExportEngineDialog

        print("Showing ExportEngineDialog...")
        export = ExportEngineDialog(root)
        print(f"ExportEngineDialog closed — exported={export.exported_path!r}")
        root.destroy()
        print("PASS: export flow completed")
        return

    if launcher.choice == "annotate":
        from auto_annotator.dialogs import ModelSelectionDialog

        print("Showing ModelSelectionDialog...")
        dialog = ModelSelectionDialog(root)
        print(f"ModelSelectionDialog closed — confirmed={dialog.confirmed!r}")
        root.destroy()
        print("PASS: annotate flow completed")
        return


if __name__ == "__main__":
    manual_launcher_flow()
