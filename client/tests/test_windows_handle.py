import win32gui

hwnd = win32gui.FindWindow(None, "WRCG")
win32gui.SetForegroundWindow(hwnd)
print(hex(hwnd))