import win32gui
import win32con
import win32process
import psutil
import time
import threading
import keyboard
import pywintypes
import pyautogui
import pyperclip
import ctypes


def force_focus_and_send_commands(app_name, loops=20):
    """
    Force focus on the Valorant game window and send '/t /ff' commands.
    Verifies the window belongs to VALORANT-Win64-Shipping.exe and prevents unfocusing.

    Args:
        app_name (str): Exact or partial name of the application window (e.g., "VALORANT  ").
        loops (int): Number of times to send the command sequence.
    """
    # Ensure pyautogui failsafe is disabled
    pyautogui.FAILSAFE = False

    def get_process_name(hwnd):
        """Get the process name for a given window handle."""
        try:
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            process = psutil.Process(pid)
            return process.name()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return None

    def find_window():
        """Find the Valorant game window by exact or partial title and process name."""
        target_windows = []

        def enum_windows_callback(hwnd, results):
            if win32gui.IsWindowVisible(hwnd) and win32gui.IsWindow(hwnd):
                title = win32gui.GetWindowText(hwnd)
                process_name = get_process_name(hwnd)
                if (
                        title == app_name or app_name in title) and process_name and process_name.lower() == "valorant-win64-shipping.exe":
                    results.append(hwnd)

        win32gui.EnumWindows(enum_windows_callback, target_windows)

        # Debug: Print all windows with 'valorant' in title
        print("Debug: All visible windows with 'VALORANT' in title:")

        def debug_windows():
            def callback(hwnd, _):
                if win32gui.IsWindowVisible(hwnd):
                    title = win32gui.GetWindowText(hwnd)
                    if "valorant" in title.lower():
                        process_name = get_process_name(hwnd)
                        print(f"Window: '{title}', Process: {process_name}, HWND: {hwnd}")

            win32gui.EnumWindows(callback, None)

        debug_windows()

        return target_windows[0] if target_windows else None

    # Find the Valorant window
    hwnd = find_window()
    if not hwnd:
        print(f"Could not find {app_name} game window. Ensure Valorant is running.")
        return

    # Check if script is running with admin privileges
    def is_admin():
        try:
            return ctypes.windll.shell32.IsUserAnAdmin()
        except:
            return False

    if not is_admin():
        print("Warning: Script not running as administrator. Focus may fail. Run as admin for best results.")

    # Attempt to bring the window to the foreground
    def set_window_focus(hwnd):
        try:
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)  # Restore if minimized
            win32gui.BringWindowToTop(hwnd)
            win32gui.SetActiveWindow(hwnd)
            win32gui.SetForegroundWindow(hwnd)
            print(f"Successfully focused window: {win32gui.GetWindowText(hwnd)}")
            return True
        except pywintypes.error as e:
            print(f"Win32 focus attempt failed: {e}")
            return False

    # Fallback: Simulate a mouse click on the window
    def simulate_focus(hwnd):
        try:
            rect = win32gui.GetWindowRect(hwnd)
            x, y = rect[0] + 10, rect[1] + 10
            pyautogui.moveTo(x, y)
            pyautogui.click()
            print("Simulated mouse click to set focus.")
            return True
        except Exception as e:
            print(f"Simulated focus failed: {e}")
            return False

    # Initial focus attempt
    if not set_window_focus(hwnd):
        print("Trying simulated focus as fallback...")
        if not simulate_focus(hwnd):
            print("All focus attempts failed. Exiting.")
            return
    time.sleep(0.02)  # Allow window to focus

    # Keep focus during command sending
    focus_active = True

    def focus_thread():
        while focus_active:
            try:
                if win32gui.IsWindow(hwnd) and win32gui.GetForegroundWindow() != hwnd:
                    if not set_window_focus(hwnd):
                        simulate_focus(hwnd)
            except pywintypes.error as e:
                print(f"Error maintaining focus: {e}")
                break
            time.sleep(0.1)

    # Start focus enforcement thread
    thread = threading.Thread(target=focus_thread)
    thread.start()

    # Send commands
    def copy_and_paste(text):
        pyperclip.copy(text)
        keyboard.press_and_release('ctrl+v')

    try:
        for _ in range(loops):
            keyboard.press_and_release('enter')
            time.sleep(0.001)
            copy_and_paste('/t /ff')
            keyboard.press_and_release('enter')
            time.sleep(0.001)
            copy_and_paste('/t /ff')

    except Exception as e:
        print(f"Error sending commands: {e}")
    finally:
        focus_active = False  # Stop focus thread
        thread.join()
        print(f"Stopped focusing and sending commands to {app_name}.")

# Example usage: Focus Valorant and send commands
if __name__ == "__main__":
    force_focus_and_send_commands("VALORANT  ", 200)